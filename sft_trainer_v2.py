import torch
import torch.nn.functional as F

from transformers import Trainer
from transformers.trainer import (
    ###
    _is_peft_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    is_torch_xla_available,
    SaveStrategy
)
from typing import List, Optional, Dict
from utils.gem_triton_loss import GEMLoss


class SFTTrainer(Trainer):

    @torch.no_grad
    def compute_training_logs(self, logits, labels):
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        training_logs = {}
        if self.args.print_entropy:
            entropy = chunked_entropy_from_logits(
                shift_logits,
                batch_size=max(1, shift_logits.size(0) // 4),
            ).mean()
            training_logs["entropy"] = round(entropy.item(), 2)

        return training_logs

    def gem_loss(self, logits, labels, num_items_in_batch, beta=0.7, ignore_index=-100, h="logsigmoid"):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        with torch.no_grad():
            logits_on_labels = torch.gather(
                shift_logits, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            logits_diff = shift_logits - logits_on_labels.unsqueeze(-1)
            if h == "linear":
                weights = torch.ones_like(logits_diff)
            elif h == "logsigmoid":
                weights = F.sigmoid(0.01 * logits_diff)
            else:
                raise ValueError(h)

        gene_log_probs = F.log_softmax(shift_logits, dim=-1)
        q_probs = torch.exp(F.log_softmax(shift_logits / beta, dim=-1)).detach()

        real_log_probs = torch.gather(
            gene_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        )

        if num_items_in_batch is not None:
            loss = -torch.sum(
                q_probs * weights * (real_log_probs - gene_log_probs), dim=-1
            ).sum() / num_items_in_batch
        else:
            loss = -torch.sum(
                q_probs * weights * (real_log_probs - gene_log_probs), dim=-1
            ).mean()

        return loss

    def gem_loss_triton(self, logits, labels, num_items_in_batch, beta=0.7, ignore_index=-100, h="linear"):
        assert h == "linear", "Only linear is supported for gem_loss_triton for now."

        if num_items_in_batch is not None:
            gem_loss_func = GEMLoss(beta=beta, ignore_index=ignore_index, reduction="none")
        else:
            gem_loss_func = GEMLoss(beta=beta, ignore_index=ignore_index, reduction="mean")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        loss = gem_loss_func(shift_logits, shift_labels)

        if num_items_in_batch is not None:
            loss = loss.sum() / num_items_in_batch
        else:
            loss = loss
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        compute_loss_func = getattr(self, "compute_loss_func", None)
        model_accepts_loss_kwargs = getattr(self, "model_accepts_loss_kwargs", False)
        if (self.label_smoother is not None or compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        past_index = getattr(self.args, "past_index", -1)
        if past_index >= 0:
            self._past = outputs[past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            if self.args.loss == "ce" or self.control.should_evaluate:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            elif self.args.loss == "gem":
                loss = self.gem_loss(
                    outputs.logits, 
                    inputs["labels"],
                    num_items_in_batch=num_items_in_batch, 
                    beta=self.args.gem_beta, 
                    h=self.args.gem_h
                )
            elif self.args.loss == "gem_triton":
                loss = self.gem_loss_triton(
                    outputs.logits, 
                    inputs["labels"],
                    num_items_in_batch=num_items_in_batch, 
                    beta=self.args.gem_beta, 
                    h=self.args.gem_h
                )

        if getattr(self.args, "average_tokens_across_devices", False) and model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        # ziniu add logs
        if not self.control.should_evaluate:
            self.training_logs = self.compute_training_logs(
                outputs.logits, inputs["labels"]
            )
            self.training_logs["ce_loss"] = (
                outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            )
            self.training_logs["ce_loss"] = round(self.training_logs["ce_loss"].item(), 4)

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time=None,
        **kwargs,
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # Newer Trainer internals can rename/remove private helpers like
            # `_nested_gather`. Fall back to the local process loss so logging
            # stays functional instead of aborting training.
            nested_gather = getattr(self, "_nested_gather", None)
            if callable(nested_gather):
                tr_loss_scalar = nested_gather(tr_loss).mean().item()
            else:
                tr_loss_scalar = tr_loss.detach().float().mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()
            if getattr(self, "training_logs", None):
                logs.update(self.training_logs)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            learning_rate = kwargs.get("learning_rate")
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            if start_time is not None:
                self.log(logs, start_time)
            else:
                self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


def chunked_entropy_from_logits(chunk_logits, batch_size=None):
    """
    Compute entropy from logits in a memory-efficient manner by introducing a batch_size parameter.

    Args:
        chunk_logits (torch.Tensor): Logits tensor of shape (total_samples, num_classes).
        batch_size (int): Number of samples to process per batch.

    Returns:
        torch.Tensor: Entropy tensor of shape (total_samples,).
    """
    total_samples, num_classes = chunk_logits.shape
    entropy_list = []
    if batch_size is None:
        batch_size = total_samples

    # Process logits in batches
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        logits_batch = chunk_logits[start_idx:end_idx]  # Get a batch of logits

        # Compute logsumexp for the current batch
        logsumexp_batch = torch.logsumexp(logits_batch, dim=-1, keepdim=False)  # Shape: (batch_size,)
        # Compute probabilities in log-space without computing softmax
        normalized_logits = logits_batch - logsumexp_batch.unsqueeze(-1)       # Shape: (batch_size, num_classes)
        exp_normalized_logits = torch.exp(normalized_logits)                   # Shape: (batch_size, num_classes)
        # Compute entropy for the batch
        entropy_batch = logsumexp_batch - (logits_batch * exp_normalized_logits).sum(dim=-1)  # Shape: (batch_size,)

        entropy_list.append(entropy_batch)  # Store entropy for the current batch

    # Concatenate results from all batches
    if len(entropy_list) > 0:
        return torch.cat(entropy_list, dim=0)
    else:
        return torch.tensor(0.0)