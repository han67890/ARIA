#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
# Modified for ARIA in 2026.
# Copyright (C) 2026 Yiheng Han (ARIA modifications only).
#

"""
CLaRa SFT Trainer

This module provides the supervised fine-tuning trainer for CLaRa models.
"""

import os
import re
import string
import unicodedata
from abc import ABC
from collections import Counter
from typing import Dict, Any, Optional, List

import torch
from torch.optim import Optimizer
from tqdm import tqdm

# from openrlhf.models import SFTLoss
from openrlhf.utils.distributed_sampler import DistributedSampler

# Set torch print options for better debugging

_ARIA_AUXILIARY_LOSSES = {
    "mse_loss": "lambda_mse",
    "cfrs_loss": "lambda_cfrs",
    "qr_loss": "lambda_qr",
    "mtfrl_loss": "lambda_mtfrl",
}


def _scalar_loss_tensor(value: Any, reference: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize a model loss output without detaching its autograd graph."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar tensor, got shape {tuple(value.shape)}")
        result = value.reshape(())
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result = reference.new_tensor(float(value))
    else:
        raise TypeError(f"{name} must be a scalar tensor or number")
    return result


def compose_aria_training_loss(
    qa_loss: torch.Tensor,
    outputs: Dict[str, Any],
    args: Any,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compose Eq. (3) and expose every raw/weighted term for logging.

    The model returns *unweighted* scalar losses. Full ARIA fails closed when a
    term is absent. Separately trained ablations and the matched CLaRa control
    may omit disabled terms; those are represented by exact zero scalars.
    """
    qa_loss = _scalar_loss_tensor(qa_loss, qa_loss, "qa_loss")
    terms: Dict[str, torch.Tensor] = {"qa_loss": qa_loss}
    total_loss = qa_loss
    phase2 = getattr(args, "stage", None) == "stage2"
    strict_full = phase2 and getattr(args, "rag_configuration", "full") == "full"

    for output_name, weight_name in _ARIA_AUXILIARY_LOSSES.items():
        if phase2 and strict_full and output_name not in outputs:
            raise KeyError(
                f"Full ARIA Phase II requires model output {output_name!r} "
                "as an unweighted scalar"
            )
        raw_value = outputs.get(output_name, qa_loss.new_zeros(())) if phase2 else 0.0
        raw_loss = _scalar_loss_tensor(raw_value, qa_loss, output_name)
        weight = float(getattr(args, weight_name, 0.0)) if phase2 else 0.0
        weighted_loss = raw_loss * weight
        terms[output_name] = raw_loss
        terms[f"weighted_{output_name}"] = weighted_loss
        total_loss = total_loss + weighted_loss

    terms["total_loss"] = total_loss
    return total_loss, terms


class EvaluationMetrics:
    """Utility class for evaluation metrics."""

    _ARTICLE_RE = re.compile(
        r"(?<![\w-])(?<!\w['’])(?:a|an|the)(?![\w-])(?!['’]\w)"
    )
    _ASCII_PUNCT = "".join(ch for ch in string.punctuation if ch not in {"'", "-"})
    _PUNCT_TABLE = str.maketrans("", "", _ASCII_PUNCT)

    @staticmethod
    def bool_mapping(text: str) -> str:
        """Map boolean values to yes/no format."""
        mapping = {"True": "yes", "False": "no"}
        return mapping.get(text, text)

    @staticmethod
    def normalize_answer(text: str) -> str:
        """Appendix A.35: NFKC, lower, articles, selected ASCII punctuation, WS."""
        normalized = unicodedata.normalize("NFKC", "" if text is None else str(text))
        normalized = normalized.lower()
        normalized = EvaluationMetrics._ARTICLE_RE.sub(" ", normalized)
        normalized = normalized.translate(EvaluationMetrics._PUNCT_TABLE)
        return " ".join(normalized.split())

    @staticmethod
    def _gold_answers(ground_truth: Any) -> List[str]:
        if isinstance(ground_truth, str):
            return [ground_truth]
        if isinstance(ground_truth, (list, tuple)):
            return [str(value) for value in ground_truth]
        return [str(ground_truth)]

    @classmethod
    def exact_match_score(cls, prediction: str, ground_truth: str) -> bool:
        """Calculate exact match score."""
        pred_norm = cls.normalize_answer(cls.bool_mapping(prediction))
        gt_norm = cls.normalize_answer(cls.bool_mapping(ground_truth))
        return pred_norm == gt_norm

    @classmethod
    def cover_exact_match_score(cls, prediction: str, ground_truth: Any) -> bool:
        """Appendix A.35 contains-exact-match against the best gold answer."""
        pred_norm = cls.normalize_answer(prediction)
        return any(
            (gold_norm in pred_norm) if gold_norm else (pred_norm == gold_norm)
            for gold_norm in (
                cls.normalize_answer(gold) for gold in cls._gold_answers(ground_truth)
            )
        )

    @classmethod
    def f1_score(cls, prediction: str, ground_truth: str) -> float:
        """Calculate F1 score between prediction and ground truth."""
        pred_norm = cls.normalize_answer(cls.bool_mapping(prediction))
        gt_norm = cls.normalize_answer(cls.bool_mapping(ground_truth))

        # Handle special cases for yes/no/noanswer
        if pred_norm in ["yes", "no", "noanswer"] and pred_norm != gt_norm:
            return 0.0
        if gt_norm in ["yes", "no", "noanswer"] and pred_norm != gt_norm:
            return 0.0

        pred_tokens = pred_norm.split()
        gt_tokens = gt_norm.split()

        # Calculate common tokens
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        # Calculate precision and recall
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)

        # Calculate F1
        return (2 * precision * recall) / (precision + recall)

    @staticmethod
    def extract_answers(text: str) -> List[str]:
        """Extract answers from text using regex."""
        return [ans.strip() for ans in re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)]


class RetrievalMetrics:
    """Utility class for retrieval evaluation metrics."""

    @staticmethod
    def calculate_recall_and_precision(pos_indices: List[List[Any]],
                                     pred_indices: List[List[Any]],
                                     k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """Calculate recall and precision at different k values."""
        metrics = {f"recall_{k}": 0.0 for k in k_values}
        metrics.update({f"precision_{k}": 0.0 for k in k_values})

        valid_samples = 0

        for gold_pos, pred_pos in zip(pos_indices, pred_indices):
            if not gold_pos:  # Skip samples with no positive indices
                continue

            valid_samples += 1
            gold_set = set(gold_pos)

            for k in k_values:
                pred_k = set(pred_pos[:k])
                hits = len(gold_set & pred_k)

                # Recall: hits / total_positives
                metrics[f"recall_{k}"] += hits / len(gold_set)

                # Precision: hits / k
                metrics[f"precision_{k}"] += hits / k

        # Average across valid samples
        if valid_samples > 0:
            for key in metrics:
                metrics[key] /= valid_samples

        return metrics, valid_samples


class SFTTrainer(ABC):
    """
    Trainer for CLaRa supervised fine-tuning (SFT).

    This trainer handles multi-stage training for CLaRa models including:
    - Stage 1: Document compression training
    - Stage 2: Retrieval and generation training
    - Stage 2 Reasoning: Multi-step reasoning training
    """

    def __init__(self,
                 model,
                 strategy,
                 optim: Optimizer,
                 train_dataloader,
                 eval_dataloader,
                 scheduler,
                 max_norm: float = 1.0,
                 pretrain_mode: bool = False,
                 batch_size: int = 1,
                 max_epochs: int = 2,
                 tokenizer=None,
                 save_hf_ckpt: bool = False,
                 disable_ds_ckpt: bool = False) -> None:
        """
        Initialize the SFT trainer.

        Args:
            model: CLaRa model to train
            strategy: Training strategy (distributed, etc.)
            optim: Optimizer for training
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            scheduler: Learning rate scheduler
            max_norm: Maximum gradient norm for clipping
            pretrain_mode: Whether in pretraining mode
            batch_size: Training batch size
            max_epochs: Maximum number of training epochs
            tokenizer: Tokenizer for the model
            save_hf_ckpt: Whether to save HuggingFace format checkpoints
            disable_ds_ckpt: Whether to disable DeepSpeed checkpoints
        """
        super().__init__()

        # Core components
        self.model = model
        self.strategy = strategy
        self.optimizer = optim
        self.scheduler = scheduler
        self.tokenizer = tokenizer

        # Training configuration
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.pretrain_mode = pretrain_mode

        # Data loaders
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Checkpointing
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt

        # Training arguments
        self.args = strategy.args

        # Model-specific settings
        # self.aux_loss = self.args.aux_loss_coef > 1e-8  # For Mixtral 8x7b
       # self.packing_samples = strategy.args.packing_samples

        # Initialize logging
        self._wandb = None
        self._tensorboard = None
        self._setup_logging()

    def _setup_logging(self):
        """Setup Wandb and TensorBoard logging."""
        # Setup Wandb
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=self.strategy.args.use_wandb)

            wandb.init(
                entity=self.strategy.args.wandb_org,
                project=self.strategy.args.wandb_project,
                group=self.strategy.args.wandb_group,
                name=self.strategy.args.wandb_run_name,
                config=self.strategy.args.__dict__,
                reinit=True,
            )

            # Define metrics
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Setup TensorBoard if Wandb is not available
        if (self.strategy.args.use_tensorboard and self._wandb is None and
            self.strategy.is_rank_0()):
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(
                self.strategy.args.use_tensorboard,
                self.strategy.args.wandb_run_name
            )
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, args, consumed_samples: int = 0, num_update_steps_per_epoch: Optional[int] = None):
        """
        Main training loop.

        Args:
            args: Training arguments
            consumed_samples: Number of samples already consumed
            num_update_steps_per_epoch: Number of update steps per epoch
        """
        # Configure evaluation and saving steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")
        elif args.save_steps == -2:
            args.save_steps = num_update_steps_per_epoch

        # Calculate starting point
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        # Initialize tracking variables
        training_metrics = self._init_training_metrics()

        # Main training loop
        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        for epoch in range(start_epoch, self.epochs):
            # A paper-protocol Phase-II artifact may expose five independently
            # sampled epoch views through a lightweight scheduled dataset.
            scheduled_dataset = getattr(self.train_dataloader, "dataset", None)
            if hasattr(scheduled_dataset, "set_epoch"):
                scheduled_dataset.set_epoch(epoch)

            # Set epoch for distributed sampler
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            epoch_micro_steps = len(self.train_dataloader)
            accumulation_steps = self.strategy.accumulated_gradient
            if epoch_micro_steps % accumulation_steps != 0:
                raise RuntimeError(
                    "Training epoch would end inside a gradient-accumulation cycle: "
                    f"{epoch_micro_steps} micro-steps vs {accumulation_steps} "
                    "accumulation steps"
                )

            step_bar = tqdm(
                range(epoch_micro_steps),
                desc=f"Train step of epoch {epoch}",
                disable=not self.strategy.is_rank_0(),
            )

            # Training loop for this epoch
            self.model.train()

            for batch in self.train_dataloader:
                if self.args.stage == "stage2":
                    base_occurrences = batch.get("sample_occurrence_ids")
                    if not isinstance(base_occurrences, (list, tuple)):
                        raise ValueError(
                            "Phase-II batches require stable sample occurrence IDs"
                        )
                    # A dataset row is encountered once per scheduled epoch.
                    # Add the replayable training position so repeated epochs
                    # receive independent run-seeded support draws, while a
                    # checkpoint resume reconstructs the exact same draw.
                    batch = dict(batch)
                    batch["sample_occurrence_ids"] = [
                        f"{value}\0epoch={epoch}\0step={step}\0row={row_index}"
                        for row_index, value in enumerate(base_occurrences)
                    ]
                # Forward pass
                loss, outputs = self.model(
                    batch=batch,
                    stage2_mips=self.args.stage2_mips,
                    stage2_retrieval_top_n=self.args.stage2_retrieval_top_n
                )

                # Eq. (3): QA plus all four explicitly weighted Phase-II terms.
                total_loss, loss_terms = compose_aria_training_loss(
                    loss, outputs, args
                )

                # Backward pass
                self.strategy.backward(total_loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # Update metrics
                step_metrics = self._calculate_step_metrics(batch, outputs)
                training_metrics = self._update_training_metrics(
                    training_metrics, step_metrics, loss_terms
                )

                # Log and save
                if step % self.strategy.accumulated_gradient == 0:
                    self._process_accumulated_step(
                        args, step, training_metrics, step_bar, num_update_steps_per_epoch
                    )
                    training_metrics = self._reset_training_metrics()

                step += 1
                step_bar.update()

            epoch_bar.update()

        # Cleanup
        self._cleanup_logging()

    def _init_training_metrics(self) -> Dict[str, float]:
        """Initialize training metrics tracking."""
        return {
            "loss_sum": 0.0,
            "qa_loss_sum": 0.0,
            "mse_loss_sum": 0.0,
            "cfrs_loss_sum": 0.0,
            "qr_loss_sum": 0.0,
            "mtfrl_loss_sum": 0.0,
            "weighted_mse_loss_sum": 0.0,
            "weighted_cfrs_loss_sum": 0.0,
            "weighted_qr_loss_sum": 0.0,
            "weighted_mtfrl_loss_sum": 0.0,
            "retrieval_recall_1": 0.0,
            "retrieval_recall_3": 0.0,
            "retrieval_recall_5": 0.0,
            "retrieval_precision_1": 0.0,
            "retrieval_precision_3": 0.0,
            "retrieval_precision_5": 0.0,
            "retrieval_samples": 0.0,
        }

    def _reset_training_metrics(self) -> Dict[str, float]:
        """Reset training metrics for next accumulation cycle."""
        return self._init_training_metrics()

    def _calculate_step_metrics(self, batch: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, float]:
        """Calculate metrics for a single training step."""
        step_metrics = {
            "retrieval_recall_1": 0.0,
            "retrieval_recall_3": 0.0,
            "retrieval_recall_5": 0.0,
            "retrieval_precision_1": 0.0,
            "retrieval_precision_3": 0.0,
            "retrieval_precision_5": 0.0,
            "retrieval_samples": 0.0,
        }

        # Calculate retrieval metrics for stage2
        if self.args.stage == "stage2" and "topk_doc_ids" in outputs:
            pos_indices = batch["gold_doc_ids"]
            pred_indices = outputs["topk_doc_ids"]

            metrics, valid_samples = RetrievalMetrics.calculate_recall_and_precision(
                pos_indices, pred_indices
            )
            # The helper returns a batch mean; accumulate metric sums so the
            # later division by total valid samples happens exactly once.
            step_metrics.update(
                {key: value * valid_samples for key, value in metrics.items()}
            )
            step_metrics["retrieval_samples"] = valid_samples
        else:
            step_metrics["retrieval_samples"] = len(batch["questions"])

        return step_metrics

    def _update_training_metrics(self,
                                training_metrics: Dict[str, float],
                                step_metrics: Dict[str, float],
                                loss_terms: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update accumulated training metrics."""
        training_metrics["loss_sum"] += loss_terms["total_loss"].detach().item()
        for name in (
            "qa_loss",
            "mse_loss",
            "cfrs_loss",
            "qr_loss",
            "mtfrl_loss",
            "weighted_mse_loss",
            "weighted_cfrs_loss",
            "weighted_qr_loss",
            "weighted_mtfrl_loss",
        ):
            training_metrics[f"{name}_sum"] += loss_terms[name].detach().item()

        for key in step_metrics:
            if key in training_metrics:
                training_metrics[key] += step_metrics[key]

        return training_metrics

    def _process_accumulated_step(self,
                                 args,
                                 step: int,
                                 training_metrics: Dict[str, float],
                                 step_bar: tqdm,
                                 num_update_steps_per_epoch: int):
        """Process accumulated gradients and log/save."""
        # Calculate averaged metrics
        logs_dict = {
            "loss": training_metrics["loss_sum"] / self.strategy.accumulated_gradient,
            "total_loss": training_metrics["loss_sum"] / self.strategy.accumulated_gradient,
            "lr": self.scheduler.get_last_lr()[0],
        }
        for name in (
            "qa_loss",
            "mse_loss",
            "cfrs_loss",
            "qr_loss",
            "mtfrl_loss",
            "weighted_mse_loss",
            "weighted_cfrs_loss",
            "weighted_qr_loss",
            "weighted_mtfrl_loss",
        ):
            logs_dict[name] = (
                training_metrics[f"{name}_sum"]
                / self.strategy.accumulated_gradient
            )

        # Add retrieval metrics
        if training_metrics["retrieval_samples"] > 0:
            for metric in ["retrieval_recall_1", "retrieval_recall_3", "retrieval_recall_5",
                          "retrieval_precision_1", "retrieval_precision_3", "retrieval_precision_5"]:
                logs_dict[metric] = training_metrics[metric] / training_metrics["retrieval_samples"]

        # Aggregate across processes
        logs_dict = self.strategy.all_reduce(logs_dict, op="mean")

        # Update progress bar
        step_bar.set_postfix(logs_dict)

        # Global step for logging
        global_step = step // self.strategy.accumulated_gradient
        client_states = {"consumed_samples": global_step * args.train_batch_size}

        # Save logs and checkpoints
        self.save_logs_and_checkpoints(
            args, global_step, step_bar, logs_dict, client_states, num_update_steps_per_epoch
        )

    def save_logs_and_checkpoints(self,
                                 args,
                                 global_step: int,
                                 step_bar: tqdm,
                                 logs_dict: Dict[str, float] = None,
                                 client_states: Dict[str, Any] = None,
                                 num_update_steps_per_epoch: Optional[int] = None):
        """Save logs and checkpoints."""
        logs_dict = logs_dict or {}
        client_states = client_states or {}

        # Logging
        if global_step % args.logging_steps == 0:
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # Checkpointing
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model,
                    args.ckpt_path,
                    tag=tag,
                    max_num=args.max_ckpt_num,
                    max_mem=args.max_ckpt_mem,
                    client_state=client_states,
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

        # Evaluation
        if global_step % args.eval_steps == 0:
            self._run_evaluation(args, global_step, num_update_steps_per_epoch)

    def _run_evaluation(self, args, global_step: int, num_update_steps_per_epoch: Optional[int]):
        """Run evaluation based on schedule."""
        if self.eval_dataloader is None or len(self.eval_dataloader) == 0:
            return

        print("Starting evaluation")

        # Determine if we should do generation evaluation
        eval_gen = False
        if global_step % (args.eval_steps * 5) == 0 and args.do_eval_gen:
            eval_gen = True
        elif (num_update_steps_per_epoch and
              global_step % num_update_steps_per_epoch == 0 and
              args.do_eval_gen):
            eval_gen = False  # Loss-only eval at epoch end

        self.evaluate(self.eval_dataloader, global_step, eval_gen=eval_gen)

    def evaluate(self, eval_dataloader, steps: int = 0, eval_gen: bool = False):
        """
        Evaluate the model on the evaluation dataset.

        Args:
            eval_dataloader: Evaluation data loader
            steps: Current training step
            eval_gen: Whether to perform generation evaluation
        """
        print(f"Starting evaluation at step {steps}")
        self.model.eval()

        # Initialize evaluation metrics
        eval_metrics = {
            "loss_sum": 0.0,
            "qa_loss_sum": 0.0,
            "mse_loss_sum": 0.0,
            "cfrs_loss_sum": 0.0,
            "qr_loss_sum": 0.0,
            "mtfrl_loss_sum": 0.0,
            "weighted_mse_loss_sum": 0.0,
            "weighted_cfrs_loss_sum": 0.0,
            "weighted_qr_loss_sum": 0.0,
            "weighted_mtfrl_loss_sum": 0.0,
            "samples": 0,
            "correct": 0,
            "retrieval_recall_1": 0.0,
            "retrieval_recall_3": 0.0,
            "retrieval_recall_5": 0.0,
            "retrieval_precision_1": 0.0,
            "retrieval_precision_3": 0.0,
            "retrieval_precision_5": 0.0,
            "retrieval_samples": 0.0,
        }

        with torch.no_grad():
            step_bar = tqdm(
                range(len(eval_dataloader)),
                desc=f"Eval stage of steps {steps}",
                disable=not self.strategy.is_rank_0(),
            )

            for batch in eval_dataloader:
                # Forward pass
                loss, outputs = self.model(batch=batch)
                _, loss_terms = compose_aria_training_loss(loss, outputs, self.args)

                # Basic metrics
                batch_size = len(batch["answers"])
                eval_metrics["loss_sum"] += (
                    loss_terms["total_loss"].detach().item() * batch_size
                )
                for name in (
                    "qa_loss",
                    "mse_loss",
                    "cfrs_loss",
                    "qr_loss",
                    "mtfrl_loss",
                    "weighted_mse_loss",
                    "weighted_cfrs_loss",
                    "weighted_qr_loss",
                    "weighted_mtfrl_loss",
                ):
                    eval_metrics[f"{name}_sum"] += (
                        loss_terms[name].detach().item() * batch_size
                    )
                eval_metrics["samples"] += batch_size

                # Retrieval metrics
                if self.args.stage == "stage2" and "topk_doc_ids" in outputs:
                    retrieval_metrics, valid_samples = RetrievalMetrics.calculate_recall_and_precision(
                        batch["gold_doc_ids"], outputs["topk_doc_ids"]
                    )

                    for key, value in retrieval_metrics.items():
                        eval_metrics[key] += value * valid_samples
                    eval_metrics["retrieval_samples"] += valid_samples
                else:
                    eval_metrics["retrieval_samples"] += batch_size

                # Generation evaluation
                if eval_gen:
                    predictions = self._generate_predictions(batch)
                    correct = self._calculate_accuracy(
                        predictions, batch.get("gold_answers", batch["answers"])
                    )
                    eval_metrics["correct"] += correct

                step_bar.update()

        # Aggregate metrics across processes
        eval_metrics = self.strategy.all_reduce(eval_metrics, op="sum")

        # Calculate final metrics
        final_metrics = self._calculate_final_eval_metrics(eval_metrics, eval_gen)

        # Log evaluation results
        self._log_evaluation_results(final_metrics, steps)

        # Update progress bar
        step_bar.set_postfix(final_metrics)

        self.model.train()  # Reset to training mode

    def _generate_predictions(self, batch: Dict[str, Any]) -> List[str]:
        """Generate predictions for evaluation."""
        questions = batch["questions"]
        docs = batch["docs"]
        answers = batch["answers"]

        if self.args.stage in ["stage1", "stage1_2"]:
            return self.model.generate_from_text(
                questions=questions,
                documents=docs,
                max_new_tokens=64
            )
        elif self.args.stage == "stage2":
            predictions, _ = self.model.generate_from_questions(
                questions=questions,
                max_new_tokens=64,
                stage2_mips=self.args.stage2_mips,
            )
            return predictions
        elif self.args.stage == "stage2_reasoning":
            predictions = []
            for question, answer in zip(questions, answers):
                prediction, _ = self.model.generate_from_reasoning(
                    questions=[question],
                    max_new_tokens=1024,
                    answers=[answer],
                    save_dir=self.args.save_path,
                )
                predictions.extend(prediction)
            return predictions
        else:
            return [""] * len(questions)

    def _calculate_accuracy(self, predictions: List[str], answers: List[str]) -> int:
        """Calculate accuracy for predictions."""
        correct = 0
        for pred, ans in zip(predictions, answers):
            if EvaluationMetrics.cover_exact_match_score(pred, ans):
                correct += 1
        return correct

    def _calculate_final_eval_metrics(self, eval_metrics: Dict[str, float], eval_gen: bool) -> Dict[str, float]:
        """Calculate final evaluation metrics."""
        final_metrics = {}

        # Basic metrics
        if eval_metrics["samples"] > 0:
            final_metrics["eval_loss"] = eval_metrics["loss_sum"] / eval_metrics["samples"]
            for name in (
                "qa_loss",
                "mse_loss",
                "cfrs_loss",
                "qr_loss",
                "mtfrl_loss",
                "weighted_mse_loss",
                "weighted_cfrs_loss",
                "weighted_qr_loss",
                "weighted_mtfrl_loss",
            ):
                final_metrics[f"eval_{name}"] = (
                    eval_metrics[f"{name}_sum"] / eval_metrics["samples"]
                )

            if eval_gen:
                final_metrics["eval_acc"] = eval_metrics["correct"] / eval_metrics["samples"]

        # Retrieval metrics
        if eval_metrics["retrieval_samples"] > 0:
            for metric in ["retrieval_recall_1", "retrieval_recall_3", "retrieval_recall_5",
                          "retrieval_precision_1", "retrieval_precision_3", "retrieval_precision_5"]:
                final_metrics[f"eval_{metric}"] = eval_metrics[metric] / eval_metrics["retrieval_samples"]

        return final_metrics

    def _log_evaluation_results(self, metrics: Dict[str, float], steps: int):
        """Log evaluation results to wandb/tensorboard."""
        if self.strategy.is_rank_0():
            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**metrics, "global_step": steps}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in metrics.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, steps)

    def _cleanup_logging(self):
        """Cleanup logging resources."""
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()
