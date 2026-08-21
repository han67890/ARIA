#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
# Modified for ARIA in 2026.
# Copyright (C) 2026 Yiheng Han (ARIA modifications only).
#

"""
CLaRa Dataset and Collate Functions

This module provides dataset handling and batch collation for CLaRa training.
"""

import hashlib
import re
import torch
from typing import Callable, List, Tuple, Dict, Any, Optional
from collections import defaultdict
from torch.utils.data import Dataset


PHASE1_DATA_TYPES = {
    "simple_qa",
    "complex_qa",
    "paraphrase",
    "entity_augmented",
}


def _answer_to_text(answer: Any, *, location: str) -> str:
    """Normalize one supervised target to ARIA's canonical text form.

    Canonical artifacts store ``answer`` as a non-empty string. A one-element
    string sequence is normalized equivalently, preserving exactly one target
    for every training row.
    """

    if isinstance(answer, str):
        if not answer.strip():
            raise ValueError(f"{location} must be a non-empty string")
        return answer
    if isinstance(answer, (list, tuple)):
        if len(answer) != 1 or not isinstance(answer[0], str) or not answer[0].strip():
            raise ValueError(
                f"{location} sequence form must contain exactly one non-empty string"
            )
        return answer[0]
    raise ValueError(f"{location} must be a string or one-element string sequence")


def _validate_pos_index(
    pos_index: Any,
    *,
    n_docs: int,
    location: str,
    allow_empty: bool = False,
) -> List[int]:
    if not isinstance(pos_index, (list, tuple)):
        raise ValueError(f"{location} must be a list of document indices")
    if not pos_index and not allow_empty:
        raise ValueError(f"{location} must be a non-empty list of document indices")
    normalized: List[int] = []
    seen = set()
    for position, index in enumerate(pos_index):
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{location}[{position}] must be an integer")
        if index < 0 or index >= n_docs:
            raise ValueError(
                f"{location}[{position}]={index} is outside the candidate range [0, {n_docs})"
            )
        if index in seen:
            raise ValueError(f"{location} contains duplicate index {index}")
        seen.add(index)
        normalized.append(index)
    return normalized


def _validate_document_batch(
    docs_list: List[List[str]],
    pos_indices: List[List[int]],
    *,
    expected_docs: int,
    stage: str,
    allow_empty_positives: bool = False,
) -> None:
    if expected_docs < 1:
        raise ValueError(f"{stage} expected document count must be positive, got {expected_docs}")
    for row_index, (docs, pos_index) in enumerate(zip(docs_list, pos_indices)):
        if not isinstance(docs, (list, tuple)):
            raise ValueError(f"{stage} row {row_index}.docs must be a list")
        if len(docs) != expected_docs:
            raise ValueError(
                f"{stage} row {row_index} has {len(docs)} documents; expected exactly "
                f"{expected_docs} for the fixed-shape candidate protocol."
            )
        for doc_index, document in enumerate(docs):
            if not isinstance(document, str) or not document.strip():
                raise ValueError(
                    f"{stage} row {row_index}.docs[{doc_index}] must be a non-empty string"
                )
        _validate_pos_index(
            pos_index,
            n_docs=len(docs),
            location=f"{stage} row {row_index}.pos_index",
            allow_empty=allow_empty_positives,
        )


def make_collate_fn(
    clara_model,
    enc_max_len: Optional[int] = None,
    dec_max_len: Optional[int] = None,
    qa_loss: bool = False,
    *,
    passage_max_len: Optional[int] = None,
    query_max_len: int = 256,
    input_max_len: Optional[int] = None,
    target_max_len: int = 128,
):
    """
    Create a collate function for CLaRa training.

    Args:
        clara_model: CLaRa model instance
        passage_max_len: Maximum passage/compressor source length
        query_max_len: Maximum independent Query Reasoner length
        input_max_len: Maximum decoder prompt/input length
        target_max_len: Maximum supervised answer/target length
        enc_max_len/dec_max_len: Deprecated aliases retained for callers outside
            the paper launchers. ``dec_max_len`` maps to ``input_max_len``.
        qa_loss: Whether to use QA loss for joint training

    Returns:
        Collate function that processes batches for training
    """
    tokenizer = clara_model.decoder_tokenizer
    generation_top_k = clara_model.generation_top_k
    passage_max_len = (
        passage_max_len
        if passage_max_len is not None
        else (enc_max_len if enc_max_len is not None else 768)
    )
    input_max_len = (
        input_max_len
        if input_max_len is not None
        else (dec_max_len if dec_max_len is not None else 1024)
    )
    for name, value in (
        ("passage_max_len", passage_max_len),
        ("query_max_len", query_max_len),
        ("input_max_len", input_max_len),
        ("target_max_len", target_max_len),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    # Legacy reasoning-only code below still uses these local names. The two
    # paper phases use the independent limits above.
    enc_max_len = passage_max_len
    dec_max_len = input_max_len + target_max_len

    def _tokenize_supervised_rows(
        instructions: List[str],
        prompt_lengths: List[int],
        *,
        return_offsets: bool,
        stage: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], Optional[torch.Tensor]]:
        """Truncate prompt and target independently, then pad without re-tokenizing."""
        if len(instructions) != len(prompt_lengths):
            raise ValueError(f"{stage} instructions and prompt lengths are misaligned")

        rows: List[List[int]] = []
        row_offsets: List[List[Tuple[int, int]]] = []
        cropped_prompt_lengths: List[int] = []
        for row_index, (instruction, prompt_length) in enumerate(
            zip(instructions, prompt_lengths)
        ):
            encoded = tokenizer(
                instruction,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=return_offsets,
            )
            token_ids = list(encoded["input_ids"])
            if not 0 < prompt_length < len(token_ids):
                raise ValueError(
                    f"{stage} row {row_index} has invalid prompt/target boundary "
                    f"{prompt_length} for {len(token_ids)} tokens"
                )
            prompt_end = min(prompt_length, input_max_len)
            target_end = min(len(token_ids), prompt_length + target_max_len)
            target_ids = token_ids[prompt_length:target_end]
            if not target_ids:
                raise ValueError(f"{stage} row {row_index} has an empty target")
            rows.append(token_ids[:prompt_end] + target_ids)
            cropped_prompt_lengths.append(prompt_end)
            if return_offsets:
                offsets = [tuple(value) for value in encoded["offset_mapping"]]
                row_offsets.append(offsets[:prompt_end] + offsets[prompt_length:target_end])

        width = max(len(row) for row in rows)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("ARIA supervised collation requires tokenizer.pad_token_id")
        input_ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        padded_offsets = (
            torch.zeros((len(rows), width, 2), dtype=torch.long)
            if return_offsets
            else None
        )
        left_padding = getattr(tokenizer, "padding_side", "right") == "left"
        for row_index, row in enumerate(rows):
            start = width - len(row) if left_padding else 0
            end = start + len(row)
            input_ids[row_index, start:end] = torch.tensor(row, dtype=torch.long)
            attention_mask[row_index, start:end] = 1
            if padded_offsets is not None:
                padded_offsets[row_index, start:end] = torch.tensor(
                    row_offsets[row_index], dtype=torch.long
                )
        return input_ids, attention_mask, cropped_prompt_lengths, padded_offsets

    def _make_labels(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: List[int],
        *,
        stage: str,
    ) -> torch.Tensor:
        """Build answer-only labels and keep all padding outside the CE loss."""

        labels = input_ids.clone()
        labels.masked_fill_(~attention_mask.bool(), -100)
        labels = _mask_prompt(
            labels,
            attention_mask,
            prompt_lengths,
            tokenizer.pad_token_id,
        )
        for row_index in range(labels.size(0)):
            if not torch.any(labels[row_index] != -100):
                raise ValueError(
                    f"{stage} row {row_index} has no supervised answer token after truncation; "
                    f"target_max_len={target_max_len}, input_max_len={input_max_len}"
                )
        return labels

    def _mask_prompt(labels: torch.Tensor,
                     attention_mask: torch.Tensor,
                     prompt_lengths: List[int],
                     pad_token_id: int) -> torch.Tensor:
        """Mask prompt tokens in labels to only compute loss on answer tokens."""
        for i, prompt_len in enumerate(prompt_lengths):
            attn = attention_mask[i]
            valid_positions = attn.nonzero(as_tuple=True)[0]

            if len(valid_positions) == 0:
                continue

            first_valid = valid_positions[0].item()
            last_valid_plus1 = valid_positions[-1].item() + 1
            end_pos = min(first_valid + prompt_len, last_valid_plus1)
            labels[i, :end_pos] = -100

        return labels

    def _find_subsequence(sequence: List[int], pattern: List[int], start: int = 0) -> int:
        """Find subsequence pattern in sequence starting from start index."""
        if not pattern:
            return -1

        n, m = len(sequence), len(pattern)
        for i in range(start, n - m + 1):
            if sequence[i:i+m] == pattern:
                return i
        return -1

    def _mask_information_spans(labels: torch.Tensor,
                               input_ids: torch.Tensor,
                               attention_mask: torch.Tensor,
                               tokenizer) -> torch.Tensor:
        """Mask <information>...</information> spans in labels."""
        open_pattern = tokenizer.encode("<information>", add_special_tokens=False)
        close_pattern = tokenizer.encode("</information>", add_special_tokens=False)

        B, L = input_ids.size()

        for i in range(B):
            ids = input_ids[i].tolist()

            # Find valid token range
            valid_positions = attention_mask[i].nonzero(as_tuple=True)[0]
            if len(valid_positions) == 0:
                continue
            last_valid = valid_positions[-1].item()

            # Find first non-masked position
            non_masked_positions = (labels[i] != -100).nonzero(as_tuple=True)
            if len(non_masked_positions[0]) == 0:
                continue
            pos = non_masked_positions[0][0].item()

            # Find and mask information spans
            while True:
                start = _find_subsequence(ids, open_pattern, pos)
                if start == -1 or start > last_valid:
                    break

                end = _find_subsequence(ids, close_pattern, start + len(open_pattern))
                if end == -1 or end > last_valid:
                    # No closing tag found, mask to end
                    labels[i, start:last_valid+1] = -100
                    break
                else:
                    # Mask the entire span including tags
                    end_inclusive = end + len(close_pattern) - 1
                    end_inclusive = min(end_inclusive, last_valid)
                    labels[i, start:end_inclusive+1] = -100
                    pos = end_inclusive + 1

        return labels

    def _process_stage1_batch(batch_data: Tuple) -> Dict[str, Any]:
        """Process batch for stage 1 training."""
        (
            docs_list, questions, answers, data_types, pos_indices,
            gold_doc_ids, gold_answers, occurrence_ids,
        ) = batch_data
        B = len(questions)

        # Flatten documents for encoding
        flat_docs = [doc for doc_list in docs_list for doc in doc_list]

        # Prepare encoder inputs
        enc_inputs = clara_model._prepare_encoder_inputs(
            flat_docs, max_length=passage_max_len
        )
        enc_input_ids = enc_inputs["input_ids"]
        enc_attention_mask = enc_inputs["attention_mask"]
        memory_token_counts = enc_inputs.get("memory_token_counts")
        if memory_token_counts is None or memory_token_counts.numel() != B:
            raise ValueError(
                "Phase-I compressor inputs require one real memory-token count per row"
            )

        if enc_input_ids.size(0) != B:
            raise ValueError(
                "Phase-I Eq. (2) requires one input document per (d, y) pair; "
                f"encoder produced {enc_input_ids.size(0)} rows for batch size {B}"
            )

        # Prepare decoder inputs
        prompt_responses = []
        for row_index, (q, a, data_type) in enumerate(
            zip(questions, answers, data_types)
        ):
            row_memory_counts = [int(memory_token_counts[row_index].item())]
            answer_text = _answer_to_text(
                a, location=f"Phase-I {data_type} paraphrase target"
            )
            # All four source families keep their released target y, but the
            # decoder sees no source instruction/question: Eq. (2) conditions
            # the reconstruction only on F(d).
            prompt_responses.append(
                clara_model._blend_prompt_and_memory_tokens(
                    query="",
                    answer=answer_text,
                    stage="stage1_2",
                    memory_counts=row_memory_counts,
                )
            )

        prompt_lengths = [pr[0] for pr in prompt_responses]
        instructions = [pr[1] for pr in prompt_responses]

        (
            dec_input_ids,
            dec_attention_mask,
            prompt_lengths,
            _,
        ) = _tokenize_supervised_rows(
            instructions,
            prompt_lengths,
            return_offsets=False,
            stage="Phase-I",
        )

        labels = _make_labels(
            dec_input_ids,
            dec_attention_mask,
            prompt_lengths,
            stage="Phase-I",
        )

        return {
            "stage": clara_model.training_stage,
            "enc_input_ids": enc_input_ids,
            "enc_attention_mask": enc_attention_mask,
            "memory_token_counts": memory_token_counts,
            "dec_input_ids": dec_input_ids,
            "dec_attention_mask": dec_attention_mask,
            "labels": labels,
            "questions": questions,
            "answers": answers,
            "gold_answers": gold_answers,
            "docs": docs_list,
            "sample_occurrence_ids": occurrence_ids,
        }

    def _process_stage2_batch(batch_data: Tuple) -> Dict[str, Any]:
        """Process batch for stage 2 training."""
        (
            docs_list, questions, answers, data_types, pos_indices,
            gold_doc_ids, gold_answers, occurrence_ids,
        ) = batch_data

        # Prepare query inputs
        query_inputs = clara_model._prepare_query_inputs(
            questions, max_length=query_max_len
        )
        # The realized CCEF survivor count and ACR allocation do not exist at
        # collation time.  Building a fixed five-block decoder prompt here can
        # silently consume the 1024-token input budget before the question.
        # The model therefore constructs the one true supervised sequence only
        # after retrieval, final ordering, and memory allocation.  Validate the
        # raw targets now and pass them through without redundant tokenization.
        for row_index, answer in enumerate(answers):
            _answer_to_text(answer, location=f"Phase-II answer row {row_index}")

        return {
            "stage": clara_model.training_stage,
            "query_input_ids": query_inputs["input_ids"],
            "query_attention_mask": query_inputs["attention_mask"],
            "questions": questions,
            "answers": answers,
            "gold_answers": gold_answers,
            "docs": docs_list,
            "pos_index": pos_indices,
            "gold_doc_ids": gold_doc_ids,
            "sample_occurrence_ids": occurrence_ids,
        }

    def _process_reasoning_batch(batch_data: Tuple) -> Dict[str, Any]:
        """Process batch for reasoning training."""
        (
            docs_list, questions, answers, data_types, pos_indices,
            gold_doc_ids, gold_answers, occurrence_ids,
        ) = batch_data

        # Parse reasoning paths from answers
        thinking_paths = []
        for answer_index, answer in enumerate(answers):
            answer = _answer_to_text(
                answer, location=f"Phase-II reasoning answer row {answer_index}"
            )
            # Extract structured reasoning components
            pattern_full = r"<(?:information|think|answer|search)>.*?</(?:information|think|answer|search)>"
            tags = re.findall(r"<(information|think|answer|search)>.*?</\1>", answer, flags=re.DOTALL)
            fulls = re.findall(pattern_full, answer, flags=re.DOTALL)

            counter = defaultdict(int)
            result = {}
            for tag, full in zip(tags, fulls):
                counter[tag] += 1
                key = f"<{tag}>{counter[tag]}"
                result[key] = full.strip()

            thinking_paths.append(result)

        # Extract documents from information tags
        flat_docs = []
        docs_counts = []

        for thinking_path in thinking_paths:
            doc_count = 0
            for key, value in thinking_path.items():
                if 'information' in key:
                    # Extract information content
                    info_match = re.search(r"<information>(.*?)</information>", value, flags=re.DOTALL)
                    if info_match:
                        info_content = info_match.group(1)
                        # Split by document markers
                        temp_docs = re.split(r"(?m)^\(\d+\)", info_content)
                        temp_docs = [doc.strip() for doc in temp_docs if doc.strip()]
                        flat_docs.extend(temp_docs)
                        thinking_path[key] = "".join(temp_docs)
                        doc_count += len(temp_docs)

            docs_counts.append(doc_count)

        # Prepare encoder inputs
        enc_inputs = clara_model._prepare_encoder_inputs(flat_docs, max_length=enc_max_len)
        enc_input_ids = enc_inputs["input_ids"]
        enc_attention_mask = enc_inputs["attention_mask"]

        # Prepare decoder inputs with reasoning
        prompt_responses = [
            clara_model._blend_prompt_and_selected_memory_tokens_for_reasoning(
                query=q, answer=tp
            )
            for q, tp in zip(questions, thinking_paths)
        ]

        prompt_lengths = [pr[0] for pr in prompt_responses]
        instructions = [pr[1] for pr in prompt_responses]

        # Tokenize decoder inputs
        dec_inputs = tokenizer(
            instructions,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
            truncation=True,
            max_length=dec_max_len,
        )

        dec_input_ids = dec_inputs["input_ids"]
        dec_attention_mask = dec_inputs["attention_mask"]

        labels = _make_labels(
            dec_input_ids,
            dec_attention_mask,
            prompt_lengths,
            stage="Phase-II reasoning",
        )
        labels = _mask_information_spans(labels, dec_input_ids, dec_attention_mask, tokenizer)

        for row_index in range(labels.size(0)):
            if not torch.any(labels[row_index] != -100):
                raise ValueError(
                    f"Phase-II reasoning row {row_index} has no supervised token after information masking"
                )

        return {
            "stage": clara_model.training_stage,
            "enc_input_ids": enc_input_ids,
            "enc_attention_mask": enc_attention_mask,
            "dec_input_ids": dec_input_ids,
            "dec_attention_mask": dec_attention_mask,
            "labels": labels,
            "questions": questions,
            "answers": answers,
            "docs": docs_list,
            "pos_index": pos_indices,
            "docs_num": docs_counts
        }

    def collate(batch: List[Tuple]) -> Dict[str, Any]:
        """Main collate function that routes to appropriate stage processor."""
        if batch and all(len(row) == 7 for row in batch):
            # Compatibility for programmatic callers that predate view_id.
            batch = [
                tuple(row)
                + (
                    hashlib.sha256(
                        (str(row[1]) + "\0" + "\0".join(row[5])).encode("utf-8")
                    ).hexdigest(),
                )
                for row in batch
            ]
        if not batch or any(len(row) != 8 for row in batch):
            raise ValueError("ARIA collate rows must contain seven fields plus occurrence ID")
        # Unpack batch
        (
            docs_list,
            questions,
            answers,
            data_types,
            pos_indices,
            gold_doc_ids,
            gold_answers,
            occurrence_ids,
        ) = zip(*batch)

        # Convert to lists
        docs_list = list(docs_list)
        questions = list(questions)
        answers = list(answers)
        data_types = list(data_types)
        pos_indices = list(pos_indices)
        gold_doc_ids = list(gold_doc_ids)
        gold_answers = list(gold_answers)
        occurrence_ids = list(occurrence_ids)

        stage = clara_model.training_stage
        if stage in ["stage1", "stage1_2"]:
            invalid_types = sorted(set(data_types) - PHASE1_DATA_TYPES)
            if invalid_types:
                raise ValueError(
                    "Phase-I data_type must be one of the four conditional-generation "
                    f"categories; got {invalid_types}"
                )
            # Eq. (2) is defined on one (document, held-out target) pair.
            _validate_document_batch(
                docs_list,
                pos_indices,
                expected_docs=1,
                stage="Phase-I",
            )
        elif stage in ["stage2", "stage2_pretrain_retrieval"]:
            if any(data_type != "qa" for data_type in data_types):
                raise ValueError("Every Phase-II row must use data_type='qa'")
            # The canonical artifact carries five BGE candidates. CCEF retains
            # exactly five real documents at runtime, so the Arrow row schema
            # and decoder document count remain aligned.
            expected_candidates = generation_top_k
            _validate_document_batch(
                docs_list,
                pos_indices,
                expected_docs=expected_candidates,
                stage="Phase-II",
                allow_empty_positives=True,
            )
            # Appendix supporting-passage supervision is optional: P(x) may be
            # empty. Non-empty ID lists were validated row-wise by SFTDataset.

        batch_data = (
            docs_list,
            questions,
            answers,
            data_types,
            pos_indices,
            gold_doc_ids,
            gold_answers,
            occurrence_ids,
        )

        # Route to appropriate processor
        if stage in ["stage1", "stage1_2"]:
            return _process_stage1_batch(batch_data)
        elif stage in ["stage2", "stage2_pretrain_retrieval"]:
            return _process_stage2_batch(batch_data)
        elif stage == "stage2_reasoning":
            return _process_reasoning_batch(batch_data)
        else:
            raise ValueError(f"Unknown training stage: {stage}")

    return collate


def preprocess_data(data: Dict[str, Any],
                   input_template: Optional[str] = None,
                   input_key: str = "input",
                   output_key: Optional[str] = None,
                   apply_chat_template: Optional[Callable] = None,
                   multiturn: bool = False) -> Tuple[List[str], str, str, str, List[int]]:
    """
    Preprocess raw data into format expected by CLaRa dataset.

    Args:
        data: Raw data dictionary
        input_template: Template for input formatting
        input_key: Key for input data
        output_key: Key for output data
        apply_chat_template: Chat template function
        multiturn: Whether this is multiturn data

    Returns:
        Tuple of (docs, question, answer, data_type, pos_index)
    """
    # Extract already-normalized document text.  Candidate dictionaries must be
    # normalized by aria_data.py first so their page-URL provenance is not lost.
    if "docs" in data and isinstance(data["docs"], list):
        docs = data["docs"]
    elif "context" in data and isinstance(data["context"], list):
        docs = data["context"]
    elif "content" in data and isinstance(data["content"], list):
        docs = data["content"]
    else:
        raise ValueError(
            f"Data requires a list-valued docs, context, or content field; "
            f"received fields {list(data.keys())}"
        )
    if not docs:
        raise ValueError("Document list must not be empty")
    for doc_index, document in enumerate(docs):
        if not isinstance(document, str) or not document.strip():
            raise ValueError(f"docs[{doc_index}] must be a non-empty string")

    # Extract answers
    if "answer" in data:
        raw_answer = data["answer"]
    elif "answers" in data:
        raw_answer = data["answers"]
    elif "golden_answers" in data:
        raw_answer = data["golden_answers"]
    else:
        raise ValueError(
            f"Data requires an answer, answers, or golden_answers field; "
            f"received fields {list(data.keys())}"
        )
    answers = _answer_to_text(raw_answer, location="answer")

    # Extract data type
    data_type = data.get('data_type', 'qa')

    # ``question`` carries q in Phase II and the explicit task instruction I
    # in every Phase-I category, including paraphrase.
    if (
        "question" not in data
        or not isinstance(data["question"], str)
        or not data["question"].strip()
    ):
        raise ValueError("Training data requires a non-empty string question/instruction")
    questions = data["question"]

    # Extract positive indices
    pos_index = _validate_pos_index(
        data.get("pos_index", []),
        n_docs=len(docs),
        location="pos_index",
        allow_empty=data_type == "qa",
    )

    return docs, questions, answers, data_type, pos_index


class SFTDataset(Dataset):
    """
    Dataset for CLaRa Supervised Fine-Tuning.

    This dataset handles data preprocessing and loading for different CLaRa training stages.
    """

    def __init__(self,
                 dataset,
                 tokenizer: Callable,
                 max_length: int,
                 strategy,
                 input_template: Optional[str] = None,
                 pretrain_mode: bool = False,
                 num_processors: int = 8,
                 multiturn: bool = False) -> None:
        """
        Initialize the SFT dataset.

        Args:
            dataset: HuggingFace dataset object
            tokenizer: Tokenizer function
            max_length: Maximum sequence length
            strategy: Training strategy object
            input_template: Template for input formatting
            pretrain_mode: Whether in pretraining mode
            num_processors: Number of processors for data processing
            multiturn: Whether to handle multiturn conversations
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiturn = multiturn
        self.training_stage = getattr(self.strategy.args, "stage", None)
        self.generation_top_k = int(getattr(self.strategy.args, "generation_top_k", 1) or 1)
        self.stage2_retrieval_top_n = int(
            getattr(self.strategy.args, "stage2_retrieval_top_n", None)
            or self.generation_top_k
        )

        # Chat template configuration
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # ``aria_data`` already materializes a normalized Arrow artifact. Keep
        # that memory-mapped table intact and validate rows lazily in __getitem__.
        # Mapping 7.8M rows through a bound method on every torchrun rank both
        # duplicated the full dataset into Python lists and attempted to pickle
        # the attached distributed strategy into worker processes.
        self.dataset = dataset

    def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single data sample."""
        if self.multiturn and self.output_key:
            data[self.input_key].append(data[self.output_key])
            data[self.output_key] = None

        if self.multiturn:
            assert (
                not self.output_key or not data[self.output_key]
            ), "For multiturn data, put the whole trajectory in input_key and don't set output_key"

            # Process multiturn conversation
            input_key = self.input_key
            apply_chat_template = self.apply_chat_template
            response_ranges = []

            for idx, message in enumerate(data[input_key]):
                if message["role"] == "assistant":
                    prompt = apply_chat_template(
                        data[input_key][:idx],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    response = apply_chat_template(
                        data[input_key][:idx + 1],
                        tokenize=False
                    )[len(prompt):]

                    # Calculate token ranges
                    start_idx = (
                        self.tokenizer(
                            prompt,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                    )

                    end_idx = (
                        start_idx
                        + self.tokenizer(
                            response,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                        - 1
                    )

                    response_ranges.append((start_idx, end_idx))

        # Preprocess the data
        docs, questions, answers, data_type, pos_index = preprocess_data(data)

        if self.training_stage in ["stage1", "stage1_2"]:
            if data_type not in PHASE1_DATA_TYPES:
                raise ValueError(
                    "ARIA Phase-I data_type must be simple_qa, complex_qa, "
                    "paraphrase, or entity_augmented"
                )
            if len(docs) != 1:
                raise ValueError(
                    f"ARIA Phase-I requires exactly one document per (d, y) pair, got {len(docs)}"
                )
        elif self.training_stage in ["stage2", "stage2_pretrain_retrieval"]:
            if data_type != "qa":
                raise ValueError(f"ARIA Phase-II expects data_type='qa', got {data_type!r}")
            if len(docs) != self.generation_top_k:
                raise ValueError(
                    f"ARIA Phase-II requires exactly {self.generation_top_k} candidate-ceiling rows, "
                    f"got {len(docs)}"
                )
            if not 1 <= self.stage2_retrieval_top_n <= self.generation_top_k:
                raise ValueError(
                    "stage2_retrieval_top_n must be between 1 and generation_top_k"
                )

        raw_gold_doc_ids = data.get("gold_doc_ids", [])
        if not isinstance(raw_gold_doc_ids, list):
            raise ValueError("gold_doc_ids must be a list of stable corpus document IDs")
        gold_doc_ids = [str(value).strip() for value in raw_gold_doc_ids]
        if any(not value for value in gold_doc_ids) or len(set(gold_doc_ids)) != len(gold_doc_ids):
            raise ValueError("gold_doc_ids must contain unique non-empty IDs")
        raw_gold_answers = data.get("gold_answers", [answers])
        if not isinstance(raw_gold_answers, list) or not raw_gold_answers:
            raise ValueError("gold_answers must be a non-empty list")
        gold_answers = [
            _answer_to_text(value, location=f"gold_answers[{index}]")
            for index, value in enumerate(raw_gold_answers)
        ]

        raw_page_url = data.get("page_url", [])
        if isinstance(raw_page_url, str):
            page_url = [raw_page_url]
        elif isinstance(raw_page_url, list):
            page_url = raw_page_url
        else:
            raise ValueError("page_url must be a string or a list aligned with docs")
        if page_url and len(page_url) != len(docs):
            raise ValueError(
                f"page_url contains {len(page_url)} values but docs contains {len(docs)}"
            )
        if any(not isinstance(url, str) or not url.strip() for url in page_url):
            raise ValueError("Every page_url value must be a non-empty string")
        if self.training_stage in [
            "stage1",
            "stage1_2",
            "stage2",
            "stage2_pretrain_retrieval",
        ] and not page_url:
            raise ValueError(
                "Paper-protocol ARIA training requires page_url provenance for train/test deduplication"
            )

        result = {
            "docs": docs,
            "questions": questions,
            "answers": answers,
            "data_type": data_type,
            "pos_index": pos_index,
            "page_url": page_url,
            "gold_doc_ids": gold_doc_ids,
            "gold_answers": gold_answers,
            "sample_occurrence_id": str(
                data.get("view_id")
                or data.get("source_row_id")
                or hashlib.sha256(
                    (questions + "\0" + "\0".join(gold_doc_ids)).encode("utf-8")
                ).hexdigest()
            ),
        }

        # Multiturn records expose response token ranges to the collate function.
        if self.multiturn:
            result["response_ranges"] = response_ranges

        return result

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[
        List[str], str, str, str, List[int], List[str], List[str], str
    ]:
        """Get a single item from the dataset."""
        processed = self._process_data(dict(self.dataset[idx]))
        return (
            processed["docs"],
            processed["questions"],
            processed["answers"],
            processed["data_type"],
            processed["pos_index"],
            processed["gold_doc_ids"],
            processed["gold_answers"],
            processed["sample_occurrence_id"],
        )
