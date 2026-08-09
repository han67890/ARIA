#!/usr/bin/env python3
"""
ARIA End-to-End Inference

Single-query and batch inference with the full ARIA pipeline:
QCA → AHR → IGFR(if Multi-Hop) → MADS → CCEF →
ACR budget → Compressor → MTFRL 2nd round → MADS → CCEF →
ACR budget → Compressor → CFRS rerank → Generator.

Usage:
    # Single question
    python -m openrlhf.cli.infer_aria \
        --model_path ./checkpoints/aria_phase2_cr16 \
        --corpus_path /data/kilt_corpus.jsonl \
        --doc_embeddings /data/kilt_bge.pt \
        --question "What is the tallest mountain in Washington state?"

    # Batch from file
    python -m openrlhf.cli.infer_aria \
        --model_path ./checkpoints/aria_phase2_cr16 \
        --corpus_path /data/kilt_corpus.jsonl \
        --doc_embeddings /data/kilt_bge.pt \
        --input_file questions.json \
        --output_file outputs/answers.json
"""

import argparse
import json
import os
import time
from typing import List, Optional, Tuple

import torch

from openrlhf.models.modeling_aria import (
    CLaRa,
    CLaRaConfig,
    RAGPipelineConfig,
    RAGDiagnostics,
)
from openrlhf.cli.evaluate_aria import (
    PAPER_COMPRESSION_RATES,
    PAPER_MAX_NEW_TOKENS,
    _assert_normal_retrieval_is_not_training_index,
    _text_sha256,
    _corpus_sha256,
    _validate_checkpoint_protocol,
    load_bge_projection,
    load_doc_embeddings,
)
from openrlhf.utils.aria_provenance import (
    CORPUS_TEXT_FIELDS,
    corpus_id as _shared_corpus_id,
    corpus_page_url as _shared_corpus_page_url,
    corpus_text as _shared_corpus_text,
)
def load_model(
    model_path: str,
    decoder_model: Optional[str] = None,
    compression_rate: int = 16,
    device: str = "cuda",
    bge_projection_path: Optional[str] = None,
) -> CLaRa:
    """Load ARIA model from checkpoint."""
    print(f"Loading model from {model_path}...")
    model_overrides = {
        "pure_inference": True,
    }
    model = CLaRa.from_pretrained(
        model_path,
        strict_aria_artifacts=True,
        external_bge_artifact=bge_projection_path is not None,
        **model_overrides,
    )
    if decoder_model is not None and decoder_model != model.decoder_model_name:
        raise ValueError("decoder_model may only assert the checkpoint's exact backbone")
    if bge_projection_path is not None:
        load_bge_projection(model, bge_projection_path, expected_output_dim=1024)
    _validate_checkpoint_protocol(
        model,
        model_path,
        training_seed=None,
        compression_rate=compression_rate,
        expected_configuration="full",
    )
    model = model.to(device)
    model.eval()
    print(f"Model loaded. Adapters: {model.adapter_keys}")
    return model


def setup_corpus(
    corpus_path: Optional[str] = None,
    embeddings_path: Optional[str] = None,
) -> Tuple[List[str], List[str], List[str], torch.Tensor, str, str]:
    """
    Load corpus for RAG pipeline.

    Returns:
        corpus_docs, corpus_ids, corpus_page_ids, doc_embeddings, corpus/index digests
    """
    if corpus_path and os.path.exists(corpus_path):
        if corpus_path.endswith(".jsonl"):
            with open(corpus_path, encoding="utf-8") as handle:
                data = [json.loads(line) for line in handle if line.strip()]
        elif corpus_path.endswith(".json"):
            with open(corpus_path, encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            data = None

        if data is not None:
            if isinstance(data, dict):
                for container_key in ("documents", "corpus", "passages", "data"):
                    if container_key in data:
                        data = data[container_key]
                        break
            if isinstance(data, dict) and any(field in data for field in CORPUS_TEXT_FIELDS):
                rows = [data]
            elif isinstance(data, dict):
                rows = list(data.values())
            elif isinstance(data, list):
                rows = data
            else:
                raise ValueError("corpus JSON must contain a list or object of rows")
            if any(not isinstance(item, dict) for item in rows):
                raise ValueError("every corpus row must be an object with provenance")
            corpus_docs = [
                _shared_corpus_text(item, location=f"corpus row {index}")
                for index, item in enumerate(rows)
            ]
            corpus_ids = [
                _shared_corpus_id(item, location=f"corpus row {index}")
                for index, item in enumerate(rows)
            ]
            corpus_urls = [
                _shared_corpus_page_url(item, location=f"corpus row {index}")
                for index, item in enumerate(rows)
            ]
        else:
            from datasets import load_dataset
            ds = load_dataset("json", data_files=corpus_path, split="train")
            corpus_docs = [
                _shared_corpus_text(item, location=f"corpus row {index}")
                for index, item in enumerate(ds)
            ]
            corpus_ids = [
                _shared_corpus_id(item, location=f"corpus row {index}")
                for index, item in enumerate(ds)
            ]
            corpus_urls = [
                _shared_corpus_page_url(item, location=f"corpus row {index}")
                for index, item in enumerate(ds)
            ]
    else:
        raise ValueError("full ARIA inference requires --corpus_path")

    if not corpus_docs or any(not text for text in corpus_docs):
        raise ValueError("corpus contains empty documents")
    if len(corpus_ids) != len(set(corpus_ids)):
        raise ValueError("corpus document IDs must be unique")
    if embeddings_path is None:
        raise ValueError("full ARIA inference requires --doc_embeddings")
    doc_embeddings, index_sha256 = load_doc_embeddings(
        embeddings_path,
        len(corpus_docs),
        expected_ids=corpus_ids,
        expected_hashes=[_text_sha256(text) for text in corpus_docs],
        expected_page_ids=corpus_urls,
        return_index_sha256=True,
    )
    corpus_digest = _corpus_sha256(
        corpus_ids,
        [_text_sha256(text) for text in corpus_docs],
        corpus_urls,
    )
    return (
        corpus_docs,
        corpus_ids,
        corpus_urls,
        doc_embeddings,
        corpus_digest,
        index_sha256,
    )


class ARIAInference:
    """
    Convenience wrapper for ARIA inference with diagnostics.
    """

    def __init__(
        self,
        model: CLaRa,
        corpus_docs: Optional[List[str]] = None,
        corpus_ids: Optional[List[str]] = None,
        doc_embeddings: Optional[torch.Tensor] = None,
        corpus_page_ids: Optional[List[str]] = None,
        use_full_pipeline: bool = True,
        compression_rate: int = 16,
    ):
        self.model = model
        self.compression_rate = compression_rate

        if use_full_pipeline:
            if not corpus_docs or doc_embeddings is None or corpus_page_ids is None:
                raise ValueError(
                    "full ARIA pipeline requires corpus text, page IDs, and dense embeddings"
                )
            rag_cfg = RAGPipelineConfig(
                top_k=5,
                compression_rate=compression_rate,
                use_cfrs=True,
                cfrs_weight=0.3,
                use_acr=True,
                acr_min_token_ratio=0.25,
                acr_max_token_ratio=1.0,
                use_mtfrl=True,
                mtfrl_second_top_k=200,
                igfr_gap_threshold=0.50,
                igfr_max_iterations=None,
                ccef_discount_alpha=0.5,
                ccef_filter_threshold=0.30,
                verbose=False,
            )
            self.model.setup_rag_pipeline(
                corpus_docs=corpus_docs,
                corpus_doc_ids=corpus_ids,
                corpus_page_ids=corpus_page_ids,
                doc_embeddings=doc_embeddings,
                rag_config=rag_cfg,
            )
            print(f"RAG pipeline enabled: corpus={len(corpus_docs)} docs")

    @torch.no_grad()
    def answer(
        self,
        question: str,
        documents: Optional[List[str]] = None,
        max_new_tokens: int = PAPER_MAX_NEW_TOKENS,
        return_diagnostics: bool = False,
    ) -> str | Tuple[str, RAGDiagnostics]:
        """
        Answer a single question.

        Args:
            question: The question to answer
            documents: Optional pre-retrieved documents. If None, uses RAG pipeline.
            max_new_tokens: Max tokens to generate
            return_diagnostics: Whether to return RAG pipeline diagnostics

        Returns:
            answer string, or (answer, diagnostics) if return_diagnostics=True
        """
        if documents is not None:
            raise ValueError(
                "Full ARIA inference retrieves from its configured corpus; supplied "
                "documents would bypass QR/QCA/AHR/IGFR"
            )
        if max_new_tokens != PAPER_MAX_NEW_TOKENS:
            raise ValueError("Paper-protocol inference requires max_new_tokens=64")
        predictions, _ = self.model.generate_from_questions(
            questions=[question],
            documents=None,
            max_new_tokens=max_new_tokens,
        )
        answer = predictions[0]

        if return_diagnostics:
            diags = self.model.get_rag_diagnostics()
            diag = diags[-1] if diags else RAGDiagnostics()
            return answer, diag
        return answer

    @torch.no_grad()
    def answer_batch(
        self,
        questions: List[str],
        documents_list: Optional[List[List[str]]] = None,
        max_new_tokens: int = PAPER_MAX_NEW_TOKENS,
        batch_size: int = 8,
    ) -> List[str]:
        """
        Answer a batch of questions.

        Args:
            questions: List of questions
            documents_list: Optional pre-retrieved documents
            max_new_tokens: Max tokens to generate
            batch_size: Batch size for generation

        Returns:
            List of answer strings
        """
        if documents_list is not None:
            raise ValueError(
                "Full ARIA inference retrieves evidence from its configured corpus; "
                "leave documents_list unset"
            )
        if max_new_tokens != PAPER_MAX_NEW_TOKENS:
            raise ValueError("Paper-protocol inference requires max_new_tokens=64")
        all_answers = []
        for i in range(0, len(questions), batch_size):
            batch_qs = questions[i:i + batch_size]
            predictions, _ = self.model.generate_from_questions(
                questions=batch_qs,
                documents=None,
                max_new_tokens=max_new_tokens,
            )
            all_answers.extend(predictions)
        return all_answers


def _load_batch_questions(path: str) -> List[str]:
    """Load a strict questions-only JSON/JSONL-like or plain-text batch."""
    with open(path, encoding="utf-8") as handle:
        raw = handle.read()
    if not raw.strip():
        raise ValueError("--input_file is empty")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        questions = [line.strip() for line in raw.splitlines() if line.strip()]
    else:
        if isinstance(payload, dict):
            if "documents" in payload or "docs" in payload:
                raise ValueError("Batch inference accepts questions only, not documents")
            payload = payload.get("questions")
            if not isinstance(payload, list):
                raise ValueError("Batch JSON objects must contain a 'questions' list")
        if not isinstance(payload, list):
            raise ValueError("Batch JSON must be a list or an object with 'questions'")
        questions = []
        for index, item in enumerate(payload):
            if isinstance(item, str):
                question = item
            elif isinstance(item, dict):
                if item.get("documents") is not None or item.get("docs") is not None:
                    raise ValueError(
                        f"Batch row {index} supplies documents and would bypass ARIA retrieval"
                    )
                question = item.get("question", item.get("q"))
            else:
                raise ValueError(f"Batch row {index} must be a string or question object")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Batch row {index} has no non-empty question")
            questions.append(question.strip())
    if not questions:
        raise ValueError("--input_file contains no questions")
    return questions


def main():
    parser = argparse.ArgumentParser(description="ARIA Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to ARIA checkpoint")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--question", type=str, default=None, help="Single question")
    input_group.add_argument("--input_file", type=str, default=None, help="JSON/text questions file")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus JSON")
    parser.add_argument(
        "--doc_embeddings",
        type=str,
        required=True,
        help="Aligned .pt/.pth/.npz BGE artifact with document IDs and text hashes",
    )
    parser.add_argument("--bge_projection_path", type=str, default=None,
                        help="Optional W_BGE artifact if not bundled in checkpoint")
    parser.add_argument(
        "--decoder_model",
        type=str,
        default=None,
        help="Optional base-model override; by default use the checkpoint configuration",
    )
    parser.add_argument("--compression_rate", type=int, default=16)
    parser.add_argument(
        "--max_new_tokens", type=int, default=PAPER_MAX_NEW_TOKENS
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true", help="Show diagnostics")
    args = parser.parse_args()

    if args.compression_rate not in PAPER_COMPRESSION_RATES:
        parser.error("--compression_rate must be one of 4, 16, 32, 64, 128")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("--batch_size and --max_new_tokens must be positive")
    if args.max_new_tokens != PAPER_MAX_NEW_TOKENS:
        parser.error("Paper-protocol inference requires --max_new_tokens=64")
    if args.question is not None and not args.question.strip():
        parser.error("--question must be non-empty")
    if args.input_file is not None and not os.path.isfile(args.input_file):
        parser.error("--input_file must name an existing file")
    if args.output_file is not None and args.input_file is None:
        parser.error("--output_file is only valid with --input_file")

    checkpoint_config = CLaRaConfig.from_pretrained(args.model_path)
    training_index_sha256 = getattr(
        checkpoint_config, "aria_training_retrieval_index_sha256", None
    )
    if not isinstance(training_index_sha256, str) or len(training_index_sha256) != 64:
        parser.error("checkpoint lacks its Phase-II training BGE-index fingerprint")

    # Validate and align the large corpus artifacts before loading the model.
    (
        corpus_docs,
        corpus_ids,
        corpus_page_ids,
        doc_embeddings,
        corpus_digest,
        index_sha256,
    ) = setup_corpus(args.corpus_path, args.doc_embeddings)
    _assert_normal_retrieval_is_not_training_index(
        checkpoint_config,
        evaluation_corpus_sha256=corpus_digest,
        evaluation_index_sha256=index_sha256,
    )

    # Load model
    model = load_model(
        args.model_path,
        args.decoder_model,
        args.compression_rate,
        args.device,
        args.bge_projection_path,
    )

    # Setup inference
    infer = ARIAInference(
        model=model,
        corpus_docs=corpus_docs,
        corpus_ids=corpus_ids,
        corpus_page_ids=corpus_page_ids,
        doc_embeddings=doc_embeddings,
        use_full_pipeline=True,
        compression_rate=args.compression_rate,
    )

    # Process queries
    if args.question:
        # Single question
        print(f"\nQ: {args.question}")
        t0 = time.time()
        answer, diag = infer.answer(
            args.question, return_diagnostics=True,
            max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.time() - t0
        print(f"A: {answer}")
        print(f"Latency: {elapsed:.2f}s")
        if args.verbose:
            print(f"  Type: {diag.question_type}, QCA conf: {diag.qca_confidence:.2f}")
            print(f"  BM25/dense weights: {diag.bm25_weight:.2f}/{diag.dense_weight:.2f}")
            print(
                f"  IGFR iters: {diag.igfr_iterations}, Coverage: "
                f"{diag.initial_coverage:.0%}→{diag.final_coverage:.0%}"
            )
            print(f"  CCEF filtered: {diag.ccef_filtered}, Avg conf: {diag.ccef_avg_confidence:.2f}")

    elif args.input_file:
        questions = _load_batch_questions(args.input_file)

        print(f"Processing {len(questions)} questions...")
        t0 = time.time()
        answers = infer.answer_batch(
            questions=questions,
            documents_list=None,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - t0
        print(f"Completed in {elapsed:.1f}s ({elapsed/len(questions):.2f}s/query)")

        results = [
            {"question": q, "answer": a}
            for q, a in zip(questions, answers)
        ]

        output_file = args.output_file or "outputs/aria_outputs.json"
        output_path = os.path.abspath(os.path.expanduser(output_file))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")

    else:
        # Interactive mode
        print("\nARIA Interactive Mode (type 'quit' to exit)")
        print("─" * 50)
        while True:
            try:
                q = input("\nQ: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue

            t0 = time.time()
            answer, diag = infer.answer(
                q, return_diagnostics=True,
                max_new_tokens=args.max_new_tokens,
            )
            elapsed = time.time() - t0
            print(f"A: {answer}")
            if args.verbose:
                print(f"  [{diag.question_type}, {elapsed:.2f}s, cov={diag.final_coverage:.0%}]")


if __name__ == "__main__":
    main()
