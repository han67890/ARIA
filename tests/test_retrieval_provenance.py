import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from openrlhf.cli.aria_data import build_parser
from openrlhf.cli.train_sft import load_bge_embeddings, load_corpus as load_training_corpus
from openrlhf.cli.evaluate_aria import (
    _assert_normal_retrieval_is_not_training_index,
    _validate_checkpoint_protocol,
    load_bge_projection,
    load_doc_embeddings,
)
from openrlhf.models.modeling_aria import (
    CLARA_DOCUMENT_REPRESENTATION_SCHEME,
    CLARA_ARCHIVE_DOCUMENT_ID_SCHEME,
    CLARA_ARCHIVE_PAGE_ID_SCHEME,
    CLARA_EVALUATION_CANDIDATE_PROTOCOL,
    CLARA_MEMORY_ALLOCATION_SCHEME,
    CLARA_PHASE2_OBJECTIVE,
    CLARA_SELECTOR_SCHEME,
    CFRS_RECONSTRUCTION_SCHEME,
    MTFRL_INITIALIZATION_SCHEME,
    QR_INPUT_SCHEME,
    RETRIEVAL_STRAIGHT_THROUGH_SCHEME,
)
from openrlhf.utils.aria_provenance import (
    CORPUS_SHA256_SCHEME,
    TEXT_SHA256_SCHEME,
    corpus_sha256,
    text_sha256,
)


def _embedding_index_sha256(embeddings: torch.Tensor) -> str:
    embeddings = embeddings.float().contiguous()
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            {"shape": list(embeddings.shape), "dtype": "float32"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    hasher.update(embeddings.numpy().tobytes())
    return hasher.hexdigest()


def _paper_checkpoint_model() -> SimpleNamespace:
    digest = "a" * 64
    base_model = "mistralai/Mistral-7B-v0.1"
    config = SimpleNamespace(
        decoder_model_resolved_revision="0" * 40,
        mads_semantic_model_name="BAAI/bge-large-en-v1.5",
        mtfrl_hidden_width=2048,
        aria_compression_rate=16,
        aria_rag_configuration="full",
        aria_training_seed=42,
        aria_dataset_manifest_sha256="b" * 64,
        aria_epoch_seed_schedule=[42, 123, 456, 789, 2024],
        doc_max_length=768,
        query_max_length=256,
        stage2_input_max_length=1024,
        max_new_tokens=64,
        aria_passage_max_length=768,
        aria_query_max_length=256,
        aria_input_max_length=1024,
        aria_target_max_length=128,
        lora=True,
        lora_r=16,
        lora_r_compressor=16,
        lora_target_modules=["q_proj"],
        sep=True,
        different_mem_tokens=True,
        optimize_mem_tokens=False,
        compr_model_name=None,
        compr_n_layers=5,
        compr_use_mlp=False,
        compr_linear_type=None,
        compr_rms_norm=False,
        training_form="both_separately",
        stage2_retrieval_top_n=5,
        aria_text_sha256_scheme=TEXT_SHA256_SCHEME,
        qr_input_scheme=QR_INPUT_SCHEME,
        mtfrl_initialization_scheme=MTFRL_INITIALIZATION_SCHEME,
        cfrs_reconstruction_scheme=CFRS_RECONSTRUCTION_SCHEME,
        retrieval_straight_through_scheme=RETRIEVAL_STRAIGHT_THROUGH_SCHEME,
        aria_loss_weights={"lambda_mse": 0.10},
        aria_test_url_sha256=digest,
        aria_phase1_training_seed=42,
        aria_phase1_dataset_manifest_sha256="c" * 64,
        aria_phase1_test_url_sha256=digest,
        aria_phase1_base_model=base_model,
        aria_phase1_base_model_resolved_revision="0" * 40,
        aria_phase1_compression_rate=16,
        aria_training_retrieval_index_sha256="d" * 64,
        aria_training_candidate_order_sha256="e" * 64,
        aria_training_corpus_sha256="f" * 64,
        aria_training_corpus_count=123,
        aria_training_corpus_sha256_scheme=CORPUS_SHA256_SCHEME,
        aria_training_corpus_scope="page_url_deduplicated",
        aria_source_snapshot_scheme="aria-source-snapshot-v1",
        aria_source_tree_sha256="9" * 64,
        aria_source_git_commit="8" * 40,
        aria_source_git_dirty=True,
        aria_source_file_count=42,
    )
    projection = torch.nn.Linear(4, 1024, bias=False)
    projection_metadata = {
        "base_model": base_model,
        "base_model_revision_resolved": "0" * 40,
        "bge_model": "BAAI/bge-large-en-v1.5",
        "sample_count": 50_000,
        "epochs": 2,
        "batch_size": 128,
        "learning_rate": 5e-4,
        "seed": 42,
        "query_sha256": "1" * 64,
        "passage_id_sha256": "2" * 64,
        "passage_text_sha256": "3" * 64,
        "test_url_sha256": digest,
        "text_sha256_scheme": TEXT_SHA256_SCHEME,
        "qr_input_scheme": QR_INPUT_SCHEME,
    }
    return SimpleNamespace(
        config=config,
        compr_rate=16,
        training_stage="stage2",
        generation_top_k=5,
        decoder_model_name=base_model,
        doc_max_length=768,
        decoder=SimpleNamespace(config=SimpleNamespace(hidden_size=4096)),
        _bge_projection=projection,
        _bge_projection_metadata=projection_metadata,
    )


def test_full_kilt_embedding_index_is_self_aligned_not_training_bound(tmp_path):
    embeddings = torch.arange(2 * 1024, dtype=torch.float32).reshape(2, 1024)
    document_ids = ["full-1", "full-2"]
    hashes = [text_sha256("first"), text_sha256("second")]
    page_ids = ["https://example.org/wiki/First", "https://example.org/wiki/Second"]
    index_sha256 = _embedding_index_sha256(embeddings)
    artifact_path = tmp_path / "full-kilt.pt"
    torch.save(
        {
            "doc_embeddings": embeddings,
            "document_ids": document_ids,
            "text_sha256": hashes,
            "page_urls": page_ids,
            "bge_model": "BAAI/bge-large-en-v1.5",
            "text_sha256_scheme": TEXT_SHA256_SCHEME,
            "index_sha256": index_sha256,
        },
        artifact_path,
    )

    loaded, loaded_digest = load_doc_embeddings(
        str(artifact_path),
        2,
        expected_ids=document_ids,
        expected_hashes=hashes,
        expected_page_ids=page_ids,
        return_index_sha256=True,
    )
    assert torch.equal(loaded, embeddings)
    assert loaded_digest == index_sha256

    with pytest.raises(ValueError, match="explicitly expected BGE index"):
        load_doc_embeddings(
            str(artifact_path),
            2,
            expected_ids=document_ids,
            expected_hashes=hashes,
            expected_index_sha256="0" * 64,
        )

    with pytest.raises(ValueError, match="page_urls do not match"):
        load_doc_embeddings(
            str(artifact_path),
            2,
            expected_ids=document_ids,
            expected_hashes=hashes,
            expected_page_ids=list(reversed(page_ids)),
        )


def test_phase2_training_corpus_rejects_duplicate_canonical_page_urls(tmp_path):
    corpus_path = tmp_path / "training-corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "id": "doc-0",
                    "text": "first passage",
                    "page_url": "HTTPS://EXAMPLE.ORG/wiki/Page/",
                },
                {
                    "id": "doc-1",
                    "text": "second passage",
                    "page_url": "https://example.org/wiki/Page",
                },
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be page-URL deduplicated"):
        load_training_corpus(str(corpus_path))


def test_normal_retrieval_rejects_training_corpus_or_index():
    config = SimpleNamespace(
        aria_training_corpus_sha256="a" * 64,
        aria_training_retrieval_index_sha256="b" * 64,
    )
    _assert_normal_retrieval_is_not_training_index(
        config,
        evaluation_corpus_sha256="c" * 64,
        evaluation_index_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="full KILT corpus"):
        _assert_normal_retrieval_is_not_training_index(
            config,
            evaluation_corpus_sha256="a" * 64,
            evaluation_index_sha256="d" * 64,
        )
    with pytest.raises(ValueError, match="full-KILT BGE index"):
        _assert_normal_retrieval_is_not_training_index(
            config,
            evaluation_corpus_sha256="c" * 64,
            evaluation_index_sha256="b" * 64,
        )


def test_checkpoint_fingerprint_labels_training_retrieval_provenance():
    model = _paper_checkpoint_model()
    fingerprint = _validate_checkpoint_protocol(
        model,
        "checkpoint",
        training_seed=42,
        compression_rate=16,
        expected_configuration="full",
    )
    assert fingerprint["training_retrieval"] == {
        "corpus_sha256": "f" * 64,
        "corpus_count": 123,
        "corpus_sha256_scheme": CORPUS_SHA256_SCHEME,
        "corpus_scope": "page_url_deduplicated",
        "index_sha256": "d" * 64,
        "candidate_order_sha256": "e" * 64,
    }


def test_checkpoint_protocol_accepts_only_canonical_clara_baseline():
    model = _paper_checkpoint_model()
    model.config.aria_rag_configuration = "clara_baseline"
    model.config.aria_loss_weights = {"lambda_mse": 0.0}
    model.config.lora_target_modules = "all-linear"
    model.config.clara_selector_scheme = CLARA_SELECTOR_SCHEME
    model.config.clara_document_representation_scheme = (
        CLARA_DOCUMENT_REPRESENTATION_SCHEME
    )
    model.config.clara_phase2_objective = CLARA_PHASE2_OBJECTIVE
    model.config.clara_phase2_trainable_adapters = [
        "query_reasoner_adapter",
        "decoder_adapter",
    ]
    model.config.clara_phase2_frozen_adapter = "encoder_adapter"
    model.config.clara_phase2_adapter_initialization = (
        "both-exact-copy-of-corresponding-phase1-compressor-v1"
    )
    model.config.clara_memory_allocation_scheme = CLARA_MEMORY_ALLOCATION_SCHEME
    model.config.clara_max_memory_tokens = 48
    model.config.clara_training_candidate_count = 5
    model.config.clara_evaluation_candidate_protocol = (
        CLARA_EVALUATION_CANDIDATE_PROTOCOL
    )
    model.config.clara_evaluation_candidate_count = 20
    model.config.clara_selection_count = 5
    model.config.clara_archive_document_id_scheme = CLARA_ARCHIVE_DOCUMENT_ID_SCHEME
    model.config.clara_archive_page_id_scheme = CLARA_ARCHIVE_PAGE_ID_SCHEME
    model.config.aria_trainable_parameter_names = [
        "decoder.lora_A.query_reasoner_adapter.weight",
        "decoder.lora_A.decoder_adapter.weight",
    ]
    model.adapter_keys = [
        "encoder_adapter",
        "query_reasoner_adapter",
        "decoder_adapter",
    ]
    model._bge_projection = None
    model._bge_projection_metadata = None

    fingerprint = _validate_checkpoint_protocol(
        model,
        "clara-checkpoint",
        training_seed=42,
        compression_rate=16,
        expected_configuration="clara_baseline",
    )
    assert fingerprint["w_bge"] is None

    model.config.lora_target_modules = ["q_proj"]
    with pytest.raises(ValueError, match="all-linear"):
        _validate_checkpoint_protocol(
            model,
            "clara-checkpoint",
            training_seed=42,
            compression_rate=16,
            expected_configuration="clara_baseline",
        )


def test_checkpoint_protocol_requires_exact_base_decoder_revision():
    model = _paper_checkpoint_model()
    model.config.decoder_model_resolved_revision = None
    with pytest.raises(ValueError, match="exact resolved base-model revision"):
        _validate_checkpoint_protocol(
            model,
            "checkpoint",
            training_seed=42,
            compression_rate=16,
            expected_configuration="full",
        )


def test_checkpoint_protocol_binds_w_bge_to_base_decoder_revision():
    model = _paper_checkpoint_model()
    model._bge_projection_metadata["base_model_revision_resolved"] = "1" * 40
    with pytest.raises(ValueError, match="base_model_revision_resolved"):
        _validate_checkpoint_protocol(
            model,
            "checkpoint",
            training_seed=42,
            compression_rate=16,
            expected_configuration="full",
        )


def test_checkpoint_protocol_requires_shared_bge_semantic_space():
    model = _paper_checkpoint_model()
    model.config.mads_semantic_model_name = "legacy-independent-semantic-encoder"
    with pytest.raises(ValueError, match="requires MADS model"):
        _validate_checkpoint_protocol(
            model,
            "checkpoint",
            training_seed=42,
            compression_rate=16,
            expected_configuration="full",
        )


def test_explicit_w_bge_loader_rejects_decoder_revision_mismatch(tmp_path):
    model = _paper_checkpoint_model()
    model.hidden_size = 4
    model.setup_bge_projection = lambda bge_dim: setattr(
        model, "_bge_projection", torch.nn.Linear(4, bge_dim, bias=False)
    )
    artifact = {
        **model._bge_projection_metadata,
        "base_model_revision_resolved": "1" * 40,
        "state_dict": {"weight": torch.ones((1024, 4))},
    }
    path = tmp_path / "wrong-base-revision.pt"
    torch.save(artifact, path)

    with pytest.raises(ValueError, match="base_model_revision_resolved"):
        load_bge_projection(model, str(path), expected_output_dim=1024)


def test_ordered_corpus_fingerprint_changes_between_train_and_full_roles():
    train_digest = corpus_sha256(
        ["one"],
        [text_sha256("training passage")],
        ["https://en.wikipedia.org/wiki/Training"],
    )
    full_digest = corpus_sha256(
        ["one", "test"],
        [text_sha256("training passage"), text_sha256("held-out passage")],
        [
            "https://en.wikipedia.org/wiki/Training",
            "https://en.wikipedia.org/wiki/Held-out",
        ],
    )
    assert train_digest != full_digest


def test_phase2_index_cli_uses_explicit_training_role_with_legacy_alias():
    digest = "9" * 64
    parser = build_parser()
    explicit = parser.parse_args(
        ["--stage", "eval", "--training-retrieval-index-sha256", digest]
    )
    legacy = parser.parse_args(
        ["--stage", "eval", "--normal-retrieval-index-sha256", digest]
    )
    assert explicit.training_retrieval_index_sha256 == digest
    assert legacy.training_retrieval_index_sha256 == digest


def test_w_bge_loader_propagates_verified_encoder_revision(tmp_path):
    embeddings = torch.arange(2 * 1024, dtype=torch.float32).reshape(2, 1024)
    document_ids = ["passage-1", "passage-2"]
    hashes = [text_sha256("first"), text_sha256("second")]
    commit = "1" * 40
    artifact_path = tmp_path / "alignment-v2.pt"
    torch.save(
        {
            "artifact_format": "aria-bge-artifact-v2",
            "target_embeddings": embeddings,
            "document_ids": document_ids,
            "text_sha256": hashes,
            "bge_model": "BAAI/bge-large-en-v1.5",
            "text_sha256_scheme": TEXT_SHA256_SCHEME,
            "index_sha256": _embedding_index_sha256(embeddings),
            "encoder_source": "BAAI/bge-large-en-v1.5",
            "encoder_source_kind": "huggingface-hub",
            "encoder_revision_declared": "paper-tag",
            "encoder_revision_resolved": commit,
            "encoder_revision_was_explicit": True,
            "encoder_source_sha256": None,
            "encoder_source_sha256_scheme": None,
        },
        artifact_path,
    )

    loaded, metadata = load_bge_embeddings(
        str(artifact_path),
        expected_rows=2,
        expected_ids=document_ids,
        expected_hashes=hashes,
        return_metadata=True,
    )
    assert torch.equal(loaded, embeddings)
    assert metadata["bge_embedding_artifact_format"] == "aria-bge-artifact-v2"
    assert metadata["bge_encoder_revision_declared"] == "paper-tag"
    assert metadata["bge_encoder_revision_resolved"] == commit

    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    payload["encoder_revision_resolved"] = "not-a-commit"
    bad_path = tmp_path / "alignment-invalid-v2.pt"
    torch.save(payload, bad_path)
    with pytest.raises(ValueError, match="exact resolved commit"):
        load_bge_embeddings(
            str(bad_path),
            expected_rows=2,
            expected_ids=document_ids,
            expected_hashes=hashes,
            return_metadata=True,
        )
