import re
from types import SimpleNamespace

import pytest
import torch
import openrlhf.models.modeling_aria as modeling_aria

from openrlhf.models.modeling_aria import (
    AdaptiveCompressionAllocator,
    CLaRa,
    CLaRaConfig,
    CompressionFidelityReranker,
    MemoryTokenFeedbackRetriever,
    ORACLE_TOP100_PROTOCOL,
    QCAResult,
    QuestionComplexityAssessor,
    QuestionType,
    RAGEnhancementPipeline,
    RAGPipelineConfig,
    _BM25Index,
    _AdaptiveHybridRetriever,
    _CCEF,
    _EntityAgent,
    _RetrievedDoc,
    _SemanticAgent,
    _ahr_get_weights,
    _chunked_inner_product_topk,
    _chunked_inner_product_topk_unique_pages,
    _construct_oracle_top100_indices,
    _fixed_memory_prompt_max_length,
    _merge_bounded_retrieval_pool,
    _mtfrl_hidden_width,
    _pack_variable_encoder_memory_rows,
    _prune_padded_memory_slots,
    _tensor_is_finite_in_chunks,
)


class _FakeEntity:
    def __init__(self, text, label="PERSON"):
        self.text = text
        self.label_ = label


class _FakeToken:
    def __init__(self, text, idx):
        self.text = text
        self.lower_ = text.casefold()
        self.idx = idx
        self.tag_ = "NNS" if self.lower_ in {"applications", "components"} else "NN"
        self.pos_ = "NOUN"
        self.children = []


class _FakeSpacyDocument:
    def __init__(self, text):
        names = ("Alice", "Bob", "Carol", "Ada Lovelace")
        self.ents = [_FakeEntity(name) for name in names if name in text]
        self._tokens = [
            _FakeToken(match.group(0), match.start())
            for match in re.finditer(r"[A-Za-z]+", text)
        ]

    def __iter__(self):
        return iter(self._tokens)

    def __getitem__(self, index):
        return self._tokens[index]

    def __len__(self):
        return len(self._tokens)


class _FakeSpacyPipeline:
    def __init__(self):
        self.pipe_batches = []

    def __call__(self, text):
        return _FakeSpacyDocument(text)

    def pipe(self, texts, batch_size):
        texts = list(texts)
        self.pipe_batches.append((texts, batch_size))
        return [_FakeSpacyDocument(text) for text in texts]


@pytest.mark.parametrize(
    "question, plural_noun",
    [
        ("What are the main applications of solar energy?", "applications"),
        ("What are the main components of a compiler?", "components"),
    ],
)
def test_qca_a07_accepts_any_pos_tagged_plural_aspect_noun(
    monkeypatch, question, plural_noun
):
    nlp = _FakeSpacyPipeline()
    monkeypatch.setattr(modeling_aria, "_qca_get_spacy", lambda: nlp)

    result = QuestionComplexityAssessor().assess(question)

    assert plural_noun in question.casefold()
    assert result.question_type is QuestionType.MULTI_ASPECT
    assert "A07" in result.matched_rules


def test_qca_h05_where_template_precedes_generic_where_splitter(monkeypatch):
    nlp = _FakeSpacyPipeline()
    monkeypatch.setattr(modeling_aria, "_qca_get_spacy", lambda: nlp)

    result = QuestionComplexityAssessor().assess(
        "Where was Ada Lovelace born before meeting Bob?"
    )

    assert result.question_type is QuestionType.MULTI_HOP
    assert "H05" in result.matched_rules
    assert result.sub_questions == [
        "Who is Ada Lovelace?",
        "Where was {BRIDGE} born?",
    ]


def test_submission_qca_hop_rule_wins_aspect_conflict_with_two_entities(monkeypatch):
    nlp = _FakeSpacyPipeline()
    monkeypatch.setattr(modeling_aria, "_qca_get_spacy", lambda: nlp)

    result = QuestionComplexityAssessor().assess(
        "Compare Alice, Bob, and Carol."
    )

    assert "A01" in result.matched_rules
    assert any(rule.startswith("H") for rule in result.matched_rules)
    assert result.question_type is QuestionType.MULTI_HOP


def test_submission_qca_hop_match_requires_two_entities(monkeypatch):
    nlp = _FakeSpacyPipeline()
    monkeypatch.setattr(modeling_aria, "_qca_get_spacy", lambda: nlp)

    result = QuestionComplexityAssessor().assess("Where was Ada Lovelace born?")

    assert "H05" in result.matched_rules
    assert result.entity_count == 1
    assert result.question_type is not QuestionType.MULTI_HOP


def test_qca_explicit_hop_phrase_overrides_explicit_aspect_phrase(monkeypatch):
    nlp = _FakeSpacyPipeline()
    monkeypatch.setattr(modeling_aria, "_qca_get_spacy", lambda: nlp)

    result = QuestionComplexityAssessor().assess(
        "Compare Alice and Bob in which country they were born."
    )

    assert {"H02", "A01"}.issubset(result.matched_rules)
    assert result.question_type is QuestionType.MULTI_HOP


def test_release_convention_ahr_confidence_interpolates_balanced_to_type_endpoint():
    low = QCAResult(
        question="q",
        question_type=QuestionType.MULTI_HOP,
        confidence=0.0,
        hop_count=2,
        entity_count=2,
    )
    high = QCAResult(
        question="q",
        question_type=QuestionType.MULTI_HOP,
        confidence=1.0,
        hop_count=2,
        entity_count=2,
    )
    middle = QCAResult(
        question="q",
        question_type=QuestionType.MULTI_HOP,
        confidence=0.5,
        hop_count=2,
        entity_count=2,
    )

    assert _ahr_get_weights(low) == pytest.approx((0.5, 0.5))
    assert _ahr_get_weights(high) == pytest.approx((0.25, 0.75))
    assert _ahr_get_weights(middle) == pytest.approx((0.375, 0.625))


def test_qca_who_action_phrase_uses_entity_count_disambiguation(monkeypatch):
    nlp = _FakeSpacyPipeline()
    monkeypatch.setattr(modeling_aria, "_qca_get_spacy", lambda: nlp)
    assessor = QuestionComplexityAssessor()

    single = assessor.assess("Who wrote Alice?")
    multiple = assessor.assess("Who wrote Alice about Bob?")

    assert single.question_type is QuestionType.SIMPLE
    assert "S03" in single.matched_rules
    assert "H04" not in single.matched_rules
    assert multiple.question_type is QuestionType.MULTI_HOP
    assert "H04" in multiple.matched_rules
    assert "S03" not in multiple.matched_rules


def test_qca_starts_requires_a_token_boundary():
    assert modeling_aria._qca_starts("what is ada?", "what is")
    assert not modeling_aria._qca_starts("what island is ada on?", "what is")
    assert not modeling_aria._qca_starts("who issued the order?", "who is")


def test_qca_h11_requires_the_verb_immediately_after_the_entity(monkeypatch):
    def parsed(text):
        verb = "expanded"
        return [
            SimpleNamespace(
                idx=match.start(),
                pos_="VERB" if match.group(0).casefold() == verb else "NOUN",
            )
            for match in re.finditer(r"[A-Za-z]+", text)
        ]

    monkeypatch.setattr(modeling_aria, "_qca_get_spacy", lambda: parsed)
    assert modeling_aria._qca_has_temporal_entity_verb(
        "After Alice expanded rapidly", {"alice"}
    )
    assert not modeling_aria._qca_has_temporal_entity_verb(
        "After Alice research expanded rapidly", {"alice"}
    )


def test_qca_h12_uses_pos_superlatives_without_treating_forest_as_one(monkeypatch):
    def parsed(text):
        tokens = []
        for match in re.finditer(r"[A-Za-z]+", text):
            value = match.group(0).casefold()
            tokens.append(
                SimpleNamespace(
                    idx=match.start(),
                    lower_=value,
                    tag_="JJS" if value in {"best", "worst"} else "NN",
                    pos_="ADJ" if value in {"best", "worst"} else "NOUN",
                )
            )
        return tokens

    monkeypatch.setattr(modeling_aria, "_qca_get_spacy", lambda: parsed)
    assert modeling_aria._qca_has_superlative_entity_pattern(
        "What is the best museum in the Paris?", {"paris"}
    )
    assert not modeling_aria._qca_has_superlative_entity_pattern(
        "What is the forest museum in the Paris?", {"paris"}
    )


def test_stage2_training_refuses_to_truncate_an_overlength_prompt():
    class _Tokenizer:
        pad_token_id = 0
        padding_side = "right"

        def __call__(self, _text, **_kwargs):
            return {
                "input_ids": list(range(1026)),
                "offset_mapping": [(index, index + 1) for index in range(1026)],
            }

    model = SimpleNamespace(
        generation_top_k=1,
        stage2_input_max_length=1024,
        stage2_target_max_length=128,
        decoder_tokenizer=_Tokenizer(),
        _blend_prompt_and_selected_memory_tokens=lambda **_kwargs: (1025, "x" * 1026),
    )

    with pytest.raises(ValueError, match="refusing to truncate any part"):
        CLaRa._prepare_stage2_supervised_decoder_inputs(
            model,
            ["question"],
            ["answer"],
            torch.tensor([[1]]),
        )


def test_mads_minmax_tie_uses_paper_half_score_fallback():
    assert _CCEF._minmax([3.0, 3.0, 3.0]) == [0.5, 0.5, 0.5]


def test_submission_mtfrl_initialization_realizes_the_fitted_bge_map():
    bge = torch.nn.Linear(12, 3, bias=False)
    with torch.no_grad():
        bge.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        )
    model = SimpleNamespace(
        hidden_size=12,
        _bge_projection=bge,
        decoder=SimpleNamespace(device=torch.device("cpu")),
        config=SimpleNamespace(mtfrl_hidden_width=None),
    )

    CLaRa.setup_mtfrl_projection(model, initialize_from_bge=True)

    values = torch.tensor(
        [
            [0.5, -1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-2.0, 0.25, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.allclose(
        model._mtfrl_projection(values),
        bge(values),
        atol=1e-6,
        rtol=1e-6,
    )


def test_mtfrl_width_is_exactly_half_the_backbone_hidden_size():
    assert _mtfrl_hidden_width(4096, 1024) == 2048
    assert _mtfrl_hidden_width(3584, 1024) == 1792


def test_submission_mtfrl_uses_five_hard_effective_prefixes():
    class _CaptureLinear(torch.nn.Linear):
        def forward(self, values):
            self.last_input = values.detach().clone()
            return super().forward(values)

    projection = _CaptureLinear(2, 2, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.tensor([[2.0, 0.0], [1.0, 1.0]]))
    model = CLaRa.__new__(CLaRa)
    torch.nn.Module.__init__(model)
    model._mtfrl_projection = torch.nn.Sequential(projection)
    model._rag_config = SimpleNamespace(numerical_epsilon=1e-6)
    memory = torch.tensor(
        [[
            [[1.0, 0.0], [99.0, 99.0]],
            [[0.0, 2.0], [0.0, 2.0]],
            [[3.0, 0.0], [99.0, 99.0]],
            [[0.0, 4.0], [0.0, 4.0]],
            [[5.0, 0.0], [99.0, 99.0]],
        ]]
    )
    counts = torch.tensor([[1, 2, 1, 2, 1]])

    actual = model._compute_mtfrl_feedback_query(
        memory,
        effective_counts=counts,
    )

    expected_pool = torch.tensor(
        [[(1 + 0 + 3 + 0 + 5) / 5, (0 + 2 + 0 + 4 + 0) / 5]]
    )
    expected_input = torch.nn.functional.normalize(expected_pool, dim=-1)
    expected_output = expected_input @ projection.weight.detach().T
    assert torch.allclose(projection.last_input, expected_input)
    assert torch.allclose(
        actual,
        torch.nn.functional.normalize(expected_output, dim=-1),
    )


def test_release_convention_mtfrl_rejects_non_five_document_input():
    model = CLaRa.__new__(CLaRa)
    torch.nn.Module.__init__(model)
    model._mtfrl_projection = torch.nn.Identity()
    model._rag_config = SimpleNamespace(numerical_epsilon=1e-6)

    with pytest.raises(ValueError, match="five"):
        model._compute_mtfrl_feedback_query(
            torch.ones(1, 4, 2, 3),
            effective_counts=torch.ones(1, 4, dtype=torch.long),
        )


def test_submission_cfrs_matches_teacher_forced_squared_probability_error():
    logits = torch.tensor(
        [
            [[2.0, 0.0, -1.0], [0.1, 0.2, 0.3], [1.0, -1.0, 0.0]],
            [[-0.5, 0.5, 1.5], [3.0, 0.0, -2.0], [0.0, 0.0, 0.0]],
        ],
        requires_grad=True,
    )
    target_ids = torch.tensor([[0, 2, 1], [2, 0, 1]])
    target_mask = torch.tensor([[True, True, False], [True, False, True]])

    actual = CompressionFidelityReranker.squared_probability_error(
        logits, target_ids, target_mask
    )
    probabilities = logits.softmax(dim=-1)
    targets = torch.nn.functional.one_hot(
        target_ids, num_classes=logits.size(-1)
    ).to(probabilities)
    per_token = (probabilities - targets).square().sum(dim=-1)
    expected = torch.stack(
        [per_token[0, :2].mean(), per_token[1, [0, 2]].mean()]
    )

    assert torch.allclose(actual, expected, atol=1e-7, rtol=0)
    actual.sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_oracle_top100_tail_injection_preserves_order_and_gold_annotation_order():
    base = list(range(100))
    record = _construct_oracle_top100_indices(base, [50, 102, 101])

    assert record.protocol == ORACLE_TOP100_PROTOCOL
    assert record.injected_indices == (102, 101)
    assert record.evicted_indices == (98, 99)
    assert record.pool_indices[:-2] == tuple(range(98))
    assert record.pool_indices[-2:] == (102, 101)
    assert len(record.pool_indices) == len(set(record.pool_indices)) == 100
    assert {50, 102, 101}.issubset(record.pool_indices)
    assert record.pool_sha256 == _construct_oracle_top100_indices(
        base, [50, 102, 101]
    ).pool_sha256


def test_oracle_pool_keeps_first_bge_occurrence_per_page_and_injects_gold_page():
    page_ids = ["p0", "p0", "p1", "p2", "p3", "p4"]
    record = _construct_oracle_top100_indices(
        [0, 1, 2, 3, 4],
        [1, 5],
        corpus_page_ids=page_ids,
        pool_size=3,
    )

    assert record.base_indices == (0, 2, 3)
    assert record.gold_indices == (1, 5)
    assert record.injected_indices == (5,)
    assert record.evicted_indices == (3,)
    assert record.pool_indices == (0, 2, 5)
    assert record.pool_page_ids == ("p0", "p1", "p4")


def test_dense_topk_unique_pages_expands_ranked_prefix_exactly():
    query = torch.tensor([[1.0, 0.0]])
    corpus = torch.tensor(
        [[1.0, 0.0], [0.99, 0.0], [0.98, 0.0], [0.97, 0.0]]
    )
    indices = _chunked_inner_product_topk_unique_pages(
        query,
        corpus,
        ["same", "same", "page-1", "page-2"],
        3,
        chunk_size=2,
    )

    assert indices.tolist() == [[0, 2, 3]]


def test_mads_cutoff_keeps_ranked_passage_occurrences_from_the_same_page():
    documents = ["duplicate high", "duplicate low", "page one", "page two"]
    pipeline = RAGEnhancementPipeline.from_corpus(
        documents,
        corpus_doc_ids=["d0", "d1", "d2", "d3"],
        corpus_page_ids=["shared", "shared", "p1", "p2"],
        config=RAGPipelineConfig(
            use_mads=False,
            use_ccef=False,
            use_mtfrl=False,
        ),
    )
    ranked = pipeline._mads_ccef(
        "query",
        [
            _RetrievedDoc("d0", documents[0], 0, hybrid_score=1.0),
            _RetrievedDoc("d1", documents[1], 1, hybrid_score=0.9),
            _RetrievedDoc("d2", documents[2], 2, hybrid_score=0.8),
            _RetrievedDoc("d3", documents[3], 3, hybrid_score=0.7),
        ],
        top_k=3,
    )

    assert [document.doc_id for document in ranked] == ["d0", "d1", "d2"]


def test_mtfrl_oracle_scope_searches_exact_pool_and_can_overlap_first_pass():
    corpus_embeddings = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.99, 0.01]],
        dtype=torch.float32,
    )
    retriever = MemoryTokenFeedbackRetriever(
        corpus_embeddings=corpus_embeddings,
        corpus_docs=["d0", "d1", "d2", "outside"],
        corpus_ids=["d0", "d1", "d2", "outside"],
    )
    result = retriever.second_round_retrieve(
        torch.tensor([[1.0, 0.0]]),
        already_retrieved_ids=[["d0"]],
        top_k=200,
        allowed_corpus_indices=[[0, 1, 2]],
    )

    # Algorithm 1 searches D2 independently.  D1/D2 duplicate removal happens
    # only when their union is formed for the second MADS+CCEF pass.
    assert [doc.doc_id for doc in result[0]] == ["d0", "d1", "d2"]
    assert all(doc.corpus_index in {0, 1, 2} for doc in result[0])


def test_mads_bge_model_revision_is_forwarded_and_part_of_cache_key(monkeypatch):
    import transformers

    calls = []
    commit = "0123456789abcdef0123456789abcdef01234567"

    class _FakeModel:
        config = SimpleNamespace(_commit_hash=commit)

        def to(self, device):
            calls.append(("device", device))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

    def load_tokenizer(name, **kwargs):
        calls.append(("tokenizer", name, kwargs))
        return SimpleNamespace(init_kwargs={"_commit_hash": commit})

    def load_model(name, **kwargs):
        calls.append(("model", name, kwargs))
        return _FakeModel()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_model)
    modeling_aria._SEMANTIC_MODEL_CACHE.clear()

    agent = _SemanticAgent("org/semantic", "revision-a")
    agent._device = "cpu"
    agent._lazy_load()
    assert ("tokenizer", "org/semantic", {"revision": "revision-a"}) in calls
    assert ("model", "org/semantic", {"revision": "revision-a"}) in calls
    assert ("org/semantic", "revision-a", "cpu") in modeling_aria._SEMANTIC_MODEL_CACHE
    assert agent.resolved_revision == commit

    with pytest.raises(ValueError, match="revision must be non-empty"):
        _SemanticAgent("org/semantic", "")


def test_mads_external_documents_use_projected_query_and_frozen_bge_encoder():
    agent = _SemanticAgent()
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        ).numpy()

    agent._embed = embed
    scores = agent.score_projected(
        torch.tensor([1.0, 0.0]),
        ["aligned", "orthogonal", "opposed"],
    )

    assert calls == [["aligned", "orthogonal", "opposed"]]
    assert scores == pytest.approx([1.0, 0.0, -1.0])


def test_mads_online_bge_documents_use_passage_cap_and_cls_pooling():
    tokenizer_calls = []

    class _EncodedBatch(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class _Tokenizer:
        def __call__(self, texts, **kwargs):
            tokenizer_calls.append((list(texts), kwargs))
            return _EncodedBatch(
                input_ids=torch.ones(len(texts), 3, dtype=torch.long),
                attention_mask=torch.ones(len(texts), 3, dtype=torch.long),
            )

    class _Encoder:
        def __call__(self, **kwargs):
            batch_size = kwargs["input_ids"].size(0)
            states = torch.tensor(
                [
                    [[1.0, 2.0], [100.0, 200.0], [300.0, 400.0]],
                    [[3.0, 4.0], [500.0, 600.0], [700.0, 800.0]],
                ]
            )[:batch_size]
            return SimpleNamespace(last_hidden_state=states)

    agent = _SemanticAgent()
    agent._device = "cpu"
    agent._tok = _Tokenizer()
    agent._enc = _Encoder()

    encoded = torch.from_numpy(agent._embed(["first", "second"]))

    assert tokenizer_calls == [
        (
            ["first", "second"],
            {
                "return_tensors": "pt",
                "truncation": True,
                "max_length": 768,
                "padding": True,
            },
        )
    ]
    assert torch.equal(encoded, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_mads_ccef_consumes_shared_bge_scores_without_a_second_semantic_model():
    ccef = _CCEF()
    ccef.lex.score = lambda query, docs: [0.0, 1.0]
    ccef.ent.score = lambda query, docs, doc_ids=None: [0.0, 1.0]
    ccef.sem.score_projected = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("MADS must consume the caller's shared BGE scores")
    )
    scored = ccef.score_and_fuse(
        "query",
        [
            _RetrievedDoc("row-1", "second", corpus_index=1),
            _RetrievedDoc("row-0", "first", corpus_index=0),
        ],
        semantic_scores=[0.0, 1.0],
    )

    by_id = {document.doc_id: document for document in scored}
    assert by_id["row-0"].sem_raw == pytest.approx(1.0)
    assert by_id["row-1"].sem_raw == pytest.approx(0.0)
    with pytest.raises(ValueError, match=r"W_BGE\(QR\)"):
        ccef.score_and_fuse(
            "query",
            [_RetrievedDoc("row-0", "first", corpus_index=0)],
        )


def test_checkpoint_config_round_trips_declared_and_resolved_bge_revisions(tmp_path):
    declared = "release-tag"
    resolved = "0123456789abcdef0123456789abcdef01234567"
    config = CLaRaConfig(
        mads_semantic_model_revision=declared,
        mads_semantic_model_resolved_revision=resolved,
    )
    config.save_pretrained(tmp_path)
    loaded = CLaRaConfig.from_pretrained(tmp_path)
    assert loaded.mads_semantic_model_revision == declared
    assert loaded.mads_semantic_model_resolved_revision == resolved


def test_checkpoint_config_round_trips_base_decoder_revisions(tmp_path):
    declared = "release-tag"
    resolved = "fedcba9876543210fedcba9876543210fedcba98"
    config = CLaRaConfig(
        decoder_model_revision=declared,
        decoder_model_resolved_revision=resolved,
    )
    config.save_pretrained(tmp_path)
    loaded = CLaRaConfig.from_pretrained(tmp_path)
    assert loaded.decoder_model_revision == declared
    assert loaded.decoder_model_resolved_revision == resolved


def test_base_decoder_and_tokenizer_must_resolve_to_same_commit():
    model_commit = "0" * 40
    tokenizer_commit = "1" * 40
    model = CLaRa.__new__(CLaRa)
    torch.nn.Module.__init__(model)
    model.decoder = SimpleNamespace(config=SimpleNamespace(_commit_hash=model_commit))
    model.decoder_tokenizer = SimpleNamespace(
        init_kwargs={"_commit_hash": tokenizer_commit}
    )

    with pytest.raises(RuntimeError, match="different Hub commits"):
        model._record_decoder_model_revision(CLaRaConfig())


def test_exact_declared_base_commit_is_persisted_without_hub_metadata():
    commit = "abcdef0123456789abcdef0123456789abcdef01"
    model = CLaRa.__new__(CLaRa)
    torch.nn.Module.__init__(model)
    model.decoder = SimpleNamespace(config=SimpleNamespace(_commit_hash=None))
    model.decoder_tokenizer = SimpleNamespace(init_kwargs={})
    config = CLaRaConfig(decoder_model_revision=commit)

    model._record_decoder_model_revision(config)

    assert config.decoder_model_resolved_revision == commit
    assert model.decoder_model_resolved_revision == commit


def test_checkpoint_save_records_resolved_bge_commit_without_losing_declared_tag(tmp_path):
    declared = "release-tag"
    resolved = "0123456789abcdef0123456789abcdef01234567"
    model = CLaRa.__new__(CLaRa)
    torch.nn.Module.__init__(model)
    model.lora = True
    model.config = CLaRaConfig(
        lora=True,
        mads_semantic_model_revision=declared,
    )
    model._mtfrl_projection = None
    model._bge_projection = None
    model._get_all_adapters_state_dict = lambda: {
        "encoder_adapter": {"weight": torch.ones(1)}
    }
    model._get_decoder_first_and_last_layer_state_dict = lambda: {
        "weight": torch.ones(1)
    }
    model.rag_pipeline = SimpleNamespace(
        ccef=SimpleNamespace(
            sem=SimpleNamespace(
                model_name="BAAI/bge-large-en-v1.5",
                model_revision=declared,
                resolved_revision=resolved,
            )
        )
    )

    model.save_pretrained(tmp_path)
    loaded = CLaRaConfig.from_pretrained(tmp_path)
    assert loaded.mads_semantic_model_revision == declared
    assert loaded.mads_semantic_model_resolved_revision == resolved


def test_mads_entity_agent_batches_misses_and_bounds_lru_cache():
    nlp = _FakeSpacyPipeline()
    agent = _EntityAgent(cache_max_entries=2)
    agent._nlp = nlp

    assert agent.score(
        "Alice",
        ["Alice met Bob", "Carol travelled"],
        doc_ids=["a", "b"],
    ) == [1.0, 0.0]
    assert nlp.pipe_batches == [
        (["Alice met Bob", "Carol travelled"], 128),
    ]

    assert agent.score(
        "Alice",
        ["Alice met Bob", "Alice visited Carol"],
        doc_ids=["a", "c"],
    ) == [1.0, 1.0]
    assert nlp.pipe_batches[-1] == (["Alice visited Carol"], 128)
    assert list(agent._document_cache) == ["a", "c"]

    # External-document IDs can be reused by a later inference batch; a changed
    # text must invalidate the cached entity set rather than leak old evidence.
    assert agent.score("Alice", ["Bob"], doc_ids=["a"]) == [0.0]
    assert nlp.pipe_batches[-1] == (["Bob"], 128)


def test_igfr_entities_are_batched_and_lru_bounded(monkeypatch):
    nlp = _FakeSpacyPipeline()
    monkeypatch.setattr(modeling_aria, "_qca_get_spacy", lambda: nlp)
    pipeline = RAGEnhancementPipeline(
        qca=None,
        ahr=None,
        ccef=None,
        config=RAGPipelineConfig(entity_cache_max_entries=2),
    )

    entity_sets = pipeline._document_entity_sets([
        _RetrievedDoc("a", "Alice", 0),
        _RetrievedDoc("b", "Bob", 1),
        _RetrievedDoc("c", "Carol", 2),
    ])
    assert entity_sets == [{"alice"}, {"bob"}, {"carol"}]
    assert nlp.pipe_batches == [(["Alice", "Bob", "Carol"], 128)]
    assert list(pipeline._igfr_entity_cache) == ["b", "c"]

    pipeline._document_entity_sets([
        _RetrievedDoc("b", "Bob", 1),
        _RetrievedDoc("d", "Alice and Carol", 3),
    ])
    assert nlp.pipe_batches[-1] == (["Alice and Carol"], 128)
    assert list(pipeline._igfr_entity_cache) == ["b", "d"]


def test_submission_cfrs_ranking_keeps_the_reconstruction_gradient_path():
    relevance = torch.tensor([[0.2, 0.8, 0.5]], requires_grad=True)
    error = torch.tensor([0.1, 0.9, 0.5], requires_grad=True)
    scores = CompressionFidelityReranker.rerank(relevance, error)

    epsilon = 1e-6
    span = 0.8
    fidelity = torch.tensor(
        [[(0.9 - 0.1) / (span + epsilon), 0.0, (0.9 - 0.5) / (span + epsilon)]]
    )
    assert torch.allclose(scores, 0.7 * relevance + 0.3 * fidelity)
    downstream_weights = torch.tensor([[1.0, 2.0, 4.0]])
    (scores * downstream_weights).sum().backward()

    # ARIA_old.tex Appendix A.1 detaches the normalization statistics, not
    # g_i itself: every local derivative is -1 / Delta.
    expected_gradient = -0.3 * downstream_weights.reshape(-1) / (span + epsilon)
    assert torch.allclose(error.grad, expected_gradient)


def test_submission_cfrs_tied_errors_follow_literal_epsilon_formula():
    error = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
    scores = CompressionFidelityReranker.rerank(
        torch.tensor([[0.4, 0.4, 0.4]]),
        error,
    )
    assert torch.equal(scores, torch.tensor([[0.28, 0.28, 0.28]]))
    scores.sum().backward()
    assert torch.allclose(error.grad, torch.full_like(error, -0.3 / 1e-6))


def test_release_convention_cfrs_permutation_is_hard_forward_and_soft_backward():
    model = CLaRa.__new__(CLaRa)
    torch.nn.Module.__init__(model)
    memory = torch.tensor([[[[1.0]], [[2.0]], [[4.0]], [[8.0]], [[16.0]]]])
    scores = torch.tensor(
        [[0.2, 0.9, 0.5, 0.1, 0.7]],
        requires_grad=True,
    )

    ordered, hard_order = model._straight_through_cfrs_permutation(
        memory,
        scores,
        torch.ones_like(scores, dtype=torch.bool),
    )

    assert hard_order.tolist() == [[1, 4, 2, 0, 3]]
    assert torch.equal(
        ordered.detach().reshape(-1),
        torch.tensor([2.0, 16.0, 4.0, 1.0, 8.0]),
    )
    downstream_weights = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    (ordered.reshape(-1) * downstream_weights).sum().backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum() > 0


def test_submission_acr_uses_literal_epsilon_normalization_and_soft_gate():
    allocator = AdaptiveCompressionAllocator(min_ratio=0.25, max_ratio=1.0)
    ratios = allocator.ratios_from_scores(torch.tensor([[0.0, 1.0]]))
    assert ratios[0, 0] == 0.25
    assert ratios[0, 1] == pytest.approx(0.25 + 0.75 / 1.000001)
    embeddings = torch.ones(2, 4, 3, requires_grad=True)
    gated, gates, counts = allocator.apply_ratios(
        embeddings, ratios, torch.tensor([4, 4])
    )
    expected_positions = torch.arange(1, 5, dtype=torch.float32)
    expected = torch.sigmoid(
        10.0 * (ratios.squeeze(0).float().unsqueeze(1) * 4 - expected_positions)
    )
    assert torch.allclose(gates, expected)
    assert torch.allclose(gated, embeddings * gates.unsqueeze(-1))
    assert counts.tolist() == [1, 3]
    gated.sum().backward()
    assert embeddings.grad is not None and embeddings.grad.abs().sum() > 0


def test_release_convention_mtfrl_clamps_an_empty_effective_prefix_to_one():
    allocator = AdaptiveCompressionAllocator(min_ratio=0.25, max_ratio=1.0)
    embeddings = torch.ones(1, 2, 1, requires_grad=True)

    gated, gates, counts = allocator.apply_ratios(
        embeddings,
        torch.tensor([0.25]),
        base_token_counts=torch.tensor([1]),
    )

    # The paper's 1/T_i mean is undefined at zero. The release keeps one token
    # for MTFRL auditing while the generator still receives the literal soft
    # sigmoid state from Eq. (8).
    assert counts.tolist() == [1]
    assert torch.allclose(
        gates,
        torch.tensor([[torch.sigmoid(torch.tensor(-7.5)), 0.0]]),
    )
    assert torch.allclose(gated, embeddings * gates.unsqueeze(-1))


def test_submission_acr_does_not_add_singleton_or_tie_special_cases():
    allocator = AdaptiveCompressionAllocator(min_ratio=0.25, max_ratio=1.0)
    singleton = allocator.ratios_from_scores(torch.tensor([[3.0]]))
    tied = allocator.ratios_from_scores(torch.tensor([[3.0, 3.0, 3.0]]))
    assert singleton.tolist() == [[0.25]]
    assert tied.tolist() == [[0.25, 0.25, 0.25]]


def test_chunked_dense_search_matches_full_matrix_topk():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    corpus = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.7], [-1.0, 0.0]]
    )
    values, indices = _chunked_inner_product_topk(
        queries, corpus, top_k=3, chunk_size=2
    )
    expected_values, expected_indices = torch.topk(queries @ corpus.T, k=3, dim=1)
    assert torch.equal(indices, expected_indices)
    assert torch.allclose(values, expected_values)


def test_chunked_dense_search_breaks_score_ties_by_corpus_index():
    values, indices = _chunked_inner_product_topk(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.5, 0.0]]),
        top_k=2,
        chunk_size=1,
    )
    assert indices.tolist() == [[0, 1]]
    assert values.tolist() == [[1.0, 1.0]]


def test_chunked_dense_search_keeps_lowest_indices_at_large_cutoff_tie():
    corpus = torch.ones(257, 2)
    values, indices = _chunked_inner_product_topk(
        torch.tensor([[1.0, 0.0]]),
        corpus,
        top_k=7,
        chunk_size=31,
    )
    assert values.tolist() == [[1.0] * 7]
    assert indices.tolist() == [list(range(7))]


def test_large_tensor_finite_validation_uses_bounded_chunks(monkeypatch):
    values = torch.zeros(10, 4)
    values[-1, -1] = float("nan")
    observed_sizes = []
    original_isfinite = torch.isfinite

    def recording_isfinite(chunk):
        observed_sizes.append(chunk.numel())
        return original_isfinite(chunk)

    monkeypatch.setattr(modeling_aria.torch, "isfinite", recording_isfinite)
    assert not _tensor_is_finite_in_chunks(values, max_chunk_elements=12)
    assert observed_sizes == [12, 12, 12, 4]


def test_ahr_reuses_normalized_float32_cpu_contiguous_embedding_storage():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    pipeline = RAGEnhancementPipeline.from_corpus(
        ["first document", "second document"],
        corpus_doc_ids=["first", "second"],
        doc_embeddings=embeddings,
    )

    prepared = pipeline.ahr.dense_embeddings
    assert prepared is not None
    assert prepared.untyped_storage().data_ptr() == embeddings.untyped_storage().data_ptr()
    assert not prepared.requires_grad


def test_ahr_normalizes_and_copies_embeddings_outside_zero_copy_contract():
    embeddings = torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float32)
    pipeline = RAGEnhancementPipeline.from_corpus(
        ["first document", "second document"],
        corpus_doc_ids=["first", "second"],
        doc_embeddings=embeddings,
    )

    prepared = pipeline.ahr.dense_embeddings
    assert prepared is not None
    assert prepared.untyped_storage().data_ptr() != embeddings.untyped_storage().data_ptr()
    assert torch.allclose(
        prepared,
        torch.tensor([[0.6, 0.8], [0.0, 1.0]], dtype=torch.float32),
    )
    assert torch.equal(embeddings, torch.tensor([[3.0, 4.0], [0.0, 2.0]]))


def test_pipeline_reuses_prebuilt_bm25_only_for_identical_corpus_list():
    documents = ["first document", "second document"]
    bm25 = _BM25Index().build(documents)
    pipeline = RAGEnhancementPipeline.from_corpus(
        documents,
        corpus_doc_ids=["first", "second"],
        bm25_index=bm25,
    )
    assert pipeline.ahr.bm25 is bm25

    try:
        RAGEnhancementPipeline.from_corpus(
            list(documents),
            corpus_doc_ids=["first", "second"],
            bm25_index=bm25,
        )
    except ValueError as exc:
        assert "exact corpus_docs list" in str(exc)
    else:
        raise AssertionError("misaligned prebuilt BM25 index was accepted")


def test_ahr_batches_dense_queries_into_one_exact_corpus_scan(monkeypatch):
    documents = ["alpha document", "beta document", "gamma document"]
    retriever = _AdaptiveHybridRetriever(
        _BM25Index().build(documents),
        documents,
        ["a", "b", "c"],
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
    )
    original_search = modeling_aria._chunked_inner_product_topk
    query_batch_sizes = []

    def recording_search(queries, corpus, top_k, chunk_size=65_536):
        query_batch_sizes.append(queries.shape[0])
        return original_search(queries, corpus, top_k, chunk_size)

    monkeypatch.setattr(
        modeling_aria, "_chunked_inner_product_topk", recording_search
    )
    results = retriever.retrieve_batch(
        ["unseen one", "unseen two"],
        [
            QCAResult("unseen one", QuestionType.SIMPLE, 1.0, 1, 0),
            QCAResult("unseen two", QuestionType.MULTI_HOP, 1.0, 2, 0),
        ],
        query_embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        top_k=2,
    )

    assert query_batch_sizes == [2]
    assert [document.doc_id for document in results[0]] == ["a", "b"]
    assert [document.doc_id for document in results[1]] == ["b", "a"]


def test_igfr_secondary_pool_is_globally_bounded_across_iterations():
    first = [
        _RetrievedDoc(str(index), str(index), index, hybrid_score=float(index))
        for index in range(4)
    ]
    second = [
        _RetrievedDoc(str(index), str(index), index, hybrid_score=float(index))
        for index in range(4, 8)
    ]

    merged = _merge_bounded_retrieval_pool(first, second, limit=5)

    assert [doc.doc_id for doc in merged] == ["7", "6", "5", "4", "3"]


def test_ccef_never_reinserts_documents_below_threshold():
    pipeline = RAGEnhancementPipeline(
        qca=None,
        ahr=SimpleNamespace(corpus_page_ids=["0", "1", "2"]),
        ccef=None,
        config=RAGPipelineConfig(
            use_mads=False,
            use_ccef=True,
            ccef_filter_threshold=0.9,
        ),
    )
    retrieved = [
        _RetrievedDoc(str(index), str(index), index, hybrid_score=score)
        for index, score in enumerate([1.0, 0.5, 0.1])
    ]

    survivors = pipeline._mads_ccef("query", retrieved, top_k=3)

    assert [doc.doc_id for doc in survivors] == ["0"]


def test_submission_ccef_retains_exactly_the_top_five_survivors():
    pipeline = RAGEnhancementPipeline(
        qca=None,
        ahr=SimpleNamespace(corpus_page_ids=[str(index) for index in range(6)]),
        ccef=None,
        config=RAGPipelineConfig(
            use_mads=False,
            use_ccef=True,
            ccef_filter_threshold=0.0,
        ),
    )
    retrieved = [
        _RetrievedDoc(str(index), str(index), index, hybrid_score=float(6 - index))
        for index in range(6)
    ]

    survivors = pipeline._mads_ccef("query", retrieved, top_k=5)

    assert [document.doc_id for document in survivors] == ["0", "1", "2", "3", "4"]


def test_cr4_clara_prompt_reserves_all_five_memory_blocks():
    max_length = _fixed_memory_prompt_max_length(5, 1024 // 4, 1024)
    assert max_length == 2560
    assert max_length > 5 * (1024 // 4)


def test_encoder_memory_is_adjacent_to_source_before_batch_padding():
    ids, attention = _pack_variable_encoder_memory_rows(
        source_rows=[[1, 2, 3], [4, 5, 6, 7, 8]],
        memory_counts=[1, 2],
        memory_token_ids=[10, 11, 12],
        pad_token_id=0,
    )

    assert ids.tolist() == [[1, 2, 3, 10, 0, 0, 0], [4, 5, 6, 7, 8, 10, 11]]
    assert attention.tolist() == [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]]


def test_phase2_prunes_only_slots_beyond_each_documents_base_count():
    input_ids = torch.tensor([
        [99, 10, 11, 12, 13, 90, 10, 11, 12, 13, 90, 20, 30],
        [99, 10, 11, 12, 13, 90, 10, 11, 12, 13, 90, 20, 30],
    ])
    attention = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[:, -1] = 30
    query_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    query_mask[:, -2] = True

    ids, pruned_attention, pruned_labels, pruned_query = _prune_padded_memory_slots(
        input_ids,
        attention,
        labels,
        query_mask,
        base_counts=torch.tensor([[2, 4], [1, 3]]),
        memory_token_ids=torch.tensor([10, 11, 12, 13]),
        slots_per_document=4,
        pad_token_id=0,
        padding_side="left",
    )

    assert ids.tolist() == [
        [99, 10, 11, 90, 10, 11, 12, 13, 90, 20, 30],
        [0, 0, 99, 10, 90, 10, 11, 12, 90, 20, 30],
    ]
    assert pruned_attention.sum(dim=1).tolist() == [11, 9]
    assert (
        torch.isin(ids, torch.tensor([10, 11, 12, 13])) & pruned_attention.bool()
    ).sum(dim=1).tolist() == [6, 4]
    assert pruned_labels[:, -1].tolist() == [30, 30]
    assert pruned_query[:, -2].tolist() == [True, True]


def test_base_slot_pruning_preserves_acr_soft_gate_after_effective_floor():
    class _EmbeddingDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(64, 2)

        def get_input_embeddings(self):
            return self.embedding

    model = CLaRa.__new__(CLaRa)
    torch.nn.Module.__init__(model)
    model.decoder = _EmbeddingDecoder()
    model.decoder_tokenizer = SimpleNamespace(
        mem_token_ids_pt=torch.tensor([10, 11, 12, 13])
    )
    raw_memory = torch.ones(1, 4, 2, requires_grad=True)
    allocator = AdaptiveCompressionAllocator(min_ratio=0.75, max_ratio=1.0)
    gated, _, effective_counts = allocator.apply_ratios(
        raw_memory,
        torch.tensor([0.75]),
        base_token_counts=torch.tensor([4]),
    )
    assert effective_counts.tolist() == [3]

    # Training inserts all four real base slots.  Slot four is below the hard
    # inference floor but still carries the paper's differentiable soft gate.
    input_ids = torch.tensor([[10, 11, 12, 13, 20]])
    embeds = model._replace_variable_memory_embeddings(
        gated.view(1, 1, 4, 2),
        torch.tensor([[4]]),
        input_ids,
    )
    embeds[:, :4].sum().backward()
    assert raw_memory.grad is not None
    assert raw_memory.grad[0, 3].abs().sum() > 0
