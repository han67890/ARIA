from types import SimpleNamespace

import torch
from peft import LoraConfig
from transformers import MistralConfig, MistralForCausalLM

from openrlhf.cli.train_sft import enable_gradient_checkpointing


class _FakeDecoder:
    def __init__(self):
        self.config = SimpleNamespace(use_cache=True)
        self.is_gradient_checkpointing = False
        self.kwargs = None
        self.input_grads_enabled = False
        self.active = ["encoder_adapter"]
        self._aria_test_adapters_enabled = True

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.kwargs = gradient_checkpointing_kwargs
        self.is_gradient_checkpointing = True

    def enable_input_require_grads(self):
        self.input_grads_enabled = True

    def active_adapters(self):
        return list(self.active)

    def set_adapter(self, names):
        self.active = [names] if isinstance(names, str) else list(names)

    def enable_adapters(self):
        self._aria_test_adapters_enabled = True

    def disable_adapters(self):
        self._aria_test_adapters_enabled = False


def test_gradient_checkpointing_launcher_flag_changes_decoder_runtime():
    decoder = _FakeDecoder()
    enable_gradient_checkpointing(SimpleNamespace(decoder=decoder))
    assert decoder.config.use_cache is False
    assert decoder.is_gradient_checkpointing is True
    assert decoder.kwargs == {"use_reentrant": False}
    assert decoder.input_grads_enabled is True


def test_multi_adapter_checkpoint_recompute_restores_forward_adapter():
    decoder = _FakeDecoder()
    model = SimpleNamespace(
        decoder=decoder,
        adapter_keys=[
            "encoder_adapter",
            "decoder_adapter",
            "query_reasoner_adapter",
        ],
    )
    enable_gradient_checkpointing(model)
    assert decoder.kwargs["use_reentrant"] is False
    assert callable(decoder.kwargs["context_fn"])

    decoder.set_adapter("query_reasoner_adapter")
    forward_context, recompute_context = decoder.kwargs["context_fn"]()
    decoder.set_adapter(model.adapter_keys)
    with forward_context:
        assert decoder.active == model.adapter_keys
    with recompute_context:
        assert decoder.active == ["query_reasoner_adapter"]
    assert decoder.active == model.adapter_keys


def test_checkpoint_recompute_restores_disabled_adapter_state():
    decoder = _FakeDecoder()
    model = SimpleNamespace(decoder=decoder, adapter_keys=["encoder_adapter"])
    enable_gradient_checkpointing(model)

    decoder.disable_adapters()
    _, recompute_context = decoder.kwargs["context_fn"]()
    decoder.enable_adapters()
    with recompute_context:
        assert decoder._aria_test_adapters_enabled is False
        assert decoder.active == ["encoder_adapter"]
    assert decoder._aria_test_adapters_enabled is True


def test_multi_adapter_checkpointing_rejects_legacy_decoder_api():
    class _LegacyDecoder(_FakeDecoder):
        def _set_gradient_checkpointing(self, module, value=False):
            pass

    decoder = _LegacyDecoder()
    model = SimpleNamespace(
        decoder=decoder,
        adapter_keys=["encoder_adapter", "decoder_adapter"],
    )
    try:
        enable_gradient_checkpointing(model)
    except RuntimeError as exc:
        assert "unsafe legacy API" in str(exc)
    else:
        raise AssertionError("legacy multi-adapter checkpointing was accepted")


def _tiny_multi_adapter_decoder():
    torch.manual_seed(7)
    decoder = MistralForCausalLM(
        MistralConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )
    adapter_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=2,
        lora_alpha=4,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
    )
    adapter_names = [
        "encoder_adapter",
        "decoder_adapter",
        "query_reasoner_adapter",
    ]
    for name in adapter_names:
        decoder.add_adapter(adapter_config, name)
    return decoder, adapter_names


def _multi_adapter_gradients(*, checkpointed):
    decoder, adapter_names = _tiny_multi_adapter_decoder()
    wrapper = SimpleNamespace(decoder=decoder, adapter_keys=adapter_names)
    if checkpointed:
        enable_gradient_checkpointing(wrapper)
    token_ids = torch.tensor([[1, 2, 3, 4, 5]])
    loss = torch.zeros(())
    for name in adapter_names:
        decoder.set_adapter(name)
        loss = loss + decoder(input_ids=token_ids, labels=token_ids).loss
    # ARIA makes every adapter trainable before returning to the trainer.  The
    # checkpoint context must nevertheless replay each forward with its own one.
    decoder.set_adapter(adapter_names)
    loss.backward()
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in decoder.named_parameters()
        if any(f".{adapter}." in name for adapter in adapter_names)
    }


def test_checkpointed_multi_adapter_gradients_match_non_checkpointed_forward():
    expected = _multi_adapter_gradients(checkpointed=False)
    actual = _multi_adapter_gradients(checkpointed=True)
    assert actual.keys() == expected.keys()
    assert actual
    for name in actual:
        assert torch.allclose(actual[name], expected[name], rtol=1e-5, atol=1e-6), name


def _single_adapter_with_frozen_generator_gradients(*, checkpointed):
    decoder, _ = _tiny_multi_adapter_decoder()
    # Keep one registered adapter to mirror Phase I, where the generator branch
    # runs with all adapters disabled after the compressor branch.
    adapter_names = ["encoder_adapter"]
    wrapper = SimpleNamespace(decoder=decoder, adapter_keys=adapter_names)
    if checkpointed:
        enable_gradient_checkpointing(wrapper)
    token_ids = torch.tensor([[1, 2, 3, 4, 5]])
    decoder.set_adapter("encoder_adapter")
    loss = decoder(input_ids=token_ids, labels=token_ids).loss
    decoder.disable_adapters()
    try:
        loss = loss + decoder(input_ids=token_ids, labels=token_ids).loss
    finally:
        decoder.enable_adapters()
        decoder.set_adapter("encoder_adapter")
    loss.backward()
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in decoder.named_parameters()
        if ".encoder_adapter." in name
    }


def test_checkpointed_disabled_generator_matches_phase1_non_checkpointed_gradients():
    expected = _single_adapter_with_frozen_generator_gradients(checkpointed=False)
    actual = _single_adapter_with_frozen_generator_gradients(checkpointed=True)
    assert actual.keys() == expected.keys()
    for name in actual:
        assert torch.allclose(actual[name], expected[name], rtol=1e-5, atol=1e-6), name
