import torch

from openrlhf.models.modeling_aria import CLaRa, CLaRaConfig, QR_INPUT_SCHEME


class _RecordingTokenizer:
    def __init__(self):
        self.questions = None
        self.kwargs = None

    def __call__(self, questions, **kwargs):
        self.questions = list(questions)
        self.kwargs = dict(kwargs)
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


def test_qr_tokenization_is_native_and_compression_rate_independent():
    model = CLaRa.__new__(CLaRa)
    torch.nn.Module.__init__(model)
    tokenizer = _RecordingTokenizer()
    model.decoder_tokenizer = tokenizer

    encoded = model._prepare_query_inputs(["Who wrote the novel?"], max_length=128)

    assert tokenizer.questions == ["Who wrote the novel?"]
    assert tokenizer.kwargs == {
        "return_tensors": "pt",
        "padding": "longest",
        "truncation": True,
        "max_length": 128,
        "add_special_tokens": True,
    }
    assert encoded["attention_mask"].tolist() == [[1, 1, 1]]


def test_checkpoint_config_records_qr_input_scheme():
    config = CLaRaConfig(qr_input_scheme=QR_INPUT_SCHEME)
    assert config.qr_input_scheme == "native-tokenizer-final-token-v1"
