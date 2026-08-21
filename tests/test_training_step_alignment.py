from torch.utils.data import Dataset

from openrlhf.cli.train_sft import _training_step_counts
from openrlhf.utils.distributed_sampler import DistributedSampler


class _LengthOnlyDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index


class _LengthOnlyLoader:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


def test_aligned_sampler_drops_deterministic_global_tail():
    dataset = _LengthOnlyDataset(65)
    per_rank = [
        list(
            DistributedSampler(
                dataset,
                num_replicas=8,
                rank=rank,
                shuffle=False,
                drop_last=True,
                drop_last_multiple=4,
            )
        )
        for rank in range(8)
    ]

    assert all(len(indices) == 8 for indices in per_rank)
    assert sorted(index for indices in per_rank for index in indices) == list(range(64))


def test_scheduler_rejects_epoch_with_partial_accumulation():
    try:
        _training_step_counts(
            _LengthOnlyLoader(976_058),
            accumulated_gradient=4,
            max_epochs=3,
        )
    except ValueError as exc:
        assert "must be divisible by gradient accumulation" in str(exc)
    else:
        raise AssertionError("unaligned epoch was accepted")
