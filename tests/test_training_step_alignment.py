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


def test_phase1_sampler_materializes_one_complete_global_batch_per_step():
    sampler = DistributedSampler(
        _LengthOnlyDataset(7_808_465),
        num_replicas=8,
        rank=0,
        shuffle=True,
        drop_last=True,
        # 8 ranks * 16 examples/rank = the paper's global batch 128.
        drop_last_multiple=16,
    )

    assert sampler.num_samples == 976_048
    assert sampler.total_size == 7_808_384
    assert len(sampler) % 16 == 0

    # Checkpoint state records whole global batches, so a resumed partial epoch
    # also starts and ends on a physical minibatch boundary.
    sampler.set_epoch(1, consumed_samples=5 * 128)
    assert len(sampler) == 975_968
    assert len(sampler) % 16 == 0


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


def test_scheduler_step_count_matches_aligned_phase1_loader():
    updates_per_epoch, total_updates = _training_step_counts(
        _LengthOnlyLoader(61_003),
        accumulated_gradient=1,
        max_epochs=3,
    )

    assert updates_per_epoch == 61_003
    assert total_updates == 183_009


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
