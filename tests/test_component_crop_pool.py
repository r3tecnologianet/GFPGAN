"""ComponentCropPool: the FIFO history of (real, fake) component crops."""
import torch

from gfpgan.models.component_crop_pool import ComponentCropPool


def _crops(value, n=2):
    return torch.full((n, 3, 4, 4), float(value)), torch.full((n, 3, 4, 4), -float(value))


def test_the_pool_keeps_the_newest_crops_up_to_capacity():
    pool = ComponentCropPool(capacity=5)
    for i in range(1, 5):  # 4 pushes of 2 crops: 8 crops into a pool of 5
        pool.push(*_crops(i))
    assert len(pool) == 5
    real, _ = pool.sample(5)
    assert sorted(real[:, 0, 0, 0].tolist()) == [2.0, 3.0, 3.0, 4.0, 4.0]  # value 1 and one of the 2s are gone


def test_sample_returns_pairs_pushed_together():
    pool = ComponentCropPool(capacity=8)
    for i in range(1, 5):
        pool.push(*_crops(i))
    real, fake = pool.sample(6)
    assert torch.equal(real, -fake)


def test_sample_returns_at_most_n_and_at_most_the_pool_size():
    pool = ComponentCropPool(capacity=8)
    pool.push(*_crops(1, n=3))
    assert pool.sample(2)[0].shape[0] == 2
    real, fake = pool.sample(10)
    assert real.shape[0] == fake.shape[0] == 3
    # without replacement: all three distinct crops come back
    pool = ComponentCropPool(capacity=8)
    pool.push(torch.arange(3.0).view(3, 1, 1, 1), torch.arange(3.0).view(3, 1, 1, 1))
    assert sorted(pool.sample(3)[0].flatten().tolist()) == [0.0, 1.0, 2.0]


def test_sample_on_an_empty_pool_returns_none():
    assert ComponentCropPool(capacity=4).sample(2) is None


def test_the_pool_stores_detached_tensors():
    pool = ComponentCropPool(capacity=4)
    real = torch.rand(2, 3, 4, 4, requires_grad=True)
    fake = torch.rand(2, 3, 4, 4, requires_grad=True)
    pool.push(real, fake)
    out_real, out_fake = pool.sample(2)
    assert not out_real.requires_grad and not out_fake.requires_grad
