"""A FIFO history of (real, fake) facial component crops for the component discriminators.

The component discriminators see only `batch_size` real and fake crops per step. Mixing in crops from earlier steps
gives each discriminator update more samples without enlarging the generator batch, the history buffer of
Shrivastava et al. 2017 (SimGAN) and the image pool of Zhu et al. 2017 (CycleGAN). This file does not end in
`_model.py`, so `gfpgan/models/__init__.py` does not auto-import it.
"""
import torch


class ComponentCropPool():
    """Holds up to `capacity` (real, fake) crop pairs, dropping the oldest first.

    Args:
        capacity (int): Maximum number of pairs kept.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.real = None
        self.fake = None

    def __len__(self):
        return 0 if self.real is None else self.real.size(0)

    def push(self, real, fake):
        """Append a batch of pairs (detached), keeping only the newest `capacity`."""
        real, fake = real.detach(), fake.detach()
        if self.real is None:
            self.real, self.fake = real, fake
        else:
            self.real = torch.cat([self.real, real], dim=0)
            self.fake = torch.cat([self.fake, fake], dim=0)
        self.real = self.real[-self.capacity:]
        self.fake = self.fake[-self.capacity:]

    def sample(self, n):
        """Draw `min(n, len(self))` pairs uniformly at random, without replacement.

        Returns:
            tuple[Tensor, Tensor] | None: (real, fake), or None when the pool is empty.
        """
        size = len(self)
        if size == 0:
            return None
        idx = torch.randperm(size, device=self.real.device)[:n]
        return self.real[idx], self.fake[idx]
