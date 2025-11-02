# utils/normalizer.py
import numpy as np
import torch

class RunningMeanStd:
    """
    Running mean and variance estimator (for states/observations).
    Keeps track of mean, var and count and can normalize inputs.
    """
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.eps = 1e-8

    def update(self, x):
        """
        x: numpy array with shape (n, *shape) or (*shape,) for single
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == self.mean.ndim:
            # single example
            batch_mean = x
            batch_var = np.zeros_like(self.var)
            batch_count = 1.0
        else:
            # batch of examples
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]

        # Welford / stable combine
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        arr = np.asarray(x, dtype=np.float32)
        return (arr - self.mean.astype(np.float32)) / (np.sqrt(self.var + self.eps).astype(np.float32))

    def normalize_torch(self, t: torch.Tensor):
        # normalize a torch tensor (on CPU or GPU). Uses stored numpy mean/var.
        mean = torch.from_numpy(self.mean.astype(np.float32)).to(t.device)
        std = torch.from_numpy(np.sqrt(self.var + self.eps).astype(np.float32)).to(t.device)
        return (t - mean) / std