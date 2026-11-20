from __future__ import annotations

import copy
import unittest

import torch

from esx.backends.fura_s_pretransform import FuraSPretransform
from esx.backends.seeded_model_es import SeedReplayModelES


class TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(24, 36, bias=False)
        self.down_proj = torch.nn.Linear(36, 24, bias=True)

    def forward(self, x):
        return self.down_proj(torch.tanh(self.up_proj(x)))


class FuraSPretransformTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.base = TinyMLP().double()
        self.x = torch.randn(5, 24, dtype=torch.float64)

    def dense(self):
        model = copy.deepcopy(self.base)
        es = SeedReplayModelES()
        es.init(model, parameter_scope="fura_s", verbose=False)
        return model, es

    def pre(self, mode="fused"):
        model = copy.deepcopy(self.base)
        return model, FuraSPretransform(model, mode=mode, s_floor=0.0)

    def test_identity_at_zero_is_bit_exact(self):
        model, _ = self.pre()
        self.assertTrue(torch.equal(model(self.x), self.base(self.x)))

    def test_matches_dense_fura_s(self):
        for mode in ("fused", "exact"):
            dense_model, dense = self.dense()
            pre_model, pre = self.pre(mode)
            dense.apply(seed=7, sigma=0.05)
            pre.perturb(seed=7, sigma=0.05)
            torch.testing.assert_close(pre_model(self.x), dense_model(self.x), rtol=1e-5, atol=1e-5)
            dense.revert(seed=7, sigma=0.05)
            pre.restore()
            dense.update(seeds=[1, 2, 3], rewards=[0.1, -0.3, 0.7], alpha=0.2)
            weights = SeedReplayModelES._normalize_rewards([0.1, -0.3, 0.7], "zscore")
            pre.update(seeds=[1, 2, 3], weights=weights, eta=0.2)
            torch.testing.assert_close(pre_model(self.x), dense_model(self.x), rtol=1e-5, atol=1e-5)
            pre.reset()
            self.assertTrue(torch.equal(pre_model(self.x), self.base(self.x)))

    def test_population_batch_matches_sequential_members(self):
        model, pre = self.pre()
        pre.update(seeds=[4], weights=[1.0], eta=0.1)
        x = torch.randn(3, 2, 7, 24, dtype=torch.float64)  # 3 members x 2 rows each
        seeds, sigmas = [11, 12, 13], [0.05, -0.05, 0.02]
        pre.perturb_members(seeds=seeds, sigmas=sigmas)
        batched = model(x.reshape(6, 7, 24)).reshape(3, 2, 7, 24)
        pre.restore()
        for m, (seed, sigma) in enumerate(zip(seeds, sigmas)):
            pre.perturb(seed=seed, sigma=sigma)
            torch.testing.assert_close(batched[m], model(x[m]), rtol=1e-12, atol=1e-12)
            pre.restore()

    def test_trainable_coefficients_get_gradients(self):
        model = copy.deepcopy(self.base)
        pre = FuraSPretransform(model, trainable=True)
        for p in model.parameters():
            p.requires_grad_(False)
        model(self.x).pow(2).sum().backward()
        self.assertTrue(all(c.grad is not None and c.grad.abs().sum() > 0 for c in pre.parameters()))

    def test_quantized_core_keeps_exact_start(self):
        model = copy.deepcopy(self.base)
        FuraSPretransform(model, core_quantizer=lambda vh: vh.half().double())
        self.assertTrue(torch.equal(model(self.x), self.base(self.x)))


if __name__ == "__main__":
    unittest.main()
