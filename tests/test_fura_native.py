import unittest

import torch

from esx.backends.fura_native import FuraNative
from esx.backends.structured_es import StructuredES


class _Method:
    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


class _Linear(torch.nn.Module):
    """Minimal stand-in for a vLLM linear: weight + quant_method.apply."""

    def __init__(self, out_f, in_f, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(out_f, in_f, generator=g) * 0.05, requires_grad=False)
        self.input_size, self.output_size = in_f, out_f
        self.quant_method = _Method()

    def forward(self, x):
        return self.quant_method.apply(self, x)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
        for i, layer in enumerate(self.layers):
            layer.mlp = torch.nn.Module()
            layer.mlp.gate_up_proj = _Linear(96, 48, 2 * i)
            layer.mlp.down_proj = _Linear(48, 64, 2 * i + 1)


class FuraNativeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _Model().double()
        self.fura = FuraNative(self.model, targets="mlp", core_dtype=torch.float64)
        for layer in self.fura.layers:
            layer.vh = layer.vh.double()
            layer.vh_t = layer.vh_t.double()

    def _set_table(self, entries):
        # fp64 tables so the comparison is exact up to fp64 rounding
        self.fura.set_slots(entries)
        for li, layer in enumerate(self.fura.layers):
            rows = [torch.zeros(layer.n_blk, layer.b, dtype=torch.float64)]
            for uid, seed, sigma in entries:
                c = self.fura.adapters[uid][li].double() + sigma * layer.noise(seed).double()
                rows.append(c)
            layer.table = torch.stack(rows) * layer.inv_s.double()

    def test_slot_zero_and_zero_adapter_are_exact(self):
        self.fura.new_adapter("u")
        self._set_table([("u", 0, 0.0)])
        x = torch.randn(5, 48, dtype=torch.float64)
        lin = self.model.layers[0].mlp.gate_up_proj
        base = torch.nn.functional.linear(x, lin.weight)
        for slot in (0, 1):
            self.fura.set_token_slots(torch.full((5,), slot, dtype=torch.int64))
            self.assertTrue(torch.equal(lin(x), base))
        self.fura.set_token_slots(None)

    def test_matches_materialized_and_structured_es(self):
        self.fura.new_adapter("u")
        g = torch.Generator().manual_seed(1)
        for c, layer in zip(self.fura.adapters["u"], self.fura.layers):
            c.copy_(torch.randn(c.shape, generator=g) * 0.01)
        self._set_table([("u", 11, 0.02), ("u", 11, -0.02)])
        names = [n for n, m in self.model.named_modules() if hasattr(m, "quant_method")]
        infos = [(n, l.tensor_id, dict(self.model.named_modules())[n].weight) for n, l in zip(names, self.fura.layers)]
        for li, layer in enumerate(self.fura.layers):
            lin = layer.module
            x = torch.randn(7, layer.in_f, dtype=torch.float64)
            for slot, sign in ((1, 1.0), (2, -1.0)):
                coef = [self.fura.adapters["u"][i].double() + sign * 0.02 * self.fura.layers[i].noise(11).double()
                        for i in range(len(self.fura.layers))]
                ref = self.fura.materialize(coef, li, lin.weight.data)
                self.fura.set_token_slots(torch.full((7,), slot, dtype=torch.int64))
                y = lin(x)
                self.fura.set_token_slots(None)
                torch.testing.assert_close(y, x @ ref.t().double(), rtol=1e-5, atol=1e-6)  # fp32 inv_s
        # The materialized FuRA-S of StructuredES renders U diag(S0 + c + sigma eps) Vh: same weight.
        model2 = _Model().double()
        infos2 = [(n, self.fura.layers[i].tensor_id, dict(model2.named_modules())[n].weight)
                  for i, n in enumerate(names)]
        ses = StructuredES(model2, infos2, scope="fura_s")
        for st, c in zip(ses.states, self.fura.adapters["u"]):
            st["coef"].copy_(c)
        ses.perturb(seed=11, sigma=0.02)
        for li, layer in enumerate(self.fura.layers):
            coef = [self.fura.adapters["u"][i].double() + 0.02 * self.fura.layers[i].noise(11).double()
                    for i in range(len(self.fura.layers))]
            ref = self.fura.materialize(coef, li, dict(self.model.named_modules())[names[li]].weight.data)
            w2 = dict(model2.named_modules())[names[li]].weight.data
            torch.testing.assert_close(w2.double(), ref.double(), rtol=1e-5, atol=1e-6)

    def test_update_is_seeded_sum(self):
        self.fura.new_adapter("u")
        self.fura.update("u", [3, 4], [1.0, -0.5], eta=0.2)
        for c, layer in zip(self.fura.adapters["u"], self.fura.layers):
            torch.testing.assert_close(c, 0.1 * (layer.noise(3) - 0.5 * layer.noise(4)), rtol=1e-6, atol=1e-7)

    def test_slot_groups_give_every_sequence_its_own_direction(self):
        for uid in ("u", "v"):
            self.fura.new_adapter(uid)
        g = torch.Generator().manual_seed(2)
        for c in self.fura.adapters["v"]:
            c.copy_(torch.randn(c.shape, generator=g) * 0.01)
        firsts = self.fura.set_slot_groups([("u", 5, 3, 0.02), ("v", 6, 2, 0.03)])
        self.assertEqual(firsts, [1, 7])
        for li, layer in enumerate(self.fura.layers):
            self.assertEqual(layer.table.shape[0], 1 + 2 * 3 + 2 * 2)
            for uid, seed, n, sigma, first in (("u", 5, 3, 0.02, 1), ("v", 6, 2, 0.03, 7)):
                eps = layer.noise_block(seed, n)
                for k in range(n):
                    for s, sign in ((0, 1.0), (1, -1.0)):
                        ref = ((self.fura.adapters[uid][li] + sign * sigma * eps[k]) * layer.inv_s).to(torch.bfloat16)
                        self.assertTrue(torch.equal(layer.table[first + 2 * k + s], ref))
            # Distinct directions within a block, and rows of different blocks are independent.
            eps = layer.noise_block(5, 3)
            self.assertFalse(torch.allclose(eps[0], eps[1]))
            self.assertTrue(torch.equal(layer.table[0], torch.zeros_like(layer.table[0])))

    def test_update_block_is_weighted_direction_mean(self):
        self.fura.new_adapter("u")
        self.fura.update_block("u", 9, [1.0, -2.0, 0.5], eta=0.3)
        for c, layer in zip(self.fura.adapters["u"], self.fura.layers):
            eps = layer.noise_block(9, 3)
            torch.testing.assert_close(c, 0.1 * (eps[0] - 2.0 * eps[1] + 0.5 * eps[2]), rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
