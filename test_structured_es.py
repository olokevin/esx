from __future__ import annotations

import copy
import unittest

import torch

from algorithms.es.seeded_model_es import SeedReplayModelES
from algorithms.es.structured_es import (
    StructuredES,
    _cayley,
    _rotate_columns,
    _rotate_rows,
    _skew,
    closest_factor_pair,
    normalize_scope_options,
)
from vllm_math_es_worker import WorkerExtension


class FusedLinear(torch.nn.Linear):
    """Stands in for a vLLM fused projection such as ``qkv_proj``."""

    def __init__(self, in_features: int, output_sizes: list[int]):
        super().__init__(in_features, sum(output_sizes), bias=False)
        self.output_sizes = output_sizes


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(10, 24)
        self.q_proj = torch.nn.Linear(24, 24, bias=False)
        self.down_proj = torch.nn.Linear(36, 24, bias=True)
        self.qkv_proj = FusedLinear(24, [16, 8, 8])
        self.norm = torch.nn.LayerNorm(24)


def singular_values(weight: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(weight.double())


def block_singular_values(weight: torch.Tensor) -> torch.Tensor:
    out_f, in_f = weight.shape
    n_blk, b = closest_factor_pair(in_f)
    return torch.linalg.svdvals(weight.double().reshape(out_f, n_blk, b).permute(1, 0, 2))


def make_es(scope: str, **options) -> tuple[TinyModel, SeedReplayModelES, dict]:
    torch.manual_seed(7)
    model = TinyModel()
    es = SeedReplayModelES()
    result = es.init(model, parameter_scope=scope, scope_options=options or None, verbose=False)
    return model, es, result


def weights(model: TinyModel) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


class KernelTest(unittest.TestCase):
    def test_closest_factor_pair(self) -> None:
        self.assertEqual(closest_factor_pair(3584), (56, 64))
        self.assertEqual(closest_factor_pair(36), (6, 6))
        self.assertEqual(closest_factor_pair(7), (1, 7))

    def test_cayley_is_orthogonal(self) -> None:
        generator = torch.Generator().manual_seed(1)
        omega = _skew(3, 8, generator, "cpu").double()
        for scale in (1e-4, 5e-2, 1.0):
            c = _cayley(omega, scale)
            eye = torch.eye(8, dtype=torch.float64).expand(3, 8, 8)
            self.assertLess(float((c @ c.transpose(1, 2) - eye).abs().max()), 1e-12)

    def test_block_kernels_match_the_dense_operator(self) -> None:
        generator = torch.Generator().manual_seed(2)
        x = torch.randn(12, 8, generator=generator, dtype=torch.float64)
        perm_l = torch.randperm(12, generator=generator)
        perm_r = torch.randperm(8, generator=generator)
        c_l = _cayley(_skew(3, 4, generator, "cpu").double(), 0.3)
        c_r = _cayley(_skew(2, 4, generator, "cpu").double(), 0.3)
        dense_l = torch.zeros(12, 12, dtype=torch.float64)
        dense_l[perm_l[:, None], perm_l[None, :]] = torch.block_diag(*c_l)
        dense_r = torch.zeros(8, 8, dtype=torch.float64)
        dense_r[perm_r[:, None], perm_r[None, :]] = torch.block_diag(*c_r)
        actual = _rotate_columns(_rotate_rows(x, perm_l, c_l), perm_r, c_r)
        self.assertLess(float((actual - dense_l @ x @ dense_r.T).abs().max()), 1e-12)

    def test_unknown_scope_option_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_scope_options({"block": 4})


class StructuredScopeTest(unittest.TestCase):
    def test_structured_scopes_select_linear_weights_only(self) -> None:
        for scope in ("fura", "fura_s", "iso", "isobtt"):
            model, es, result = make_es(scope, iso_block_size=4)
            names = {name for name, _, _ in es.parameter_infos}
            self.assertEqual(names, {"q_proj.weight", "down_proj.weight", "qkv_proj.weight"}, scope)
            self.assertEqual(result["n_tensors"], 3)
            self.assertGreater(result["manifold_dim"], 0)

    def test_scope_options_require_a_structured_scope(self) -> None:
        with self.assertRaises(ValueError):
            SeedReplayModelES().init(TinyModel(), parameter_scope="full", scope_options={"iso_perm": False})

    def test_untouched_parameters_stay_bit_identical(self) -> None:
        for scope in ("fura", "fura_s", "iso", "isobtt"):
            torch.manual_seed(7)
            reference = weights(TinyModel())
            model, es, _ = make_es(scope, iso_block_size=4)
            es.apply(seed=5, sigma=5e-2)
            es.revert(seed=5, sigma=5e-2)
            es.update(seeds=[5, 6], rewards=[0.0, 1.0], alpha=1e-2)
            for name in ("embed_tokens.weight", "down_proj.bias", "norm.weight"):
                self.assertTrue(torch.equal(dict(model.named_parameters())[name], reference[name]), (scope, name))

    def test_init_keeps_the_base_model(self) -> None:
        torch.manual_seed(7)
        reference = weights(TinyModel())
        for scope, tolerance in (("iso", 0.0), ("fura", 1e-5), ("fura_s", 1e-5), ("isobtt", 1e-5)):
            model, _, _ = make_es(scope, iso_block_size=4)
            for name, value in weights(model).items():
                self.assertLessEqual(float((value - reference[name]).abs().max()), tolerance, (scope, name))

    def test_perturbation_reverts_exactly(self) -> None:
        for scope in ("fura", "fura_s", "iso", "isobtt"):
            model, es, _ = make_es(scope, iso_block_size=4)
            before = weights(model)
            es.apply(seed=11, sigma=5e-2)
            self.assertFalse(torch.equal(model.q_proj.weight, before["q_proj.weight"]), scope)
            es.revert(seed=11, sigma=5e-2)
            for name, value in weights(model).items():
                self.assertTrue(torch.equal(value, before[name]), (scope, name))

    def test_same_seed_reproduces_the_perturbation(self) -> None:
        for scope in ("fura", "fura_s", "iso", "isobtt"):
            first, es_first, _ = make_es(scope, iso_block_size=4)
            second, es_second, _ = make_es(scope, iso_block_size=4)
            es_first.apply(seed=11, sigma=5e-2)
            es_second.apply(seed=11, sigma=5e-2)
            self.assertTrue(torch.equal(first.q_proj.weight, second.q_proj.weight), scope)
            es_second.revert(seed=11, sigma=5e-2)
            es_second.apply(seed=12, sigma=5e-2)
            self.assertFalse(torch.equal(first.q_proj.weight, second.q_proj.weight), scope)

    def test_update_can_be_reset(self) -> None:
        for scope, tolerance in (("fura", 0.0), ("fura_s", 0.0), ("isobtt", 0.0), ("iso", 1e-5)):
            model, es, _ = make_es(scope, iso_block_size=4)
            before = weights(model)
            es.update(seeds=[11, 12, 13], rewards=[0.0, 1.0, 0.3], alpha=5e-2)
            es.update(seeds=[14, 15, 16], rewards=[0.7, 0.1, 0.3], alpha=5e-2)
            self.assertFalse(torch.equal(model.q_proj.weight, before["q_proj.weight"]), scope)
            es.apply(seed=3, sigma=5e-2)
            result = es.reset()
            self.assertEqual(result["reverted_updates"], 2)
            self.assertEqual(result["reverted_perturbation"], {"seed": 3, "sigma": 5e-2})
            for name, value in weights(model).items():
                self.assertLessEqual(float((value - before[name]).abs().max()), tolerance, (scope, name))

    def test_update_requires_reverted_perturbation(self) -> None:
        _, es, _ = make_es("isobtt")
        es.apply(seed=3, sigma=1e-2)
        with self.assertRaises(RuntimeError):
            es.update(seeds=[3], rewards=[1.0], alpha=1e-2)

    def test_fura_update_is_the_seed_replay_sum_on_the_core(self) -> None:
        model, es, _ = make_es("fura")
        state = next(state for state in es.structured.states if state["name"] == "q_proj.weight")
        noise = [es.structured._coef_noise(state, seed) for seed in (11, 12)]
        es.update(seeds=[11, 12], rewards=[2.0, -1.0], alpha=0.5, reward_normalization="none")
        expected = (0.5 / 2) * (2.0 * noise[0] - 1.0 * noise[1])
        self.assertTrue(torch.allclose(state["coef"], expected, atol=1e-7, rtol=0.0))
        rendered = torch.bmm(state["A"], state["R0"] + state["coef"]).permute(1, 0, 2).reshape(24, 24)
        self.assertTrue(torch.allclose(model.q_proj.weight, rendered, atol=1e-6, rtol=0.0))

    def test_fura_s_moves_only_the_block_singular_values(self) -> None:
        model, es, _ = make_es("fura_s")
        state = next(state for state in es.structured.states if state["name"] == "q_proj.weight")
        noise = [es.structured._coef_noise(state, seed) for seed in (11, 12)]
        self.assertEqual(noise[0].shape, state["S0"].shape)

        def rendered(spectrum: torch.Tensor) -> torch.Tensor:
            blocks = torch.bmm(state["U"] * spectrum.unsqueeze(1), state["Vh"])
            return blocks.permute(1, 0, 2).reshape(24, 24)

        es.apply(seed=11, sigma=0.1)
        self.assertTrue(
            torch.allclose(model.q_proj.weight, rendered(state["S0"] + 0.1 * noise[0]), atol=1e-6, rtol=0.0)
        )
        es.revert(seed=11, sigma=0.1)
        es.update(seeds=[11, 12], rewards=[2.0, -1.0], alpha=0.5, reward_normalization="none")
        expected = (0.5 / 2) * (2.0 * noise[0] - 1.0 * noise[1])
        self.assertTrue(torch.allclose(state["coef"], expected, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(model.q_proj.weight, rendered(state["S0"] + expected), atol=1e-6, rtol=0.0))
        # The frames are untouched, so the block singular values are exactly |S0 + c|.
        after = block_singular_values(model.q_proj.weight.detach())
        target = (state["S0"] + expected).abs().double().sort(dim=1, descending=True).values
        self.assertLess(float((after - target).norm() / target.norm()), 1e-5)

    def test_iso_preserves_the_singular_values(self) -> None:
        model, es, _ = make_es("iso", iso_block_size=4)
        before = {name: singular_values(value) for name, value in weights(model).items() if value.ndim == 2}
        segments_before = [singular_values(part) for part in model.qkv_proj.weight.detach().split([16, 8, 8])]

        def check(label: str) -> None:
            for name in ("q_proj.weight", "down_proj.weight"):
                after = singular_values(dict(model.named_parameters())[name].detach())
                self.assertLess(float((after - before[name]).norm() / before[name].norm()), 1e-5, (label, name))
            # A fused projection keeps the spectrum of each matrix it was built from.
            for part, expected in zip(model.qkv_proj.weight.detach().split([16, 8, 8]), segments_before):
                self.assertLess(float((singular_values(part) - expected).norm() / expected.norm()), 1e-5, label)

        es.apply(seed=11, sigma=0.2)
        check("perturbed")
        es.revert(seed=11, sigma=0.2)
        for generation in range(5):
            result = es.update(
                seeds=[generation * 3 + offset for offset in range(3)],
                rewards=[0.0, 1.0, 0.4],
                alpha=0.2,
            )
            self.assertLess(result["frob_drift"], 1e-5)
        check("updated")

    def test_isobtt_preserves_the_block_singular_values(self) -> None:
        model, es, _ = make_es("isobtt")
        names = ("q_proj.weight", "down_proj.weight", "qkv_proj.weight")
        before = {name: block_singular_values(dict(model.named_parameters())[name].detach()) for name in names}

        def check(label: str) -> None:
            for name in names:
                after = block_singular_values(dict(model.named_parameters())[name].detach())
                self.assertLess(float((after - before[name]).norm() / before[name].norm()), 1e-5, (label, name))

        es.apply(seed=11, sigma=0.2)
        check("perturbed")
        es.revert(seed=11, sigma=0.2)
        for generation in range(5):
            result = es.update(
                seeds=[generation * 3 + offset for offset in range(3)],
                rewards=[0.0, 1.0, 0.4],
                alpha=0.2,
            )
            self.assertLess(result["orth_err"], 1e-5)
        check("updated")

    def test_fura_does_not_preserve_the_block_singular_values(self) -> None:
        model, es, _ = make_es("fura")
        before = block_singular_values(model.q_proj.weight.detach())
        es.apply(seed=11, sigma=0.2)
        after = block_singular_values(model.q_proj.weight.detach())
        self.assertGreater(float((after - before).norm() / before.norm()), 1e-2)

    def test_fixed_spectrum_sigma_is_the_relative_displacement(self) -> None:
        torch.manual_seed(3)
        layer = torch.nn.Linear(256, 256, bias=False)
        base = layer.weight.detach().clone()
        for scope in ("iso", "isobtt"):
            layer.weight.data.copy_(base)
            structured = StructuredES(layer, [("weight", 1, layer.weight)], scope=scope, options={"iso_block_size": 64})
            reference = layer.weight.detach().clone()
            structured.perturb(seed=5, sigma=5e-2)
            relative = float((layer.weight.detach() - reference).norm() / reference.norm())
            self.assertAlmostEqual(relative, 5e-2, delta=1e-2, msg=scope)

    def test_fixed_spectrum_update_matches_the_first_order_estimator(self) -> None:
        for scope in ("iso", "isobtt"):
            model, es, _ = make_es(scope, iso_block_size=4)
            base = model.q_proj.weight.detach().clone()
            step = 1e-3
            directions = []
            for seed in (11, 12):
                es.apply(seed=seed, sigma=step)
                directions.append((model.q_proj.weight.detach() - base) / step)
                es.revert(seed=seed, sigma=step)
            es.update(seeds=[11, 12], rewards=[2.0, -1.0], alpha=2 * step, reward_normalization="none")
            expected = step * (2.0 * directions[0] - 1.0 * directions[1])
            actual = model.q_proj.weight.detach() - base
            self.assertLess(float((actual - expected).norm() / expected.norm()), 2e-2, scope)


class WorkerParityTest(unittest.TestCase):
    """The vLLM worker and the HTTP-server class must produce the same weights."""

    def test_worker_matches_seed_replay_model_es(self) -> None:
        for scope in ("fura", "fura_s", "iso", "isobtt"):
            torch.manual_seed(7)
            server_model = TinyModel()
            worker = WorkerExtension()
            worker.model = copy.deepcopy(server_model)
            options = {"iso_block_size": 4}

            es = SeedReplayModelES()
            server_init = es.init(server_model, parameter_scope=scope, scope_options=options, verbose=False)
            worker_init = worker.init_math_es(parameter_scope=scope, scope_options=options, verbose=False)
            self.assertEqual(server_init["total_params"], worker_init["total_params"])

            es.apply(seed=21, sigma=5e-2)
            worker.apply_math_es(21, 5e-2)
            self.assertTrue(torch.equal(server_model.q_proj.weight, worker.model.q_proj.weight), scope)
            es.revert(seed=21, sigma=5e-2)
            worker.revert_math_es(21, 5e-2)

            seeds, rewards = [21, 22, 23], [0.2, 0.9, 0.4]
            es.update(seeds=seeds, rewards=rewards, alpha=2e-2)
            zscore = SeedReplayModelES._normalize_rewards(rewards, "zscore")
            worker.dipu(seeds, zscore, 2e-2)
            for name, value in weights(server_model).items():
                self.assertTrue(torch.equal(value, dict(worker.model.named_parameters())[name]), (scope, name))

            worker.reset_math_es()
            es.reset()
            for name, value in weights(server_model).items():
                self.assertTrue(torch.equal(value, dict(worker.model.named_parameters())[name]), (scope, name))
            self.assertEqual(worker.status_math_es()["update_history"], 0)


if __name__ == "__main__":
    unittest.main()
