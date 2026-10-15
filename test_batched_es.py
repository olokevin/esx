import unittest

import torch

from algorithms.es.batched_es import BatchedStructuredES, group_seed, stochastic_round
from algorithms.es.structured_es import StructuredES, _cayley


def _tiny_model(dtype=torch.float32):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(24, 16, bias=True),
        torch.nn.Linear(16, 12, bias=False),
    )
    return model.to(dtype)


def _infos(model):
    return [
        (f"{name}.weight", 1000 + index, module.weight)
        for index, (name, module) in enumerate(model.named_modules())
        if isinstance(module, torch.nn.Linear)
    ]


def _batched(scope, dtype=torch.float32):
    model = _tiny_model(dtype)
    dense = [module.weight.detach().clone() for module in model if isinstance(module, torch.nn.Linear)]
    es = BatchedStructuredES(model, _infos(model), scope=scope, options={"batched": True})
    return model, es, dense


class FactoredForwardTest(unittest.TestCase):
    def test_unperturbed_forward_matches_dense_weight_and_drops_it(self):
        for scope in ("isobtt", "fura_s"):
            model, es, dense = _batched(scope)
            for layer, weight in zip(es.layers, dense):
                self.assertLess(float((layer.dense() - weight).abs().max()), 1e-5)
            self.assertEqual(model[0].weight.numel(), 0)
            x = torch.randn(7, 24)
            expected = torch.nn.functional.linear(
                torch.nn.functional.linear(x, dense[0], model[0].bias), dense[1]
            )
            self.assertLess(float((model(x) - expected).abs().max()), 1e-4)

    def test_each_token_gets_its_own_member(self):
        for scope in ("isobtt", "fura_s"):
            model, es, _ = _batched(scope)
            es.set_population(seeds=[11, 22, 33], sigma=0.05)
            x = torch.randn(5, 24)
            slots = torch.tensor([0, 1, 2, 3, 2])
            es.set_token_slots(slots)
            mixed = model(x)
            for row, slot in enumerate(slots.tolist()):
                weights = [layer.dense(slot) for layer in es.layers]
                expected = torch.nn.functional.linear(
                    torch.nn.functional.linear(x[row : row + 1], weights[0], model[0].bias), weights[1]
                )
                self.assertLess(float((mixed[row : row + 1] - expected).detach().abs().max()), 1e-4, scope)
            self.assertGreater(float((mixed[1] - mixed[2]).abs().max()), 0.0)

    def test_untagged_tokens_and_cleared_population_are_unperturbed(self):
        model, es, _ = _batched("isobtt")
        x = torch.randn(4, 24)
        base = model(x)
        es.set_population(seeds=[5, 6], sigma=0.1)
        es.set_token_slots(torch.zeros(4, dtype=torch.int64))
        self.assertTrue(torch.equal(model(x), base))
        es.set_token_slots(torch.ones(4, dtype=torch.int64))
        self.assertGreater(float((model(x) - base).abs().max()), 1e-4)
        es.clear_population()
        self.assertTrue(torch.equal(model(x), base))


class IsobttFlipoutTest(unittest.TestCase):
    def test_member_rotation_is_the_sign_flipped_shared_generator(self):
        _, es, _ = _batched("isobtt")
        seeds = [3, 4, 5]
        es.set_population(seeds=seeds, sigma=0.07)
        for layer in es.layers:
            omega = layer.omega(seeds)
            for slot, seed in enumerate(seeds, 1):
                sign = layer.signs(seed)
                member = omega * sign.unsqueeze(1) * sign.unsqueeze(2)
                core = torch.bmm(_cayley(member, 0.07), layer.core.float())
                frame = layer.frame.view(layer.out_f, layer.n_blk, layer.k).permute(1, 0, 2)
                expected = torch.bmm(frame, core).permute(1, 0, 2).reshape(layer.out_f, layer.in_f)
                self.assertLess(float((layer.dense(slot) - expected).abs().max()), 1e-5)

    def test_perturbation_and_update_preserve_block_spectra(self):
        _, es, dense = _batched("isobtt")
        layer, weight = es.layers[0], dense[0]

        def spectra(w):
            blocks = w.reshape(layer.out_f, layer.n_blk, layer.b).permute(1, 0, 2)
            return torch.linalg.svdvals(blocks)

        es.set_population(seeds=[1, 2], sigma=0.2)
        self.assertLess(float((spectra(layer.dense(1)) - spectra(weight)).abs().max()), 1e-4)
        self.assertGreater(float((layer.dense(1) - weight).abs().max()), 1e-3)
        es.clear_population()
        metrics = es.update(seeds=[1, 2, 3, 4], weights=[1.0, -0.5, 0.25, -0.75], eta=0.3)
        self.assertLess(metrics["orth_err"], 1e-5)
        self.assertLess(float((spectra(layer.dense()) - spectra(weight)).abs().max()), 1e-4)
        self.assertGreater(float((layer.dense() - weight).abs().max()), 1e-3)

    def test_update_is_first_order_weighted_sum_of_member_generators(self):
        _, es, _ = _batched("isobtt")
        seeds, weights, eta = [7, 8, 9, 10], [0.9, -1.1, 0.4, -0.2], 1e-3
        layer = es.layers[0]
        before = layer.core.float().clone()
        omega = layer.omega(seeds)
        total = torch.zeros_like(omega)
        for seed, weight in zip(seeds, weights):
            sign = layer.signs(seed)
            total += eta / len(seeds) * weight * omega * sign.unsqueeze(1) * sign.unsqueeze(2)
        es.update(seeds=seeds, weights=weights, eta=eta)
        step = layer.core.float() - before
        expected = torch.bmm(total, before)
        self.assertLess(float((step - expected).norm() / expected.norm()), 1e-2)


class FuraSCompatibilityTest(unittest.TestCase):
    def test_matches_unbatched_scope_for_the_same_seeds(self):
        seeds, weights, eta, sigma = [21, 22, 23], [1.0, -1.0, 0.5], 0.4, 0.03
        reference_model = _tiny_model()
        reference = StructuredES(reference_model, _infos(reference_model), scope="fura_s")
        _, es, _ = _batched("fura_s")

        es.set_population(seeds=seeds, sigma=sigma)
        for slot, seed in enumerate(seeds, 1):
            reference.perturb(seed=seed, sigma=sigma)
            for layer, module in zip(es.layers, [reference_model[0], reference_model[1]]):
                self.assertLess(float((layer.dense(slot) - module.weight).abs().max()), 1e-5)
            reference.restore()
        es.clear_population()

        reference.update(seeds=seeds, weights=weights, eta=eta)
        es.update(seeds=seeds, weights=weights, eta=eta)
        for layer, module in zip(es.layers, [reference_model[0], reference_model[1]]):
            self.assertLess(float((layer.dense() - module.weight).abs().max()), 1e-5)


class LifecycleTest(unittest.TestCase):
    def test_update_requires_cleared_population_and_reset_restores_cores(self):
        for scope in ("isobtt", "fura_s"):
            model, es, _ = _batched(scope)
            x = torch.randn(3, 24)
            base = model(x)
            es.set_population(seeds=[1, 2], sigma=0.1)
            with self.assertRaises(RuntimeError):
                es.update(seeds=[1, 2], weights=[1.0, -1.0], eta=0.1)
            es.clear_population()
            es.update(seeds=[1, 2], weights=[1.0, -1.0], eta=0.5)
            self.assertGreater(float((model(x) - base).abs().max()), 1e-4)
            es.reset()
            self.assertTrue(torch.equal(model(x), base))

    def test_update_replays_to_identical_cores(self):
        for scope in ("isobtt", "fura_s"):
            cores = []
            for _ in range(2):
                _, es, _ = _batched(scope, torch.bfloat16)
                es.update(seeds=[4, 5, 6], weights=[0.3, -0.9, 0.6], eta=0.01)
                es.update(seeds=[7, 8, 9], weights=[-0.3, 0.2, 0.1], eta=0.01)
                cores.append([layer.core.clone() for layer in es.layers])
            for first, second in zip(*cores):
                self.assertTrue(torch.equal(first, second))

    def test_token_count_mismatch_is_an_error(self):
        model, es, _ = _batched("fura_s")
        es.set_population(seeds=[1], sigma=0.1)
        es.set_token_slots(torch.ones(9, dtype=torch.int64))
        with self.assertRaises(RuntimeError):
            model(torch.randn(4, 24))

    def test_structured_es_rejects_the_batched_option(self):
        model = _tiny_model()
        with self.assertRaises(ValueError):
            StructuredES(model, _infos(model), scope="isobtt", options={"batched": True})


class RoundingTest(unittest.TestCase):
    def test_stochastic_round_is_unbiased_and_lands_on_neighbours(self):
        generator = torch.Generator().manual_seed(0)
        x = torch.full((200_000,), 0.50123, dtype=torch.float32)
        x[100_000:] = -0.50123
        rounded = stochastic_round(x, torch.bfloat16, generator).float()
        self.assertEqual(len(rounded[:100_000].unique()), 2)
        self.assertLess(abs(float(rounded[:100_000].mean()) - 0.50123), 2e-5)
        self.assertLess(abs(float(rounded[100_000:].mean()) + 0.50123), 2e-5)
        nearest = x.to(torch.bfloat16).float()
        self.assertGreater(abs(float(nearest[:100_000].mean()) - 0.50123), 1e-4)

    def test_sub_ulp_updates_survive_bf16_storage_on_average(self):
        # Steps of about a tenth of a bf16 ulp: round-to-nearest would drop all of them.
        _, es, _ = _batched("fura_s", torch.bfloat16)
        layer = es.layers[0]
        before = layer.core.float().clone()
        expected = torch.zeros_like(before)
        eta = 1e-3
        for step in range(100):
            seeds = [1000 + step, 2000 + step]
            for seed, weight in zip(seeds, [1.0, 0.5]):
                expected += eta / 2 * weight * layer.noise(seed)
            es.update(seeds=seeds, weights=[1.0, 0.5], eta=eta)
        moved = layer.core.float() - before
        cosine = float((moved * expected).sum() / (moved.norm() * expected.norm()))
        # Unbiased, but each step adds rounding noise of about sqrt(step * ulp).
        self.assertGreater(cosine, 0.4)
        nearest = (before + expected / 100).to(torch.bfloat16).float() - before
        self.assertLess(float(nearest.abs().max()), 1e-6)

    def test_group_seed_depends_on_every_member(self):
        self.assertEqual(group_seed([1, 2, 3]), group_seed([1, 2, 3]))
        self.assertNotEqual(group_seed([1, 2, 3]), group_seed([1, 2, 4]))
        self.assertNotEqual(group_seed([1, 2, 3]), group_seed([3, 2, 1]))


if __name__ == "__main__":
    unittest.main()
