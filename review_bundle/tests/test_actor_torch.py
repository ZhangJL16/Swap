from __future__ import annotations

import unittest
from dataclasses import replace

try:
    import torch

    from cert_runtime.actor import FeedForwardAffineTanhActor
    from cert_runtime.synthetic import build_synthetic_closure_fixture
    from cert_runtime.trainer import GeneratorSACTrainer

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "Torch is not installed in this interpreter")
class TorchActorTests(unittest.TestCase):
    def test_stable_jacobian_and_logdet_match_direct_formula(self):
        torch.manual_seed(3)
        actor = FeedForwardAffineTanhActor(4, 8)
        observation = torch.zeros(4)
        center = torch.tensor([0.1, -0.2, 0.3], requires_grad=True)
        generator_scales = torch.tensor([0.4, 0.5, 0.6], requires_grad=True)
        generators = torch.diag(generator_scales)
        action, log_density, u = actor.action_and_log_density(observation, center, generators)
        distribution = actor.distribution(observation)
        direct = (
            distribution.log_prob(u).sum()
            - torch.log(1.0 - torch.tanh(u).square()).sum()
            - torch.logdet(generators.detach())
        )
        self.assertTrue(torch.allclose(log_density, direct, atol=1e-6, rtol=1e-6))
        loss = action.square().sum() + log_density
        loss.backward()
        self.assertIsNone(center.grad)
        self.assertIsNone(generator_scales.grad)

    def test_log_density_u_gradient_matches_finite_difference(self):
        actor = FeedForwardAffineTanhActor(2, 4)
        observation = torch.zeros(2)
        distribution = actor.distribution(observation)
        generators = torch.diag(torch.tensor([0.4, 0.5, 0.6]))
        u = torch.tensor([0.2, -0.1, 0.3], requires_grad=True)
        value = actor.log_density_from_u(distribution, u, generators)
        value.backward()
        analytical = u.grad.detach().clone()
        epsilon = 1e-4
        finite = []
        for index in range(3):
            plus = u.detach().clone()
            minus = u.detach().clone()
            plus[index] += epsilon
            minus[index] -= epsilon
            finite.append(
                (
                    actor.log_density_from_u(distribution, plus, generators)
                    - actor.log_density_from_u(distribution, minus, generators)
                )
                / (2.0 * epsilon)
            )
        finite_tensor = torch.stack(finite)
        self.assertTrue(torch.allclose(analytical, finite_tensor, atol=2e-3, rtol=2e-3))

    def _accepted_record(self):
        fixture = build_synthetic_closure_fixture()
        result = fixture.closure.close(
            fixture.state, fixture.geometry, fixture.corridor, fixture.cells,
            fixture.operating_point, fixture.device_version, fixture.firmware_version,
            fixture.timestamp, allow_synthetic=True,
        )
        self.assertTrue(result.closed)
        decision = fixture.runtime.step(fixture.state, (0.0, 0.0, 0.0, 0.0))
        self.assertTrue(decision.accepted)
        return fixture, fixture.replay.records[-1]

    def test_generator_sac_critic_uses_executed_action_and_epoch_is_frozen(self):
        fixture, record = self._accepted_record()

        class RecordingCritic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.last_action = None

            def forward(self, state, action):
                self.last_action = action.detach().clone()
                return action.sum(dim=-1, keepdim=True)

        actor = FeedForwardAffineTanhActor(4, 8)
        critic = RecordingCritic()
        trainer = GeneratorSACTrainer(actor, critic, 0.2)
        trainer.begin_epoch(record.certificate_state)
        states = torch.zeros((1, 2))
        trainer.critic_loss(states, (record,), torch.zeros((1, 1)))
        expected = torch.tensor([record.executed_action], dtype=states.dtype)
        self.assertTrue(torch.allclose(critic.last_action, expected))
        stale = replace(record, bound_versions=record.bound_versions + (("energy", "changed"),))
        with self.assertRaises(ValueError):
            trainer.validate_replay((stale,))

    def test_fallback_atom_is_excluded_from_generator_entropy(self):
        _, record = self._accepted_record()

        class Critic(torch.nn.Module):
            def forward(self, state, action):
                return action.sum().reshape(())

        actor = FeedForwardAffineTanhActor(4, 8)
        trainer = GeneratorSACTrainer(actor, Critic(), 0.2)
        trainer.begin_epoch(record.certificate_state)
        fallback = replace(
            record,
            accepted=False,
            nominal_pre_squash_u=None,
            squashed_eta=None,
            candidate_action=None,
            zonotope_center=None,
            zonotope_generators=None,
            inclusion_certificate_hash=None,
            fallback_reason="WATCHDOG_DEADLINE",
        )
        result = trainer.actor_loss(torch.zeros((1, 4)), torch.zeros((1, 2)), (fallback,))
        self.assertEqual(result.accepted_count, 0)
        self.assertEqual(result.fallback_count, 1)
        self.assertIn("fallback atom excluded", result.semantics)


if __name__ == "__main__":
    unittest.main()
