"""Gradient contracts for local state learning without changing action planning."""

import unittest

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from h_jepa.hierarchical_solver import LevelCostModel
from h_jepa.hjepa_utils import create_world_model, hjepa_forward
from h_jepa.models.encoders.hf_vit import create_hf_vit
from h_jepa.models.jepa import JEPA
from h_jepa.models.module import ConditionalBlock


def make_config(fusion: bool) -> DictConfig:
    config = {
        "num_levels": 4,
        "img_size": 4,
        "patch_size": 2,
        "data": {"dataset": {"level1": {"frameskip": 2}}},
    }
    for level in range(1, 5):
        if fusion:
            encoder = {
                "pixel_encoder": {
                    "encoder": {"type": "mlp", "input_dim": 4, "hidden_dim": 8, "output_dim": 4},
                    "projector": {"type": "identity"},
                },
                "proprio_encoder": {
                    "encoder": {"type": "mlp", "input_dim": 3 if level == 1 else 4,
                                "hidden_dim": 8, "output_dim": 4},
                    "projector": {"type": "identity"},
                },
                "projector": {"type": "identity"},
            }
        else:
            encoder = {
                "type": "mlp" if level == 1 else "latent_mlp",
                "input_dim": 4 if level == 1 else 8,
                "hidden_dim": 8,
                "output_dim": 8,
                "projector": {"type": "identity"},
            }
        spec = {
            "stride": 1 if level == 1 else 2,
            "window_size": 1,
            "wm": {"history_size": 2, "rollout_n": 1, "embed_dim": 8, "use_proprio": fusion,
                   "action_dim": 2, "proprio_dim": 3 if level == 1 else 4, "proprio_emb_dim": 4},
            "encoder": encoder,
            "predictor": {"depth": 1, "heads": 2, "dim_head": 4, "mlp_dim": 16,
                          "hidden_dim": 8, "dropout": 0.0, "emb_dropout": 0.0,
                          "projector": {"type": "mlp" if fusion else "identity"}},
            "action_embed": {"patch_embed": False, "input_dim": 4},
            "loss": {"embed": {"sigreg": {"enabled": True, "weight": 0.09, "knots": 5, "num_proj": 8}}},
        }
        if level > 1:
            spec["action_encoder"] = {
                "input_dim": 8, "output_dim": 4, "d_model": 8, "nhead": 2,
                "num_layers": 1, "dropout": 0.0, "max_chunk": 2,
            }
        config[f"level{level}"] = spec
    return OmegaConf.create(config)


class LossModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model, regularizers = create_world_model(config)
        for name, regularizer in regularizers.items():
            self.add_module(name, regularizer)
        self.logged = {}
        # At exact AdaLN-zero initialization, actions have no effect yet. Use a
        # deterministic nonzero learned-conditioning state to exercise that route.
        for module in self.model.modules():
            if isinstance(module, ConditionalBlock):
                linear = module.adaLN_modulation[-1]
                assert isinstance(linear, nn.Linear)
                nn.init.normal_(linear.weight, std=0.1)
                nn.init.normal_(linear.bias, std=0.1)
        self.eval()

    def log_dict(self, values, **_kwargs):
        self.logged.update(values)


def make_batch():
    return {
        "pixels_level1": torch.randn(2, 32, 4),
        "proprio_level1": torch.randn(2, 32, 3),
        "action_level1": torch.randn(2, 32, 4),
    }


def forward(module, config, batch, *, detach=None):
    # Inputs are already floating point features; no image normalization applies.
    kwargs = {} if detach is None else {"detach_lower_state": detach}
    return hjepa_forward(module, dict(batch), "train", config, normalize_batch=nn.Identity(), **kwargs)


def gradient_size(gradients):
    return sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None)


class StateStopGradientTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(torch.random.fork_rng(devices=[]))
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(123)

    def test_unmasked_vit_backward_reaches_all_trainable_parameters(self):
        encoder = create_hf_vit(
            image_size=4,
            patch_size=2,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
        embeddings = encoder(torch.randn(2, 3, 4, 4)).last_hidden_state
        embeddings.square().mean().backward()
        missing = [
            name for name, parameter in encoder.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(missing, [], f"DDP cannot reduce unused ViT parameters: {missing}")

    def test_each_loss_only_updates_its_own_state_encoder(self):
        for fusion in (False, True):
            config = make_config(fusion)
            module = LossModule(config)
            batch = make_batch()
            for detached in (False, True):
                output = forward(module, config, batch, detach=detached)
                for loss_level in range(1, 5):
                    for encoder_level in range(1, 5):
                        with self.subTest(fusion=fusion, detached=detached,
                                          loss=loss_level, encoder=encoder_level):
                            parameters = tuple(module.model.get_level(encoder_level).encoder.parameters())
                            gradients = torch.autograd.grad(output[f"loss_level{loss_level}"], parameters,
                                                            retain_graph=True, allow_unused=True)
                            active = encoder_level == loss_level if detached else encoder_level <= loss_level
                            if active:
                                self.assertGreater(gradient_size(gradients), 0)
                            else:
                                self.assertEqual(gradient_size(gradients), 0)

    def test_summed_loss_still_learns_every_encoder_and_fusion_branch(self):
        for fusion in (False, True):
            config = make_config(fusion)
            module = LossModule(config)
            output = forward(module, config, make_batch(), detach=True)
            optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
            before = [parameter.detach().clone() for parameter in module.model.parameters()]
            output["loss"].backward()
            for level in range(1, 5):
                encoder = module.model.get_level(level).encoder
                branches = [encoder.pixel_encoder, encoder.proprio_encoder] if fusion else [encoder]
                for branch in branches:
                    gradients = [parameter.grad for parameter in branch.parameters()]
                    self.assertGreater(gradient_size(gradients), 0)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients if g is not None))
            optimizer.step()
            self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, module.model.parameters())))

    def test_forward_values_and_action_gradients_match_the_original_mode(self):
        for fusion in (False, True):
            config = make_config(fusion)
            module = LossModule(config)
            batch = make_batch()
            state = torch.random.get_rng_state()
            original = forward(module, config, batch)
            torch.random.set_rng_state(state)
            detached = forward(module, config, batch, detach=True)
            for key in original:
                if torch.is_tensor(original[key]):
                    torch.testing.assert_close(original[key], detached[key], rtol=0, atol=0)
            parameters = tuple(parameter for level in range(1, 5)
                               for network in (module.model.get_level(level).action_encoder,
                                               module.model.get_level(level).action_embed)
                               for parameter in network.parameters())
            expected = torch.autograd.grad(original["loss_level4"], parameters)
            actual = torch.autograd.grad(detached["loss_level4"], parameters)
            self.assertGreater(gradient_size(expected), 0)
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right)

    def test_sg_training_does_not_disable_action_gradients_through_upper_planning_cost(self):
        config = make_config(False)
        module = LossModule(config)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        forward(module, config, make_batch(), detach=True)["loss"].backward()
        optimizer.step()
        model = module.model.requires_grad_(False)
        pixels = torch.randn(2, 1, 4)
        observation = model.get_level(1).encode({"pixels": pixels})["embed_0"]
        candidate = torch.randn(2, 3, 8, 4, requires_grad=True)
        adapter = LevelCostModel(model.get_level(1), [model.get_level(level) for level in range(2, 5)])
        cost = adapter.get_cost({
            "pixels": pixels[:, None], "embed_0": observation[:, None],
            "goal_embed_0": torch.randn(2, 1, 1, 8), "goal_embed_0_level": 4,
            "goal_embed_0_in_upper_space": True, "dense_upper_goal_projection": True,
        }, candidate)
        gradient, = torch.autograd.grad(cost.sum(), candidate)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)


class LayerwiseTrainingTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(torch.random.fork_rng(devices=[]))
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(456)

    def _make_module(self, fusion):
        config = make_config(fusion)
        module = LossModule(config)
        for level in range(1, 5):
            jepa = module.model.get_level(level)
            # Set queue capacity at construction, not through runtime reconfiguration.
            module.model.jepas[level - 1] = JEPA(
                encoder=jepa.encoder, predictor=jepa.predictor,
                action_embed=jepa.action_embed, action_encoder=jepa.action_encoder,
                level=level, action_queue_size=2,
                temporal_stride=jepa.temporal_stride,
                temporal_window_size=jepa.temporal_window_size,
                target_length=jepa.target_length,
            )
        return config, module

    @staticmethod
    def _state_snapshot(level):
        return {name: value.detach().clone() for name, value in level.state_dict().items()}

    def assertStateDictUnchanged(self, expected, level, *, message):
        actual = level.state_dict()
        self.assertEqual(set(expected), set(actual), message)
        for name, before in expected.items():
            after = actual[name]
            self.assertEqual(before.dtype, after.dtype, f"{message}: {name} dtype")
            self.assertEqual(before.shape, after.shape, f"{message}: {name} shape")
            self.assertTrue(torch.equal(before, after), f"{message}: {name} changed")

    def assertLayerTrainingModes(self, module, active_level, *, training):
        for level in range(1, 5):
            self.assertEqual(
                module.model.get_level(level).training,
                training and level == active_level,
                f"level {level} training mode",
            )

    @staticmethod
    def _state_encoders(level):
        encoder = level.encoder
        if hasattr(encoder, "pixel_encoder"):
            return (encoder.pixel_encoder, encoder.proprio_encoder)
        return (encoder,)

    def _mark_stale_gradients(self, module):
        for level in range(1, 5):
            for parameter in module.model.get_level(level).parameters():
                parameter.grad = torch.ones_like(parameter)

    def test_each_stage_updates_only_its_current_level(self):
        for fusion in (False, True):
            for active_level in range(1, 5):
                with self.subTest(fusion=fusion, active_level=active_level):
                    config, module = self._make_module(fusion)
                    model = module.model
                    self._mark_stale_gradients(module)
                    model.set_training_level(active_level)

                    for level in range(1, 5):
                        parameters = tuple(model.get_level(level).parameters())
                        self.assertTrue(
                            all(parameter.grad is None for parameter in parameters),
                            f"level {level} retained a stale gradient",
                        )
                        self.assertEqual(
                            [parameter.requires_grad for parameter in parameters],
                            [level == active_level] * len(parameters),
                        )

                    module.train()
                    self.assertLayerTrainingModes(module, active_level, training=True)
                    module.eval()
                    self.assertLayerTrainingModes(module, active_level, training=False)
                    module.train()
                    self.assertLayerTrainingModes(module, active_level, training=True)

                    snapshots = {
                        level: self._state_snapshot(model.get_level(level)) for level in range(1, 5)
                    }
                    for snapshot in snapshots.values():
                        self.assertIn("latent_action_queue", snapshot)

                    future_encoder_calls = {level: 0 for level in range(active_level + 1, 5)}
                    handles = []
                    for level in range(active_level + 1, 5):
                        for encoder in self._state_encoders(model.get_level(level)):
                            def record_call(_module, _inputs, _output, *, level=level, calls=future_encoder_calls):
                                calls[level] += 1

                            handles.append(encoder.register_forward_hook(record_call))

                    batch = {
                        name: value.clone().requires_grad_(True)
                        for name, value in make_batch().items()
                    }
                    try:
                        output = hjepa_forward(
                            module,
                            batch,
                            "train",
                            config,
                            normalize_batch=nn.Identity(),
                            active_level=active_level,
                        )
                    finally:
                        for handle in handles:
                            handle.remove()

                    self.assertEqual(future_encoder_calls, {level: 0 for level in future_encoder_calls})
                    observed_loss_levels = {
                        level
                        for level in range(1, 5)
                        if any("loss" in name and name.endswith(f"_level{level}") for name in output)
                    }
                    self.assertEqual(observed_loss_levels, {active_level})
                    self.assertEqual(output["loss"], output[f"loss_level{active_level}"])
                    for level in range(active_level + 1, 5):
                        self.assertNotIn(f"embed_{level}", output)

                    for level in range(1, active_level):
                        for output_name in ("embed", "action"):
                            frozen_output = output[f"{output_name}_{level}"]
                            self.assertFalse(frozen_output.requires_grad)
                            self.assertIsNone(frozen_output.grad_fn)
                    active_embed = output[f"embed_{active_level}"]
                    self.assertTrue(active_embed.requires_grad)
                    active_embed.retain_grad()

                    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
                    output["loss"].backward()
                    self.assertIsNotNone(active_embed.grad)
                    self.assertGreater(float(active_embed.grad.abs().sum()), 0)
                    active_gradients = [
                        parameter.grad for parameter in model.get_level(active_level).parameters()
                    ]
                    self.assertGreater(gradient_size(active_gradients), 0)
                    self.assertTrue(
                        all(torch.isfinite(gradient).all() for gradient in active_gradients if gradient is not None)
                    )
                    for level in range(1, 5):
                        if level == active_level:
                            continue
                        self.assertTrue(
                            all(parameter.grad is None for parameter in model.get_level(level).parameters()),
                            f"frozen level {level} received a gradient",
                        )

                    if active_level == 1:
                        self.assertGreater(
                            gradient_size(
                                [batch["pixels_level1"].grad, batch["action_level1"].grad]
                            ),
                            0,
                        )
                    else:
                        self.assertIsNone(batch["pixels_level1"].grad)
                        self.assertIsNone(batch["action_level1"].grad)

                    optimizer.step()
                    active_state = model.get_level(active_level).state_dict()
                    active_parameter_names = set(dict(model.get_level(active_level).named_parameters()))
                    self.assertTrue(
                        any(
                            not torch.equal(snapshots[active_level][name], active_state[name])
                            for name in active_parameter_names
                        ),
                        "the active level did not update",
                    )
                    self.assertGreater(active_state["latent_action_queue"].numel(), 0)
                    self.assertFalse(
                        torch.equal(
                            snapshots[active_level]["latent_action_queue"],
                            active_state["latent_action_queue"],
                        )
                    )
                    for level in range(1, 5):
                        if level != active_level:
                            self.assertStateDictUnchanged(
                                snapshots[level],
                                model.get_level(level),
                                message=f"frozen level {level}",
                            )

    def test_train_without_a_selected_stage_keeps_ordinary_behavior(self):
        _config, module = self._make_module(False)
        module.train()
        self.assertTrue(all(module.model.get_level(level).training for level in range(1, 5)))
        self.assertTrue(
            all(
                parameter.requires_grad
                for level in range(1, 5)
                for parameter in module.model.get_level(level).parameters()
            )
        )
        module.eval()
        self.assertTrue(all(not module.model.get_level(level).training for level in range(1, 5)))
        module.train()
        self.assertTrue(all(module.model.get_level(level).training for level in range(1, 5)))

    def test_candidate_action_gradient_survives_frozen_lower_hierarchy(self):
        _config, module = self._make_module(False)
        model = module.model
        model.set_training_level(4)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for level in range(1, 4)
                for parameter in model.get_level(level).parameters()
            )
        )
        self.assertTrue(all(parameter.requires_grad for parameter in model.get_level(4).parameters()))
        model.eval()

        pixels = torch.randn(2, 1, 4)
        observation = model.get_level(1).encode({"pixels": pixels})["embed_0"]
        candidate = torch.randn(2, 3, 8, 4, requires_grad=True)
        adapter = LevelCostModel(
            model.get_level(1), [model.get_level(level) for level in range(2, 5)]
        )
        cost = adapter.get_cost(
            {
                "pixels": pixels[:, None],
                "embed_0": observation[:, None],
                "goal_embed_0": torch.randn(2, 1, 1, 8),
                "goal_embed_0_level": 4,
                "goal_embed_0_in_upper_space": True,
                "dense_upper_goal_projection": True,
            },
            candidate,
        )
        gradient, = torch.autograd.grad(cost.sum(), candidate)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
