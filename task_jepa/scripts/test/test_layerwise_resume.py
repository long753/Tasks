"""Stage completion must preserve optimizer state and never skip interrupted fits."""

import tempfile
import unittest
from pathlib import Path

import lightning as pl
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from h_jepa.layerwise_training import LayerwiseStage


class InterruptedStage(RuntimeError):
    pass


class InterruptAfterFirstStep(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 1:
            raise InterruptedStage("Intentional interruption after the durable batch checkpoint")


class StopAfterFirstStep(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        trainer.should_stop = True


class ScalarWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.normalizer_artifact = {"stats": {}}

    def forward(self, inputs):
        return self.weight * inputs


class RegressionModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ScalarWorldModel()

    def training_step(self, batch, batch_idx):
        return self.model(batch[0]).square().mean() / 2

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


class LayerwiseResumeTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(torch.random.fork_rng(devices=[]))
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="layerwise resume ")))
        self.cfg = OmegaConf.create({
            "models": str(self.directory), "run_name": "run", "num_levels": 2,
            "trainer": {"max_epochs": 1, "max_steps": 2}, "resume_every_n_steps": 1,
        })
        self.loader = DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=1)

    def trainer(self, *callbacks, max_steps=2):
        return pl.Trainer(
            accelerator="cpu", devices=1, max_epochs=1, max_steps=max_steps,
            logger=False, enable_checkpointing=False, enable_model_summary=False,
            enable_progress_bar=False, num_sanity_val_steps=0,
            default_root_dir=str(self.directory), callbacks=list(callbacks),
        )

    def test_interrupted_stage_restores_weights_momentum_scheduler_and_step_budget(self):
        stage = LayerwiseStage(self.cfg, 1)
        interrupted = self.trainer(stage, InterruptAfterFirstStep())
        with self.assertRaises(InterruptedStage):
            interrupted.fit(RegressionModule(), self.loader)
        self.assertFalse((stage.directory / "progress.json").exists())

        resumed_model = RegressionModule()
        resumed_stage = LayerwiseStage(self.cfg, 1)
        resumed = self.trainer(resumed_stage)
        resumed.fit(resumed_model, self.loader, ckpt_path=str(stage.path))
        self.assertEqual(resumed.global_step, 2)
        # Step 1: w=.9, momentum=1, lr=.05; step 2: momentum=1.8, w=.81.
        self.assertAlmostEqual(float(resumed_model.model.weight.detach()), 0.81, places=6)
        self.assertTrue(resumed_stage.complete)
        self.assertTrue((stage.directory / "progress.json").is_file())

    def test_partial_epoch_stop_is_not_committed_as_a_completed_stage(self):
        stage = LayerwiseStage(self.cfg, 1)
        trainer = self.trainer(stage, StopAfterFirstStep())
        with self.assertRaisesRegex(RuntimeError, "before its training budget"):
            trainer.fit(RegressionModule(), self.loader)
        self.assertFalse(stage.complete)
        self.assertFalse((stage.directory / "progress.json").exists())

    def test_resume_rejects_a_different_stage_budget(self):
        stage = LayerwiseStage(self.cfg, 1)
        trainer = self.trainer(stage, InterruptAfterFirstStep())
        with self.assertRaises(InterruptedStage):
            trainer.fit(RegressionModule(), self.loader)
        self.cfg.trainer.max_steps = 3
        changed = self.trainer(LayerwiseStage(self.cfg, 1), max_steps=3)
        with self.assertRaisesRegex(ValueError, "stage or training budget"):
            changed.fit(RegressionModule(), self.loader, ckpt_path=str(stage.path))


if __name__ == "__main__":
    unittest.main()
