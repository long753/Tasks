"""Storage isolation regressions exercised with actual files and datasets."""

import os
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from h_jepa.planning_eval import resolve_results_dir
from h_jepa.runtime import storage_config
from h_jepa.utils import resolve_model_checkpoint_path
from stable_worldmodel.data import HDF5Dataset
from stable_worldmodel.paths import PROJECT_ROOT, StoragePaths


class StorageTests(unittest.TestCase):
    def test_relative_storage_is_independent_of_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                paths = StoragePaths.resolve("datasets", "models", "outputs")
                self.assertEqual(paths.datasets, PROJECT_ROOT / "datasets")
                self.assertEqual(paths.models, PROJECT_ROOT / "models")
                self.assertEqual(paths.outputs, PROJECT_ROOT / "outputs")
            finally:
                os.chdir(previous)

    def test_dataset_reads_selected_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.arange(12, dtype=np.float32).reshape(6, 2)
            with h5py.File(root / "trajectory.h5", "w") as dataset:
                dataset["ep_len"] = [6]
                dataset["ep_offset"] = [0]
                dataset["action"] = values
                dataset["state"] = values + 100
            data = HDF5Dataset(
                "trajectory",
                cache_dir=root,
                level1={"frameskip": 2, "num_steps": 2},
            )
            try:
                sample = data[0]
                torch.testing.assert_close(sample["action_level1"], torch.from_numpy(values[:4].reshape(2, 4)))
                torch.testing.assert_close(sample["state_level1"], torch.from_numpy(values[[0, 2]] + 100))
            finally:
                if data.h5_file is not None:
                    data.h5_file.close()

    def test_model_lookup_does_not_use_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            checkpoint = models / "selected_object.ckpt"
            torch.save({"value": torch.tensor([7])}, checkpoint)
            torch.save({"value": torch.tensor([-1])}, root / checkpoint.name)
            try:
                os.chdir(root)
                selected = resolve_model_checkpoint_path(checkpoint.name, models)
                self.assertEqual(torch.load(selected)["value"].item(), 7)
            finally:
                os.chdir(previous)

    def test_results_are_separated_from_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = StoragePaths.resolve(*(str(Path(directory) / name) for name in ("data", "models", "outputs")))
            run = paths.models / "ant" / "model" / "seed42"
            run.mkdir(parents=True)
            checkpoint = run / "model_object.ckpt"
            torch.save({}, checkpoint)
            cfg = storage_config(OmegaConf.create({
                **paths.as_dict(),
                "policy": "ant/model/seed42/model_object.ckpt",
                "output": {"dir": "evaluation"},
            }))
            destination = resolve_results_dir(cfg)
            self.assertEqual(destination, paths.outputs / "ant/model/seed42/evaluation")
            self.assertFalse(destination.is_relative_to(paths.models))


if __name__ == "__main__":
    unittest.main()
