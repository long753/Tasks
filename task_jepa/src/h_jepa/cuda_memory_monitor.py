"""Opt-in, rank-local allocator telemetry for H-JEPA training."""

import json
import logging
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.utils.hooks import RemovableHandle

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class CudaMemoryMonitor(Callback):
    # Keep recent allocation/free events and capture the first unusually high peak.
    _HISTORY_ENTRIES = 32768
    _HIGH_WATER_FRACTION = 0.75
    _TENSORBOARD_STATS = ("allocated_mib", "peak_mib", "reserved_mib", "inactive_split_mib", "device_free_mib")

    def __init__(self, output_dir: Path, image_encoder: nn.Module) -> None:
        self._output_dir = output_dir / "cuda_memory"
        self._image_encoder = image_encoder
        self._stream: TextIO | None = None
        self._tensorboard: SummaryWriter | None = None
        self._history_enabled = False
        self._high_water_seen = False
        self._batch: dict[str, int | float | str] | None = None
        self._encoder_hooks: tuple[RemovableHandle, ...] = ()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._device = pl_module.device
        if self._device.type != "cuda":
            if trainer.is_global_zero:
                logger.warning("CUDA memory monitoring is disabled for a non-CUDA Trainer")
            return
        self._rank = trainer.global_rank
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._output_dir / f"rank-{self._rank}-{time.time_ns()}.jsonl"
        self._stream = self._path.open("x", encoding="utf-8", buffering=1)
        properties = torch.cuda.get_device_properties(self._device)
        self._high_water_bytes = int(properties.total_memory * self._HIGH_WATER_FRACTION)
        self._high_water_seen = False
        torch.cuda.memory._record_memory_history(
            enabled="all", context="alloc", stacks="all", max_entries=self._HISTORY_ENTRIES,
        )
        self._history_enabled = True
        self._write(
            "start", device=str(self._device), gpu=properties.name,
            allocator=torch.cuda.get_allocator_backend(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            capacity_mib=self._mib(properties.total_memory), high_water_mib=self._mib(self._high_water_bytes),
        )
        tensorboard_logger = next((log for log in trainer.loggers if isinstance(log, TensorBoardLogger)), None)
        if tensorboard_logger is not None:
            from torch.utils.tensorboard import SummaryWriter

            # Lightning's logger writes only on rank zero. Each rank owns its own event files.
            log_dir = Path(tensorboard_logger.root_dir) / "cuda" / self._path.stem
            self._tensorboard = SummaryWriter(str(log_dir), max_queue=100, flush_secs=10)

    @staticmethod
    def _mib(size: int) -> float:
        return round(size / (1 << 20), 2)

    def _write(self, event: str, **fields: object) -> None:
        if self._stream is not None:
            self._stream.write(json.dumps({"event": event, "rank": self._rank, **fields}) + "\n")

    def _stats(self) -> dict[str, int | float]:
        stats = torch.cuda.memory_stats(self._device)
        return {
            "allocated_mib": self._mib(stats["allocated_bytes.all.current"]),
            "peak_mib": self._mib(stats["allocated_bytes.all.peak"]),
            "reserved_mib": self._mib(stats["reserved_bytes.all.current"]),
            "peak_reserved_mib": self._mib(stats["reserved_bytes.all.peak"]),
            "inactive_split_mib": self._mib(stats["inactive_split_bytes.all.current"]),
            "device_free_mib": self._mib(torch.cuda.mem_get_info(self._device)[0]),
            "allocation_retries": stats["num_alloc_retries"],
            "ooms": stats["num_ooms"],
        }

    def _snapshot(self, reason: str, step: int, phase: str) -> None:
        path = self._path.with_suffix(f".{reason}-step-{step}.pickle")
        tmp = path.with_suffix(".tmp")
        try:
            torch.cuda.memory._dump_snapshot(str(tmp))
            tmp.replace(path)
        except (OSError, RuntimeError) as error:
            with suppress(OSError):
                tmp.unlink(missing_ok=True)
            self._write("snapshot_error", reason=reason, step=step, phase=phase, error=str(error))
            logger.warning("Could not save CUDA allocator snapshot %s: %s", path, error)
            return
        self._write("snapshot", reason=reason, step=step, phase=phase, file=path.name)
        logger.warning("CUDA allocator %s snapshot (rank %d, step %d): %s", reason, self._rank, step, path)

    def _phase(self, phase: str, *, snapshot: bool = True) -> None:
        if self._batch is None:
            return
        self._batch["phase"] = phase
        self._batch[f"{phase}_mib"] = self._mib(torch.cuda.memory_allocated(self._device))
        if snapshot and not self._high_water_seen and torch.cuda.max_memory_allocated(self._device) >= self._high_water_bytes:
            self._high_water_seen = True
            self._snapshot("peak", int(self._batch["step"]), phase)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int) -> None:
        if self._stream is None:
            return
        torch.cuda.reset_peak_memory_stats(self._device)
        self._batch = {"epoch": trainer.current_epoch, "batch": batch_idx, "step": trainer.global_step}
        self._phase("start", snapshot=False)
        # Remove these before epoch/checkpoint hooks ever serialize the model.
        self._encoder_hooks = (
            self._image_encoder.register_forward_pre_hook(self._encoder_start),
            self._image_encoder.register_forward_hook(self._encoder_end),
        )

    def _encoder_start(self, module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        if self._batch is not None:
            self._batch["l1_images"] = args[0].size(0)
            self._phase("l1_start", snapshot=False)

    def _encoder_end(self, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self._phase("l1_end")

    def on_before_backward(self, trainer: Trainer, pl_module: LightningModule, loss: torch.Tensor) -> None:
        self._phase("forward")

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._phase("backward")

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:
        if self._batch is None:
            return
        self._remove_hooks()
        self._batch["step"] = trainer.global_step
        self._phase("optimizer")
        stats = self._stats()
        self._write("batch", **self._batch, **stats)
        if self._tensorboard is not None:
            for stat in self._TENSORBOARD_STATS:
                self._tensorboard.add_scalar(f"cuda/{stat}", stats[stat], trainer.global_step)
            if "l1_images" in self._batch:
                self._tensorboard.add_scalar("cuda/l1_images", self._batch["l1_images"], trainer.global_step)
        self._batch = None

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        if self._stream is None:
            return
        try:
            self._remove_hooks()
            fields = dict(self._batch or {})
            fields["step"] = trainer.global_step
            phase = str(fields.setdefault("phase", "validation" if trainer.validating or trainer.sanity_checking else "fit"))
            oom = isinstance(exception, torch.OutOfMemoryError)
            self._write("oom" if oom else "exception", **fields, **self._stats(), error=str(exception))
            if oom and self._history_enabled:
                self._snapshot("oom", trainer.global_step, phase)
        except Exception:
            logger.exception("Could not record CUDA diagnostics for the original training exception")
        finally:
            try:
                self._close()
            except Exception:
                logger.exception("Could not close CUDA memory monitor")

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._write("end", step=trainer.global_step)

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self._close()

    def _remove_hooks(self) -> None:
        for hook in self._encoder_hooks:
            hook.remove()
        self._encoder_hooks = ()

    def _close(self) -> None:
        self._remove_hooks()
        self._batch = None
        try:
            if self._history_enabled:
                torch.cuda.memory._record_memory_history(enabled=None)
                self._history_enabled = False
        finally:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            if self._tensorboard is not None:
                self._tensorboard.close()
                self._tensorboard = None
