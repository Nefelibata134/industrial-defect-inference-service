from __future__ import annotations

from collections.abc import Iterable, Sized
from itertools import islice
from time import perf_counter
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from industrial_defect.metrics import SegmentationMetrics


def run_epoch(
    *,
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    class_count: int,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
    max_batches: int | None = None,
    description: str | None = None,
) -> dict[str, float | list[float]]:
    training = optimizer is not None
    model.train(training)
    meter = SegmentationMetrics(class_count, threshold=threshold)
    total_loss = 0.0
    sample_count = 0
    start_time = perf_counter()

    batch_iterator = iter(batches)
    total_batches = len(batches) if isinstance(batches, Sized) else None
    if max_batches is not None:
        batch_iterator = islice(batch_iterator, max_batches)
        total_batches = (
            min(max_batches, total_batches) if total_batches is not None else max_batches
        )
    progress = tqdm(
        batch_iterator,
        total=total_batches,
        desc=description,
        leave=False,
        disable=description is None,
    )

    for batch_index, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        amp_enabled = amp and device.type == "cuda"
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at batch {batch_index}")
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
        meter.update(logits, targets)

    if sample_count == 0:
        raise ValueError("epoch received no samples")

    results = meter.compute()
    elapsed_seconds = perf_counter() - start_time
    results["loss"] = total_loss / sample_count
    results["samples"] = float(sample_count)
    results["elapsed_seconds"] = elapsed_seconds
    results["samples_per_second"] = sample_count / elapsed_seconds
    return results
