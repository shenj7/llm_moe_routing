"""
Entry point for LLaDA progressive distillation training.

Quick-start
-----------
Run all three rounds (128 → 64 → 32 → 16 steps):

    python train_distill.py

Run only a specific round:

    python train_distill.py --round 0      # 128 → 64 steps

Point to a custom config:

    python train_distill.py --config config/distill_config.yaml

Override individual config values at the CLI:

    python train_distill.py --lr 1e-4 --batch-size 2

Memory tips for the A30 (24 GB)
---------------------------------
  • Teacher is quantised to 8-bit by default (~8 GB).  If bitsandbytes
    is not installed it uses bf16 (~16 GB) — switch to the 400 m preset.
  • For 3 B students, see the memory notes in config/student_configs/student_3b.yaml.
  • Pass --no-gradient-checkpointing to trade memory for speed (not recommended).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
import typer

from distillation.progressive_distill import ProgressiveDistiller

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility helper
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    config: Path = typer.Option(
        "config/distill_config.yaml",
        "--config", "-c",
        help="Path to the distillation config YAML.",
    ),
    round: Optional[int] = typer.Option(
        None,
        "--round", "-r",
        help="Run a single distillation round by index (0-based). "
             "Omit to run all rounds.",
    ),
    device: str = typer.Option(
        "cuda",
        help="Compute device: 'cuda' or 'cpu'.",
    ),
    # ---- Fine-grained overrides (optional) ---------------------------------
    lr: Optional[float] = typer.Option(
        None, "--lr",
        help="Override learning_rate in config.",
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size",
        help="Override training.batch_size in config.",
    ),
    grad_accum: Optional[int] = typer.Option(
        None, "--grad-accum",
        help="Override training.gradient_accumulation_steps in config.",
    ),
    max_steps: Optional[int] = typer.Option(
        None, "--max-steps",
        help="Override max_train_steps for every round.",
    ),
    student_preset: Optional[str] = typer.Option(
        None, "--student-preset",
        help="Override student size preset for every round: 400m | 1b | 3b.",
    ),
    no_gradient_checkpointing: bool = typer.Option(
        False, "--no-gradient-checkpointing",
        help="Disable gradient checkpointing (faster but uses more VRAM).",
    ),
) -> None:
    """Train LLaDA progressive distillation: 128 → 64 → 32 → 16 steps."""

    # ---- Load config -------------------------------------------------------
    with open(config) as fh:
        cfg = yaml.safe_load(fh)

    # ---- Apply CLI overrides -----------------------------------------------
    if lr is not None:
        cfg["training"]["learning_rate"] = lr
    if batch_size is not None:
        cfg["training"]["batch_size"] = batch_size
    if grad_accum is not None:
        cfg["training"]["gradient_accumulation_steps"] = grad_accum
    if no_gradient_checkpointing:
        cfg["training"]["gradient_checkpointing"] = False
    if max_steps is not None:
        for r in cfg["distillation"]["rounds"]:
            r["max_train_steps"] = max_steps
    if student_preset is not None:
        for r in cfg["distillation"]["rounds"]:
            r.pop("student_config", None)   # remove YAML path override
            r["student_preset"] = student_preset

    # ---- Reproducibility ---------------------------------------------------
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    # ---- Device ------------------------------------------------------------
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = "cpu"
    dev = torch.device(device)

    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(
            "GPU: %s  |  VRAM: %.1f GB",
            props.name,
            props.total_memory / 1e9,
        )

    # ---- Build distiller ---------------------------------------------------
    logger.info("Initialising ProgressiveDistiller …")
    distiller = ProgressiveDistiller(cfg=cfg, device=dev)

    # ---- Run ---------------------------------------------------------------
    mode = cfg["distillation"].get("mode", "progressive")

    if mode == "simultaneous":
        if round is not None:
            logger.warning("--round is ignored in simultaneous mode.")
        distiller.train_simultaneous()
    else:
        if round is not None:
            rounds = cfg["distillation"]["rounds"]
            if round >= len(rounds):
                raise typer.BadParameter(
                    f"--round {round} is out of range "
                    f"(config has {len(rounds)} rounds: 0 … {len(rounds) - 1})."
                )
            round_cfg = rounds[round]
            logger.info(
                "Running single round %d: %s",
                round,
                round_cfg.get("name", f"round{round}"),
            )
            distiller.train_round(
                round_idx     = round,
                round_cfg     = round_cfg,
                teacher_model = distiller.teacher,
            )
        else:
            distiller.train_all_rounds()


if __name__ == "__main__":
    app()
