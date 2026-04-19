"""
Progressive distillation trainer for LLaDA.

Algorithm (Salimans & Ho 2022 adapted to masked diffusion):

  Given a teacher running at N steps, train a student to match the teacher's
  2-step output in a single step → student runs at N/2 steps.

  Repeat three times:  128 → 64 → 32 → 16  steps.

Training step (one micro-batch):
  1. Sample a clean token sequence  x_0  from the dataset.
  2. Sample a time index  k  uniformly from  {2, …, N_student}.
  3. Compute the masking fraction  t = k / N_student.
  4. Apply forward masking:  x_t = mask(x_0, t).
  5. Run teacher 2 steps from  x_t  to get target distribution  p_T  (no grad).
  6. Run student 1 step from  x_t  to get  p_S  (with grad).
  7. Compute score-matching loss:  L = MSE( p_S[masked] , p_T[masked] ).
  8. Accumulate gradients; optimiser step every grad_accumulation_steps.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_scheduler

from data.masked_diffusion_dataset import MaskedDiffusionDataset, collate_fn
from distillation.losses import combined_loss
from distillation.sampler import (
    LLADA_MASK_TOKEN_ID,
    apply_masking,
    sample_time_index,
    teacher_two_step,
    teacher_multi_step,
)
from models.student_arch import create_student_model

logger = logging.getLogger(__name__)

# Optional bitsandbytes for 8-bit Adam and 8-bit model loading.
try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False
    logger.warning(
        "bitsandbytes not installed. "
        "Falling back to standard AdamW and bf16 teacher loading. "
        "Install it with:  pip install bitsandbytes"
    )


# ---------------------------------------------------------------------------
# Helper: resolve mask token ID
# ---------------------------------------------------------------------------

def _get_mask_token_id(tokenizer, fallback: int = LLADA_MASK_TOKEN_ID) -> int:
    """Return the [MASK] token ID from the tokenizer, or use the fallback."""
    if tokenizer.mask_token_id is not None:
        return int(tokenizer.mask_token_id)
    mid = tokenizer.convert_tokens_to_ids("[MASK]")
    if isinstance(mid, int) and mid != tokenizer.unk_token_id:
        return mid
    logger.warning(
        "Could not resolve mask token from tokenizer; "
        "using fallback id=%d", fallback
    )
    return fallback


# ---------------------------------------------------------------------------
# Helper: extract logits from model output
# ---------------------------------------------------------------------------

def _get_logits(output) -> torch.Tensor:
    """
    Extract token logits from a HuggingFace model output.

    LLaDA exposes `.logits` directly.  This helper also handles the case
    where only `.last_hidden_state` is available (no LM head on the model).
    """
    if hasattr(output, "logits") and output.logits is not None:
        return output.logits
    # Fallback: if only hidden states are returned the caller should add a
    # language-model head.  Raise a clear error so the issue is obvious.
    raise AttributeError(
        "Model output has no '.logits' attribute. "
        "Ensure the student/teacher is loaded as a full masked-LM model "
        "(e.g. AutoModelForMaskedLM or the LLaDA model class with its LM head)."
    )


# ---------------------------------------------------------------------------
# Main distiller class
# ---------------------------------------------------------------------------

class ProgressiveDistiller:
    """
    Manages all progressive distillation rounds for LLaDA.

    Usage::

        distiller = ProgressiveDistiller(cfg, device)
        distiller.train_all_rounds()          # runs all 3 rounds
        # — or —
        distiller.train_round(0, cfg["distillation"]["rounds"][0],
                              distiller.teacher)   # single round
    """

    def __init__(self, cfg: Dict, device: torch.device) -> None:
        self.cfg    = cfg
        self.device = device

        # Resolve mask token id early so we can pass it everywhere.
        self._mask_token_id: Optional[int] = cfg["teacher"].get("mask_token_id")

        # Load teacher (frozen).
        self.teacher = self._load_teacher()

        # Load shared tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["teacher"]["model_id"],
            trust_remote_code=True,
        )
        if self._mask_token_id is None:
            self._mask_token_id = _get_mask_token_id(self.tokenizer)
        logger.info("Mask token id: %d", self._mask_token_id)

        # Build shared dataset / dataloader.
        data_cfg = cfg["data"]
        dataset  = MaskedDiffusionDataset(
            tokenizer      = self.tokenizer,
            dataset_name   = data_cfg["dataset_name"],
            dataset_config = data_cfg.get("dataset_config", "sample-10BT"),
            text_column    = data_cfg.get("text_column", "text"),
            max_seq_length = data_cfg["max_seq_length"],
            streaming      = data_cfg.get("streaming", True),
            seed           = cfg["training"].get("seed", 42),
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size  = cfg["training"]["batch_size"],
            collate_fn  = collate_fn,
            num_workers = data_cfg.get("num_workers", 2),
        )

    # ------------------------------------------------------------------
    # Teacher loading
    # ------------------------------------------------------------------

    def _load_teacher(self) -> nn.Module:
        t_cfg    = self.cfg["teacher"]
        model_id = t_cfg["model_id"]
        use_8bit = t_cfg.get("load_in_8bit", True) and _HAS_BNB

        logger.info("Loading teacher %s (8-bit=%s) …", model_id, use_8bit)

        load_kwargs: Dict = {"trust_remote_code": True}
        if use_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"]   = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"]  = "auto"

        teacher = AutoModel.from_pretrained(model_id, **load_kwargs)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        logger.info("Teacher loaded and frozen.")
        return teacher

    # ------------------------------------------------------------------
    # Student creation
    # ------------------------------------------------------------------

    def _create_student(self, round_cfg: Dict) -> nn.Module:
        student = create_student_model(
            student_config_path = round_cfg.get("student_config"),
            preset              = round_cfg.get("student_preset", "1b"),
        )

        # Optionally resume from a checkpoint.
        ckpt_path = round_cfg.get("student_ckpt")
        if ckpt_path and os.path.isfile(ckpt_path):
            logger.info("Loading student weights from checkpoint: %s", ckpt_path)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            # Support both full training checkpoints (dict with "model" key)
            # and legacy weight-only checkpoints (plain state dict).
            student.load_state_dict(state.get("model", state))

        student = student.to(self.device, dtype=torch.bfloat16)

        if self.cfg["training"].get("gradient_checkpointing", True):
            if hasattr(student, "gradient_checkpointing_enable"):
                student.gradient_checkpointing_enable()

        return student

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def _create_optimizer(self, student: nn.Module) -> torch.optim.Optimizer:
        lr      = self.cfg["training"]["learning_rate"]
        wd      = self.cfg["training"].get("weight_decay", 0.01)
        opt_key = self.cfg["training"].get("optimizer", "adamw_bnb_8bit")

        if opt_key == "adamw_bnb_8bit" and _HAS_BNB:
            return bnb.optim.AdamW8bit(student.parameters(), lr=lr, weight_decay=wd)
        else:
            if opt_key == "adamw_bnb_8bit" and not _HAS_BNB:
                logger.warning("bitsandbytes unavailable; falling back to standard AdamW.")
            return torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=wd)

    # ------------------------------------------------------------------
    # LR scheduler
    # ------------------------------------------------------------------

    def _create_scheduler(self, optimizer, num_training_steps: int):
        t_cfg = self.cfg["training"]
        warmup = int(num_training_steps * t_cfg.get("warmup_ratio", 0.05))
        return get_scheduler(
            name                = t_cfg.get("lr_scheduler", "cosine"),
            optimizer           = optimizer,
            num_warmup_steps    = warmup,
            num_training_steps  = num_training_steps,
        )

    # ------------------------------------------------------------------
    # Core training round
    # ------------------------------------------------------------------

    def train_round(
        self,
        round_idx:     int,
        round_cfg:     Dict,
        teacher_model: nn.Module,
    ) -> nn.Module:
        """
        Train one distillation round: teacher at N steps → student at N/2 steps.

        Args:
            round_idx:     Index of this round (0, 1, 2, …).
            round_cfg:     Sub-dict from distillation.rounds[round_idx].
            teacher_model: Frozen model used to generate 2-step targets.

        Returns:
            student: Trained student model (useful as the teacher for the
                     next round when use_prev_student_as_teacher is true).
        """
        teacher_steps   = round_cfg["teacher_steps"]
        student_steps   = round_cfg["student_steps"]
        max_steps       = round_cfg["max_train_steps"]
        round_name      = round_cfg.get("name", f"round{round_idx}")
        t_cfg           = self.cfg["training"]
        d_cfg           = self.cfg["distillation"]

        grad_accum      = t_cfg.get("gradient_accumulation_steps", 8)
        max_grad_norm   = t_cfg.get("max_grad_norm", 1.0)
        loss_type       = d_cfg.get("loss_type", "score_matching")
        mdm_weight      = d_cfg.get("loss_weight_original", 0.0)
        log_every       = t_cfg.get("log_every", 50)
        save_every      = t_cfg.get("save_every", 2000)
        save_dir        = t_cfg.get("save_dir", "checkpoints/distill")

        os.makedirs(save_dir, exist_ok=True)

        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "[Round %d] %s  —  %d → %d steps",
            round_idx, round_name, teacher_steps, student_steps,
        )
        logger.info("=" * 60)

        student   = self._create_student(round_cfg)
        optimizer = self._create_optimizer(student)
        scheduler = self._create_scheduler(optimizer, max_steps)

        # Student step size (fraction of masking removed per student step).
        delta_s: float = 1.0 / student_steps

        student.train()
        teacher_model.eval()

        data_iter        = iter(self.dataloader)
        global_step      = 0
        start_micro_step = 0
        running_loss     = 0.0
        optimizer.zero_grad()

        # ---- Restore optimizer / scheduler state if resuming -----------------
        ckpt_path = round_cfg.get("student_ckpt")
        if ckpt_path and os.path.isfile(ckpt_path):
            resume = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if isinstance(resume, dict) and "optimizer" in resume:
                optimizer.load_state_dict(resume["optimizer"])
                scheduler.load_state_dict(resume["scheduler"])
                global_step      = resume["global_step"]
                start_micro_step = resume["micro_step"] + 1
                logger.info(
                    "Resumed optimizer/scheduler — continuing from "
                    "global_step=%d, micro_step=%d.",
                    global_step, resume["micro_step"],
                )
            else:
                logger.info(
                    "Checkpoint contains weights only — "
                    "optimizer/scheduler state not restored."
                )

        micro_steps_total = max_steps * grad_accum
        last_micro_step   = max(start_micro_step - 1, 0)

        for micro_step in range(start_micro_step, micro_steps_total):
            # ---- Fetch batch -------------------------------------------------
            try:
                x0 = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                x0 = next(data_iter)

            x0 = x0.to(self.device)  # [B, L]

            # ---- Sample time ------------------------------------------------
            # Use a single t for the whole micro-batch (simpler; the teacher
            # 2-step loop is parameterised by a scalar t).
            k_tensor = sample_time_index(
                batch_size=1,
                num_steps=student_steps,
                device=self.device,
            )
            k      = int(k_tensor.item())
            t_frac = k / student_steps

            # ---- Forward masking  (x_0 → x_t) --------------------------------
            x_t = apply_masking(x0, t_frac, self._mask_token_id)

            # ---- Teacher 2-step target (no grad) -----------------------------
            with torch.no_grad():
                target_probs, masked_positions = teacher_two_step(
                    teacher   = teacher_model,
                    x_t       = x_t,
                    t_frac    = t_frac,
                    delta_s   = delta_s,
                    mask_token_id = self._mask_token_id,
                )
            # target_probs:    [B, L, V]  soft probability distribution
            # masked_positions:[B, L]     bool — positions masked in x_t

            # ---- Student forward pass (with grad) ----------------------------
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                out            = student(input_ids=x_t)
                student_logits = _get_logits(out)                # [B, L, V]

                loss = combined_loss(
                    student_logits   = student_logits.float(),
                    target_probs     = target_probs,
                    x0               = x0,
                    masked_positions = masked_positions,
                    t                = t_frac,
                    loss_type        = loss_type,
                    mdm_weight       = mdm_weight,
                )
                loss = loss / grad_accum

            if not torch.isfinite(loss):
                logger.error(
                    "Non-finite loss (%.4g) at micro_step %d — stopping round.",
                    loss.item(), micro_step,
                )
                break

            loss.backward()
            running_loss  += loss.item()
            last_micro_step = micro_step

            # ---- Gradient accumulation step ----------------------------------
            if (micro_step + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                avg_loss     = running_loss  # already divided by grad_accum
                running_loss = 0.0

                if global_step % log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        "[Round %d | %s] step %d/%d  loss=%.4f  lr=%.2e  t=%.3f",
                        round_idx, round_name, global_step, max_steps,
                        avg_loss, lr, t_frac,
                    )

                if global_step % save_every == 0:
                    ckpt = os.path.join(save_dir, f"{round_name}_step{global_step}.pt")
                    torch.save({
                        "model":       student.state_dict(),
                        "optimizer":   optimizer.state_dict(),
                        "scheduler":   scheduler.state_dict(),
                        "global_step": global_step,
                        "micro_step":  micro_step,
                    }, ckpt)
                    logger.info("Checkpoint saved: %s", ckpt)

                if global_step >= max_steps:
                    break

        # ---- Save final model ------------------------------------------------
        final_path = os.path.join(save_dir, f"{round_name}_final.pt")
        torch.save({
            "model":       student.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "global_step": global_step,
            "micro_step":  last_micro_step,
        }, final_path)
        logger.info("Round %d complete.  Final model: %s", round_idx, final_path)

        return student

    # ------------------------------------------------------------------
    # Simultaneous multi-student training
    # ------------------------------------------------------------------

    def train_simultaneous(self) -> None:
        """
        Train all students simultaneously from a single shared teacher trajectory.

        For each micro-batch the teacher runs once for max(teacher_steps_per_student)
        denoising steps, capturing intermediate targets for every student along the
        way.  Each student then independently does its own forward + backward pass
        against its target.  This avoids re-running the teacher for each student.

        Requires distillation.mode == "simultaneous" and distillation.students
        list in the config.
        """
        d_cfg = self.cfg["distillation"]
        t_cfg = self.cfg["training"]

        student_cfgs      = d_cfg["students"]
        base_teacher_steps = d_cfg["base_teacher_steps"]
        base_delta_t      = 1.0 / base_teacher_steps
        max_steps         = d_cfg["max_train_steps"]
        loss_type         = d_cfg.get("loss_type", "score_matching")
        mdm_weight        = d_cfg.get("loss_weight_original", 0.0)
        grad_accum        = t_cfg.get("gradient_accumulation_steps", 8)
        max_grad_norm     = t_cfg.get("max_grad_norm", 1.0)
        log_every         = t_cfg.get("log_every", 50)
        save_every        = t_cfg.get("save_every", 2000)
        save_dir          = t_cfg.get("save_dir", "checkpoints/distill")

        os.makedirs(save_dir, exist_ok=True)

        # Number of teacher steps needed per student:
        #   student at S steps covers delta_s = 1/S per step;
        #   teacher step size = base_delta_t = 1/base_teacher_steps;
        #   teacher steps needed = delta_s / base_delta_t = base_teacher_steps / S
        n_teacher_steps = {
            s["student_steps"]: base_teacher_steps // s["student_steps"]
            for s in student_cfgs
        }
        checkpoint_steps  = sorted(set(n_teacher_steps.values()))
        min_student_steps = min(s["student_steps"] for s in student_cfgs)

        logger.info("")
        logger.info("=" * 60)
        logger.info("Simultaneous distillation — %d students", len(student_cfgs))
        for s in student_cfgs:
            logger.info(
                "  %s  (%d steps, %d teacher steps/batch)",
                s["name"], s["student_steps"], n_teacher_steps[s["student_steps"]],
            )
        logger.info("=" * 60)

        students   = [self._create_student(s) for s in student_cfgs]
        optimizers = [self._create_optimizer(s) for s in students]
        schedulers = [self._create_scheduler(opt, max_steps) for opt in optimizers]

        for s in students:
            s.train()
        self.teacher.eval()

        data_iter     = iter(self.dataloader)
        global_step   = 0
        running_losses = [0.0] * len(students)

        for opt in optimizers:
            opt.zero_grad()

        micro_steps_total = max_steps * grad_accum
        last_micro_step   = 0

        for micro_step in range(micro_steps_total):
            try:
                x0 = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                x0 = next(data_iter)
            x0 = x0.to(self.device)

            # Sample t valid for all students (most restrictive = fewest steps)
            k_tensor = sample_time_index(
                batch_size=1,
                num_steps=min_student_steps,
                device=self.device,
            )
            t_frac = int(k_tensor.item()) / min_student_steps

            x_t = apply_masking(x0, t_frac, self._mask_token_id)

            # One teacher trajectory shared across all students
            with torch.no_grad():
                targets = teacher_multi_step(
                    teacher          = self.teacher,
                    x_t              = x_t,
                    t_frac           = t_frac,
                    base_delta_t     = base_delta_t,
                    checkpoint_steps = checkpoint_steps,
                    mask_token_id    = self._mask_token_id,
                )

            # Each student trains against its own checkpoint target
            for i, (student, optimizer, s_cfg) in enumerate(
                zip(students, optimizers, student_cfgs)
            ):
                chk = n_teacher_steps[s_cfg["student_steps"]]
                target_probs, masked_positions = targets[chk]

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    out            = student(input_ids=x_t)
                    student_logits = _get_logits(out)

                    loss = combined_loss(
                        student_logits   = student_logits.float(),
                        target_probs     = target_probs,
                        x0               = x0,
                        masked_positions = masked_positions,
                        t                = t_frac,
                        loss_type        = loss_type,
                        mdm_weight       = mdm_weight,
                    )
                    loss = loss / grad_accum

                if not torch.isfinite(loss):
                    logger.error(
                        "Non-finite loss for student '%s' at micro_step %d — skipping.",
                        s_cfg["name"], micro_step,
                    )
                    continue

                loss.backward()
                running_losses[i] += loss.item()

            last_micro_step = micro_step

            if (micro_step + 1) % grad_accum == 0:
                for student, optimizer, scheduler in zip(students, optimizers, schedulers):
                    nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step  += 1
                lr            = optimizers[0].param_groups[0]["lr"]

                if global_step % log_every == 0:
                    loss_str = "  ".join(
                        f"{s['name']}={running_losses[i]:.4f}"
                        for i, s in enumerate(student_cfgs)
                    )
                    logger.info(
                        "step %d/%d  %s  lr=%.2e  t=%.3f",
                        global_step, max_steps, loss_str, lr, t_frac,
                    )

                running_losses = [0.0] * len(students)

                if global_step % save_every == 0:
                    for i, s_cfg in enumerate(student_cfgs):
                        ckpt = os.path.join(
                            save_dir, f"{s_cfg['name']}_step{global_step}.pt"
                        )
                        torch.save({
                            "model":       students[i].state_dict(),
                            "optimizer":   optimizers[i].state_dict(),
                            "scheduler":   schedulers[i].state_dict(),
                            "global_step": global_step,
                            "micro_step":  micro_step,
                        }, ckpt)
                    logger.info("Checkpoints saved at step %d", global_step)

                if global_step >= max_steps:
                    break

        for i, s_cfg in enumerate(student_cfgs):
            final_path = os.path.join(save_dir, f"{s_cfg['name']}_final.pt")
            torch.save({
                "model":       students[i].state_dict(),
                "optimizer":   optimizers[i].state_dict(),
                "scheduler":   schedulers[i].state_dict(),
                "global_step": global_step,
                "micro_step":  last_micro_step,
            }, final_path)
            logger.info("Saved: %s", final_path)

        logger.info("Simultaneous distillation complete.")

    # ------------------------------------------------------------------
    # Orchestrate all rounds
    # ------------------------------------------------------------------

    def train_all_rounds(self) -> None:
        """
        Run all progressive distillation rounds sequentially.

        After each round, if use_prev_student_as_teacher is true, the trained
        student is frozen and promoted to be the teacher for the next round.
        This chains 128 → 64 → 32 → 16 step models progressively.
        """
        rounds          = self.cfg["distillation"]["rounds"]
        current_teacher = self.teacher

        for round_idx, round_cfg in enumerate(rounds):
            student = self.train_round(
                round_idx     = round_idx,
                round_cfg     = round_cfg,
                teacher_model = current_teacher,
            )

            # Promote trained student to teacher for the next round.
            if round_cfg.get("use_prev_student_as_teacher", False):
                logger.info(
                    "Promoting round-%d student to teacher for round %d.",
                    round_idx, round_idx + 1,
                )
                student.eval()
                for p in student.parameters():
                    p.requires_grad_(False)
                current_teacher = student

        logger.info("All distillation rounds complete.")
