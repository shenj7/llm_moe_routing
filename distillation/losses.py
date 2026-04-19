"""
Loss functions for LLaDA progressive distillation.

The core idea (analogous to Salimans & Ho 2022, adapted for discrete masked
diffusion):

    The "score" in masked diffusion is the denoising function
        D_θ(x_t) = p_θ(x_0 | x_t)
    — i.e. the predicted probability distribution over the vocabulary at
    each masked position.

    Score-matching distillation loss:
        L = E_{ x_0, t, x_t } [
              Σ_i  1[x_t^i = MASK]  ‖ p_student(x_0^i|x_t) − p_teacher_2step^i ‖²
            ]

    where p_teacher_2step is the teacher's predicted distribution after 2
    denoising steps from x_t (see distillation/sampler.py).

Three loss variants are provided:
    "score_matching"  – MSE between softmax probability vectors  (default)
    "kl"              – KL( teacher ‖ student )
    "cross_entropy"   – hard-label CE using teacher argmax

An optional auxiliary MDM training loss (cross-entropy against the TRUE
tokens) can be mixed in to prevent the student from drifting from the data
distribution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def score_matching_loss(
    student_logits:  torch.Tensor,   # [B, L, V]
    target_probs:    torch.Tensor,   # [B, L, V]
    masked_positions: torch.Tensor,  # [B, L]  bool
) -> torch.Tensor:
    """
    MSE between student and teacher probability distributions at masked positions.

        L = (1 / N_masked) * Σ_masked ‖ softmax(student_logits_i) − target_probs_i ‖²

    This is the discrete-diffusion analogue of denoising-score-matching loss.
    """
    student_probs = F.softmax(student_logits.float(), dim=-1)  # [B, L, V]

    s = student_probs[masked_positions]   # [N_masked, V]
    t = target_probs[masked_positions]    # [N_masked, V]

    if s.shape[0] == 0:
        return student_logits.sum() * 0.0  # keeps grad graph alive

    return F.mse_loss(s, t)


def kl_distillation_loss(
    student_logits:  torch.Tensor,   # [B, L, V]
    target_probs:    torch.Tensor,   # [B, L, V]
    masked_positions: torch.Tensor,  # [B, L]  bool
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    KL divergence  KL( teacher ‖ student )  at masked positions.

    Using teacher-as-target (forward KL) so we cover the full support of
    the teacher distribution rather than the student's modes.
    """
    log_student = F.log_softmax(student_logits.float() / temperature, dim=-1)

    ls = log_student[masked_positions]   # [N_masked, V]
    t  = target_probs[masked_positions]  # [N_masked, V]

    if ls.shape[0] == 0:
        return student_logits.sum() * 0.0

    # F.kl_div expects (log_input, target); reduction="batchmean" averages over batch dim.
    return F.kl_div(ls, t, reduction="batchmean", log_target=False)


def cross_entropy_distillation_loss(
    student_logits:  torch.Tensor,   # [B, L, V]
    target_probs:    torch.Tensor,   # [B, L, V]
    masked_positions: torch.Tensor,  # [B, L]  bool
) -> torch.Tensor:
    """
    Hard-label cross-entropy using the teacher's argmax as the target class.
    Faster to compute than KL; less informative about soft uncertainty.
    """
    hard_targets = target_probs.argmax(dim=-1)  # [B, L]

    B, L, V = student_logits.shape
    logits_flat  = student_logits.reshape(B * L, V)
    targets_flat = hard_targets.reshape(B * L)
    mask_flat    = masked_positions.reshape(B * L)

    if mask_flat.sum() == 0:
        return student_logits.sum() * 0.0

    return F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])


def mdm_training_loss(
    student_logits:  torch.Tensor,   # [B, L, V]
    x0:              torch.Tensor,   # [B, L]   clean tokens
    masked_positions: torch.Tensor,  # [B, L]   bool
    t: float,
) -> torch.Tensor:
    """
    Original MDM (Masked Diffusion Model) training loss — cross-entropy against
    the TRUE tokens, importance-weighted by 1/t:

        L_MDM = −(1/t) · E[ Σ_i 1[masked] · log p_θ(x_0^i | x_t) ]

    Adding a small fraction of this loss alongside the distillation objective
    helps the student stay anchored to the real data distribution.
    """
    if masked_positions.sum() == 0:
        return student_logits.sum() * 0.0

    B, L, V = student_logits.shape
    logits_flat  = student_logits.reshape(B * L, V)
    targets_flat = x0.reshape(B * L)
    mask_flat    = masked_positions.reshape(B * L)

    ce = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
    return ce / max(t, 1e-8)


# ---------------------------------------------------------------------------
# Combined loss (main entry point used by the trainer)
# ---------------------------------------------------------------------------

def combined_loss(
    student_logits:  torch.Tensor,   # [B, L, V]
    target_probs:    torch.Tensor,   # [B, L, V]
    x0:              torch.Tensor,   # [B, L]
    masked_positions: torch.Tensor,  # [B, L]  bool
    t: float,
    loss_type:       str   = "score_matching",
    mdm_weight:      float = 0.0,
) -> torch.Tensor:
    """
    Compute the distillation loss (+ optional MDM regularisation).

    Args:
        student_logits:   Raw logits from student forward pass [B, L, V].
        target_probs:     Teacher 2-step soft targets [B, L, V].
        x0:               Clean token sequence [B, L].
        masked_positions: Bool mask of positions that were masked in x_t [B, L].
        t:                Masking fraction used to construct x_t (scalar float).
        loss_type:        One of "score_matching" | "kl" | "cross_entropy".
        mdm_weight:       Weight λ for the auxiliary MDM loss (0 = disabled).

    Returns:
        total_loss: Scalar loss tensor with gradient graph attached.
    """
    if loss_type == "score_matching":
        distill = score_matching_loss(student_logits, target_probs, masked_positions)
    elif loss_type == "kl":
        distill = kl_distillation_loss(student_logits, target_probs, masked_positions)
    elif loss_type == "cross_entropy":
        distill = cross_entropy_distillation_loss(student_logits, target_probs, masked_positions)
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. "
            "Choose from: 'score_matching', 'kl', 'cross_entropy'."
        )

    if mdm_weight > 0.0:
        aux = mdm_training_loss(student_logits, x0, masked_positions, t)
        return distill + mdm_weight * aux

    return distill
