"""
Masking schedule and sampling utilities for LLaDA masked diffusion.

LLaDA uses a *linear* (uniform) masking schedule:

    t  ~  Uniform(0, 1)
    x_t^i  =  [MASK]  with prob t,  else  x_0^i

The model never receives t as an explicit input (it is marginalised out),
so no time embedding is needed in the student either.

Inference (reverse process) uses "low-confidence remasking":
  at each step from time t to t – Δ, the (Δ / t) fraction of masked
  positions with the HIGHEST predicted confidence are unmasked;
  the rest remain masked for the next step.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# LLaDA's special [MASK] token ID (fallback; look up from tokenizer at runtime).
LLADA_MASK_TOKEN_ID: int = 126336


# ---------------------------------------------------------------------------
# Forward process
# ---------------------------------------------------------------------------

def apply_masking(
    x0: torch.Tensor,
    t: float,
    mask_token_id: int = LLADA_MASK_TOKEN_ID,
) -> torch.Tensor:
    """
    Apply the forward masking process: each token is independently replaced
    with [MASK] with probability *t*.

    Args:
        x0:            Clean token sequence, shape [B, L].
        t:             Masking fraction in (0, 1].
        mask_token_id: Vocabulary index of the [MASK] token.

    Returns:
        x_t: Partially-masked sequence, shape [B, L].
    """
    mask = torch.bernoulli(torch.full(x0.shape, t, dtype=torch.float, device=x0.device)).bool()
    x_t = x0.clone()
    x_t[mask] = mask_token_id
    return x_t


def sample_time_index(
    batch_size: int,
    num_steps: int,
    device: torch.device,
    min_t_idx: int = 2,
) -> torch.Tensor:
    """
    Sample a discrete time index k uniformly from [min_t_idx, num_steps].

    We require k ≥ 2 so there is at least one valid teacher step below t
    (the teacher always needs Δ_t = 1/N_teacher headroom).

    Args:
        batch_size:  Number of samples.
        num_steps:   Student total step count (N_student).
        device:      Target device.
        min_t_idx:   Minimum time index (default 2).

    Returns:
        t_idx: Integer tensor of shape [batch_size] in [min_t_idx, num_steps].
    """
    return torch.randint(min_t_idx, num_steps + 1, (batch_size,), device=device)


# ---------------------------------------------------------------------------
# Teacher denoising steps
# ---------------------------------------------------------------------------

@torch.no_grad()
def _teacher_single_step(
    teacher: torch.nn.Module,
    x_t: torch.Tensor,
    t_frac: float,
    delta_t: float,
    mask_token_id: int = LLADA_MASK_TOKEN_ID,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run one teacher denoising step: x_t → x_{t – delta_t}.

    Uses low-confidence remasking (the LLaDA default): accept the
    (delta_t / t_frac) fraction of masked positions that have the
    *highest* predicted confidence; leave the rest masked.

    Args:
        teacher:       Frozen teacher model.
        x_t:           Current noisy tokens, shape [B, L].
        t_frac:        Current masking fraction (scalar float).
        delta_t:       Step size (scalar float).
        mask_token_id: [MASK] token ID.

    Returns:
        x_next:  Partially denoised tokens, shape [B, L].
        logits:  Teacher logits from this step, shape [B, L, V].
    """
    out = teacher(input_ids=x_t)
    logits: torch.Tensor = out.logits          # [B, L, V]
    probs = F.softmax(logits.float(), dim=-1)  # [B, L, V]

    pred_tokens = probs.argmax(dim=-1)          # [B, L]
    confidence  = probs.max(dim=-1).values      # [B, L]

    is_masked = (x_t == mask_token_id)          # [B, L]  bool
    unmask_fraction = min(delta_t / max(t_frac, 1e-8), 1.0)

    x_next = x_t.clone()
    for b in range(x_t.shape[0]):
        masked_pos = is_masked[b].nonzero(as_tuple=True)[0]  # [n_masked]
        n_masked = masked_pos.shape[0]
        if n_masked == 0:
            continue

        n_to_unmask = max(1, int(round(n_masked * unmask_fraction)))
        conf_at_masked = confidence[b, masked_pos]
        top_idx = conf_at_masked.argsort(descending=True)[:n_to_unmask]
        positions_to_unmask = masked_pos[top_idx]
        x_next[b, positions_to_unmask] = pred_tokens[b, positions_to_unmask]

    return x_next, logits


@torch.no_grad()
def teacher_two_step(
    teacher: torch.nn.Module,
    x_t: torch.Tensor,
    t_frac: float,
    delta_s: float,
    mask_token_id: int = LLADA_MASK_TOKEN_ID,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run 2 teacher steps from x_t, covering the same time interval as
    1 student step (delta_s = 2 × delta_t).

    This implements the *progressive distillation* target: the student
    should predict in one shot what 2 teacher steps produce.

    Target construction at originally-masked positions:
      • Positions that the teacher UNMASKED in step 1:
          → one-hot of the accepted token  (teacher was confident here)
      • Positions still masked after step 1:
          → teacher's step-2 soft probability distribution

    Args:
        teacher:       Frozen teacher model.
        x_t:           Current noisy tokens, shape [B, L].
        t_frac:        Current masking fraction (scalar float, same for whole batch).
        delta_s:       Student step size = 2 × teacher step size.
        mask_token_id: [MASK] token ID.

    Returns:
        target_probs:     Soft target distributions, shape [B, L, V].
        originally_masked: Boolean mask of positions masked in x_t, shape [B, L].
    """
    delta_t = delta_s / 2.0  # teacher step size

    # ---- Step 1: x_t → x_{t – delta_t} ----------------------------------------
    x_mid, _ = _teacher_single_step(teacher, x_t, t_frac, delta_t, mask_token_id)

    # ---- Step 2: predict from x_{t – delta_t} -----------------------------------
    out2 = teacher(input_ids=x_mid)
    logits2: torch.Tensor = out2.logits                       # [B, L, V]
    target_probs = F.softmax(logits2.float(), dim=-1)         # [B, L, V]

    # ---- Build combined target --------------------------------------------------
    originally_masked     = (x_t   == mask_token_id)  # [B, L]
    still_masked_at_mid   = (x_mid == mask_token_id)  # [B, L]
    newly_unmasked        = originally_masked & ~still_masked_at_mid  # [B, L]

    # For positions the teacher already accepted in step 1, set target to one-hot.
    # The model was confident enough to unmask them; we treat that as a hard label.
    if newly_unmasked.any():
        vocab_size = target_probs.shape[-1]
        one_hot = F.one_hot(x_mid.long(), num_classes=vocab_size).float()  # [B, L, V]
        target_probs[newly_unmasked] = one_hot[newly_unmasked]

    return target_probs, originally_masked


@torch.no_grad()
def teacher_multi_step(
    teacher: torch.nn.Module,
    x_t: torch.Tensor,
    t_frac: float,
    base_delta_t: float,
    checkpoint_steps: List[int],
    mask_token_id: int = LLADA_MASK_TOKEN_ID,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Run teacher for max(checkpoint_steps) denoising steps of size base_delta_t,
    capturing a soft target distribution at each step in checkpoint_steps.

    This lets multiple students share a single teacher trajectory: e.g. for
    students at [64, 32, 16] steps the teacher runs 8 steps total and returns
    targets at steps [2, 4, 8], rather than running 2+4+8=14 passes separately.

    Args:
        teacher:          Frozen teacher model.
        x_t:              Noisy input tokens, shape [B, L].
        t_frac:           Current masking fraction (scalar float).
        base_delta_t:     Teacher step size = 1 / base_teacher_steps.
        checkpoint_steps: Sorted list of step counts at which to capture targets.
        mask_token_id:    [MASK] token ID.

    Returns:
        Dict mapping each n in checkpoint_steps to
            (target_probs [B, L, V], originally_masked [B, L]).
    """
    originally_masked = (x_t == mask_token_id)
    checkpoint_set    = set(checkpoint_steps)
    x_cur  = x_t
    cur_t  = t_frac
    results: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    for step in range(1, max(checkpoint_steps) + 1):
        x_cur, _ = _teacher_single_step(teacher, x_cur, cur_t, base_delta_t, mask_token_id)
        cur_t = max(cur_t - base_delta_t, 0.0)

        if step in checkpoint_set:
            out          = teacher(input_ids=x_cur)
            target_probs = F.softmax(out.logits.float(), dim=-1)  # [B, L, V]

            still_masked   = (x_cur == mask_token_id)
            newly_unmasked = originally_masked & ~still_masked
            if newly_unmasked.any():
                vocab_size = target_probs.shape[-1]
                one_hot    = F.one_hot(x_cur.long(), num_classes=vocab_size).float()
                target_probs[newly_unmasked] = one_hot[newly_unmasked]

            results[step] = (target_probs.clone(), originally_masked.clone())

    return results
