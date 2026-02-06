import torch
from torch import nn, Tensor
import torch.distributed as dist
from trl import SFTTrainer, SFTConfig

import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def smooth_asymmetric_power_ratio_math(
    ratio: float,
    beta_pos: float = 0.5,   # r > 1 时的放大强度
    beta_neg: float = 0.25,  # r < 1 时的压缩强度（更接近 1）
    tau: float = 0.4,        # log 域平滑宽度
    eps: float = 1e-12,
) -> float:
    """
    Smoothly maps ratio=v/mean_v (ratio>0) to a multiplicative factor g(ratio),
    where negative log-ratios are compressed (smaller slope) and positive side
    keeps stronger scaling.

    Properties:
      - g(1) = 1
      - For large ratio: g(r) ≈ r^{beta_pos}
      - For small ratio: g(r) ≈ r^{beta_neg}
      - Continuous and differentiable everywhere
    """
    # 数值安全
    r = max(ratio, eps)
    x = math.log(r)  # log-ratio

    # 平滑插值系数 in (0,1)
    s = 0.5 * (1.0 + math.tanh(x / tau))

    # 非对称 beta
    beta_x = beta_neg + (beta_pos - beta_neg) * s

    # 指数映射回比例
    return math.exp(beta_x * x)

from typing import Iterable, Tuple, List

def _quantile(sorted_x: List[float], q: float) -> float:
    """Linear-interpolated quantile for q in [0,1]. Assumes sorted_x is sorted."""
    n = len(sorted_x)
    if n == 0:
        raise ValueError("Empty data.")
    if q <= 0:
        return sorted_x[0]
    if q >= 1:
        return sorted_x[-1]
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_x[lo]
    w = pos - lo
    return sorted_x[lo] * (1 - w) + sorted_x[hi] * w


def infer_betas_from_ratios(
    var_of_layers: Iterable[float],
    alpha_min: float,
    alpha_max: float,
    *,
    use_quantiles: bool = True,
    q_low: float = 0.01,
    q_high: float = 0.99,
    eps: float = 1e-12,
) -> Tuple[float, float, float, float]:
    """
    Infer beta_pos and beta_neg from ratio statistics so that:
      r_high^beta_pos ≈ alpha_max  (for r_high > 1)
      r_low^beta_neg  ≈ alpha_min  (for r_low < 1)

    Returns:
      (beta_pos, beta_neg, r_low_used, r_high_used)

    Notes:
      - If use_quantiles=True, uses (q_low, q_high) quantiles for robustness
        instead of raw min/max.
      - If data do not contain ratios <1 or >1, it falls back to beta=0 on that side.
    """
    mean_v = sum(var_of_layers) / len(var_of_layers)
    ratios = [v / mean_v for v in var_of_layers]
    if alpha_min <= 0 or alpha_max <= 0:
        raise ValueError("alpha_min and alpha_max must be > 0.")
    if alpha_min >= 1.0:
        raise ValueError("For meaningful compression of r<1, alpha_min should be < 1.")
    if alpha_max <= 1.0:
        raise ValueError("For meaningful expansion of r>1, alpha_max should be > 1.")

    xs = [max(float(r), eps) for r in ratios]
    if not xs:
        raise ValueError("ratios is empty.")

    xs.sort()
    r_low = _quantile(xs, q_low) if use_quantiles else xs[0]
    r_high = _quantile(xs, q_high) if use_quantiles else xs[-1]

    # Ensure we have usable sides; otherwise set the corresponding beta to 0.
    # Positive side: need r_high > 1
    if r_high <= 1.0 + 1e-15:
        beta_pos = 0.0
        r_high_used = 1.0
    else:
        beta_pos = math.log(alpha_max) / math.log(r_high)
        r_high_used = r_high

    # Negative side: need r_low < 1
    if r_low >= 1.0 - 1e-15:
        beta_neg = 0.0
        r_low_used = 1.0
    else:
        # log(r_low) < 0 and log(alpha_min) < 0 -> beta_neg > 0
        beta_neg = math.log(alpha_min) / math.log(r_low)
        r_low_used = r_low

    # Guard against pathological huge betas (can happen if r_low ~ 1 or r_high ~ 1)
    # You can adjust these caps if you like.
    beta_pos = max(0.0, min(beta_pos, 10.0))
    beta_neg = max(0.0, min(beta_neg, 10.0))

    return beta_pos, beta_neg, r_low_used, r_high_used


def iter_lora_factors_with_names(model: nn.Module):
    for module_name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for name in module.lora_A.keys():
                yield module_name, name, module.lora_B[name].weight, module.lora_A[name].weight


class DistributedSvdRefactorTrainer(SFTTrainer):
    """
    使用分布式低秩 SVD 重构 LoRA 因子，并在重构后清空 Adam moments。
    """

    def __init__(
        self,
        *args,
        adjust_lora_alpha_at: List[int] = [],
        basic_alpha: float = 2.0,
        min_alpha_ratio: float = 0.8,
        max_alpha_ratio: float = 1.6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.adjust_lora_alpha_at = adjust_lora_alpha_at
        self.min_alpha_ratio = float(min_alpha_ratio)
        self.max_alpha_ratio = float(max_alpha_ratio)
        self._last_lr_values = None
        self._prev_lr_values = None
        self._lr_restart_last_checked_step = -1
        self._refactor_times = 0
        self.alpha_log = {}
        self.basic_alpha = float(basic_alpha)

    def _zero_adam_moments_for_param(self, param: Tensor) -> None:
        if not hasattr(self, "optimizer") or self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p is param:
                    state = self.optimizer.state.get(p, None)
                    if not state:
                        return
                    exp_avg = state.get("exp_avg", None)
                    if exp_avg is not None:
                        exp_avg.zero_()
                    exp_avg_sq = state.get("exp_avg_sq", None)
                    if exp_avg_sq is not None:
                        exp_avg_sq.zero_()
                    max_exp_avg_sq = state.get("max_exp_avg_sq", None)
                    if max_exp_avg_sq is not None:
                        max_exp_avg_sq.zero_()
                    return

    @torch.no_grad()
    def distributed_low_rank_refactor(
        self,
        adjust_lora_alpha: bool = True,
        min_alpha_ratio: float = 0.8,
        max_alpha_ratio: float = 1.6,
    ):
        is_dist = self.accelerator.num_processes > 1 and dist.is_initialized()
        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index

        model = self.model
        was_training = model.training
        model.eval()

        variance_of_layers = {}
        device_for_broadcast = None
        broadcast_works = []

        for idx, (module_name, name, B, A) in enumerate(iter_lora_factors_with_names(model)):
            lora_r = A.shape[0]
            compute_here = (not is_dist) or (rank == idx % world_size)
            if device_for_broadcast is None:
                device_for_broadcast = B.device

            if compute_here:
                # 始终对 B @ A 做 SVD，作为正交基
                base = B @ A
                U, S, Vh = torch.svd_lowrank(base.float(), q=lora_r)
                variance_of_layers[f"{module_name}.{name}"] = float((S ** 2).sum().item())
                B_new = U.to(B.dtype)
                A_new = Vh.t().to(A.dtype)
                B.copy_(B_new)
                A.copy_(A_new)

            if is_dist:
                broadcast_works.append(
                    dist.broadcast(B, src=idx % world_size, async_op=True).get_future()
                )
                broadcast_works.append(
                    dist.broadcast(A, src=idx % world_size, async_op=True).get_future()
                )

        if is_dist and broadcast_works:
            torch.futures.wait_all(broadcast_works)

        if is_dist:
            gathered_variances = [None] * world_size
            dist.all_gather_object(gathered_variances, variance_of_layers)
            variance_of_layers = {}
            for part in gathered_variances:
                variance_of_layers.update(part)
        if not variance_of_layers:
            if was_training:
                model.train()
            return

        beta_pos, beta_neg, r_low_used, r_high_used = infer_betas_from_ratios(
            variance_of_layers.values(),
            min_alpha_ratio,
            max_alpha_ratio,
        )

        if adjust_lora_alpha and variance_of_layers:
            if rank == 0:
                avg_of_global_variance = sum(variance_of_layers.values()) / len(variance_of_layers)
                #clip_ratio = self.alpha_clip_ratio
                #beta = self.alpha_beta #* (self.state.max_steps - self.state.global_step) / self.state.max_steps
                for module_name, sub_module in model.named_modules():
                    if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B") and hasattr(sub_module, "lora_alpha"):
                        for adapter_name in sub_module.lora_A.keys():
                            if adapter_name not in sub_module.lora_alpha:
                                continue
                            layer_key = f"{module_name}.{adapter_name}"
                            if layer_key not in variance_of_layers:
                                continue
                            layer_var = variance_of_layers[layer_key]
                            ratio = layer_var / avg_of_global_variance
                            ratio_new = smooth_asymmetric_power_ratio_math(ratio, beta_pos=beta_pos, beta_neg=beta_neg)
                            sub_module.lora_alpha[adapter_name] = ratio_new * self.basic_alpha

            alpha_values = []
            alpha_indices = []
            for module_name, sub_module in model.named_modules():
                if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B") and hasattr(sub_module, "lora_alpha"):
                    for adapter_name in sub_module.lora_A.keys():
                        if adapter_name not in sub_module.lora_alpha:
                            continue
                        alpha_indices.append((module_name, adapter_name))
                        if rank == 0:
                            alpha_values.append(float(sub_module.lora_alpha[adapter_name]))
                        else:
                            alpha_values.append(0.0)

            if alpha_values:
                if device_for_broadcast is None:
                    device_for_broadcast = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                alpha_tensor = torch.tensor(alpha_values, device=device_for_broadcast, dtype=torch.float32)
                if is_dist:
                    dist.broadcast(alpha_tensor, src=0)

                modules_dict = dict(model.named_modules())
                for idx, (module_name, adapter_name) in enumerate(alpha_indices):
                    sub_module = modules_dict[module_name]
                    if f"{module_name}.{adapter_name}" not in self.alpha_log:
                        self.alpha_log[f"{module_name}.{adapter_name}"] = [sub_module.lora_alpha[adapter_name]]
                    sub_module.lora_alpha[adapter_name] = float(alpha_tensor[idx].item())    
                    sub_module.set_scale(adapter_name, 1.0)  # 更新 scale
                    self.alpha_log[f"{module_name}.{adapter_name}"].append(sub_module.lora_alpha[adapter_name])

        for module_name, name, B, A in iter_lora_factors_with_names(model):
            self._zero_adam_moments_for_param(B)
            self._zero_adam_moments_for_param(A)

        if was_training:
            model.train()

    def _is_lr_restart(self):
        if not hasattr(self, "lr_scheduler") or self.lr_scheduler is None:
            return False

        step = self.state.global_step
        scheduler_step = getattr(self.lr_scheduler, "last_epoch", step)
        # Avoid double-processing the same optimizer step when using gradient accumulation
        if self._lr_restart_last_checked_step == step:
            return False
        self._lr_restart_last_checked_step = step

        if not hasattr(self.lr_scheduler, "get_last_lr"):
            return False

        current_lrs = list(self.lr_scheduler.get_last_lr())
        if self._last_lr_values is None:
            self._last_lr_values = current_lrs
            self._prev_lr_values = None
            return False

        # Prefer using LambdaLR's lambda to detect a true "restart" boundary for
        # get_warmup_restart_then_final_decay_scheduler_ratio():
        # a restart is when the schedule stops decreasing and starts increasing again.
        eps = 1e-12
        try:
            from torch.optim.lr_scheduler import LambdaLR

            if isinstance(self.lr_scheduler, LambdaLR) and getattr(self.lr_scheduler, "lr_lambdas", None):
                if scheduler_step >= 2:
                    is_restart = False
                    for lr_lambda in self.lr_scheduler.lr_lambdas:
                        r2 = float(lr_lambda(int(scheduler_step - 2)))
                        r1 = float(lr_lambda(int(scheduler_step - 1)))
                        r0 = float(lr_lambda(int(scheduler_step)))
                        delta_prev = r1 - r2
                        delta_cur = r0 - r1
                        if (delta_cur > eps) and (delta_prev <= eps):
                            is_restart = True
                            break
                    self._prev_lr_values = self._last_lr_values
                    self._last_lr_values = current_lrs
                    return is_restart
        except Exception:
            pass

        # Fallback: detect the point where LR stops decreasing and starts increasing.
        if self._prev_lr_values is None:
            self._prev_lr_values = self._last_lr_values
            self._last_lr_values = current_lrs
            return False

        is_restart = any(
            ((cur_lr - prev_lr) > eps) and ((prev_lr - prev2_lr) <= eps)
            for cur_lr, prev_lr, prev2_lr in zip(current_lrs, self._last_lr_values, self._prev_lr_values)
        )
        self._prev_lr_values = self._last_lr_values
        self._last_lr_values = current_lrs
        return is_restart

    def training_step(self, model: nn.Module, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        step = self.state.global_step

        if self.optimizer is None:
            return loss

        if self._is_lr_restart():
            if self.accelerator.process_index == 0:
                print(f"Step {step}: Detected LR restart {self._refactor_times},  performing distributed low-rank refactor...")
            if self._refactor_times in self.adjust_lora_alpha_at:
                self.distributed_low_rank_refactor(
                    adjust_lora_alpha = True,
                    min_alpha_ratio = self.min_alpha_ratio,
                    max_alpha_ratio = self.max_alpha_ratio,
                    )
            else:
                self.distributed_low_rank_refactor(
                    adjust_lora_alpha = False,
                    min_alpha_ratio = self.min_alpha_ratio,
                    max_alpha_ratio = self.max_alpha_ratio,
                    )
            self._refactor_times += 1 
        return loss

    def save_alpha_log(self, filepath: str):
        if not self.alpha_log or self.accelerator.process_index != 0:
            return
        import json
        with open(filepath, "w") as f:
            json.dump(self.alpha_log, f, indent=4)

def get_warmup_restart_then_final_decay_scheduler_ratio(
    optimizer,
    num_training_steps,
    repeat_n,
    repeat_warmup_ratio,
    repeat_decay_ratio,
    repeat_end_lr_rate,
    final_warmup_ratio,
    min_lr_rate,
    repeat_decay_type="cosine",
    final_decay_type="cosine",
    warmup_start_lr_rate=0.0,
    first_warmup_start_lr_rate=0.0,
    last_epoch=-1,
):

    T = num_training_steps

    repeat_warmup_steps = int(round(repeat_warmup_ratio * T))
    repeat_decay_steps  = int(round(repeat_decay_ratio  * T))
    final_warmup_steps  = int(round(final_warmup_ratio  * T))

    cycle_len = repeat_warmup_steps + repeat_decay_steps
    repeat_total_steps = repeat_n * cycle_len

    def _decay_factor(t, kind):
        t = min(max(t, 0.0), 1.0)
        if kind == "linear":
            return 1.0 - t
        return 0.5 * (1.0 + math.cos(math.pi * t))

    # Baseline: schedule without restarts; align peaks with standard cosine/linear.
    def _baseline_lr_ratio(step):
        step = max(0, min(step, T))
        baseline_warmup_steps = repeat_warmup_steps
        if baseline_warmup_steps > 0 and step < baseline_warmup_steps:
            t = step / baseline_warmup_steps
            return first_warmup_start_lr_rate + (1.0 - first_warmup_start_lr_rate) * t

        decay_total = T - baseline_warmup_steps
        if decay_total <= 0:
            return min_lr_rate
        dpos = step - baseline_warmup_steps
        t = dpos / decay_total
        f = _decay_factor(t, final_decay_type)
        return min_lr_rate + (1.0 - min_lr_rate) * f

    def lr_lambda(step):
        step = max(0, min(step, T))

        # repeated phase
        if step < repeat_total_steps:
            pos = step % cycle_len
            time = step // cycle_len
            peak_step = time * cycle_len + repeat_warmup_steps
            peak_lr_ratio = _baseline_lr_ratio(peak_step)
            peak_lr_ratio = 1.0 # (1+peak_lr_ratio)/2.0 
            cycle_end_lr_ratio = min(peak_lr_ratio, repeat_end_lr_rate * peak_lr_ratio)

            if time <=0 and pos < repeat_warmup_steps : # first warmup
                if repeat_warmup_steps == 0:
                    return peak_lr_ratio
                t = pos / repeat_warmup_steps
                return first_warmup_start_lr_rate + (peak_lr_ratio - first_warmup_start_lr_rate) * t

            if pos < repeat_warmup_steps:
                if repeat_warmup_steps == 0:
                    return peak_lr_ratio
                t = pos / repeat_warmup_steps
                return warmup_start_lr_rate + (peak_lr_ratio - warmup_start_lr_rate) * t

            dpos = pos - repeat_warmup_steps
            if repeat_decay_steps == 0:
                return cycle_end_lr_ratio
            t = dpos / repeat_decay_steps
            f = _decay_factor(t, repeat_decay_type)
            return cycle_end_lr_ratio + (peak_lr_ratio - cycle_end_lr_ratio) * f

        # final phase
        final_pos = step - repeat_total_steps
        final_total = T - repeat_total_steps
        peak_step = repeat_total_steps + final_warmup_steps
        peak_lr_ratio = _baseline_lr_ratio(peak_step)
        peak_lr_ratio = 1.0 # (1+peak_lr_ratio)/2.0

        if final_pos < final_warmup_steps:
            if final_warmup_steps == 0:
                return peak_lr_ratio
            t = final_pos / final_warmup_steps
            return warmup_start_lr_rate + (peak_lr_ratio - warmup_start_lr_rate) * t

        decay_left = final_total - final_warmup_steps
        if decay_left <= 0:
            return min_lr_rate

        dpos = final_pos - final_warmup_steps
        t = dpos / decay_left
        f = _decay_factor(t, final_decay_type)
        return min_lr_rate + (peak_lr_ratio - min_lr_rate) * f

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def get_cleaned_svd_ref_trainer(
    *,
    model: nn.Module,
    args: SFTConfig,
    train_dataset,
    global_batch_size: int = 32,
    eval_dataset=None,
    adjust_lora_alpha_at: List[int] = [2],
    basic_alpha: float = 2.0,
    min_alpha_ratio: float = 0.8,
    max_alpha_ratio: float = 1.25,
    repeat_n: int = 3,
    repeat_warmup_ratio: float = 0.03,
    repeat_decay_ratio: float = 0.03,
    repeat_end_lr_rate: float = 0.97,
    final_warmup_ratio: float = 0.03,
    min_lr_rate: float = 0.001,
    repeat_decay_type: str = "cosine",
    final_decay_type: str = "linear",
    warmup_start_lr_rate: float = 0.001,
    first_warmup_start_lr_rate: float = 0.001,
    last_epoch: int = -1,
    **trainer_kwargs,
) -> DistributedSvdRefactorTrainer:

    optimizer = AdamW(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    if args.max_steps and args.max_steps > 0:
        num_training_steps = int(args.max_steps)
    else:
        num_training_steps = math.ceil(len(train_dataset) / global_batch_size) * args.num_train_epochs

    lr_scheduler = get_warmup_restart_then_final_decay_scheduler_ratio(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        repeat_n=repeat_n,
        repeat_warmup_ratio=repeat_warmup_ratio,
        repeat_decay_ratio=repeat_decay_ratio,
        repeat_end_lr_rate=repeat_end_lr_rate,
        final_warmup_ratio=final_warmup_ratio,
        min_lr_rate=min_lr_rate,
        repeat_decay_type=repeat_decay_type,
        final_decay_type=final_decay_type,
        warmup_start_lr_rate=warmup_start_lr_rate,
        first_warmup_start_lr_rate=first_warmup_start_lr_rate,
        last_epoch=last_epoch,
    )
    
    trainer = DistributedSvdRefactorTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers =(optimizer, lr_scheduler),
        adjust_lora_alpha_at=adjust_lora_alpha_at,
        basic_alpha=basic_alpha,
        min_alpha_ratio=min_alpha_ratio,
        max_alpha_ratio=max_alpha_ratio,
        **trainer_kwargs,
    )
    print("Using DistributedSvdRefactorTrainer with cleaned SVD refactor.")
    return trainer
