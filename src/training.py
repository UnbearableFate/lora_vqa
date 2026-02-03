from __future__ import annotations

import datetime
from typing import Optional, Sequence, Union

from accelerate import Accelerator
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

from .lora_loader import LoraHyperparameters, attach_vlm_lora_adapter, get_lora_config
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
from trl import SFTConfig, SFTTrainer
from .data_process import get_val_split_name, load_and_preprocess_dataset
from .common import (
    build_structured_output_dir,
    build_wandb_project_run_tags,
    maybe_enable_wandb,
    normalize_init_lora_weights,
    resolve_output_dir,
)

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def _parse_list(value: Union[str, Sequence[str], None]) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        return [p for p in parts if p]
    return [str(v).strip() for v in value if str(v).strip()]

def _coerce_int(value: Optional[Union[str, int]]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        return int(value)
    return int(value)

def _resolve_precision_flags(bf16: bool, fp16: bool) -> tuple[bool, bool]:
    if bf16 and fp16:
        logger.warning("Both bf16 and fp16 requested; using bf16.")
        fp16 = False
    if not torch.cuda.is_available():
        if bf16 or fp16:
            logger.warning("CUDA not available; disabling bf16/fp16.")
        return False, False
    if bf16:
        bf16_supported = False
        if hasattr(torch.cuda, "is_bf16_supported"):
            bf16_supported = torch.cuda.is_bf16_supported()
        if not bf16_supported:
            logger.warning("bf16 requested but not supported; falling back to fp16.")
            bf16 = False
            if not fp16:
                fp16 = True
    return bf16, fp16


def _resolve_dtype(bf16: bool, fp16: bool) -> Optional[torch.dtype]:
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return None

def train(
    *,
    dataset_name: str,
    model_name: str,
    output_dir: str = "outputs",
    subset_name: Optional[str] = None,
    trust_remote_code: bool = True,
    peft_variant: str = "lora",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_bias: str = "none",
    target_modules: Union[str, Sequence[str]] = ("q_proj", "v_proj" ,"o_proj", "k_proj"),
    modules_to_save: Optional[Union[str, Sequence[str]]] = None,
    init_lora_weights: Union[bool, str, None] = True,
    init_num_samples: int = 512,
    init_batch_size: int = 8,
    init_seed: Optional[int] = None,
    corda_method: str = "kpm",
    learning_rate: float = 5e-4,
    lr_scheduler_type: str = "linear",
    weight_decay: float = 0.0,
    warmup_ratio: float = 0.03,
    num_train_epochs: float = 3.0,
    max_steps: Optional[int] = None,
    global_batch_size: int = 32,
    per_device_batch_size: int = 2,
    eval_batch_size: Optional[int] = None,
    logging_steps: int = 50,
    eval_steps: int = 500,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_online: bool = False,
    fp16: bool = False,
    bf16: bool = True,
    attn_implementation: Optional[str] = "flash_attention_2",
    use_fast: bool = True,
    compile_model: bool = False,
    gradient_checkpointing: bool = False,
    cache_dir: Optional[str] = None,
    use_cleaned_svd_ref_trainer: bool = False,
    adjust_lora_alpha_at: Union[str, Sequence[int]] = (2,),
    min_alpha_ratio: float = 0.8,
    max_alpha_ratio: float = 1.25,
    repeat_n: int = 3,
    repeat_warmup_ratio: float = 0.03,
    repeat_decay_ratio: float = 0.03,
    repeat_end_lr_rate: float = 0.97,
    final_warmup_ratio: float = 0.03,
    min_lr_rate: float = 0.001,
    warmup_start_lr_rate: float = 0.1,
    first_warmup_start_lr_rate: float = 0.001,
    last_epoch: int = -1,
    timestamp: Optional[str] = None,
    skip_eval: bool = False,
):
    accelerator = Accelerator()
    set_seed(seed)

    timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    max_steps = _coerce_int(max_steps)
    eval_steps = _coerce_int(eval_steps) or eval_steps
    logging_steps = _coerce_int(logging_steps) or logging_steps

    target_modules_list = _parse_list(target_modules) or []
    if not target_modules_list:
        raise ValueError("target_modules must contain at least one module name fragment.")
    modules_to_save_list = _parse_list(modules_to_save)

    parsed_adjust_lora_alpha_at = None
    if adjust_lora_alpha_at is not None:
        parsed_adjust_lora_alpha_at = [int(v) for v in _parse_list(adjust_lora_alpha_at) or []]

    effective_init_seed = init_seed if init_seed is not None else seed * 2 + 1
    parsed_init_lora_weights = normalize_init_lora_weights(init_lora_weights)

    bf16, fp16 = _resolve_precision_flags(bf16, fp16)
    dtype = _resolve_dtype(bf16, fp16)

    derived_project, derived_run_name, tags = build_wandb_project_run_tags(
        model_name=model_name,
        dataset_id=dataset_name,
        peft_variant=peft_variant,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        init_lora_weights=parsed_init_lora_weights,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        global_batch_size=global_batch_size,
        per_device_batch_size=per_device_batch_size,
        eval_steps=int(eval_steps) if eval_steps is not None else 0,
        logging_steps=int(logging_steps) if logging_steps is not None else 0,
        seed=seed,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        use_cleaned_svd_ref_trainer=use_cleaned_svd_ref_trainer,
        repeat_n=repeat_n,
        adjust_lora_alpha_at=parsed_adjust_lora_alpha_at,
        timestamp=timestamp,
    )

    resolved_output_dir = build_structured_output_dir(
        output_root= output_dir,
        dataset_id=dataset_name,
        model_name=model_name,
        lora_r=lora_r,
        run_name=derived_run_name,
    ) 

    if accelerator.is_main_process:
        maybe_enable_wandb(
            use_wandb,
            project=derived_project,
            run_name=derived_run_name,
            online=wandb_online,
            tags=tags,
            config={
                "dataset": dataset_name,
                "model_name": model_name,
                "subset_name": subset_name,
                "peft_variant": peft_variant,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_bias": lora_bias,
                "target_modules": target_modules_list,
                "modules_to_save": modules_to_save_list,
                "init_lora_weights": parsed_init_lora_weights,
                "init_num_samples": init_num_samples,
                "init_batch_size": init_batch_size,
                "init_seed": effective_init_seed,
                "corda_method": corda_method,
                "learning_rate": learning_rate,
                "lr_scheduler_type": lr_scheduler_type,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "num_train_epochs": num_train_epochs,
                "max_steps": max_steps,
                "global_batch_size": global_batch_size,
                "per_device_batch_size": per_device_batch_size,
                "eval_steps": eval_steps,
                "logging_steps": logging_steps,
                "seed": seed,
                "fp16": fp16,
                "bf16": bf16,
                "gradient_checkpointing": gradient_checkpointing,
                "use_cleaned_svd_ref_trainer": use_cleaned_svd_ref_trainer,
                "repeat_n": repeat_n,
                "adjust_lora_alpha_at": parsed_adjust_lora_alpha_at,
                "timestamp": timestamp,
                "output_dir": resolved_output_dir,
            },
        )

    ### Load model and dataset

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    if accelerator.is_main_process:
        print(f"Model loaded {model}")

    train_dataset, val_dataset, _ = load_and_preprocess_dataset(dataset_name, subset_name, splits=["train", "val" if not skip_eval else "none"], num_proc=8)
    print(train_dataset.column_names)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    lora_hparams = LoraHyperparameters(
        variant=peft_variant,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        bias=lora_bias,
        target_modules=target_modules_list,
        init_lora_weights=parsed_init_lora_weights,
        init_num_samples=init_num_samples,
        init_batch_size=init_batch_size,
        corda_method=corda_method,
        cache_dir=cache_dir or "data_cache",
        model_name_or_path=model_name,
        dataset_name=dataset_name,
        subdataset_name=subset_name,
        init_seed=effective_init_seed,
    )
    peft_config = get_lora_config(lora_hparams)
    if modules_to_save_list:
        if hasattr(peft_config, "modules_to_save"):
            peft_config.modules_to_save = modules_to_save_list
        else:
            logger.warning("modules_to_save is not supported by this LoRA config.")

    collator = DataCollatorForVisionLanguageModeling(
        processor=processor,
    )

    model = attach_vlm_lora_adapter(
        base_model= model,
        lora_cfg= peft_config,
        train_dataset=train_dataset,
        data_collator=collator,
        init_num_samples=init_num_samples,
        batch_size=init_batch_size,
        seed=effective_init_seed,
        accelerator=accelerator,
    )

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    world_size = accelerator.num_processes
    gradient_accumulation_steps = max(1, global_batch_size // max(1, per_device_batch_size * world_size))
    if global_batch_size <= world_size * per_device_batch_size:
        gradient_accumulation_steps = 1
        assert global_batch_size % world_size == 0, (
            f"global_batch_size {global_batch_size} must be divisible by world size {world_size} "
            f"when it is less than or equal to per-device batch size {per_device_batch_size}."
        )
        real_per_device_batch_size = global_batch_size // world_size
    else:
        assert global_batch_size % (world_size * per_device_batch_size) == 0, (
            f"global_batch_size {global_batch_size} must be divisible by world size {world_size} * "
            f"per-device batch size {per_device_batch_size} = {world_size * per_device_batch_size}."
        )
        real_per_device_batch_size = per_device_batch_size
        gradient_accumulation_steps = global_batch_size // (world_size * per_device_batch_size)

    training_args = SFTConfig(
        output_dir=resolved_output_dir,
        per_device_train_batch_size=real_per_device_batch_size,
        per_device_eval_batch_size=eval_batch_size or real_per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps or -1,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        eval_strategy="steps" if not skip_eval else "no",
        save_strategy="steps",
        save_total_limit=2,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
        max_length=None,
        packing=False,
        assistant_only_loss=False,
        data_seed=seed,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
    )

    if use_cleaned_svd_ref_trainer:
        trainer = get_cleaned_svd_ref_trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if not skip_eval else None,
            processing_class=processor,
            data_collator=collator,
            global_batch_size=global_batch_size,
            adjust_lora_alpha_at=parsed_adjust_lora_alpha_at or [2],
            min_alpha_ratio=min_alpha_ratio,
            max_alpha_ratio=max_alpha_ratio,
            repeat_n=repeat_n,
            repeat_warmup_ratio=repeat_warmup_ratio,
            repeat_decay_ratio=repeat_decay_ratio,
            repeat_end_lr_rate=repeat_end_lr_rate,
            final_warmup_ratio=final_warmup_ratio,
            min_lr_rate=min_lr_rate,
            warmup_start_lr_rate=warmup_start_lr_rate,
            first_warmup_start_lr_rate=first_warmup_start_lr_rate,
            last_epoch=last_epoch,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if not skip_eval else None,
            processing_class=processor,
        )

    if compile_model and hasattr(torch, "compile"):
        trainer.model = torch.compile(trainer.model)
    trainer.train()
    trainer.save_model(resolved_output_dir)
    processor.save_pretrained(resolved_output_dir)
    if accelerator.is_main_process:
        print(f"TRAIN_OUTPUT_DIR\t{resolved_output_dir}", flush=True)
        print(f"Training complete at {timestamp} -> {resolved_output_dir}")
