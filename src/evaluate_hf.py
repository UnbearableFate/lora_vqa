from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

from peft import PeftModel

from .common import (
    append_row_to_csv,
    json_safe,
    now_timestamp,
    parse_run_name_fields,
    read_adapter_config,
    resolve_eval_csv_path,
)


def _compute_metrics(predictions: Iterable[str], references: Iterable[List[str]]) -> Dict[str, float]:
    em_scores = []
    vqa_scores = []
    f1_scores = []
    for pred, answers in zip(predictions, references):
        em_scores.append(exact_match(pred, answers))
        vqa_scores.append(vqa_soft_accuracy(pred, answers))
        f1_scores.append(token_f1(pred, answers))
    return {
        "exact_match": sum(em_scores) / max(1, len(em_scores)),
        "vqa_soft": sum(vqa_scores) / max(1, len(vqa_scores)),
        "token_f1": sum(f1_scores) / max(1, len(f1_scores)),
    }


def _resolve_precision_flags(bf16: bool, fp16: bool) -> tuple[bool, bool]:
    if bf16 and fp16:
        fp16 = False
    if not torch.cuda.is_available():
        return False, False
    if bf16 and hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
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


def _infer_dataset_from_path(adapter_path: Optional[str]) -> Optional[str]:
    if not adapter_path:
        return None
    parts = os.path.normpath(adapter_path).split(os.sep)
    for marker in ("output", "outputs"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    if len(parts) >= 4:
        return parts[-4]
    return None


def _infer_model_from_path(adapter_path: Optional[str]) -> Optional[str]:
    if not adapter_path:
        return None
    parts = os.path.normpath(adapter_path).split(os.sep)
    for marker in ("output", "outputs"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 2 < len(parts):
                return parts[idx + 2]
    if len(parts) >= 3:
        return parts[-3]
    return None


def _load_processor(model_name: str, *, trust_remote_code: bool, use_fast: Optional[bool]) -> object:
    if use_fast is None:
        try:
            return AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )
        except (TypeError, ValueError):
            return AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                use_fast=False,
            )
    try:
        return AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )
    except TypeError:
        return AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )


def _load_model(
    model_name: str,
    *,
    trust_remote_code: bool,
    dtype: Optional[torch.dtype],
    attn_implementation: Optional[str],
):
    model_kwargs = {"trust_remote_code": trust_remote_code}
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    try:
        return AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
    except (TypeError, ValueError):
        if "attn_implementation" in model_kwargs:
            model_kwargs.pop("attn_implementation")
            return AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        raise


def _resolve_model_type(model) -> str:
    if getattr(getattr(model, "config", None), "is_encoder_decoder", False):
        return "seq2seq"
    return "causal_lm"


def _move_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _decode_outputs(
    outputs: torch.Tensor,
    *,
    tokenizer,
    model_type: str,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> List[str]:
    if model_type == "seq2seq":
        return [text.strip() for text in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
    if attention_mask is not None:
        input_lengths = attention_mask.sum(dim=1)
    else:
        pad_token_id = tokenizer.pad_token_id or 0
        input_lengths = (input_ids != pad_token_id).sum(dim=1)
    decoded: List[str] = []
    for idx, length in enumerate(input_lengths.tolist()):
        gen_ids = outputs[idx, length:]
        decoded.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return decoded


def evaluate_model_hf(
    *,
    dataset: Optional[str] = None,
    model_name: Optional[str] = None,
    adapter_path: Optional[str] = None,
    subset_name: Optional[str] = None,
    eval_split: str = "validation",
    image_column: Optional[str] = None,
    question_column: Optional[str] = None,
    answer_column: Optional[str] = None,
    trust_remote_code: bool = True,
    image_placeholder: Optional[str] = None,
    max_length: int = 512,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eval_batch_size: int = 2,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    csv_path_dir: Optional[str] = "experiments",
    device: Optional[str] = None,
    bf16: bool = True,
    fp16: bool = False,
    use_fast: Optional[bool] = None,
    attn_implementation: Optional[str] = None,
    num_workers: int = 0,
    examples_num_batches: int = 3,
    examples_json_path: Optional[str] = None,
):
    set_seed(seed)

    adapter_cfg = read_adapter_config(adapter_path)
    if not model_name:
        model_name = str(adapter_cfg.get("base_model_name_or_path") or _infer_model_from_path(adapter_path) or "")
        model_name = model_name.strip() or None
    if not dataset:
        dataset = _infer_dataset_from_path(adapter_path)
    if not model_name:
        raise ValueError("model_name is required (or provide adapter_path with base_model_name_or_path).")
    if not dataset:
        raise ValueError("dataset is required (or provide adapter_path under output/<dataset>/...).")

    dataset_id = resolve_dataset_name(dataset)
    dataset_bundle = load_vqa_dataset(
        dataset_name=dataset,
        cache_dir=cache_dir,
        streaming=False,
        subset_name=subset_name,
        train_split="train",
        eval_split=eval_split,
        image_column=image_column,
        question_column=question_column,
        answer_column=answer_column,
    )
    eval_dataset = dataset_bundle["eval"]

    bf16, fp16 = _resolve_precision_flags(bf16, fp16)
    dtype = _resolve_dtype(bf16, fp16)

    processor_source = adapter_path if adapter_path and os.path.isdir(adapter_path) else model_name
    processor = _load_processor(processor_source, trust_remote_code=trust_remote_code, use_fast=use_fast)
    tokenizer = getattr(processor, "tokenizer", processor)

    model = _load_model(
        model_name,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()

    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device_name)
    model.to(device_obj)

    model_type = _resolve_model_type(model)
    if model_type == "causal_lm" and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    
    prompt_style = "chat" # plain or chat
    if model_name.lower().find("paligemma") >= 0:
        prompt_style = "plain"
    collator = VqaEvalCollator(
        processor,
        model_type=model_type,
        prompt_style=prompt_style,
        image_placeholder=image_placeholder,
        max_length=max_length,
        return_answers=True,
    )
    def _collate_with_questions(features):
        batch = collator(features)
        batch["questions"] = [str(feature.get("question", "")) for feature in features]
        return batch

    dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=_collate_with_questions,
        num_workers=num_workers,
    )

    do_sample = temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    predictions: List[str] = []
    references: List[List[str]] = []
    examples: List[Dict[str, object]] = []
    max_examples_batches = max(0, int(examples_num_batches))
    seen_batches = 0
    autocast_ctx = (
        torch.autocast(device_type=device_obj.type, dtype=dtype)
        if device_obj.type == "cuda" and dtype is not None
        else torch.autocast(device_type=device_obj.type, dtype=torch.float32, enabled=False)
    )

    with torch.inference_mode(), autocast_ctx:
        for batch_idx, batch in enumerate(dataloader):
            answers = batch.pop("answers", [])
            questions = batch.pop("questions", [])
            references.extend(answers)
            batch = _move_to_device(batch, device_obj)
            outputs = model.generate(**batch, **gen_kwargs)
            decoded = _decode_outputs(
                outputs,
                tokenizer=tokenizer,
                model_type=model_type,
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            predictions.extend(decoded)
            if max_examples_batches and batch_idx < max_examples_batches:
                seen_batches = batch_idx + 1
                for idx, pred in enumerate(decoded):
                    answer_list = answers[idx] if idx < len(answers) else []
                    question = questions[idx] if idx < len(questions) else ""
                    is_correct = bool(exact_match(pred, answer_list))
                    examples.append(
                        {
                            "batch_index": batch_idx,
                            "sample_index": idx,
                            "problem": question,
                            "answers": answer_list,
                            "prediction": pred,
                            "is_correct": is_correct,
                        }
                    )

    metrics = _compute_metrics(predictions, references)

    base_model = (
        str(adapter_cfg.get("base_model_name_or_path") or model_name).split("/")[-1]
        if isinstance(adapter_cfg, dict)
        else model_name.split("/")[-1]
    )
    run_name = os.path.basename(os.path.normpath(adapter_path)) if adapter_path else ""
    run_fields = parse_run_name_fields(run_name) if run_name else {}
    train_timestamp = str(run_fields.get("timestamp") or "")

    resolved_csv_path = resolve_eval_csv_path(
        dataset_id=dataset_id,
        model_name=model_name,
        adapter_path=adapter_path,
        csv_path_dir=csv_path_dir,
    )
    resolved_examples_path = None
    if max_examples_batches:
        resolved_examples_path = examples_json_path
        if not resolved_examples_path:
            base_path, _ = os.path.splitext(resolved_csv_path)
            resolved_examples_path = f"{base_path}_examples.json"
        examples_dir = os.path.dirname(resolved_examples_path)
        if examples_dir:
            os.makedirs(examples_dir, exist_ok=True)
        payload = {
            "dataset_name": dataset_id,
            "model_name": model_name,
            "adapter_path": adapter_path or "",
            "eval_split": eval_split,
            "num_batches": seen_batches,
            "num_samples": len(examples),
            "examples": examples,
        }
        with open(resolved_examples_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    row: Dict[str, object] = {
        "timestamp": train_timestamp or now_timestamp(),
        "base_model": base_model,
        "dataset_name": dataset_id,
        "adapter_path": adapter_path or "",
        "run_name": run_name,
        "eval_split": eval_split,
        "backend": "hf",
        "seed": run_fields.get("seed", seed),
        "eval_seed": seed,
        "num_samples": len(predictions),
    }

    init_lora_weights = adapter_cfg.get("init_lora_weights") if isinstance(adapter_cfg, dict) else None
    if isinstance(init_lora_weights, str) and init_lora_weights.strip().lower() == "true":
        init_lora_weights = "kaiming"
    row["init_lora_weights"] = init_lora_weights if init_lora_weights is not None else ""

    for k, v in metrics.items():
        row[f"metric_{k}"] = v

    if isinstance(adapter_cfg, dict) and adapter_cfg:
        for key in ("r", "lora_alpha", "lora_dropout", "target_modules", "bias", "use_dora", "use_rslora"):
            if key in adapter_cfg:
                v = adapter_cfg[key]
                row[key] = json_safe(v) if isinstance(v, (dict, list)) else v

    append_row_to_csv(resolved_csv_path, row)
    print(f"Eval metrics (HF): {metrics}")
    print(f"Saved CSV -> {resolved_csv_path}")
    if resolved_examples_path:
        print(f"Saved examples JSON -> {resolved_examples_path}")
    return metrics
