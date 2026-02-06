from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

from .common import (
    append_row_to_csv,
    json_safe,
    now_timestamp,
    parse_run_name_fields,
    read_adapter_config,
    resolve_eval_csv_path,
)
from .data_process import get_column_names, get_val_split_name, preprocess_fn
from .metrics import exact_match, token_f1, vqa_soft_accuracy


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


def _resolve_torch_device(device: Optional[str]) -> torch.device:
    if device:
        lowered = device.strip().lower()
        if lowered == "cpu":
            return torch.device("cpu")
        if lowered.startswith("cuda"):
            index = 0
            if ":" in lowered:
                index = int(lowered.split(":", maxsplit=1)[1])
            return torch.device(f"cuda:{index}")
        raise ValueError(f"Unsupported device={device!r}; use 'cpu' or 'cuda[:index]'.")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _resolve_split(dataset_name: str, eval_split: str) -> str:
    split = (eval_split or "").strip().lower()
    if split in {"val", "validation", "dev"}:
        return get_val_split_name(dataset_name)
    if split in {"test", "train"}:
        return split
    return eval_split


def _extract_text_from_message(message: Dict[str, Any]) -> str:
    for part in message.get("content", []):
        if isinstance(part, dict) and part.get("type") == "text":
            return str(part.get("text", "")).strip()
    return ""


def _build_batch_inputs(
    processor: Any,
    batch_messages: List[List[Dict[str, Any]]],
    batch_images: List[List[Any]],
) -> Dict[str, torch.Tensor]:
    try:
        return processor(
            text=batch_messages,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )
    except Exception:
        if not hasattr(processor, "apply_chat_template"):
            raise
        rendered_prompts = [
            processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            for messages in batch_messages
        ]
        return processor(
            text=rendered_prompts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )


def _decode_predictions(
    processor: Any,
    generated_ids: torch.Tensor,
    prompt_input_ids: Optional[torch.Tensor],
) -> List[str]:
    if prompt_input_ids is not None and generated_ids.shape[1] >= prompt_input_ids.shape[1]:
        generated_ids = generated_ids[:, prompt_input_ids.shape[1] :]
    if hasattr(processor, "batch_decode"):
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    elif hasattr(processor, "tokenizer"):
        decoded = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    else:
        decoded = [str(v) for v in generated_ids]
    return [text.strip() for text in decoded]


def _load_processor_with_fallback(processor_kwargs: Dict[str, Any]) -> Any:
    try:
        return AutoProcessor.from_pretrained(**processor_kwargs)
    except ValueError as exc:
        # Some model repos expose only slow tokenizer classes; retry without fast tokenizer.
        message = str(exc)
        if "Tokenizer class" not in message or "does not exist" not in message:
            raise
        if processor_kwargs.get("use_fast") is False:
            raise
        retry_kwargs = dict(processor_kwargs)
        retry_kwargs["use_fast"] = False
        print("AutoProcessor fast tokenizer load failed; retrying with use_fast=False.")
        return AutoProcessor.from_pretrained(**retry_kwargs)


def _configure_decoder_only_padding(processor: Any, model: Any) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return
    if getattr(model.config, "is_encoder_decoder", False):
        return
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
        print("Using left padding for decoder-only generation.")
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id


def _compute_local_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    if not predictions:
        return {"exact_match": 0.0, "vqa_soft_accuracy": 0.0, "token_f1": 0.0}
    total = float(len(predictions))
    em = sum(exact_match(p, [r]) for p, r in zip(predictions, references)) / total
    soft = sum(vqa_soft_accuracy(p, [r]) for p, r in zip(predictions, references)) / total
    f1 = sum(token_f1(p, [r]) for p, r in zip(predictions, references)) / total
    return {"exact_match": em, "vqa_soft_accuracy": soft, "token_f1": f1}


def evaluate(
    *,
    dataset_name: Optional[str] = None,
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
    device: Optional[str] = "cuda:0",
    bf16: bool = True,
    fp16: bool = False,
    use_fast: Optional[bool] = None,
    attn_implementation: Optional[str] = None,
    num_workers: int = 0,
    examples_num_batches: int = 3,
    examples_json_path: Optional[str] = None,
):
    del image_column, question_column, answer_column, image_placeholder, max_length, num_workers
    if not dataset_name:
        raise ValueError("dataset_name is required.")
    if not model_name:
        raise ValueError("model_name is required.")

    set_seed(seed)
    dataset_id = dataset_name
    adapter_cfg = read_adapter_config(adapter_path)

    model_inputs: List[Dict[str, Any]] = []
    references: List[str] = []
    questions: List[str] = []
    split_name = "test"
    if "kvasir-vqa" in dataset_name.lower():
        split_name = "test"
        test_dataset = load_dataset("json", data_files={"test": "data/Kvasir-VQA-x1/Kvasir-VQA-x1-test.jsonl"}, split="test")
        for sample in test_dataset:
            user_message = sample["messages"][0]
            assistant_message = sample["messages"][1]
            model_inputs.append({"messages": [user_message], "images": sample["images"]})
            questions.append(_extract_text_from_message(user_message))
            references.append(_extract_text_from_message(assistant_message))
       
    else :
        split_name = _resolve_split(dataset_id, eval_split)
        raw_dataset = load_dataset(dataset_id, subset_name, split=split_name, cache_dir=cache_dir)
        col_names = get_column_names(dataset_id)
        for sample in raw_dataset:
            processed = preprocess_fn(sample, col_names=col_names, rgb_convert=True)
            user_message = processed["messages"][0]
            assistant_message = processed["messages"][1]
            model_inputs.append({"messages": [user_message], "images": processed["images"]})
            questions.append(_extract_text_from_message(user_message))
            references.append(_extract_text_from_message(assistant_message))

    bf16, fp16 = _resolve_precision_flags(bf16, fp16)
    dtype = _resolve_dtype(bf16, fp16)
    torch_device = _resolve_torch_device(device)

    processor_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": trust_remote_code,
    }
    if cache_dir:
        processor_kwargs["cache_dir"] = cache_dir
    if use_fast is not None:
        processor_kwargs["use_fast"] = use_fast
    processor = _load_processor_with_fallback(processor_kwargs)

    model_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": trust_remote_code,
    }
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    base_model = AutoModelForImageTextToText.from_pretrained(**model_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path) if adapter_path else base_model
    _configure_decoder_only_padding(processor, model)
    if hasattr(model, "fuse_lora"):
        try:
            model.fuse_lora()
            model.unload_lora_weights()
            model.compile()
        except Exception as e:
            print(f"Warning: Failed to fuse/unload/compile LoRA weights: {e}")
    model.eval()
    model.to(torch_device)

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
    }
    if temperature and temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = float(temperature)
        generation_kwargs["top_p"] = float(top_p)
    else:
        generation_kwargs["do_sample"] = False

    predictions: List[str] = []
    batch_size = max(1, int(eval_batch_size))
    batch_starts = range(0, len(model_inputs), batch_size)
    print("input num samples:", len(model_inputs))
    with torch.inference_mode():
        for start in tqdm(
            batch_starts,
            total=(len(model_inputs) + batch_size - 1) // batch_size,
            desc="Evaluating",
            unit="batch",
        ):
            batch_samples = model_inputs[start : start + batch_size]
            batch_messages = [sample["messages"] for sample in batch_samples]
            batch_images = [sample["images"] for sample in batch_samples]
            batch_inputs = _build_batch_inputs(processor, batch_messages, batch_images)
            batch_inputs = {
                key: value.to(torch_device) if hasattr(value, "to") else value
                for key, value in batch_inputs.items()
            }
            prompt_input_ids = batch_inputs.get("input_ids")
            generated_ids = model.generate(**batch_inputs, **generation_kwargs)
            predictions.extend(_decode_predictions(processor, generated_ids, prompt_input_ids))
            if start % (20 * batch_size) == 0:
                print(f"cuda max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    metrics = _compute_local_metrics(predictions, references)

    base_model_name = (
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
    max_examples = max(0, int(examples_num_batches)) * max(1, int(eval_batch_size))
    if max_examples > 0:
        resolved_examples_path = examples_json_path
        if not resolved_examples_path:
            base_path, _ = os.path.splitext(resolved_csv_path)
            resolved_examples_path = f"{base_path}_examples.json"
        examples_dir = os.path.dirname(resolved_examples_path)
        if examples_dir:
            os.makedirs(examples_dir, exist_ok=True)
        examples = []
        for i in range(min(max_examples, len(predictions))):
            examples.append(
                {
                    "sample_id": i,
                    "question": questions[i],
                    "reference": references[i],
                    "prediction": predictions[i],
                }
            )
        payload = {
            "dataset_name": dataset_id,
            "model_name": model_name,
            "adapter_path": adapter_path or "",
            "eval_split": split_name,
            "num_samples": len(predictions),
            "num_examples": len(examples),
            "examples": examples,
        }
        with open(resolved_examples_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    row: Dict[str, object] = {
        "timestamp": train_timestamp or now_timestamp(),
        "base_model": base_model_name,
        "dataset_name": dataset_id,
        "seed": run_fields.get("seed", seed),
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
    print(f"Eval metrics (HF model.generate): {metrics}")
    print(f"Saved CSV -> {resolved_csv_path}")
    if resolved_examples_path:
        print(f"Saved examples JSON -> {resolved_examples_path}")
    return metrics
