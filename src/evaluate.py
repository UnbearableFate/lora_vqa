from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline, set_seed

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


def _resolve_torch_and_pipeline_device(device: Optional[str]) -> tuple[torch.device, int]:
    if device:
        lowered = device.strip().lower()
        if lowered == "cpu":
            return torch.device("cpu"), -1
        if lowered.startswith("cuda"):
            index = 0
            if ":" in lowered:
                index = int(lowered.split(":", maxsplit=1)[1])
            return torch.device(f"cuda:{index}"), index
        raise ValueError(f"Unsupported device={device!r}; use 'cpu' or 'cuda[:index]'.")
    if torch.cuda.is_available():
        return torch.device("cuda:0"), 0
    return torch.device("cpu"), -1


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


def _extract_generated_text(raw_output: Any) -> str:
    if isinstance(raw_output, list):
        if not raw_output:
            return ""
        return _extract_generated_text(raw_output[0])
    if isinstance(raw_output, dict):
        generated = raw_output.get("generated_text", "")
        if isinstance(generated, str):
            return generated.strip()
        if isinstance(generated, list):
            # Some VLM pipelines return a full chat transcript.
            for msg in reversed(generated):
                if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "assistant":
                    text = _extract_text_from_message(msg)
                    if text:
                        return text
            return str(generated).strip()
        return str(generated).strip()
    return str(raw_output).strip()


class _VisionGenerationEvaluatorPipeline:
    task = "text-generation"

    def __init__(
        self,
        *,
        generation_pipe: Any,
        model_inputs: List[Dict[str, Any]],
        generation_kwargs: Dict[str, Any],
    ):
        self.generation_pipe = generation_pipe
        self.model_inputs = model_inputs
        self.generation_kwargs = generation_kwargs
        self.predictions_by_id: Dict[int, str] = {}

    def __call__(self, input_ids: Any, **kwargs) -> List[Dict[str, str]]:
        if isinstance(input_ids, list):
            sample_ids = [int(v) for v in input_ids]
        else:
            sample_ids = [int(input_ids)]
        merged_kwargs = dict(self.generation_kwargs)
        merged_kwargs.update(kwargs)
        outputs: List[Dict[str, str]] = []
        for sample_id in sample_ids:
            sample = self.model_inputs[sample_id]
            try:
                raw = self.generation_pipe(
                    text=sample["messages"],
                    images=sample["images"],
                    **merged_kwargs,
                )
            except TypeError:
                raw = self.generation_pipe(
                    sample["messages"],
                    images=sample["images"],
                    **merged_kwargs,
                )
            prediction = _extract_generated_text(raw)
            self.predictions_by_id[sample_id] = prediction
            outputs.append({"generated_text": prediction})
        return outputs


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
    device: Optional[str] = None,
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

    try:
        from evaluate import evaluator as hf_evaluator
        from evaluate import load as hf_load_metric
    except Exception as exc:
        raise RuntimeError("The `evaluate` package is required. Please install: pip install evaluate") from exc

    set_seed(seed)
    dataset_id = dataset_name
    adapter_cfg = read_adapter_config(adapter_path)

    split_name = _resolve_split(dataset_id, eval_split)
    raw_dataset = load_dataset(dataset_id, subset_name, split=split_name, cache_dir=cache_dir)
    _ = get_column_names(dataset_id)

    model_inputs: List[Dict[str, Any]] = []
    references: List[str] = []
    questions: List[str] = []
    for sample in raw_dataset:
        processed = preprocess_fn(sample, dataset_name=dataset_id)
        user_message = processed["messages"][0]
        assistant_message = processed["messages"][1]
        model_inputs.append({"messages": [user_message], "images": processed["images"]})
        questions.append(_extract_text_from_message(user_message))
        references.append(_extract_text_from_message(assistant_message))

    bf16, fp16 = _resolve_precision_flags(bf16, fp16)
    dtype = _resolve_dtype(bf16, fp16)
    torch_device, pipeline_device = _resolve_torch_and_pipeline_device(device)

    processor_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": trust_remote_code,
    }
    if cache_dir:
        processor_kwargs["cache_dir"] = cache_dir
    if use_fast is not None:
        processor_kwargs["use_fast"] = use_fast
    processor = AutoProcessor.from_pretrained(**processor_kwargs)

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
    model.eval()
    model.to(torch_device)

    generation_pipe = pipeline(
        task="image-text-to-text",
        model=model,
        processor=processor,
        device=pipeline_device,
    )
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "batch_size": max(1, int(eval_batch_size)),
    }
    if temperature and temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = float(temperature)
        generation_kwargs["top_p"] = float(top_p)
    else:
        generation_kwargs["do_sample"] = False

    wrapped_pipeline = _VisionGenerationEvaluatorPipeline(
        generation_pipe=generation_pipe,
        model_inputs=model_inputs,
        generation_kwargs=generation_kwargs,
    )

    evaluator_dataset = Dataset.from_dict(
        {
            "sample_id": list(range(len(model_inputs))),
            "label": references,
        }
    )
    task_evaluator = hf_evaluator("text-generation")
    evaluator_results = task_evaluator.compute(
        model_or_pipeline=wrapped_pipeline,
        data=evaluator_dataset,
        metric=hf_load_metric("exact_match"),
        input_column="sample_id",
        label_column="label",
        strategy="simple",
    )

    predictions = [wrapped_pipeline.predictions_by_id.get(i, "") for i in range(len(model_inputs))]
    metrics = _compute_local_metrics(predictions, references)
    for key, value in evaluator_results.items():
        if key not in metrics:
            metrics[f"evaluator_{key}"] = value

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
        "adapter_path": adapter_path or "",
        "run_name": run_name,
        "eval_split": split_name,
        "backend": "hf_pipeline_evaluator",
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
    print(f"Eval metrics (HF pipeline + evaluator): {metrics}")
    print(f"Saved CSV -> {resolved_csv_path}")
    if resolved_examples_path:
        print(f"Saved examples JSON -> {resolved_examples_path}")
    return metrics
