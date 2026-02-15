from __future__ import annotations

import json
import os
import re
import statistics
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, set_seed

from .common import (
    append_row_to_csv,
    json_safe,
    now_timestamp,
    parse_adapter_path_fields,
    read_adapter_config,
    resolve_eval_csv_path,
)
from .data_process import get_column_names, get_val_split_name, preprocess_fn

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


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _default_predictions_jsonl_path(resolved_csv_path: str, adapter_path: Optional[str]) -> str:
    base_path, _ = os.path.splitext(resolved_csv_path)
    adapter_name = (adapter_path or "").rstrip("/").split("/")[-1] or "no_adapter"
    return f"{base_path}_{adapter_name}_predictions.jsonl"


def _default_judged_jsonl_path(resolved_csv_path: str, adapter_path: Optional[str]) -> str:
    base_path, _ = os.path.splitext(resolved_csv_path)
    adapter_name = (adapter_path or "").rstrip("/").split("/")[-1] or "no_adapter"
    return f"{base_path}_{adapter_name}_judged.jsonl"


def _write_predictions_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


_FIRST_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_FIRST_NUMBER_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def _parse_judge_score(text: str) -> Optional[float]:
    raw = (text or "").strip()
    if not raw:
        return None

    match = _FIRST_JSON_OBJECT_RE.search(raw)
    if match:
        blob = match.group(0)
        try:
            obj = json.loads(blob)
            score = obj.get("score", None)
            if score is None:
                return None
            return float(score)
        except Exception:
            pass

    match = _FIRST_NUMBER_RE.search(raw)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _build_judge_prompt(tokenizer: Any, question: str, reference: str, prediction: str) -> str:
    system = (
        "You are a strict judge for visual question answering (VQA). "
        "Score the model answer against the reference answer for correctness. "
        "Return ONLY a JSON object like {\"score\": <number>} where score is a number from 0 to 10. "
        "10 = fully correct, 0 = completely incorrect. If partially correct, use an intermediate score."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Reference Answer:\n{reference}\n\n"
        f"Model Answer:\n{prediction}\n"
    )

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    return f"{system}\n\n{user}\nJSON:"


def _resolve_generation_device(model: Any) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for mapped_device in hf_device_map.values():
            if isinstance(mapped_device, int):
                return torch.device(f"cuda:{mapped_device}")
            if isinstance(mapped_device, str) and mapped_device.startswith("cuda"):
                return torch.device(mapped_device)
        return torch.device("cpu")

    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device):
        return model_device
    if isinstance(model_device, str):
        return torch.device(model_device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _score_with_transformers(
    *,
    questions: List[str],
    references: List[str],
    predictions: List[str],
    judge_model_name_or_path: str,
    judge_batch_size: int = 16,
    judge_max_tokens: int = 64,
    judge_temperature: float = 0.0,
    judge_top_p: float = 1.0,
    cache_dir: Optional[str] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
) -> Dict[str, Any]:
    _ = tensor_parallel_size, gpu_memory_utilization

    tokenizer_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": judge_model_name_or_path,
        "trust_remote_code": True,
    }
    if cache_dir:
        tokenizer_kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    model_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": judge_model_name_or_path,
        "trust_remote_code": True,
    }
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.eval()
    if not torch.cuda.is_available():
        model.to(torch.device("cpu"))
    generation_device = _resolve_generation_device(model)

    prompts = [
        _build_judge_prompt(tokenizer, q, r, p)
        for q, r, p in zip(questions, references, predictions)
    ]

    scores: List[Optional[float]] = []
    raw_outputs: List[str] = []
    batch_size = max(1, int(judge_batch_size))
    for start in tqdm(range(0, len(prompts), batch_size), desc="LLM judging", unit="batch"):
        batch_prompts = prompts[start : start + batch_size]
        tokenizer_call_kwargs: Dict[str, Any] = {
            "text": batch_prompts,
            "return_tensors": "pt",
            "padding": True,
        }
        if max_model_len is not None:
            tokenizer_call_kwargs["truncation"] = True
            tokenizer_call_kwargs["max_length"] = int(max_model_len)
        encoded_inputs = tokenizer(**tokenizer_call_kwargs)
        encoded_inputs = {
            key: value.to(generation_device) if hasattr(value, "to") else value
            for key, value in encoded_inputs.items()
        }

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(judge_max_tokens),
            "do_sample": bool(judge_temperature and judge_temperature > 0),
            "pad_token_id": tokenizer.pad_token_id,
        }
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = float(judge_temperature)
            generate_kwargs["top_p"] = float(judge_top_p)

        with torch.inference_mode():
            generated_ids = model.generate(**encoded_inputs, **generate_kwargs)

        prompt_length = encoded_inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, prompt_length:]
        batch_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for text in batch_outputs:
            text = text.strip()
            raw_outputs.append(text)
            score = _parse_judge_score(text)
            if score is not None:
                score = max(0.0, min(10.0, float(score)))
            scores.append(score)

    parsed_scores = [s for s in scores if s is not None]
    avg = float(sum(parsed_scores) / len(parsed_scores)) if parsed_scores else 0.0
    std = float(statistics.pstdev(parsed_scores)) if len(parsed_scores) >= 2 else 0.0
    parse_fail = int(len(scores) - len(parsed_scores))
    return {
        "scores": scores,
        "raw_outputs": raw_outputs,
        "avg_score": avg,
        "std_score": std,
        "parse_fail_count": parse_fail,
        "num_samples": len(scores),
    }


def evaluate(
    *,
    adapter_path: Optional[str] = None,
    subset_name: Optional[str] = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eval_batch_size: int = 2,
    cache_dir: Optional[str] = None,
    csv_path_dir: Optional[str] = "experiments",
    device: Optional[str] = "cuda:0",
    bf16: bool = True,
    fp16: bool = False,
    use_fast: Optional[bool] = None,
    attn_implementation: Optional[str] = None,
    examples_num_batches: int = 3,
    examples_json_path: Optional[str] = None,
    predictions_jsonl_path: Optional[str] = None,
    judge_model_name_or_path: Optional[str] = None,
    judge_batch_size: int = 16,
    judge_max_tokens: int = 64,
    judge_temperature: float = 0.0,
    judge_top_p: float = 1.0,
    judge_tensor_parallel_size: int = 1,
    judge_gpu_memory_utilization: float = 0.9,
    judge_max_model_len: Optional[int] = None,
):
    trust_remote_code = True
    adapter_cfg = read_adapter_config(adapter_path)
    exp_info = parse_adapter_path_fields(adapter_path)
    dataset_id = exp_info["dataset_name"]
    seed = exp_info["seed"]
    eval_split = "test"
    model_name = adapter_cfg.get("base_model_name_or_path")
    set_seed(seed)
   
    model_inputs: List[Dict[str, Any]] = []
    references: List[str] = []
    questions: List[str] = []
    split_name = "test"
    if "kvasir-vqa" in dataset_id.lower():
        split_name = "test"
        test_dataset = load_dataset("json", data_files={"test": "data/Kvasir-VQA-x1/Kvasir-VQA-x1-test.jsonl"}, split="test").select(range(512))
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
        model_kwargs["dtype"] = dtype
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    base_model = AutoModelForImageTextToText.from_pretrained(**model_kwargs)
    fake_lora_config = None
    if adapter_cfg.get("init_lora_weights") == "corda":
        from peft import LoraConfig
        fake_lora_config = LoraConfig(
            r=adapter_cfg.get("r"),
            init_lora_weights=True,
            lora_alpha=adapter_cfg.get("lora_alpha", 16),
            target_modules=adapter_cfg.get("target_modules"),
            exclude_modules=adapter_cfg.get("exclude_modules"),
            lora_dropout=adapter_cfg.get("lora_dropout", 0.0),
            bias="none",
            task_type="CAUSAL_LM",
        )
    model = PeftModel.from_pretrained(base_model, adapter_path, config=fake_lora_config)
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

    base_model_name = (
        str(adapter_cfg.get("base_model_name_or_path") or model_name).split("/")[-1]
        if isinstance(adapter_cfg, dict)
        else model_name.split("/")[-1]
    )
    train_timestamp = str(exp_info.get("timestamp") or "")

    resolved_csv_path = resolve_eval_csv_path(
        dataset_id=dataset_id,
        model_name=model_name,
        adapter_path=adapter_path,
        csv_path_dir=csv_path_dir,
    )

    resolved_predictions_path = predictions_jsonl_path
    if not resolved_predictions_path:
        resolved_predictions_path = _default_predictions_jsonl_path(resolved_csv_path, adapter_path)
    prediction_rows: List[Dict[str, Any]] = []
    for i in range(len(predictions)):
        prediction_rows.append(
            {
                "sample_id": i,
                "question": questions[i],
                "reference": references[i],
                "prediction": predictions[i],
            }
        )
    _write_predictions_jsonl(resolved_predictions_path, prediction_rows)

    if not judge_model_name_or_path or not str(judge_model_name_or_path).strip():
        raise ValueError(
            "judge_model_name_or_path is required. "
            "Set it to a local path or HF model id to run Transformers judging."
        )
    judge = _score_with_transformers(
        questions=questions,
        references=references,
        predictions=predictions,
        judge_model_name_or_path=str(judge_model_name_or_path),
        judge_batch_size=judge_batch_size,
        judge_max_tokens=judge_max_tokens,
        judge_temperature=judge_temperature,
        judge_top_p=judge_top_p,
        cache_dir=cache_dir,
        tensor_parallel_size=judge_tensor_parallel_size,
        gpu_memory_utilization=judge_gpu_memory_utilization,
        max_model_len=judge_max_model_len,
    )

    resolved_judged_path = _default_judged_jsonl_path(resolved_csv_path, adapter_path)
    judged_rows: List[Dict[str, Any]] = []
    for row, score, raw in zip(prediction_rows, judge["scores"], judge["raw_outputs"]):
        judged = dict(row)
        judged["judge_score_0_10"] = score
        judged["judge_raw_output"] = raw
        judged_rows.append(judged)
    _write_predictions_jsonl(resolved_judged_path, judged_rows)

    resolved_examples_path = None
    max_examples = max(0, int(examples_num_batches)) * max(1, int(eval_batch_size))
    if max_examples > 0:
        resolved_examples_path = examples_json_path
        if not resolved_examples_path:
            base_path, _ = os.path.splitext(resolved_csv_path)
            resolved_examples_path = f"{base_path}_{adapter_path.split('/')[-1]}_examples.json"
        examples_dir = os.path.dirname(resolved_examples_path)
        if examples_dir:
            os.makedirs(examples_dir, exist_ok=True)
        examples = []
        for i in range(min(max_examples, len(predictions))):
            score = judge["scores"][i] if i < len(judge["scores"]) else None
            examples.append(
                {
                    "sample_id": i,
                    "question": questions[i],
                    "reference": references[i],
                    "prediction": predictions[i],
                    "judge_score_0_10": score,
                }
            )
        payload = {
            "dataset_name": dataset_id,
            "model_name": model_name,
            "adapter_path": adapter_path or "",
            "eval_split": split_name,
            "num_samples": len(predictions),
            "num_examples": len(examples),
            "judge_model_name_or_path": str(judge_model_name_or_path),
            "judge_avg_score_0_10": judge["avg_score"],
            "judge_std_score_0_10": judge["std_score"],
            "judge_parse_fail_count": judge["parse_fail_count"],
            "examples": examples,
        }
        with open(resolved_examples_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    row: Dict[str, object] = {
        "timestamp": train_timestamp or now_timestamp(),
        "base_model": base_model_name,
        "dataset_name": dataset_id,
        "seed": exp_info.get("seed", seed),
    }

    init_lora_weights = adapter_cfg.get("init_lora_weights") if isinstance(adapter_cfg, dict) else None
    if isinstance(init_lora_weights, str) and init_lora_weights.strip().lower() == "true":
        init_lora_weights = "kaiming"
    row["init_lora_weights"] = init_lora_weights if init_lora_weights is not None else "none"
    row['extra'] = exp_info.get('extra', 'none')

    row["metric_llm_score_0_10_avg"] = judge["avg_score"]
    row["metric_llm_score_0_10_std"] = judge["std_score"]
    row["metric_llm_parse_fail_count"] = judge["parse_fail_count"]
    row["judge_model_name_or_path"] = str(judge_model_name_or_path)

    if isinstance(adapter_cfg, dict) and adapter_cfg:
        for key in ("r", "lora_alpha", "lora_dropout", "target_modules", "bias", "use_dora", "use_rslora"):
            if key in adapter_cfg:
                v = adapter_cfg[key]
                row[key] = json_safe(v) if isinstance(v, (dict, list)) else v

    append_row_to_csv(resolved_csv_path, row)
    print(
        "LLM judge score avg/std (0-10): "
        f"{judge['avg_score']:.3f}/{judge['std_score']:.3f} "
        f"(parse_fail={judge['parse_fail_count']}/{judge['num_samples']})"
    )
    print(f"Saved CSV -> {resolved_csv_path}")
    if resolved_examples_path:
        print(f"Saved examples JSON -> {resolved_examples_path}")
    print(f"Saved predictions JSONL -> {resolved_predictions_path}")
    print(f"Saved judged JSONL -> {resolved_judged_path}")
    return {
        "llm_score_0_10_avg": judge["avg_score"],
        "llm_score_0_10_std": judge["std_score"],
        "llm_parse_fail_count": judge["parse_fail_count"],
        "num_samples": judge["num_samples"],
        "predictions_jsonl_path": resolved_predictions_path,
        "judged_jsonl_path": resolved_judged_path,
        "examples_json_path": resolved_examples_path,
    }
