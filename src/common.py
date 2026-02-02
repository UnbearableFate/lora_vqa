from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "unknown"


def flatten_hf_id(value: str) -> str:
    return slugify("-".join(part for part in value.split("/") if part))


def truncate_with_hash(value: str, max_len: int = 128) -> str:
    if len(value) <= max_len:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    keep = max(1, max_len - (1 + len(digest)))
    return f"{value[:keep]}-{digest}"


def _format_float(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 1000:
        return f"{value:.2e}"
    return f"{value:g}"


def normalize_init_lora_weights(value: Union[bool, str, None]) -> Union[bool, str, None]:
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    if lowered in {"none", "null"}:
        return None
    return lowered


def build_wandb_project_run_tags(
    *,
    model_name: str,
    dataset_id: str,
    peft_variant: str,
    lora_r: int,
    lora_alpha: int,
    init_lora_weights: Union[bool, str, None],
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    num_train_epochs: float,
    global_batch_size: int,
    per_device_batch_size: int,
    eval_steps: int,
    logging_steps: int,
    seed: int,
    fp16: bool,
    bf16: bool,
    gradient_checkpointing: bool,
    use_cleaned_svd_ref_trainer: bool,
    repeat_n: int,
    adjust_lora_alpha_at: Optional[Sequence[int]],
    timestamp: str,
) -> Tuple[str, str, list[str]]:
    model_component = flatten_hf_id(model_name)
    dataset_component = flatten_hf_id(dataset_id)
    project = truncate_with_hash(f"{model_component}__{dataset_component}", max_len=128)

    parsed_init = normalize_init_lora_weights(init_lora_weights)
    if isinstance(parsed_init, bool) and parsed_init:
        init_lora_weights_str = "kaiming"
    elif isinstance(parsed_init, str):
        init_lora_weights_str = slugify(parsed_init)
    else:
        init_lora_weights_str = "none"

    key_parts: list[str] = [
        peft_variant,
        dataset_id.split("/")[-1],
        f"r{lora_r}",
        f"a{lora_alpha}",
        init_lora_weights_str,
    ]
    if use_cleaned_svd_ref_trainer:
        key_parts.append(f"sr#{repeat_n}rp")
    key_parts.append(f"s{seed}")
    key_parts.append(str(timestamp))

    run_name = truncate_with_hash("_".join(key_parts), max_len=128)

    tags: list[str] = [
        f"model={model_component}",
        f"dataset={dataset_component}",
        f"lr={_format_float(learning_rate)}",
        f"wd={_format_float(weight_decay)}",
        f"warmup={_format_float(warmup_ratio)}",
        f"epochs={_format_float(num_train_epochs)}",
        f"gbs={global_batch_size}",
        f"pbs={per_device_batch_size}",
        f"eval_steps={eval_steps}",
        f"logging_steps={logging_steps}",
        f"seed={seed}",
    ]
    if fp16:
        tags.append("fp16")
    if bf16:
        tags.append("bf16")
    if gradient_checkpointing:
        tags.append("grad_ckpt")
    if use_cleaned_svd_ref_trainer:
        tags.append("trainer=cleaned_svd_ref")
        if adjust_lora_alpha_at:
            tags.append(f"adjust_alpha_at={','.join(str(v) for v in adjust_lora_alpha_at)}")

    return project, run_name, tags


def maybe_enable_wandb(
    use_wandb: bool,
    *,
    project: str,
    run_name: Optional[str],
    online: bool = True,
    tags: Optional[Sequence[str]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> None:
    if not use_wandb:
        os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "disabled")
        return

    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "use_wandb=True but wandb is not importable; install wandb or set use_wandb=False."
        ) from exc

    if wandb.run is None:
        wandb.init(
            project=project,
            name=run_name,
            mode="online" if online else "offline",
            tags=list(tags) if tags else None,
            config=dict(config) if config else None,
        )


def build_structured_output_dir(
    output_root: str,
    *,
    dataset_id: str,
    model_name: str,
    lora_r: int,
    run_name: str,
) -> str:
    dataset_dir = slugify(dataset_id.split("/")[-1])
    model_dir = slugify(model_name.split("/")[-1])
    return os.path.join(output_root, dataset_dir, model_dir, f"r{lora_r}", run_name)


def resolve_output_dir(
    output_dir: str,
    *,
    dataset_id: str,
    model_name: str,
    lora_r: int,
    run_name: str,
    mode: str = "auto_if_outputs",
) -> str:
    mode = (mode or "auto_if_outputs").strip().lower()
    if mode in {"explicit", "none", "no"}:
        return output_dir
    if mode in {"auto"}:
        return build_structured_output_dir(
            output_dir, dataset_id=dataset_id, model_name=model_name, lora_r=lora_r, run_name=run_name
        )
    if mode in {"auto_if_outputs", "auto_if_default"}:
        base = os.path.basename(os.path.normpath(output_dir))
        if base in {"outputs", "output"}:
            return build_structured_output_dir(
                output_dir, dataset_id=dataset_id, model_name=model_name, lora_r=lora_r, run_name=run_name
            )
        return output_dir
    raise ValueError(f"Unsupported output_dir_mode={mode!r} (expected: explicit/auto/auto_if_outputs).")


def _rewrite_csv_with_extended_header(csv_path: str, fieldnames: Sequence[str]) -> None:
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)


def append_row_to_csv(csv_path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            header_reader = csv.reader(f)
            existing_header = next(header_reader, [])
        fieldnames = list(dict.fromkeys(existing_header + [k for k in row.keys() if k not in existing_header]))
        if fieldnames != existing_header:
            _rewrite_csv_with_extended_header(csv_path, fieldnames)
    else:
        fieldnames = list(row.keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(row)


def default_eval_csv_path(adapter_path: Optional[str]) -> str:
    if adapter_path and os.path.isdir(adapter_path):
        return os.path.join(adapter_path, "eval_results.csv")
    return "eval_results.csv"


def resolve_eval_csv_path(
    *,
    dataset_id: str,
    model_name: str,
    adapter_path: Optional[str],
    csv_path_dir: Optional[str],
) -> str:
    if csv_path_dir:
        dataset_dir = slugify(dataset_id.split("/")[-1])
        model_dir = slugify(model_name.split("/")[-1])
        return os.path.join(csv_path_dir, dataset_dir, model_dir, "eval_results.csv")
    return default_eval_csv_path(adapter_path)


def read_adapter_config(adapter_path: Optional[str]) -> Dict[str, Any]:
    if not adapter_path:
        return {}
    candidate = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(candidate):
        return {}
    try:
        with open(candidate, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


_RUN_SEED_RE = re.compile(r"(?:^|_)s(?P<seed>\d+)(?:_|$)")
_RUN_TS_RE = re.compile(r"(?P<ts>\d{8}[-_]\d{6})$")


def parse_run_name_fields(run_name: str) -> Dict[str, object]:
    """
    Best-effort parse for lora_image-style run_name suffixes like:
      ..._s42_20260127-123456
    """
    fields: Dict[str, object] = {}
    match_seed = _RUN_SEED_RE.search(run_name)
    if match_seed:
        try:
            fields["seed"] = int(match_seed.group("seed"))
        except Exception:
            pass
    match_ts = _RUN_TS_RE.search(run_name)
    if match_ts:
        fields["timestamp"] = match_ts.group("ts").replace("_", "-")
    return fields


def now_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def json_safe(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
