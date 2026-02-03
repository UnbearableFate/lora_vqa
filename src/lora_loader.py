from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import inspect
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
import tqdm
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:  # optional, only needed for init_lora_weights="eva"
    from peft import initialize_lora_eva_weights  # type: ignore
except Exception:  # pragma: no cover
    initialize_lora_eva_weights = None

try:  # optional, only needed for init_lora_weights="eva"
    from peft.tuners.lora.eva import forward_fn_dict as _peft_forward_fn_dict  # type: ignore
except Exception:  # pragma: no cover
    _peft_forward_fn_dict = None

try:  # optional, only needed for init_lora_weights="corda" / "eva"
    from peft.tuners.lora.corda import preprocess_corda  # type: ignore
    from peft.tuners.lora.config import CordaConfig, EvaConfig  # type: ignore
except Exception:  # pragma: no cover
    preprocess_corda = None
    CordaConfig = None
    EvaConfig = None

from my_peft import LoraGAConfig
from accelerate import Accelerator 

import logging
logger = logging.getLogger(__name__)

class CompletionDataCollator:
    """Pad input/label pairs for completion-style causal LM supervision."""

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id or 0
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = -100

    def _pad_sequences(self, sequences, pad_value):
        if not sequences:
            return torch.empty(0)

        max_length = max(len(seq) for seq in sequences)
        if self.pad_to_multiple_of:
            remainder = max_length % self.pad_to_multiple_of
            if remainder:
                max_length += self.pad_to_multiple_of - remainder

        padded = []
        for seq in sequences:
            pad_len = max_length - len(seq)
            padded.append(seq + [pad_value] * pad_len)
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, features):
        input_ids = [feature["input_ids"] for feature in features]
        attention_masks = [feature.get("attention_mask") for feature in features]
        labels = [feature.get("labels") for feature in features]

        batch_input_ids = self._pad_sequences(input_ids, self.pad_token_id)

        if all(mask is not None for mask in attention_masks):
            batch_attention_mask = self._pad_sequences(attention_masks, 0)
        else:
            batch_attention_mask = torch.zeros_like(batch_input_ids)
            for idx, ids in enumerate(input_ids):
                batch_attention_mask[idx, : len(ids)] = 1

        if any(label is not None for label in labels):
            batch_labels = self._pad_sequences(
                [label if label is not None else [] for label in labels],
                self.label_pad_token_id,
            )
        else:
            batch_labels = batch_input_ids.clone()

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }


@dataclass
class LoraHyperparameters:
    """Hyperparameters for the LoRA-family adapters."""

    variant: str = "lora"  # lora, dora, qalora, rslora
    task_type: str = "CAUSAL_LM"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    )
    init_lora_weights: Union[bool, str, None] = True # ["gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]", "corda", "loftq", "orthogonal"]
    init_num_samples: int = 512
    init_batch_size: int = 8
    corda_method: str = "kpm"  # kpm or ipm

    loraga_direction : str = "ArB2r"  # ArB2r, A2rB, BrA2r
    loraga_dtype : torch.dtype = torch.float32
    exclude_modules: Optional[Union[List[str], str]] = None

    cache_dir: Optional[str] = "data_cache"
    unique_cache_filename : Optional[str] = None
    model_name_or_path: Optional[str] = None
    dataset_name: Optional[str] = None
    subdataset_name: Optional[str] = None
    init_seed: int = 1337

    def __post_init__(self):
        if not self.model_name_or_path or not self.dataset_name:
            return
        unique_cache_filename = f"{self.model_name_or_path.replace('/', '-')}_{self.dataset_name.replace('/', '-')}"
        if self.subdataset_name:
            unique_cache_filename += f"_{self.subdataset_name.replace('/', '-')}"
        self.unique_cache_filename = f"{unique_cache_filename}_r{self.r}_dp{self.init_num_samples}_{self.init_seed}.pt"
        if str(self.init_lora_weights).lower() == "true":
            self.init_lora_weights = True

    def get_unique_cache_path(self,path_mid_name) -> str:
        parent_path = Path(self.cache_dir, path_mid_name)
        if not parent_path.exists():
            parent_path.mkdir(parents=True, exist_ok=True)
        return parent_path.joinpath(self.unique_cache_filename).as_posix()

_VARIANT_TO_FLAGS = {
    "lora": {"use_dora": False, "use_rslora": False, "use_qalora": False},
    "dora": {"use_dora": True, "use_rslora": False, "use_qalora": False},
    "rslora": {"use_dora": False, "use_rslora": True, "use_qalora": False},
    "qalora": {"use_dora": False, "use_rslora": False, "use_qalora": True},
}


def build_LoraHyperparameters_from_yaml_dict(cfg_dict) -> LoraHyperparameters:
    peft_config = cfg_dict.get("peft", {})
    loraga_config = cfg_dict.get("loraga", {})
    lora_init_kwargs = peft_config.get("lora_init_kwargs", {})
    return LoraHyperparameters(
        variant= peft_config['variant'],
        task_type= peft_config.get("task_type", "CAUSAL_LM"),
        r= peft_config['lora_r'],
        alpha= peft_config['lora_alpha'],
        dropout= peft_config['lora_dropout'],
        bias= peft_config['bias'],
        target_modules= peft_config['target_modules'],
        init_lora_weights= peft_config['init_lora_weights'],
        init_num_samples= lora_init_kwargs.get('init_num_samples', 512),
        init_batch_size= lora_init_kwargs.get('init_batch_size', 8),

        corda_method= lora_init_kwargs.get('corda_method', "kpm"),
        loraga_direction= lora_init_kwargs.get('loraga_direction', "ArB2r") if loraga_config else "ArB2r",
        loraga_dtype= torch.float32,
        
        cache_dir= peft_config.get('cache_dir', "data_cache"),
        exclude_modules= peft_config.get("exclude_modules", None),
        model_name_or_path= cfg_dict["model"]["name_or_path"],
        dataset_name= cfg_dict["dataset"]["name"],
        subdataset_name= cfg_dict["dataset"].get("subset", None),
        init_seed= lora_init_kwargs.get('init_seed', cfg_dict['training'].get("seed", 42) *2 +1),
    )

def get_lora_config(lora_cfg: LoraHyperparameters) -> LoraConfig | LoraGAConfig:
    variant = lora_cfg.variant.lower()
    if variant not in _VARIANT_TO_FLAGS:
        raise ValueError(f"Unsupported LoRA variant: {variant}")
    peft_config = None
    if lora_cfg.init_lora_weights != "lora_ga":
        corda_config = None
        eva_config = None
        if lora_cfg.init_lora_weights == "corda":
                if CordaConfig is None:
                    raise ImportError("init_lora_weights='corda' requires a PEFT build that ships CordaConfig.")
                corda_config = CordaConfig(
                    corda_method=lora_cfg.corda_method, # kpm or ipm
                    cache_file=lora_cfg.get_unique_cache_path("corda_cache"),
                    covariance_file=lora_cfg.get_unique_cache_path("covariance_file"),
                )
        elif lora_cfg.init_lora_weights == "eva":
            if EvaConfig is None:
                raise ImportError("init_lora_weights='eva' requires a PEFT build that ships EvaConfig.")
            eva_config = EvaConfig()

        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            bias=lora_cfg.bias,
            target_modules=list(lora_cfg.target_modules),
            exclude_modules=lora_cfg.exclude_modules,
            task_type=lora_cfg.task_type,
            init_lora_weights=lora_cfg.init_lora_weights,
            corda_config=corda_config,
            eva_config=eva_config,
            **_VARIANT_TO_FLAGS[variant],
        )
        
    else:
        peft_config = LoraGAConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            bias=lora_cfg.bias,
            target_modules=list(lora_cfg.target_modules),
            task_type=lora_cfg.task_type,
            bsz=lora_cfg.init_batch_size,
            direction=lora_cfg.loraga_direction,
            dtype= lora_cfg.loraga_dtype,
            gradient_save_path=lora_cfg.get_unique_cache_path("loraga_gradient"),
            **_VARIANT_TO_FLAGS[variant],
        )
    
    print(f"lora config: {peft_config}")
    return peft_config

"""use the following function in VLM training script
"""

def attach_vlm_lora_adapter(base_model,lora_cfg: LoraConfig|LoraGAConfig, train_dataset,data_collator, init_num_samples:int, batch_size:int,seed: int, accelerator: Accelerator, save_dir: Path = None):
    if lora_cfg.init_lora_weights not in ["corda", "eva", "lora_ga"]:
        return get_peft_model(base_model, lora_cfg)
    sub_dataset = _select_init_subset(train_dataset, init_num_samples, seed)
    if sub_dataset is None:
        return get_peft_model(base_model, lora_cfg)

    if lora_cfg.init_lora_weights == "corda":
        return get_peft_model_with_corda(base_model, lora_cfg, sub_dataset,data_collator,accelerator=accelerator)
    elif lora_cfg.init_lora_weights == "eva":
        return get_peft_model_with_eva(base_model, lora_cfg, sub_dataset,data_collator ,batch_size ,accelerator=accelerator)
    elif lora_cfg.init_lora_weights == "lora_ga":
        return get_peft_model_with_lora_ga(base_model, lora_cfg, sub_dataset,data_collator ,batch_size,accelerator=accelerator)

"""
use the following function in language modeling training script
"""
def attach_lora_adapter(base_model,lora_cfg: LoraConfig|LoraGAConfig, train_dataset,data_collator, init_num_samples:int, batch_size:int,seed: int, accelerator: Accelerator, save_dir: Path = None):
    if lora_cfg.init_lora_weights not in ["corda", "eva", "lora_ga"]:
        return get_peft_model(base_model, lora_cfg)
    sub_dataset = _select_init_subset(train_dataset, init_num_samples, seed)
    if sub_dataset is None:
        return get_peft_model(base_model, lora_cfg)

    if lora_cfg.init_lora_weights == "corda":
        return get_peft_model_with_corda(base_model, lora_cfg, sub_dataset,data_collator,accelerator=accelerator)
    elif lora_cfg.init_lora_weights == "eva":
        #if batch_size > 1:
        #    _ensure_model_pad_token_id(base_model, data_collator.tokenizer)
        return get_peft_model_with_eva(base_model, lora_cfg, sub_dataset,data_collator ,batch_size ,accelerator=accelerator)
    elif lora_cfg.init_lora_weights == "lora_ga":
        # Some decoder-only checkpoints (e.g. Qwen3) ship without a padding token in the config.
        # Transformers sequence-classification heads will error on batch_size > 1 unless
        # `model.config.pad_token_id` is set.
        if batch_size > 1:
            _ensure_model_pad_token_id(base_model, data_collator.tokenizer)
        return get_peft_model_with_lora_ga(base_model, lora_cfg, sub_dataset,data_collator ,batch_size,accelerator=accelerator)

def _ensure_model_pad_token_id(base_model, tokenizer) -> None:
    if getattr(getattr(base_model, "config", None), "pad_token_id", None) is not None:
        return
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if hasattr(base_model, "resize_token_embeddings"):
                base_model.resize_token_embeddings(len(tokenizer))
    if getattr(getattr(base_model, "config", None), "pad_token_id", None) is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    generation_config = getattr(base_model, "generation_config", None)
    if generation_config is not None and getattr(generation_config, "pad_token_id", None) is None:
        generation_config.pad_token_id = tokenizer.pad_token_id


def _select_init_subset(train_dataset, init_num_samples: int, seed: int):
    if init_num_samples <= 0:
        return None
    try:
        dataset_len = len(train_dataset)
    except TypeError:
        dataset_len = None
    sample_count = init_num_samples
    if dataset_len is not None:
        sample_count = min(init_num_samples, dataset_len)
    if sample_count <= 0:
        return None
    if hasattr(train_dataset, "shuffle") and hasattr(train_dataset, "select"):
        return train_dataset.shuffle(seed=seed).select(range(sample_count))
    return None

def freeze_lora_A_weights(peft_model):
    for name, param in peft_model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

def get_peft_model_with_corda(base_model,lora_cfg: LoraConfig,sub_dataset,data_collator,accelerator: Accelerator):
    if preprocess_corda is None:
        raise ImportError("init_lora_weights='corda' requires a PEFT build that includes preprocess_corda.")

    sub_dataset_columns = set(getattr(sub_dataset, "column_names", []) or [])
    collator_name = type(data_collator).__name__.lower()
    is_vlm_batch = {"messages", "images"}.issubset(sub_dataset_columns) or (
        "visionlanguagemodeling" in collator_name
    )

    # CorDA in current PEFT assumes 2D activations in covariance hooks.
    # Vision-tower attention often feeds 3D tensors and can crash (`t()` on 3D).
    # For VLM conversational batches, exclude vision tower modules from CorDA init.
    if is_vlm_batch:
        target_modules = lora_cfg.target_modules
        vision_excludes: List[str] = []
        for name, module in base_model.named_modules():
            if "vision_tower" not in name or not isinstance(module, torch.nn.Linear):
                continue
            target_hit = False
            if isinstance(target_modules, str):
                target_hit = re.fullmatch(target_modules, name) is not None
            elif isinstance(target_modules, (list, tuple, set)):
                target_hit = (name in target_modules) or any(name.endswith(f".{t}") for t in target_modules)
            if target_hit:
                vision_excludes.append(name)
        if vision_excludes:
            if lora_cfg.exclude_modules is None:
                lora_cfg.exclude_modules = list(dict.fromkeys(vision_excludes))
            elif isinstance(lora_cfg.exclude_modules, str):
                escaped = "|".join(re.escape(m) for m in vision_excludes)
                lora_cfg.exclude_modules = f"(?:{lora_cfg.exclude_modules})|(?:{escaped})"
            else:
                merged = list(lora_cfg.exclude_modules) + vision_excludes
                lora_cfg.exclude_modules = list(dict.fromkeys(merged))
            print(f"CorDA VLM mode: excluded {len(vision_excludes)} vision-tower target modules.")

    device = accelerator.device if accelerator is not None else next(base_model.parameters()).device

    def _collate_to_device(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = data_collator(features)
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

    calib_loader = DataLoader(
        sub_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=_collate_to_device,
    )
    if accelerator is not None:
        base_model.to(accelerator.device)
    print(f"Running Corda preprocessing on device: {device}")
    #calib_loader = accelerator.prepare(calib_loader)

    @torch.no_grad()
    def _run_model():
        was_training = base_model.training
        base_model.eval()
        # for batch in calib_loader:
        #     batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        #     base_model(**batch)
        for batch in tqdm.tqdm(calib_loader, desc="corda preprocessing"):
            if isinstance(batch, dict) and "labels" in batch:
                batch = {k: v for k, v in batch.items() if k != "labels"}
            base_model(**batch)
        if was_training:
            base_model.train()

    print(f"Starting Corda preprocessing... with sub-dataset of size {len(sub_dataset)}")
    preprocess_corda(
        base_model,
        lora_cfg,
        run_model=_run_model,
    )
    return get_peft_model(base_model, lora_cfg)

def get_peft_model_with_eva(
        base_model,
        lora_cfg: LoraConfig,
        sub_dataset,
        data_collator,
        batch_size: int,
        accelerator: Accelerator,
    ):
    if initialize_lora_eva_weights is None:
        raise ImportError(
            "init_lora_weights='eva' requires a PEFT build that exports initialize_lora_eva_weights."
        )

    device = accelerator.device if accelerator is not None else next(base_model.parameters()).device

    def _collate_to_device(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = data_collator(features)
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

    dataloader = DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_to_device,
    )
    if accelerator is not None:
        base_model.to(accelerator.device)

    peft_model = get_peft_model(base_model, lora_cfg, low_cpu_mem_usage=True)
    print(f"Initializing Eva LoRA weights... with sub-dataset of size {len(sub_dataset)}")

    def _forward_fn(model, model_input):
        if isinstance(model_input, dict) and "labels" in model_input:
            model_input = {k: v for k, v in model_input.items() if k != "labels"}
        if _peft_forward_fn_dict is not None:
            return _peft_forward_fn_dict(model, model_input)
        if isinstance(model_input, dict):
            return model(**model_input)
        return model(model_input)

    sub_dataset_columns = set(getattr(sub_dataset, "column_names", []) or [])
    collator_name = type(data_collator).__name__.lower()
    is_vlm_batch = {"messages", "images"}.issubset(sub_dataset_columns) or (
        "visionlanguagemodeling" in collator_name
    )
    try:
        if len(sub_dataset) > 0 and hasattr(sub_dataset, "__getitem__"):
            sample_batch = _collate_to_device([sub_dataset[0]])
            if isinstance(sample_batch, dict) and "pixel_values" in sample_batch:
                is_vlm_batch = True
    except Exception:
        # Fall back to dataset/collator heuristics above.
        pass

    init_kwargs: Dict[str, Any] = {}
    try:
        sig = inspect.signature(initialize_lora_eva_weights)
        if "forward_fn" in sig.parameters:
            init_kwargs["forward_fn"] = _forward_fn
        if is_vlm_batch and "prepare_model_inputs_fn" in sig.parameters:
            init_kwargs["prepare_model_inputs_fn"] = None
        if is_vlm_batch and "prepare_layer_inputs_fn" in sig.parameters:
            init_kwargs["prepare_layer_inputs_fn"] = None
    except Exception:  # pragma: no cover
        pass

    was_training = peft_model.training
    peft_model.eval()
    try:
        initialize_lora_eva_weights(peft_model, dataloader, **init_kwargs)
    except TypeError:
        initialize_lora_eva_weights(peft_model, dataloader)
    if was_training:
        peft_model.train()
    return peft_model

__all__ = [
    "load_base_model",
    "load_tokenizer",
    "get_lora_config",
]

def get_peft_model_with_lora_ga(
        model,
        lora_ga_cfg: LoraGAConfig,
        sub_dataset,
        data_collator,
        batch_size: int,
        accelerator,
    ):

    from my_peft.utils.lora_ga_utils import (
                LoraGAContext,
                estimate_gradient,
            )
    from my_peft import get_peft_model as my_get_peft_model

    gradient_loader = DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    named_grad = estimate_gradient(
        model=model,
        dataloader=gradient_loader,
        accelerator=accelerator,
        quant_flag=True,
        origin_type="bf16",
        quant_type="int8",
        no_split_module_classes=None,
        grad_save_path=lora_ga_cfg.gradient_save_path,
    )
    start_time = time.time()
    with LoraGAContext(model=model, named_grad=named_grad):
        model = my_get_peft_model(model=model, peft_config=lora_ga_cfg)
    logger.info(f"LoRA-GA initialization took {time.time() - start_time:.2f} seconds")
    
    return model
