from datasets import load_dataset
import re
from pathlib import Path
from PIL import Image

from typing import Optional

COL_NAME_MAP = {
    "geometry3k" : {
        "images": "images",
        "question": "problem",
        "answer": "answer",
    },
    "chartqa" : {
        "images": "image",
        "question": "query",
        "answer": "label",
    },
    "documentvqa" : { #HuggingFaceM4/DocumentVQA 
            "images": "image",
            "question": "question",
            "answer": "answers",
        },
    
}

SPLIT_NAME_MAP = {
    "geometry3k": {
        "train": "train",
        "val": "validation",
        "test": "test",
    },
    "chartqa": {
        "train": "train",
        "val": "val",
        "test": "test",
    },
    "documentvqa": {
        "train": "train",
        "val": "validation",
        "test": "test",
    },
}

def get_column_names(dataset_name: str) -> str:
    key = dataset_name.split("/")[-1].lower()
    if key not in COL_NAME_MAP:
        raise ValueError(f"Dataset {dataset_name} not supported for column name mapping.")
    return COL_NAME_MAP[key]

def _to_rgb(image):
    if image is None:
        return None
    # Many VLM processors only support 1 or 3 channels; Geometry3K images are RGBA.
    if hasattr(image, "mode") and hasattr(image, "convert") and image.mode != "RGB":
        return image.convert("RGB")
    return image


def _normalize_images(images):
    if isinstance(images, list):
        return [_to_rgb(img) for img in images]
    return [_to_rgb(images)]

def _normalize_images_without_convert(images):
    if isinstance(images, list):
        return images
    return [images]

def _strip_inline_image_tokens(text: str) -> str:
    if text is None:
        return ""
    # Prevent duplicated image placeholders when dataset text already contains image tags.
    text = re.sub(r"<\s*image\s*>", "", text, flags=re.IGNORECASE)
    return text.strip()


def preprocess_fn(data_point, col_names , rgb_convert: bool = True):
    """
    Input columns:
      - example["image"]: PIL.Image / image path / dataset Image object (depending on your dataset)
      - example["question"]: str
      - example["answer"]: str

    Output:
      - "messages": list of chat messages with typed content (image + text)
    """
    question_text = _strip_inline_image_tokens(data_point[col_names["question"]])
    
    if isinstance(data_point[col_names["answer"]], list):
        answer_text = str(data_point[col_names["answer"]][0])
    else:
        answer_text = str(data_point[col_names["answer"]])

    image_fn = _normalize_images if rgb_convert else _normalize_images_without_convert  
    return {
        "images": image_fn(data_point[col_names["images"]]),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer_text},
                ],
            },
        ]
    }

def load_and_preprocess_dataset(dataset_id: str, subset_name: Optional[str] = None, splits: list = ["train", "val"], num_proc: int = 16 ,rgb_convert: bool = True):
    if "documentvqa" in dataset_id.lower():
        rgb_convert = False
    elif "chartqa" in dataset_id.lower():
        rgb_convert = True
    dataset = load_dataset(dataset_id, subset_name)
    train_dataset, val_dataset, test_dataset = None, None, None
    col_names = get_column_names(dataset_id)
    cache_path = Path("./cache",dataset_id.replace('/', '_'))
    print(f"Cache path: {str(cache_path / f"{dataset_id.replace('/', '_')}_{'raw' if not rgb_convert else 'rgb'}_train.arrow" )}")
    if not cache_path.exists():
        cache_path.mkdir(parents=True, exist_ok=True)
    if "train" in splits:
        train_columns = list(dataset["train"].column_names)
        if "images" in train_columns:
            train_columns.remove("images")
        train_dataset = dataset["train"].map(
            preprocess_fn,
            fn_kwargs={"col_names": col_names , "rgb_convert": rgb_convert},
            remove_columns=train_columns,
            num_proc=num_proc,
            cache_file_name=str(cache_path / f"{dataset_id.replace('/', '_')}_{'raw' if not rgb_convert else 'rgb'}_train.arrow"),
            load_from_cache_file=True,
        )
    if "val" in splits:
        val_split_name = get_val_split_name(dataset_id)
        val_columns = list(dataset[val_split_name].column_names)
        if "images" in val_columns:
            val_columns.remove("images")
        val_dataset = dataset[val_split_name].map(
            preprocess_fn,
            fn_kwargs={"col_names": col_names, "rgb_convert": rgb_convert},
            remove_columns=val_columns,
            num_proc=num_proc,
            cache_file_name=str(cache_path / f"{dataset_id.replace('/', '_')}_{'raw' if not rgb_convert else 'rgb'}_val.arrow"),
            load_from_cache_file=True,
        )
    if "test" in splits:
        test_columns = list(dataset["test"].column_names)
        if "images" in test_columns:
            test_columns.remove("images")
        test_dataset = dataset["test"].map(
            preprocess_fn,
            fn_kwargs={"col_names": col_names, "rgb_convert": rgb_convert},
            remove_columns=test_columns,
            num_proc=num_proc,
            cache_file_name=str(cache_path / f"{dataset_id.replace('/', '_')}_{'raw' if not rgb_convert else 'rgb'}_test.arrow"),
            load_from_cache_file=True,
        )
    return train_dataset, val_dataset, test_dataset

def load_and_preprocess_kvasir_vqa(
    splits: list = ["train"],
    val_set_size: int = 1024,
    data_dir: str = "data/Kvasir-VQA-x1",
    seed: int = 42,
):
    requested_splits = set(splits or [])
    train_dataset, val_dataset, test_dataset = None, None, None

    data_dir_path = Path(data_dir)
    train_jsonl = data_dir_path / "Kvasir-VQA-x1-train.jsonl"
    test_jsonl = data_dir_path / "Kvasir-VQA-x1-test.jsonl"

    if "train" in requested_splits or "val" in requested_splits:
        if not train_jsonl.exists():
            raise FileNotFoundError(f"Kvasir train jsonl not found: {train_jsonl}")
        processed_train_dataset = load_dataset("json", data_files={"train": str(train_jsonl)}, split="train")

        if "val" in requested_splits:
            if val_set_size <= 0:
                raise ValueError("val_set_size must be > 0 when 'val' is requested.")
            if len(processed_train_dataset) <= 1:
                raise ValueError("Cannot split validation set from train: train size must be > 1.")

            actual_val_size = min(int(val_set_size), len(processed_train_dataset) - 1)
            split_result = processed_train_dataset.train_test_split(
                test_size=actual_val_size,
                shuffle=True,
                seed=seed,
            )
            val_dataset = split_result["test"]
            if "train" in requested_splits:
                train_dataset = split_result["train"]
        elif "train" in requested_splits:
            train_dataset = processed_train_dataset

    if "test" in requested_splits:
        if not test_jsonl.exists():
            raise FileNotFoundError(f"Kvasir test jsonl not found: {test_jsonl}")
        test_dataset = load_dataset("json", data_files={"test": str(test_jsonl)}, split="test")
    return train_dataset, val_dataset, test_dataset

def get_val_split_name(dataset_name: str) -> str:
    key = dataset_name.split("/")[-1].lower()
    if key not in SPLIT_NAME_MAP:
        raise ValueError(f"Dataset {dataset_name} not supported for split name mapping.")
    return SPLIT_NAME_MAP[key]["val"]


if __name__ == "__main__":
    # For quick testing
    dataset_id = "HuggingFaceM4/DocumentVQA" #cache/HuggingFaceM4_DocumentVQA
    train_ds, val_ds, test_ds = load_and_preprocess_dataset(dataset_id, splits=["train", "val", "test"], num_proc=8, rgb_convert=False)
    print(train_ds[0])