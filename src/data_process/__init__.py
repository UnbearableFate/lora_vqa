from datasets import load_dataset
from functools import partial
import re

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


def _strip_inline_image_tokens(text: str) -> str:
    if text is None:
        return ""
    # Prevent duplicated image placeholders when dataset text already contains image tags.
    text = re.sub(r"<\s*image\s*>", "", text, flags=re.IGNORECASE)
    return text.strip()


def preprocess_fn(data_point, dataset_name: str):
    """
    Input columns:
      - example["image"]: PIL.Image / image path / dataset Image object (depending on your dataset)
      - example["question"]: str
      - example["answer"]: str

    Output:
      - "messages": list of chat messages with typed content (image + text)
    """

    col_names = get_column_names(dataset_name)
    
    question_text = _strip_inline_image_tokens(data_point[col_names["question"]])
    
    if isinstance(data_point[col_names["answer"]], list):
        answer_text = str(data_point[col_names["answer"]][0])
    else:
        answer_text = str(data_point[col_names["answer"]])

    return {
        "images": _normalize_images(data_point[col_names["images"]]),
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

def load_and_preprocess_dataset(dataset_id: str, subset_name: Optional[str] = None):
    dataset = load_dataset(dataset_id, subset_name)
    remove_columns = dataset["train"].column_names
    print("Dataset columns:", remove_columns)
    if "images" in remove_columns:
        remove_columns.remove("images")
    dataset = dataset.map(
        partial(preprocess_fn, dataset_name=dataset_id),
        remove_columns=remove_columns,
        num_proc=8,
        load_from_cache_file=False,
    )
    return dataset

def get_val_split_name(dataset_name: str) -> str:
    key = dataset_name.split("/")[-1].lower()
    if key not in SPLIT_NAME_MAP:
        raise ValueError(f"Dataset {dataset_name} not supported for split name mapping.")
    return SPLIT_NAME_MAP[key]["val"]