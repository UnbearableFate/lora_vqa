from datasets import load_dataset
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
    "documentvqa" : {
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


def preprocess_fn(data_point, col_names):
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

def load_and_preprocess_dataset(dataset_id: str, subset_name: Optional[str] = None, splits: list = ["train", "val"] , num_proc: int = 16):
    dataset = load_dataset(dataset_id, subset_name)
    train_dataset, val_dataset, test_dataset = None, None, None
    col_names = get_column_names(dataset_id)
    if "train" in splits:
        train_columns = list(dataset["train"].column_names)
        if "images" in train_columns:
            train_columns.remove("images")
        train_dataset = dataset["train"].map(
            preprocess_fn,
            fn_kwargs={"col_names": col_names},
            remove_columns=train_columns,
            num_proc=num_proc,
        )
    if "val" in splits:
        val_split_name = get_val_split_name(dataset_id)
        val_columns = list(dataset[val_split_name].column_names)
        if "images" in val_columns:
            val_columns.remove("images")
        val_dataset = dataset[val_split_name].map(
            preprocess_fn,
            fn_kwargs={"col_names": col_names},
            remove_columns=val_columns,
            num_proc=num_proc,
        )
    if "test" in splits:
        test_columns = list(dataset["test"].column_names)
        if "images" in test_columns:
            test_columns.remove("images")
        test_dataset = dataset["test"].map(
            preprocess_fn,
            fn_kwargs={"col_names": col_names},
            remove_columns=test_columns,
            num_proc=num_proc,
        )
    return train_dataset, val_dataset, test_dataset

def get_val_split_name(dataset_name: str) -> str:
    key = dataset_name.split("/")[-1].lower()
    if key not in SPLIT_NAME_MAP:
        raise ValueError(f"Dataset {dataset_name} not supported for split name mapping.")
    return SPLIT_NAME_MAP[key]["val"]