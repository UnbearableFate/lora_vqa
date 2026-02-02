from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List


_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(" + "|".join(_ARTICLES) + r")\b", " ", text)
    text = " ".join(text.split())
    return text


def _tokenize(text: str) -> List[str]:
    return normalize_answer(text).split()


def exact_match(prediction: str, answers: Iterable[str]) -> float:
    pred = normalize_answer(prediction)
    for ans in answers:
        if pred == normalize_answer(str(ans)):
            return 1.0
    return 0.0


def vqa_soft_accuracy(prediction: str, answers: Iterable[str]) -> float:
    pred = normalize_answer(prediction)
    normalized = [normalize_answer(str(ans)) for ans in answers if str(ans).strip()]
    if not normalized:
        return 0.0
    counts = Counter(normalized)
    if pred not in counts:
        return 0.0
    return min(counts[pred] / 3.0, 1.0)


def token_f1(prediction: str, answers: Iterable[str]) -> float:
    pred_tokens = _tokenize(prediction)
    if not pred_tokens:
        return 0.0
    best_f1 = 0.0
    for ans in answers:
        answer_tokens = _tokenize(str(ans))
        if not answer_tokens:
            continue
        common = Counter(pred_tokens) & Counter(answer_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(answer_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1
