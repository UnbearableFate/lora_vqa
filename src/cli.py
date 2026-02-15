import fire

from .evaluate import evaluate
from .evaluate_by_llm import evaluate as evaluate_by_llm_vllm
from .evaluate_by_llm_transformers import evaluate as evaluate_by_llm_transformers
from .training import train


class App:
    def train(self, **kwargs):
        return train(**kwargs)

    def evaluate(self, **kwargs):
        return evaluate(**kwargs)

    def evaluate_by_llm_vllm(self, **kwargs):
        return evaluate_by_llm_vllm(**kwargs)

    def evaluate_by_llm_transformers(self, **kwargs):
        return evaluate_by_llm_transformers(**kwargs)


def main():
    fire.Fire(App)


if __name__ == "__main__":
    main()
