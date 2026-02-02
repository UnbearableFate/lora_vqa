import fire

from .evaluate_hf import evaluate_model_hf
from .training import train


class App:
    def train(self, **kwargs):
        return train(**kwargs)

    def evaluate(self, **kwargs):
        return evaluate_model_hf(**kwargs)

def main():
    fire.Fire(App)


if __name__ == "__main__":
    main()
