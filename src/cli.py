import fire

from .evaluate import evaluate
from .training import train


class App:
    def train(self, **kwargs):
        return train(**kwargs)

    def evaluate(self, **kwargs):
        return evaluate(**kwargs)


def main():
    fire.Fire(App)


if __name__ == "__main__":
    main()
