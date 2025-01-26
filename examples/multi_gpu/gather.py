import random

from flair.distributed_utils import aggregate, flatten, is_main_process, launch_distributed


def main():

    x = [random.randint(0, 100) for _ in range(5)]

    agg = aggregate(x, flatten)
    if is_main_process():
        print(f"Aggregate: {agg}")


if __name__ == "__main__":
    """Minimal example demonstrating how to aggregate items from multiple GPUs."""
    multi = True
    if multi:
        launch_distributed(main)
    else:
        main(n)
