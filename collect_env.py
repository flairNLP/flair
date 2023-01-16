import torch
import transformers

import flair


def main():
    print("#### Versions:")
    print(f"##### Flair\n{flair.__version__}")
    print(f"##### Pytorch\n{torch.__version__}")
    print(f"##### Transformers\n{transformers.__version__}")
    print(f"#### GPU\n{torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
