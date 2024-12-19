# Multi GPU

Training can be distributed across multiple GPUs on a local machine when using 
[`ModelTrainer`](#flair.trainers.trainer.ModelTrainer).

## Example

See the script `run_multi_gpu.py` and its comments.

## Tutorial

There are 2 changes that are always required, as well as a few things to consider

Always Required:
1) Pass the argument `multi_gpu=True` to your [`.train()`](#flair.trainers.trainer.ModelTrainer.train) or `.fine_tune()`
2) Wrap your code in [`launch_distributed`](#flair.distributed_utils.launch_distributed), e.g.
   `launch_distributed(main, *args)`. This spawns multiple processes, each driving a GPU

Other considerations:
- The corpus and other preprocessing must be the same on all processes. For example, if corpus initialization involves
  anything random, you should either 
  - Set the random seed before initializing the corpus (e.g. [`flair.set_seed(42)`) OR 
  - Initialize the corpus before calling `launch_distributed` and pass the corpus as an argument so it's serialized to
    all processes
- The effective batch size will be larger by a factor of num_gpus
  - Each GPU will now process `mini_batch_size` examples before the optimizer steps, resulting in fewer total steps
    taken relative to training with a single device. To obtain comparable results between single/multi gpu,
    both mathematically, and in terms of wall time, consider the method in the example script.
- Large batch sizes may be necessary to see faster runs, otherwise the communication overhead may dominate

Only the parameter updates in the training process will be distributed across multiple GPUs. Evaluation and prediction
are still done on a single device.
