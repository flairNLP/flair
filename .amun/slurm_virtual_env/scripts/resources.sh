#!/bin/bash
x=$(ssh slurm.zalando.net -l tbergmann squeue | grep gpu_v100 | wc -l)
echo $x
