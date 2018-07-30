#!/bin/bash
info=$(ssh slurm.zalando.net -l tbergmann squeue -n flair-trainer-${version}${job_id} | sed -n 2p)
IFS=' '
read -ra ADDR <<< "$info"
node=${ADDR[7]}
ssh slurm.zalando.net -l tbergmann srun -c1 -N1 -p gpu_v100 --mem-per-cpu=1000 -t 0-00:01 -w ${node} nvidia-smi -l 1
