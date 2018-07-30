#!/bin/bash
rsync -r --relative --progress ./${checkpoint} tbergmann@slurm.zalando.net:flair-trainer
