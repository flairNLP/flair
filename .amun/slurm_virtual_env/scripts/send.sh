#!/bin/bash
rsync -r --exclude "data" --exclude '.git' --exclude '.amun' --exclude 'checkpoints' --progress ./ tbergmann@slurm.zalando.net:flair-trainer