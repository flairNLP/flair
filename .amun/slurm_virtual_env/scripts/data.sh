#!/bin/bash
rsync -r --ignore-existing ./data --progress tbergmann@slurm.zalando.net:~/flair-trainer/
