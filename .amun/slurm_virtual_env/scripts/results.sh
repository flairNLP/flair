#!/bin/bash
rsync -r tbergmann@slurm.zalando.net:flair-trainer/${checkpoint}/* ${checkpoint}
