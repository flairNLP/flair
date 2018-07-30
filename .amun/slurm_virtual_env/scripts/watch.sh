#!/bin/bash
ssh slurm.zalando.net -l tbergmann tail -f flair-trainer/${checkpoint}/log
