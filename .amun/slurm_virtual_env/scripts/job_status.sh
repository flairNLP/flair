#!/usr/bin/env bash
LINES=$(ssh slurm.zalando.net -l tbergmann squeue -n flair-trainer-${version}${job_id} | wc -l)

if [ $LINES = 2 ]
then
    echo "running"
else
  echo "done"
fi
