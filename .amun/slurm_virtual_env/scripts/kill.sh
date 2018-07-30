#!/bin/bash
ssh slurm.zalando.net -l tbergmann scancel -n flair-trainer-${version}${job_id}
