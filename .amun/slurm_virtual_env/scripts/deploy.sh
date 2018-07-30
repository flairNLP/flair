#!/bin/bash
echo "type is ${type_}"
if [[ ${type_} = gpu ]]
then
  command_=$(echo "cd flair-trainer && " \
    "srun -c4 -N1 -p gpu_v100 --mem-per-cpu=20000 --job-name flair-trainer-${version}${job_id} --gres=gpu:1 " \
    "${checkpoint}/scripts/virtual_run.sh > ${checkpoint}/log 2> ${checkpoint}/errors && " \
    "echo amun_finished >> ${checkpoint}/log &")
elif [[ ${type_} = cpu ]]
then
  command_=$(echo "cd flair-trainer && " \
    "srun -c1 -N1 -p cpu --mem-per-cpu=20000 --job-name flair-trainer-${version}${job_id} " \
    "${checkpoint}/scripts/virtual_run.sh > ${checkpoint}/log 2> ${checkpoint}/errors && " \
    "echo amun_finished >> ${checkpoint}/log &")
fi

echo ${command_}
ssh slurm.zalando.net -l tbergmann ${command_} &
