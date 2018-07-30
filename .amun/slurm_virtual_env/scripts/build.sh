#!/bin/bash
grep -v '^tensorflow' requirements.txt > new_requirements_bar.txt
grep -v '^torch' new_requirements_bar.txt > new_requirements.txt
echo torchvision >> new_requirements.txt
rm new_requirements_bar.txt
echo tensorflow-gpu >> new_requirements.txt

pv=$(echo 3.6 | tr -d .)

if [ ${type_} = torch_legacy ]
then
  echo http://download.pytorch.org/whl/cu90/torch-0.3.1-cp${pv}-cp${pv}m-linux_x86_64.whl >> new_requirements.txt
else
  echo http://download.pytorch.org/whl/cu90/torch-0.4.0-cp${pv}-cp${pv}m-linux_x86_64.whl >> new_requirements.txt
fi

scp new_requirements.txt tbergmann@slurm.zalando.net:flair-trainer/
if [ ${type_} = gpu ]
then
  ssh slurm.zalando.net -l tbergmann srun -c1 -N1 -p gpu_v100 --mem-per-cpu=10000 --gres=gpu:1 .environments/flair-trainer/bin/pip install -r flair-trainer/new_requirements.txt
elif [ ${type} = cpu ]
then
  ssh slurm.zalando.net -l tbergmann .environments/flair-trainer/bin/pip install -r flair-trainer/requirements.txt
fi

rm new_requirements.txt
