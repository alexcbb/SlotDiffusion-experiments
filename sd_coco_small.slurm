#!/bin/bash

#SBATCH --job-name=vq_vae_coco
#SBATCH --output=logs/coco_vq_vae.%j.out
#SBATCH --error=logs/coco_vq_vae.%j.err
#SBATCH --open-mode=append

#SBATCH -A uli@a100
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2 #number of MPI tasks per node (=number of GPUs per node)
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH -t 00:20:00
#SBATCH --mail-user=alexandre.chapin@ec-lyon.fr
#SBATCH --mail-typ=FAIL
#SBATCH --qos=qos_gpu-dev

echo ${SLURM_NODELIST}

module purge

module load cpuarch/amd # To be compatible with a100 nodes
module load pytorch-gpu/py3/1.11.0
#module load pytorch-gpu/py3/1.10.1

python -m torch.distributed.launch scripts/train.py --task img_based \
    --base_folder /gpfswork/rech/uli/ujf38zl/SlotDiffusion-experiments/ \
    --params slotdiffusion/img_based/configs/sa_ldm/sa_ldm_dino_coco_params-res224i-bs32.py \
    --fp16 --ddp --cudnn
