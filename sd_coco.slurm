#!/bin/bash

#SBATCH --job-name=vq_vae_coco
#SBATCH --output=logs/coco_vq_vae.%j.out
#SBATCH --error=logs/coco_vq_vae.%j.err
#SBATCH --open-mode=append

#SBATCH -A uli@v100
#SBATCH -C v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2 #number of MPI tasks per node (=number of GPUs per node)
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH -t 20:00:00
#SBATCH --mail-user=alexandre.chapin@ec-lyon.fr
#SBATCH --mail-typ=FAIL
#SBATCH --qos=qos_gpu-t3

python --version >> logs/coco_vq_vae.%j.out                      # log Python version
gcc --version >> logs/coco_vq_vae.%j.out                           # log GCC version
nvcc --version >> logs/coco_vq_vae.%j.out                         # log NVCC version

echo ${SLURM_NODELIST}

module purge

#module load cpuarch/amd # To be compatible with a100 nodes
module load pytorch-gpu/py3/1.10.1

python -m torch.distributed.launch --nproc_per_node=2 \
    scripts/train.py --task img_based \
    --base_folder /gpfswork/rech/uli/ujf38zl/SlotDiffusion-experiments/ \
    --params slotdiffusion/img_based/configs/sa_ldm/sa_ldm_dino_coco_params-res224.py \
    --fp16 --ddp --cudnn --local_rank
