#!/bin/bash

## SLURM script for running distributed training with wav2letter
## %j is the job id, %u is the user id
#SBATCH --job-name=librivox-vad
#SBATCH --output=/checkpoint/jacobkahn/jobs/wav2letter/librivox/vad/wav2letter-vad-%j.out
#SBATCH --error=/checkpoint/jacobkahn/jobs/wav2letter/librivox/vad/wav2letter-vad-%j.err
#SBATCH --nodes=1
#SBATCH --partition=priority
#SBATCH --comment="ICASSP Deadline 10/21"
#SBATCH --gres=gpu:volta:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 8
#SBATCH --time=12:00:00

source /private/home/jacobkahn/fb-infra-benchmarking/modules

if [ -z "$1" ]
then
    echo "No lst file given as argument with which to start job"
    exit 1
fi

echo "Starting job for audio analysis on $1 with SLURM job ID ${SLURM_JOB_ID}"

/private/home/jacobkahn/wav2letter-clone/wav2letter/build/AudioAnalysis \
    -am /checkpoint/jacobkahn/librivox/vad/vad_model.bin \
    -lm /checkpoint/jacobkahn/librivox/vad/lm-4g.bin \
    -test /checkpoint/mriviere/LibriFrench_wav/vad/vad_input.lst  \
    --lexicon=/checkpoint/jacobkahn/librivox/vad/dict.lst \
    -maxload -1 \
    -sclite /checkpoint/mriviere/LibriFrench_wav/vad \
    --datadir= /checkpoint/mriviere/LibriFrench_final \
    --tokensdir=/checkpoint/jacobkahn/librivox/vad \
    --tokens english-train-all-unigram-5000.vocab-filtered \
    --show
