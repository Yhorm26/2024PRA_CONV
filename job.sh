#!/bin/bash
#SBATCH -J JOB
#SBATCH -p ty_xd
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=dcu:1
#SBATCH --mem=90G

module list

module purge
module load compiler/dtk/24.04

module list

make clean
make

./conv2ddemo 64 256 14 14 256 3 3 1 1 1 1