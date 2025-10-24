#!/bin/bash
#SBATCH --job-name=spt_train
#SBATCH --output=job_output/output/output_%j.txt
#SBATCH --error=job_output/error/error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1             
#SBATCH --time=1000:00:00
#SBATCH --partition=gpu
#SBATCH --mem=32G

mkdir -p job_output                   # 确保输出目录存在

echo "作业开始时间: $(date)"
echo "作业运行在节点: $(hostname)"
echo "作业分配的ID: $SLURM_JOB_ID"
echo "分配的GPU: $CUDA_VISIBLE_DEVICES"

# 移除不再需要的 conda init 和 activate
# conda init
# conda activate spt_py37
# echo "当前激活的Conda环境: $CONDA_DEFAULT_ENV"
# export LD_LIBRARY_PATH="/public/home/lxy/toolchain-dir/x86_64-linux-gnu/x86_64-linux-gnu/sysroot/lib:$LD_LIBRARY_PATH"
# echo "已设置 LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

cd /public/home/lxy/bao_3/UniMod1K/SPT_bk2    # 确认路径正确

echo "开始在 1 环境中运行Python脚本..."
# 使用 conda run 来确保在正确的环境中执行

export PYTHONPATH=${PYTHONPATH}:/public/home/lxy/bao_3/UniMod1K/SPT_bk2

conda run -n spt_py38_pth182 python ./lib/train/run_training.py  

echo "作业结束时间: $(date)"

