#!/bin/bash
#
# Usage:
#   1) 直接测试: sbatch train_slurm.sh
#   2) 自定义命令: sbatch --export=ALL,RUN_CMD='python your_script.py --arg x' train_slurm.sh
#   2) chmod +x train_slurm.sh
#   3) sbatch train_slurm.sh
#
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=02:00:00
#SBATCH --job-name=preprocess_cpu
#SBATCH --output=logs/slurm%j.log
#SBATCH --error=logs/slurm%j.err

set -euo pipefail

# 让脚本从你提交作业时所在目录运行
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "===== Slurm Job Info ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "Start time: $(date)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-1}"
[ -f main.py ] || { echo "ERROR: main.py not found in $(pwd)"; exit 1; }

# 你指定使用的容器镜像
CONTAINER="/data/common/images/codedev_v1.0.5.sif"

# 默认运行命令：调用仓库根目录下的 main.py 做批量预处理测试（无缓冲输出）
RUN_CMD="${RUN_CMD:-python -u main.py}"
export RUN_CMD
export PREPROCESS_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# 通过 srun + singularity 启动训练（CPU 预处理）
srun --cpu-bind=none singularity exec --pwd "$(pwd)" "${CONTAINER}" /bin/bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${PREPROCESS_THREADS}"
export OPENBLAS_NUM_THREADS="${PREPROCESS_THREADS}"
export MKL_NUM_THREADS="${PREPROCESS_THREADS}"
export NUMEXPR_NUM_THREADS="${PREPROCESS_THREADS}"
export VECLIB_MAXIMUM_THREADS="${PREPROCESS_THREADS}"
echo "Inside container on $(hostname)"
echo "Container: ${SINGULARITY_CONTAINER:-unknown}"
echo "Threads: ${PREPROCESS_THREADS}"
python -c "import os; print(\"python:\", os.sys.executable); print(\"cwd:\", os.getcwd())"
echo "Run command: ${RUN_CMD}"
eval "${RUN_CMD}"
'

echo "End time: $(date)"
