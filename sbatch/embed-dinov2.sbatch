#!/bin/bash
#SBATCH --job-name=animalclef-preprocess --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --gres=gpu:V100:1 --cpus-per-task=6 --mem-per-gpu=64G
#SBATCH --time=2:00:00 --qos=embers
#SBATCH --output=Report-%j.log --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu
#SBATCH --array=0-0%5

echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"
nvidia-smi

# activate the environment
export NO_REINSTALL=1
source ~/clef/animalclef-2025/scripts/activate

set -xe

NVIDIA_LOG_FILE=${SLURM_SUBMIT_DIR}/Report-${SLURM_JOB_ID}-nvidia-logs.ndjson
nvidia-logs monitor $NVIDIA_LOG_FILE --interval 15 &
nvidia_logs_pid=$!

export PYSPARK_EXECUTOR_CORES=8
export PYSPARK_DRIVER_MEMORY=30g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp
scratch_dir=$(realpath ~/scratch/animalclef)
project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/animalclef

TEST_MODE=${TEST_MODE:-0}
if [ $TEST_MODE -eq 1 ]; then
    output_dir=$scratch_dir/tmp/data/embeddings/dinov2/$SLURM_JOB_ID
else
    output_dir=$project_data/dir/data/embeddings/dinov2
fi

export PYSPARK_EXECUTOR_CORES=6
export PYSPARK_DRIVER_MEMORY=60g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp
animalclef embed dinov2 \
    $project_dir/data/parquet/images \
    $output_dir \
    --num-sample-ids 20 \
    --sample-id $SLURM_ARRAY_TASK_ID

kill $nvidia_logs_pid
nvidia-logs parse $NVIDIA_LOG_FILE
