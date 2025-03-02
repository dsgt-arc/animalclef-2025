#!/bin/bash
#SBATCH --job-name=animalclef-preprocess --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --cpus-per-task=16 --mem-per-cpu=4G
#SBATCH --time=2:00:00 --qos=embers
#SBATCH --output=Report-%j.log --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu

echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
export NO_REINSTALL=1
source ~/clef/animalclef-2025/scripts/activate

set -xe

# 16 * 4G = 64G
export PYSPARK_EXECUTOR_CORES=16
export PYSPARK_DRIVER_MEMORY=60g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp
scratch_dir=$(realpath ~/scratch/animalclef)
project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/animalclef
animalclef preprocess parquet $scratch_dir $project_data_dir/data
