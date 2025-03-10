#!/bin/sh
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=0:30:00
#PJM -g gb20
#PJM -j

module load gcc/8.3.1
module load python/3.10.13
source venv/bin/activate
source script/import-env.sh .env

# create cache directory
mkdir -p $HF_HOME $HF_DATASETS_CACHE

# create log directory
LOG_DIR="log/$(date '+%Y-%m-%d/%H-%M-%S')"
mkdir -p ${LOG_DIR}

# record job information
cat <<EOF >> ${LOG_DIR}/job_output.log
=== Job Information ===
Job ID: ${PJM_JOBID}
Job started at $(date)

=== NVIDIA-SMI Output ===
EOF
nvidia-smi >> ${LOG_DIR}/job_output.log

# run python script
cat <<EOF >> ${LOG_DIR}/job_output.log

=== Main Output ===
EOF
python3 src/main.py >> ${LOG_DIR}/job_output.log 2>&1

# record job information
cat <<EOF >> ${LOG_DIR}/job_output.log

=== Job Information ===
Job finished at $(date)
EOF
