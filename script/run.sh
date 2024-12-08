#!/bin/sh
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=0:10:00
#PJM -g gb20
#PJM -j

module load gcc/8.3.1
module load python/3.10.13
source venv/bin/activate
python src/sample.py