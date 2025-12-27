#!/bin/bash
# 用于自动多seed训练，结果分别输出到 outputs/exp4/seed1, seed2, seed3, seed4

CONFIG=config.yaml
OUT_BASE=outputs/exp4
ENV_PATH=/home/sunyidan/miniconda3/bin/activate
ENV_NAME=/home/sunyidan/miniconda3/envs/patchmil-gpu

for SEED in 1 2 3 4
  do
    OUT_DIR=${OUT_BASE}/seed${SEED}
    echo "==== 运行 seed ${SEED}，输出到 ${OUT_DIR} ===="
    bash -c "source $ENV_PATH $ENV_NAME && CUDA_VISIBLE_DEVICES=0,1 python train.py --config $CONFIG --out_dir $OUT_DIR --seed $SEED > ${OUT_DIR}/train.log 2>&1"
  done
