#!/bin/bash

# Default values
NUM_LAYERS=8
C_HIDDEN=256
LEARN_RATE=1e-3
MLP = 3

# Ablation: c_hidden
# for c_hidden in 16 32 64 128 256 512; do
#     exp_name="ablation_c_hidden_${c_hidden}"
#     echo "Running $exp_name"
#     python3 main.py train --exp_name="$exp_name" --c_hidden="$c_hidden" --num_layers="$NUM_LAYERS" --learn_rate="$LEARN_RATE"
# done

# Ablation: num_layers
# for num_layers in 3 4 5 6 7 8 9; do
#     exp_name="ablation_num_layers_${num_layers}"
#     echo "Running $exp_name"
#     python3 main.py train --exp_name="$exp_name" --c_hidden="$C_HIDDEN" --num_layers="$num_layers" --learn_rate="$LEARN_RATE"
# done

# Ablation: learn_rate
# for lr in 1e-4 1e-3 1e-2; do
#     exp_name="ablation_learn_rate_${lr}"
#     echo "Running $exp_name"
#     python3 main.py train --exp_name="$exp_name" --c_hidden="$C_HIDDEN" --num_layers="$NUM_LAYERS" --learn_rate="$lr"
# done

# ROUND 2

# Ablation: MLP
# for mlp in 1 3 5; do
#     exp_name="ablation_mlp_${mlp}"
#     echo "Running $exp_name"
#     python3 main.py train --exp_name="$exp_name" --c_hidden="$C_HIDDEN" --num_layers="$NUM_LAYERS" --learn_rate="$lr" --mlp=""
# done


