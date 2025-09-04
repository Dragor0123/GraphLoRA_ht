#!/bin/bash

# Usage: ./run_script.sh Cora Squirrel 42 46
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <pretrain_dataset> <test_dataset> <start_seed> <end_seed>"
    echo "Example: $0 Cora Squirrel 42 46"
    exit 1
fi

PRETRAIN_DATASET=$1
TEST_DATASET=$2
START_SEED=$3
END_SEED=$4

echo "Starting experiments: $PRETRAIN_DATASET -> $TEST_DATASET (seeds $START_SEED to $END_SEED)"
for seed in $(seq $START_SEED $END_SEED); do
    echo "Running with seed: $seed"
    python main.py --pretrain_dataset $PRETRAIN_DATASET --test_dataset $TEST_DATASET --is_transfer True --seed $seed --use_logging True --method fourier
    echo "Completed seed: $seed"
    echo "----------------------------------------"
done

echo "All experiments completed!"
