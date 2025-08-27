#!/bin/bash

echo "Testing logging functionality..."

# Test 1: With logging disabled (default)
echo "Test 1: Running with logging disabled (default behavior)"
python main.py --pretrain_dataset CiteSeer --test_dataset Cora --is_transfer True --seed 42 --num_epochs 3

echo ""
echo "============================================"
echo ""

# Test 2: With logging enabled
echo "Test 2: Running with logging enabled"
python main.py --pretrain_dataset CiteSeer --test_dataset Cora --is_transfer True --seed 42 --num_epochs 3 --use_logging True --log_dir ./logs

echo ""
echo "Checking if log file was created..."
ls -la ./logs/

echo ""
echo "If log file exists, showing content:"
if ls ./logs/*.log 1> /dev/null 2>&1; then
    echo "Log file content:"
    tail -n 20 ./logs/*.log
else
    echo "No log file found"
fi