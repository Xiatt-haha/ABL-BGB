#!/bin/bash

tasks=(
    "DocTamperV1-FCD 75"
    "DocTamperV1-FCD 80"
    "DocTamperV1-FCD 85"
    "DocTamperV1-FCD 90"
    "DocTamperV1-SCD 75"
    "DocTamperV1-SCD 80"
    "DocTamperV1-SCD 85"
    "DocTamperV1-SCD 90"
    "DocTamperV1-TestingSet 75"
    "DocTamperV1-TestingSet 80"
    "DocTamperV1-TestingSet 85"
    "DocTamperV1-TestingSet 90"
)

for task in "${tasks[@]}"
do
    lmdb_name=$(echo $task | cut -d' ' -f1)
    minq=$(echo $task | cut -d' ' -f2)

    cmd="CUDA_VISIBLE_DEVICES=0 python eval.py --lmdb_name $lmdb_name --pth outputs/BGDB/resnet18/ckpt/checkpoint-best.pth --minq $minq"

    echo "Running command: $cmd"
    eval $cmd

    echo "Task completed for $lmdb_name with minq=$minq"
    echo "======================================"
done
