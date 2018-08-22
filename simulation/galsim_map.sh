#!/bin/bash

SIM_NAME=$1
SIM_BASE=~/datasets/wlenet/simulation
JSON_PATH=~/datasets/wlenet/simulation_configs
EXEC_PATH=~/projects/wlenet/simulation/galsim_map_reduce.py
SUBMIT=slurm_submit.py
#SUBMIT=screen_submit.py
PYTHON=python3
MEM=16000m
TIME=12

mkdir ${SIM_BASE}/${SIM_NAME}/data -p
mkdir ${SIM_BASE}/${SIM_NAME}/logs -p
mkdir ${SIM_BASE}/${SIM_NAME}/configs -p
cp ${JSON_PATH}/${SIM_NAME}_*.json ${SIM_BASE}/${SIM_NAME}/configs

${PYTHON} ${EXEC_PATH} ${JSON_PATH}/${SIM_NAME}_sim.json map \
       test -p print -j ${JSON_PATH}/${SIM_NAME}_map_reduce.json \
       -o ${SIM_BASE}/${SIM_NAME}/data \
       -l ${SIM_BASE}/${SIM_NAME}/logs/map_test.stderr \
       | ${SUBMIT} -m $MEM -o ${SIM_BASE}/${SIM_NAME}/logs/slurm_test -t $TIME \
       | tee ${SIM_BASE}/${SIM_NAME}/logs/slurm_submit_test.stdout

${PYTHON} ${EXEC_PATH} ${JSON_PATH}/${SIM_NAME}_sim.json map \
       train -p print -j ${JSON_PATH}/${SIM_NAME}_map_reduce.json \
       -o ${SIM_BASE}/${SIM_NAME}/data \
       -l ${SIM_BASE}/${SIM_NAME}/logs/map_train.stderr \
       | ${SUBMIT} -m $MEM -o ${SIM_BASE}/${SIM_NAME}/logs/slurm_train -t $TIME \
       | tee ${SIM_BASE}/${SIM_NAME}/logs/slurm_submit_train.stdout
