#!/bin/bash

SIM_NAME=$1
SIM_BASE=~/datasets/wlenet/simulation
JSON_PATH=~/datasets/wlenet/simulation_configs
EXEC_PATH=~/projects/wlenet/simulation/galsim_map_reduce.py
PYTHON=python3

${PYTHON} ${EXEC_PATH} ${JSON_PATH}/${SIM_NAME}_sim.json reduce \
       -o ${SIM_BASE}/${SIM_NAME}/data \
       -l ${SIM_BASE}/${SIM_NAME}/logs/reduce_test.stderr \
       test -j ${JSON_PATH}/${SIM_NAME}_map_reduce.json \

${PYTHON} ${EXEC_PATH} ${JSON_PATH}/${SIM_NAME}_sim.json reduce \
       -o ${SIM_BASE}/${SIM_NAME}/data \
       -l ${SIM_BASE}/${SIM_NAME}/logs/reduce_train.stderr \
       train -j ${JSON_PATH}/${SIM_NAME}_map_reduce.json \
