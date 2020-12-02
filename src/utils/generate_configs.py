"""
This file defines utilities for automatically generating configurations, for instance to do
hyperparameter search.
"""

import sys
import os.path as op

from pathlib import Path

sys.path.append('..')

# number of seeds
num_seeds = 3
SEEDS = list(range(num_seeds))

expe_name = 'wmg_hparam_search'
config_save_path = op.join('..', 'specs', expe_name)
Path(config_save_path).mkdir(exist_ok=True, parents=True)

ENV = 'PickUpLocNoMem'
MAX_NUMBER_STEPS = 500000
TEST_FREQ = 3000

# Variable arguments

LEARNING_RATES = [6.3e-5]
DISCOUNT_FACTORS = [0.5]
GRADIENT_CLIPS = [512]
ENTROPY_TERM_STRENGTHS = [0.1]
REWARD_SCALES = [32.]
WMG_MEMO_SIZES = [32]
WMG_MAX_MEMOSS = [8]
WMG_NUM_LAYERSS = [2, 4]
WMG_NUM_ATTENTION_HEADSS = [2, 4, 8]
WMG_ATTENTION_HEAD_SIZES = [64, 128]
WMG_HIDDEN_SIZESS = [32, 64, 128]
AC_HIDDEN_SIZESS = [2048]

i = 0

LEARNING_RATE = LEARNING_RATES[0]
DISCOUNT_FACTOR = DISCOUNT_FACTORS[0]
REWARD_SCALE = REWARD_SCALES[0]

for SEED in SEEDS:
    for WMG_ATTENTION_HEAD_SIZE in WMG_ATTENTION_HEAD_SIZES:
        for WMG_NUM_LAYERS in WMG_NUM_LAYERSS:
            for WMG_NUM_ATTENTION_HEADS in WMG_NUM_ATTENTION_HEADSS:
                for WMG_HIDDEN_SIZES in WMG_HIDDEN_SIZESS:
                    config_text = (
                        '# Copyright (c) Microsoft Corporation.\n'
                        '# Licensed under the MIT license.\n'
                        '\n'
                        '###  CONTROLS  (non-tunable)  ###\n'
                        '\n'
                        '# general\n'
                        'TYPE_OF_RUN = train  # train, test, test_episodes, render\n'
                        'LOAD_MODEL_FROM = None\n'
                        'SAVE_MODELS_TO = models / new_wmg_factored_babyai.pth\n'
                        '\n'
                        '# worker.py\n'
                        'ENV = BabyAI_Env\n'
                        f'ENV_RANDOM_SEED = {SEED}  # Use an integer for deterministic training.\n'
                        f'AGENT_RANDOM_SEED = {SEED}\n'
                        f'REPORTING_INTERVAL = {TEST_FREQ}\n'
                        f'TOTAL_STEPS = {MAX_NUMBER_STEPS}\n'
                        'ANNEAL_LR = False\n'
                        '\n'
                        '# A3cAgent\n'
                        'AGENT_NET = WMG_Network\n'
                        '\n'
                        '# WMG\n'
                        'V2 = False\n'
                        'VARIABLE_SIZE_MATRIX = False\n'
                        'AGE_INFO = PE  # possible values: NONE, PE, ONE_HOT\n'
                        'USE_INSTRUCTION = False\n'
                        '\n'
                        '# RMC\n'
                        'GATING_RMC = MEMO  # possible values: MEMO, FEATURE\n'
                        'AGGREG_TYPE = CORE  # possible values: CORE, MEAN\n'
                        '\n'
                        '# BabyAI_Env\n'
                        f'BABYAI_ENV_LEVEL = BabyAI-{ENV}-v0\n'
                        'USE_SUCCESS_RATE = True\n'
                        'SUCCESS_RATE_THRESHOLD = 0.99\n'
                        'HELDOUT_TESTING = True\n'
                        'NUM_TEST_EPISODES = 5000\n'
                        'OBS_ENCODER = Factored\n'
                        'BINARY_REWARD = True\n'
                        '\n'
                        '###  HYPERPARAMETERS  (tunable)  ###\n'
                        '\n'
                        '# A3cAgent\n'
                        'A3C_T_MAX = 6\n'
                        f'LEARNING_RATE = {LEARNING_RATE}\n'
                        f'DISCOUNT_FACTOR = {DISCOUNT_FACTOR}\n'
                        'GRADIENT_CLIP = 512.0\n'
                        'ENTROPY_TERM_STRENGTH = 0.1\n'
                        'ADAM_EPS = 1e-12\n'
                        f'REWARD_SCALE = {REWARD_SCALE}\n'
                        'WEIGHT_DECAY = 0.\n'
                        '\n'
                        '# WMG\n'
                        'WMG_MAX_OBS = 0\n'
                        'WMG_MAX_MEMOS = 8\n'
                        f'WMG_MEMO_SIZE = {WMG_MEMO_SIZES[0]}\n'
                        f'WMG_NUM_LAYERS = {WMG_NUM_LAYERS}\n'
                        f'WMG_NUM_ATTENTION_HEADS = {WMG_NUM_ATTENTION_HEADS}\n'
                        f'WMG_ATTENTION_HEAD_SIZE = {WMG_ATTENTION_HEAD_SIZE}\n'
                        f'WMG_HIDDEN_SIZE = {WMG_HIDDEN_SIZES}\n'
                        'AC_HIDDEN_LAYER_SIZE = 2048\n'
                    )
                    with open(op.join(config_save_path, f'config{i}'), 'w') as f:
                        f.write(config_text)

                    i += 1

# generate runfiles

runfile_path = op.join('..', 'runfiles', expe_name)

Path(runfile_path).mkdir(exist_ok=True, parents=True)
for idx in range(i):
    text = (
        '#!/bin/sh\n'
        '#SBATCH --mincpus 1\n'
        '#SBATCH -t 20:00:00\n'
        f'#SBATCH -o results/{expe_name}_log_config{idx}.out\n'
        f'#SBATCH -e results/{expe_name}_log_config{idx}.err\n'
        f'python main.py --experiment {expe_name} --config-id {idx}\n'
        'wait\n'
    )
    with open(op.join(runfile_path, f'run{idx}.sh'), 'w') as f:
        f.write(text)

# generate execute file

exec_path = op.join('..', 'execute_runfiles.sh', 'w')

text = (
    '#!/bin/bash\n'
    '\n'
    f'regex="runfiles/{expe_name}/run([0-9]+).sh"\n'
    '\n'
    f'for Script in runfiles/{expe_name}/*.sh ; do\n'
    '\n'
    '  [[ $Script =~ $regex ]]\n'
    '  scriptId="${BASH_REMATCH[1]}"\n'
    '  if [ -n "$scriptId" ] ; then\n'
    '    if  [[ "$scriptId" -gt -1 ]] ; then\n'
    '      echo "running script number $scriptId"\n'
    '      sbatch "$Script" &\n'
    '    fi\n'
    '  fi\n'
    '\n'
    'done\n'
)

with open(exec_path, 'w') as f:
    f.write(text)