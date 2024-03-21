import os
import pickle
from datetime import datetime
import platform
import json
import tensorflow as tf


### Tensorflow Core
mixed_precision = False
if platform.system() != 'Darwin' and mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

### Use CPU
use_cpu = True
if use_cpu:
    tf.config.set_visible_devices([], 'GPU')

# """
#       _____   _                   _                _
#      |  __ \ (_)                 | |              (_)
#      | |  | | _  _ __  ___   ___ | |_  ___   _ __  _   ___  ___
#      | |  | || || '__|/ _ \ / __|| __|/ _ \ | '__|| | / _ \/ __|
#      | |__| || || |  |  __/| (__ | |_| (_) || |   | ||  __/\__ \
#      |_____/ |_||_|   \___| \___| \__|\___/ |_|   |_| \___||___/
# """

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.join(parent_dir, 'cser-nlp')
generations_dir = os.path.join(root_dir, 'generations')
prompts_dir = os.path.join(generations_dir, 'prompts')
outputs_dir = os.path.join(generations_dir, 'outputs')
preprocessing_dir = os.path.join(root_dir, 'preprocessing')
trained_models_dir = os.path.join(root_dir, 'trained_models')


# """
#   _____        _                 _
#  |  __ \      | |               | |
#  | |  | | __ _| |_ __ _ ___  ___| |_ ___
#  | |  | |/ _` | __/ _` / __|/ _ \ __/ __|
#  | |__| | (_| | || (_| \__ \  __/ |_\__ \
#  |_____/ \__,_|\__\__,_|___/\___|\__|___/
# """
datasets_dir = os.path.join(root_dir, 'datasets')


adcs_dataset = os.path.join(datasets_dir, 'adcs_dataset')
dataset_2 = os.path.join(datasets_dir, 'dataset_2')
dataset_3 = os.path.join(datasets_dir, 'dataset_3')


# --- Used Dataset ---
used_dataset = dataset_3




# """
#   _____                  _                               _
#  |  __ \                (_)                             | |
#  | |__) |___  __ _ _   _ _ _ __ ___ _ __ ___   ___ _ __ | |_ ___
#  |  _  // _ \/ _` | | | | | '__/ _ \ '_ ` _ \ / _ \ '_ \| __/ __|
#  | | \ \  __/ (_| | |_| | | | |  __/ | | | | |  __/ | | | |_\__ \
#  |_|  \_\___|\__, |\__,_|_|_|  \___|_| |_| |_|\___|_| |_|\__|___/
#                 | |
#                 |_|
# """

def get_subsystem_requirements(base_dir, file_name):
    file_path = os.path.join(base_dir, file_name)
    with open(file_path, 'r') as f:
        requirements = f.readlines()
    requirements = [req.strip() for req in requirements]
    requirements = list(set(requirements))
    if '' in requirements:
        requirements.remove('')
    return requirements

# --- Parse Requirements ---
req_load_dir = os.path.join(generations_dir, 'store4')
req_load = [
    'adcs_requirements.txt',
    'gs_requirements.txt',
    'comm_requirements.txt',
    'power_requirements.txt',
    'thermal_requirements.txt',
    'obdh_requirements.txt',
]
req_data = []
for req_p in req_load:
    req_data.extend(get_subsystem_requirements(req_load_dir, req_p))




# """
#       _______        _       _
#      |__   __|      (_)     (_)
#         | |_ __ __ _ _ _ __  _ _ __   __ _
#         | | '__/ _` | | '_ \| | '_ \ / _` |
#         | | | | (_| | | | | | | | | | (_| |
#         |_|_|  \__,_|_|_| |_|_|_| |_|\__, |
#                                       __/ |
#                                      |___/
# """

model_name = 'requirements_decoder'
model_path = os.path.join(trained_models_dir, model_name)

batch_size = 128



















