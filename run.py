import logging
import json
import sys
import os
from utils.kernel_functions import *
from utils.forest_functions import *
from utils.knn_functions import *
from utils.etl import etl, etl_test

# setup logging
logging.basicConfig(filename='out/script_progress.log', level=logging.DEBUG, format = '%(message)s %(asctime)s', datefmt = '%I:%M:%S %p')

# capture target
try:
    target = sys.argv[1].lower()
except:
    raise ValueError('Arguments unknown, usage: run.py [target] [config json]')

if target == 'clean':
    built_files = [
        'out/results.json',
        'test/test_results.json',
        'out/script_progress.log'
    ]
    # clean results files
    for f in built_files:
        if os.path.exists(f):
            os.remove(f)
    # end script
    exit()
elif target == 'test':
    # uses test params
    with open('test/test_params.json', 'r') as f:
        script_params = json.load(f)
elif target == 'all':
    # grabs config file
    try:
        args = sys.argv[2].lower()
    except:
        raise ValueError('Configuration json required for target all, usage: run.py all [config json]')
    with open(args, 'r') as f:
        script_params = json.load(f)
else:
    raise ValueError('Target unknown, use "test" for test or use "all" for all')

# load params
num_train = int(script_params['num_train'])
num_test = int(script_params['num_test'])
corruption_types = script_params['corruption_types']
model_types = script_params['model_types']
powers = script_params['kernels']['powers']
forest_sizes = script_params['forests']['forest_sizes']
corruption_levels = script_params['corruption_levels']
filter_sizes = script_params['filter_sizes']
sigmas = script_params['sigmas']
neighbor_count = script_params['neighbor_count']
num_labels = 10

# load and setup data
if target == 'all':
    results, data = etl(num_train, num_test, model_types)
elif target == 'test':
    results, data = etl_test(num_train, num_test, model_types)
    
train_X, train_y, test_X, test_y = data

logging.debug(f'Finished setup at')

# run kernels
if 'kernel' in model_types:

    if 'label' in corruption_types:
        results['kernel']['label'] = run_label_corruption_kernels(train_X, train_y, test_X, test_y, 10, powers, corruption_levels)
        logging.debug(f'Finished kernels with label corruption at')
        
    if 'random' in corruption_types:
        results['kernel']['random'] = run_random_corruption_kernels(train_X, train_y, test_X, test_y, 10, powers, corruption_levels)
        logging.debug(f'Finished kernels with random corruption at')

    if 'gauss' in corruption_types:
        results['kernel']['gauss'] = run_gaussian_blur_kernels(train_X, train_y, test_X, test_y, 10, powers, filter_sizes, sigmas)
        logging.debug(f'Finished kernels with gaussian corruption at')

# run forests
if 'forest' in model_types:

    if 'label' in corruption_types:
        results['forest']['label'] = run_label_corruption_forests(train_X, train_y, test_X, test_y, 10, forest_sizes, corruption_levels)
        logging.debug(f'Finished forests with label corruption at')

    if 'random' in corruption_types:
        results['forest']['random'] = run_random_corruption_forests(train_X, train_y, test_X, test_y, 10, forest_sizes, corruption_levels)
        logging.debug(f'Finished forests with random corruption at')

    if 'gauss' in corruption_types:
        results['forest']['gauss'] = run_gaussian_blur_forests(train_X, train_y, test_X, test_y, 10, forest_sizes, filter_sizes, sigmas)
        logging.debug(f'Finished forests with gaussian corruption at')

# run knns
if 'knn' in model_types:

    if 'label' in corruption_types:
        results['knn']['label'] = run_label_corruption_knn(train_X, train_y, test_X, test_y, 10, neighbor_count, corruption_levels)
        logging.debug(f'Finished knns with label corruption at')

    if 'random' in corruption_types:
        results['knn']['random'] = run_random_corruption_knn(train_X, train_y, test_X, test_y, 10, neighbor_count, corruption_levels)
        logging.debug(f'Finished knns with random corruption at')

logging.debug(f'Finished at')
        
# write results
if target == 'all':
    with open('out/results.json', 'w') as f:
        json.dump(results, f)
elif target == 'test':
    with open('test/test_results.json', 'w') as f:
        json.dump(results, f)
