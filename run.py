import json
import sys

from keras.datasets import mnist

from utils.kernel_functions import *
from utils.forest_functions import *
from utils.knn_functions import *

# capture target (not yet functional)
try:
    target = sys.argv[1].lower()
except:
    raise ValueError('Arguments unknown, usage: run.py [target] [config json]')

if target == 'test':
    with open('test/test_params.json', 'r') as f:
        script_params = json.load(f)
elif target == 'all':
    try:
        args = sys.argv[2].lower()
    except:
        raise ValueError('Configuration json required for target all, usage: run.py all [config json]')
    with open(args, 'r') as f:
        script_params = json.load(f)
else:
    raise ValueError('Target unknown, use "test" for test or use "all" for all')

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
    
# Collect data
(train_X, train_y), (test_X, test_y) = mnist.load_data()
    
# setup data
train_X = train_X[:num_train].astype(int)
train_y = train_y[:num_train].astype(int)
test_X = test_X[:num_test].astype(int)
test_y = test_y[:num_test].astype(int)
    
# setup output
results = {}
for model in model_types:
    results[model] = {}

if 'kernel' in model_types:
    
    if 'label' in corruption_types:
        results['kernel']['label'] = run_label_corruption_kernels(train_X, train_y, test_X, test_y, 10, powers, corruption_levels)
        
    if 'random' in corruption_types:
        results['kernel']['random'] = run_random_corruption_kernels(train_X, train_y, test_X, test_y, 10, powers, corruption_levels)
        
    if 'gauss' in corruption_types:
        results['kernel']['gauss'] = run_gaussian_blur_kernels(train_X, train_y, test_X, test_y, 10, powers, filter_sizes, sigmas)
        
if 'forest' in model_types:
    
    if 'label' in corruption_types:
        results['forest']['label'] = run_label_corruption_forests(train_X, train_y, test_X, test_y, 10, forest_sizes, corruption_levels)
        
    if 'random' in corruption_types:
        results['forest']['random'] = run_random_corruption_forests(train_X, train_y, test_X, test_y, 10, forest_sizes, corruption_levels)
        
    if 'gauss' in corruption_types:
        results['forest']['gauss'] = run_gaussian_blur_forests(train_X, train_y, test_X, test_y, 10, forest_sizes, filter_sizes, sigmas)

if 'knn' in model_types:

    if 'label' in corruption_types:
        results['knn']['label'] = run_label_corruption_knn(train_X, train_y, test_X, test_y, 10, neighbor_count, corruption_levels)

    if 'random' in corruption_types:
        results['knn']['random'] = run_random_corruption_knn(train_X, train_y, test_X, test_y, 10, neighbor_count, corruption_levels)

with open('out/results.json', 'w') as f:
    json.dump(results, f)