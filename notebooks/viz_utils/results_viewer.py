import matplotlib.pyplot as plt
import numpy as np
import json
import os

def display_results(model_type = 'all', corruption_type = 'all', save_figs = False):
    
    # automatically collect results
    if os.path.exists('../out/results.json'):
        with open('../out/results.json', 'r') as f:
            results = json.load(f)

        with open('../utils/params.json', 'r') as f:
            params = json.load(f)

        print('Full results have been loaded')

    elif os.path.exists('../test/test_results.json'):
        with open('../test/test_results.json', 'r') as f:
            results = json.load(f)

        with open('../test/test_params.json', 'r') as f:
            params = json.load(f)

        print('Test results have been loaded.')

    else:
        print('No results have been generated. In the main directory, use the console commands `python run.py` to generate full results or `python run.py test` to generate test results.')
        return
    
    # generate plots
    if (model_type == 'kernel' or model_type == 'all') and 'kernel' in params['model_types']:
        
        if (corruption_type == 'label' or corruption_type == 'all') and 'label' in params['corruption_types']:
            fig, axes = plt.subplots(figsize = (12, 6))
            legend = []
            max_val = 0
            min_val = 1
            for key in results['kernel']['label']:
                axes.plot(params['corruption_levels'], results['kernel']['label'][key])
                legend.append(f'power = {key}')
                cur_min = np.min(results['kernel']['label'][key])
                cur_max = np.max(results['kernel']['label'][key])
                if cur_min < min_val:
                    min_val = cur_min
                if cur_max > max_val:
                    max_val = cur_max
            axes.legend(legend)
            axes.set_xticks(params['corruption_levels'])
            axes.set_yticks(np.array(range(int(round(min_val, 1) * 10), int(round(max_val + 0.1, 1) * 10))) / 10)
            axes.set_xlabel('Proportion label proportion')
            axes.set_ylabel('Accuracy')
            axes.set_title('Kernels with label corruption')
            if save_figs:
                fig.savefig('figs/kernel-label.png', bbox_inches='tight')
            fig.show()
            
        if (corruption_type == 'random' or corruption_type == 'all') and 'random' in params['corruption_types']:
            fig, axes = plt.subplots(figsize = (12, 6))
            legend = []
            max_val = 0
            min_val = 1
            for key in results['kernel']['random']:
                axes.plot(params['corruption_levels'], results['kernel']['random'][key])
                legend.append(f'power = {key}')
                cur_min = np.min(results['kernel']['random'][key])
                cur_max = np.max(results['kernel']['random'][key])
                if cur_min < min_val:
                    min_val = cur_min
                if cur_max > max_val:
                    max_val = cur_max
            axes.legend(legend)
            axes.set_xticks(params['corruption_levels'])
            axes.set_yticks(np.array(range(int(round(min_val, 1) * 10), int(round(max_val + 0.1, 1) * 10))) / 10)
            axes.set_xlabel('Proportion data proportion')
            axes.set_ylabel('Accuracy')
            axes.set_title('Kernels with random data corruption')
            if save_figs:
                fig.savefig('figs/kernel-random.png', bbox_inches='tight')
            fig.show()
            
    if (model_type == 'forest' or model_type == 'all') and 'forest' in params['model_types']:
        
        if (corruption_type == 'label' or corruption_type == 'all') and 'label' in params['corruption_types']:
            fig, axes = plt.subplots(figsize = (12, 6))
            legend = []
            max_val = 0
            min_val = 1
            for key in results['forest']['label']:
                min_acc = np.min(results['forest']['label'][key], axis = 0)
                mean_acc = np.mean(results['forest']['label'][key], axis = 0)
                max_acc = np.max(results['forest']['label'][key], axis = 0)
                axes.plot(params['corruption_levels'], mean_acc)
                axes.errorbar(params['corruption_levels'], mean_acc, [mean_acc - min_acc, max_acc - mean_acc], capsize = 3, c = axes.lines[-1].get_color())
                legend.append(f'forest size = {key}')
                cur_min = np.min(results['forest']['label'][key])
                cur_max = np.max(results['forest']['label'][key])
                if cur_min < min_val:
                    min_val = cur_min
                if cur_max > max_val:
                    max_val = cur_max
            axes.legend(legend)
            axes.set_xticks(params['corruption_levels'])
            axes.set_yticks(np.array(range(int(round(min_val, 1) * 10), int(round(max_val + 0.1, 1) * 10))) / 10)
            axes.set_xlabel('Proportion label proportion')
            axes.set_ylabel('Accuracy')
            axes.set_title('Random Forests with label corruption')
            if save_figs:
                fig.savefig('figs/forest-label.png', bbox_inches='tight')
            fig.show()
            
        if (corruption_type == 'random' or corruption_type == 'all') and 'random' in params['corruption_types']:
            fig, axes = plt.subplots(figsize = (12, 6))
            legend = []
            max_val = 0
            min_val = 1
            for key in results['forest']['random']:
                min_acc = np.min(results['forest']['random'][key], axis = 0)
                mean_acc = np.mean(results['forest']['random'][key], axis = 0)
                max_acc = np.max(results['forest']['random'][key], axis = 0)
                axes.plot(params['corruption_levels'], mean_acc)
                axes.errorbar(params['corruption_levels'], mean_acc, [mean_acc - min_acc, max_acc - mean_acc], capsize = 3, c = axes.lines[-1].get_color())
                legend.append(f'forest size = {key}')
                cur_min = np.min(results['forest']['random'][key])
                cur_max = np.max(results['forest']['random'][key])
                if cur_min < min_val:
                    min_val = cur_min
                if cur_max > max_val:
                    max_val = cur_max
            axes.legend(legend)
            axes.set_xticks(params['corruption_levels'])
            axes.set_yticks(np.array(range(int(round(min_val, 1) * 10), int(round(max_val + 0.1, 1) * 10))) / 10)
            axes.set_xlabel('Proportion data proportion')
            axes.set_ylabel('Accuracy')
            axes.set_title('Random Forests with random data corruption')
            if save_figs:
                fig.savefig('figs/forest-random.png', bbox_inches='tight')
            fig.show()
            
    if (model_type == 'knn' or model_type == 'all') and 'knn' in params['model_types']:
        
        if (corruption_type == 'label' or corruption_type == 'all') and 'label' in params['corruption_types']:
            fig, axes = plt.subplots(figsize = (12, 6))
            legend = []
            max_val = 0
            min_val = 1
            for key in results['knn']['label']:
                axes.plot(params['corruption_levels'], results['knn']['label'][key])
                legend.append(f'num neighbors = {key}')
                cur_min = np.min(results['knn']['label'][key])
                cur_max = np.max(results['knn']['label'][key])
                if cur_min < min_val:
                    min_val = cur_min
                if cur_max > max_val:
                    max_val = cur_max
            axes.legend(legend)
            axes.set_xticks(params['corruption_levels'])
            axes.set_yticks(np.array(range(int(round(min_val, 1) * 10), int(round(max_val + 0.1, 1) * 10))) / 10)
            axes.set_xlabel('Proportion label proportion')
            axes.set_ylabel('Accuracy')
            axes.set_title('K-Nearest Neighbors with label corruption')
            if save_figs:
                fig.savefig('figs/knn-label.png', bbox_inches='tight')
            fig.show()
            
        if (corruption_type == 'random' or corruption_type == 'all') and 'random' in params['corruption_types']:
            fig, axes = plt.subplots(figsize = (12, 6))
            legend = []
            max_val = 0
            min_val = 1
            for key in results['knn']['random']:
                axes.plot(params['corruption_levels'], results['knn']['random'][key])
                legend.append(f'num neighbors = {key}')
                cur_min = np.min(results['knn']['random'][key])
                cur_max = np.max(results['knn']['random'][key])
                if cur_min < min_val:
                    min_val = cur_min
                if cur_max > max_val:
                    max_val = cur_max
            axes.legend(legend)
            axes.set_xticks(params['corruption_levels'])
            axes.set_yticks(np.array(range(int(round(min_val, 1) * 10), int(round(max_val + 0.1, 1) * 10))) / 10)
            axes.set_xlabel('Proportion data proportion')
            axes.set_ylabel('Accuracy')
            axes.set_title('K-Nearest Neighbors with random data corruption')
            if save_figs:
                fig.savefig('figs/knn-random.png', bbox_inches='tight')
            fig.show()
            
    if (model_type == 'nets' or model_type == 'all') and 'nets' in params['model_types']:
        
        if (corruption_type == 'label' or corruption_type == 'all') and 'label' in params['corruption_types']:
            fig, axes = plt.subplots(figsize = (12, 6))
            legend = []
            max_val = 0
            min_val = 1
            for key in results['nets']['label']:
                min_acc = np.min(results['nets']['label'][key], axis = 0)
                mean_acc = np.mean(results['nets']['label'][key], axis = 0)
                max_acc = np.max(results['nets']['label'][key], axis = 0)
                axes.plot(params['corruption_levels'], mean_acc)
                axes.errorbar(params['corruption_levels'], mean_acc, [mean_acc - min_acc, max_acc - mean_acc], capsize = 3, c = axes.lines[-1].get_color())
                legend.append(f'Trained for {key} epochs')
                cur_min = np.min(results['nets']['label'][key])
                cur_max = np.max(results['nets']['label'][key])
                if cur_min < min_val:
                    min_val = cur_min
                if cur_max > max_val:
                    max_val = cur_max
            axes.legend(legend)
            axes.set_xticks(params['corruption_levels'])
            axes.set_yticks(np.array(range(int(round(min_val, 1) * 10), int(round(max_val + 0.1, 1) * 10))) / 10)
            axes.set_xlabel('Proportion label proportion')
            axes.set_ylabel('Accuracy')
            axes.set_title('Convolutional Neural Networks with label corruption')
            if save_figs:
                fig.savefig('figs/net-label.png', bbox_inches='tight')
            fig.show()
            
        if (corruption_type == 'random' or corruption_type == 'all') and 'random' in params['corruption_types']:
            fig, axes = plt.subplots(figsize = (12, 6))
            legend = []
            max_val = 0
            min_val = 1
            for key in results['nets']['random']:
                min_acc = np.min(results['nets']['random'][key], axis = 0)
                mean_acc = np.mean(results['nets']['random'][key], axis = 0)
                max_acc = np.max(results['nets']['random'][key], axis = 0)
                axes.plot(params['corruption_levels'], mean_acc)
                axes.errorbar(params['corruption_levels'], mean_acc, [mean_acc - min_acc, max_acc - mean_acc], capsize = 3, c = axes.lines[-1].get_color())
                legend.append(f'Trained for {key} epochs')
                cur_min = np.min(results['nets']['random'][key])
                cur_max = np.max(results['nets']['random'][key])
                if cur_min < min_val:
                    min_val = cur_min
                if cur_max > max_val:
                    max_val = cur_max
            axes.legend(legend)
            axes.set_xticks(params['corruption_levels'])
            axes.set_yticks(np.array(range(int(round(min_val, 1) * 10), int(round(max_val + 0.1, 1) * 10))) / 10)
            axes.set_xlabel('Proportion data proportion')
            axes.set_ylabel('Accuracy')
            axes.set_title('Convolutional Neural Networks with random data corruption')
            if save_figs:
                fig.savefig('figs/net-random.png', bbox_inches='tight')
            fig.show()
