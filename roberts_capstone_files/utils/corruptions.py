import numpy as np
import math



def label_randomizer(labels, proportion, num_labels = 10):
    
    # randomly choose indices to randomize
    sample = np.random.choice(len(labels), round(len(labels) * proportion), replace = False)
    
    # randomize labels
    for idx in sample:
        labels[idx] = np.random.choice(num_labels)
        
    return labels

def random_filter(data, proportion, min_val = 0, max_val = 255):
    data_shape = data[0].shape
    
    if len(data_shape) == 1:
        data_size = data_shape[0]
        sample_size = round(proportion * data_size)
        for x in data:
            sample = np.random.choice(data_size, sample_size, replace = False)
            for idx in sample:
                x[idx] = np.random.randint(min_val, max_val + 1)
        
    elif len(data_shape) == 2:
        data_size = data_shape[0] * data_shape[1]
        sample_size = round(proportion * data_size)
        for x in data:
            sample = np.random.choice(data_size, sample_size, replace = False)
            for idx in sample:
                x[idx // data_shape[1]][idx % data_shape[1]] = np.random.randint(min_val, max_val + 1)
                
    return data

def mean_filter(data, filter_size):
    img_shape = data[0].shape

    for x in data:
        pointer1 = 0
        while pointer1 < img_shape[0]:
            pointer2 = 0

            while pointer2 < img_shape[1]:
                x[pointer1:(pointer1 + filter_size), pointer2:(pointer2 + filter_size)] = \
                    np.mean(x[pointer1:(pointer1 + filter_size), pointer2:(pointer2 + filter_size)])

                pointer2 += filter_size
            pointer1 += filter_size
    
    return data

def median_filter(data, filter_size):
    img_shape = data[0].shape

    for x in data:
        pointer1 = 0
        while pointer1 < img_shape[0]:
            pointer2 = 0

            while pointer2 < img_shape[1]:
                x[pointer1:(pointer1 + filter_size), pointer2:(pointer2 + filter_size)] = \
                    np.median(x[pointer1:(pointer1 + filter_size), pointer2:(pointer2 + filter_size)])

                pointer2 += filter_size
            pointer1 += filter_size
    
    return data

def gaussian_filter(data, kernel_size, sigma = 1):
    # set constants
    center = (kernel_size - 1) // 2
    kernel = np.zeros((kernel_size, kernel_size))
    data_shape = data[0].shape
    
    # build kernel
    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel[x, y] = np.exp(-1 * (((x - center)**2 + (y - center)**2) / (2 * sigma**2)))
            
    kernel = (1 / sum(sum(kernel))) * kernel
    
    # apply filter to data
    new_data = []
    for d in data:
        new_array = np.zeros(data_shape)
        padded = np.pad(d, kernel_size // 2, 'edge')
        
        for x in range(data_shape[0]):
            for y in range(data_shape[1]):
                new_array[x, y] = sum(sum(kernel * padded[x:(x+kernel_size), y:(y+kernel_size)]))
                
        new_data.append(new_array.flatten())
    
    return np.array(new_data)