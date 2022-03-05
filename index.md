## Exploring Noise in Data

Robert Dunn, Amelia Kawasaki, Cheolmin Hwang

### Why do we care about noise in our data?

Noise level of training data is key to the performace of a prediction model. We intend to find out how much noise impacts the performance of prediction models in order to demonstrate the effects of overfitting a model.

### What is overfitting?

Overfitting is a concept in which a model is fit exactly to the data it was trained with resulting in poor performance on unseen examples of data. This is due to the fact that new examples of data will likely have slight variations that do not appear in the training set meaning that the model will not be able to detect such variations.

### Significance between noise and overfitting

By fitting models with a given amount of noise, we can view how different models perform. Based on the understanding of overfitting a model, it would be expected that fitting noisy data exactly would significantly decrease the performance of such a model, however, as shown below, it can be observed that this is not necessarily the case in many different type of models to a certain extent.

### Methodology

Used MNIST dataset and other image datasets.

Corrupted the datasets by randomizing a set proportion of their labels or the pixels of their images.

Tested training on multiple models: Gaussian/Laplacian kernel functions, K-Nearest Neighbor classifier, Random Forest Classifier, Neural Network.

### Results - Label Corruption

![kernel-label](/img/kernel-label.png | width = 40%)

![knn-label](/img/knn-label.png)

![forest-label](/img/forest-label.png)

![net-label](/img/net-label.png)

### Results - Random Corruption

![kernel-random](/img/kernel-random.png)

![knn-random](/img/knn-random.png)

![forest-random](/img/forest-random.png)

![net-random](/img/net-random.png)
