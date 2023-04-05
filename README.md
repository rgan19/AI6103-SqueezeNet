## SqueezeNet: PyTorch Implementation, Improvement and Exploration

### Description
This is a PyTorch implementation of [SqueezeNet](https://arxiv.org/abs/1602.07360) model, a lightweight CNN architecture that achieves higher accuracy compared to traditional CNN, proposed by Forrest N. Iandola.

To improve the performance of SqueezeNet, we experimented with different techniques such as adding batch normalization and skip connections.

This is tested on CIFAR-100.

In this repository, it contains the following:

1. Base model (original)
    
    The base model follows the implementation in the paper. It consists of 8 fire modules, each with a squeeze module followed by an expand module. The network has been modified to take in CIFAR-100 dataset and outputs a 100-dimension probaility vector.

2. Batch normalisation

    Batch normalisation is a technique that helps to speed up training and improve generalization. In this implementation, batch normalisation is added after each convolutional layer in the squeeze and expand modules.


3. Skip connection

    Skip connections have been shown to improve the performance of deep neural networks by allowing information to bypass certain layers.



### Run

Use `run.sh`


### Results



### References




README To be updated
