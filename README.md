## Binarizing SqueezeNet: PyTorch Implementation and Exploration

AI6103 Deep Learning & Applications Group Project

Team Members: 
- Ang Zheng Da
- Wang Wei Sheng
- Rachel Gan 

### Description

This is tested on CIFAR-100.

In this repository, it contains the following:

1. Vanilla SqueezeNet
    
    The base model follows the implementation in the paper. It consists of 8 fire modules, each with a squeeze module followed by an expand module. The network has been modified to take in CIFAR-100 dataset and outputs a 100-dimension probaility vector. Batch normalisation is added after each convolutional layer in the squeeze and expand modules.

2. Binary Connect

    A deep learning technique that acts as a regularizer and binary weights are constrained to 2 possible values, reducing memory and computational requirements.

3. Skip Connection

    Skip connections have been shown to improve the performance of deep neural networks by allowing information to bypass certain layers.

    (Code: refer to skip connection branch)

4. Mish Activation Function

    Mish is a novel self-regularized non-monotonic activation function


### Setup

Prerequisites: 
- PyTorch
- Python, Numpy
- GPU

To note:
- Modify the parameters and setup in `run.sh`. 
- Create a folder `diagram` to store the results and diagrams

To run:
```
!bash run.sh
```



### References

[1] SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size [Paper](https://arxiv.org/abs/1602.07360)

[2] BinaryConnect: Training Deep Neural Networks with binary weights during propagation [Paper](https://arxiv.org/pdf/1511.00363.pdf)

[3] Mish: A Self Regularized Non-Monotonic Activation Function [Paper](https://arxiv.org/pdf/1908.08681.pdf)



