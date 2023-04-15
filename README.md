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

3. Binary Connect

    A deep learning technique that acts as a regularizer and binary weights are constrained to 2 possible values, reducing memory and computational requirements. It has been included into SqueezeNet to reduce parameters (to confirm by how much). 

4. Skip connection

    Skip connections have been shown to improve the performance of deep neural networks by allowing information to bypass certain layers.


5. Mish Activation Function (to confirm)

    Mish is a novel self-regularized non-monotonic activation function


### Run

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



### Results



### References

[1] SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size [Paper](https://arxiv.org/abs/1602.07360)

[2] BinaryConnect: Training Deep Neural Networks with binary weights during propagation [Paper](https://arxiv.org/pdf/1511.00363.pdf)

[3] Mish: A Self Regularized Non-Monotonic Activation Function [Paper](https://arxiv.org/pdf/1908.08681.pdf)



README To be updated
