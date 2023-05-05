# Duke-Intro-To-ML
Duke Intro to ML Coursera 

This course briefly introduced the topics of logistic regression, multilayer perceptron and CNN. 

The associated notebooks are attached in this document. In this course, [PyTorch](https://pytorch.org) was used to train the relevant machine learning models. The MNIST dataset was used in this course. 

A few brief notes:
1. The trainings in this notebook were accelerated on a GPU. Personally, I use an M1 Macbook Pro, and this required a different setup to use the GPU, outlined below.
  - [Refer to the official Pytorch documentation on installing relevant depencendices to train on M1 Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)
  - Once the dependencies have been installed, in the code, we will need to send the tensors to the correct device for computation. 
  - On M1 Mac, that means using the [MPS device](https://pytorch.org/docs/stable/notes/mps.html#:~:text=mps%20device%20enables%20high%2Dperformance,devices%20with%20Metal%20programming%20framework) which enables high-performance training on GPU for MacOS devices with Metal programming framework.
  - In Notebook 3B, this was achieved as such:
```
mps_device = torch.device("mps")
model.to(mps_device)
x = images.to(mps_device)  # <---- change here 
y = model(x)
loss = criterion(y, labels.to(mps_device) )
```

- Note that if you are cloning this repository and not on M1 Mac, the MPS device may not work on your device. In that case, set the device to either CPU or CUDA if your device supports it. 



