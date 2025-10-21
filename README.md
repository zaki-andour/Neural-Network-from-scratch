# Neural Network from Scratch

This project is a simple demonstration of how a **neural network** works.  
It helped me understand how to **build and train** a neural network **without relying on large libraries** such as TensorFlow or PyTorch.


## Files

### 1. `Activation_Function.py`

This file defines the **activation functions** used within the neural network.  
It includes implementations for:

- ReLU (Rectified Linear Unit)  
- Sigmoid  
- Tanh  

Each function provides two methods:
- `feed_forward` – computes the activation output  
- `back_prop` – computes the gradient during backpropagation  

---

### 2. `fc_layer.py`

This file implements the **fully connected (dense) layer**.  
Each layer:

- Initializes its weights randomly  
- Applies an activation function  
- Performs both forward and backward passes  
- Updates its weights using **gradient descent**

---

### 3. `nn.py`

This file defines the overall **neural network architecture** composed of fully connected layers.  
It allows you to:

- Add layers with chosen activation functions  
- Perform **forward propagation** to generate predictions  
- Perform **backpropagation** to train the model  

---

### 4. `main.py`

This is the **testing script**.  
It generates random data and evaluates the implemented neural network and activation functions.

---
