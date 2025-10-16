# BPNN Mathematical Formulations for EKG Classification

This document provides comprehensive mathematical formulations for the Backpropagation Neural Network (BPNN) implementation used in the Cloud-based Biometric Sensors project for EKG/ECG signal classification.

## Table of Contents

1. [Forward Propagation](#forward-propagation)
2. [Activation Functions](#activation-functions)
3. [Loss Functions](#loss-functions)
4. [Backpropagation](#backpropagation)
5. [Parameter Updates](#parameter-updates)
6. [Signal Preprocessing](#signal-preprocessing)
7. [Feature Extraction](#feature-extraction)
8. [Implementation Notes](#implementation-notes)

---

## Forward Propagation

The forward propagation process computes the output of the neural network given an input.

### Linear Transformation

For each layer $l$ (where $l = 1, 2, ..., L$):

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$

where:
- $z^{[l]}$ is the pre-activation output vector of layer $l$
- $W^{[l]}$ is the weight matrix of layer $l$ with dimensions $(n^{[l]}, n^{[l-1]})$
- $a^{[l-1]}$ is the activation output from the previous layer with dimensions $(n^{[l-1]}, m)$
- $b^{[l]}$ is the bias vector of layer $l$ with dimensions $(n^{[l]}, 1)$
- $m$ is the number of training examples

### Activation

After the linear transformation, an activation function is applied:

$$a^{[l]} = g^{[l]}(z^{[l]})$$

where $g^{[l]}$ is the activation function for layer $l$.

### Input Layer

For the input layer ($l=0$):

$$a^{[0]} = X$$

where $X$ is the input data matrix with dimensions $(n^{[0]}, m)$.

---

## Activation Functions

### 1. Sigmoid Function

**Function:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Derivative:**
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**Properties:**
- Output range: $(0, 1)$
- Used for binary classification
- Suffers from vanishing gradient problem

**Implementation:**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
```

### 2. Hyperbolic Tangent (tanh)

**Function:**
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{e^{2z} - 1}{e^{2z} + 1}$$

**Derivative:**
$$\tanh'(z) = 1 - \tanh^2(z)$$

**Properties:**
- Output range: $(-1, 1)$
- Zero-centered, which helps with gradient flow
- Still suffers from vanishing gradient problem

**Implementation:**
```python
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2
```

### 3. ReLU (Rectified Linear Unit)

**Function:**
$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

**Derivative:**
$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

**Properties:**
- Most commonly used in deep learning
- Computationally efficient
- Helps alleviate vanishing gradient problem
- Can suffer from "dying ReLU" problem

**Implementation:**
```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
```

### 4. Softmax (Output Layer)

**Function:**
For a vector $z$ with $K$ elements:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Properties:**
- Converts logits to probability distribution
- Output range: $(0, 1)$ with $\sum_{i=1}^{K} \text{softmax}(z)_i = 1$
- Used for multi-class classification

**Numerical Stability:**
To prevent overflow, subtract the maximum value:

$$\text{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{K} e^{z_j - \max(z)}}$$

**Implementation:**
```python
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
```

---

## Loss Functions

### 1. Binary Cross-Entropy Loss

For binary classification problems:

$$L(y, \hat{y}) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})\right]$$

where:
- $y^{(i)} \in \{0, 1\}$ is the true label
- $\hat{y}^{(i)} \in (0, 1)$ is the predicted probability
- $m$ is the number of samples

### 2. Categorical Cross-Entropy Loss

For multi-class classification (used in our EKG classifier):

$$L(y, \hat{y}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{y}_k^{(i)})$$

where:
- $y_k^{(i)} \in \{0, 1\}$ is the true label (one-hot encoded)
- $\hat{y}_k^{(i)} \in (0, 1)$ is the predicted probability for class $k$
- $K$ is the number of classes
- $m$ is the number of samples

**Note:** Add small epsilon to prevent $\log(0)$:
$$L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{y}_k^{(i)} + \epsilon)$$

where $\epsilon = 10^{-8}$.

### 3. Mean Squared Error (MSE)

For regression problems:

$$L(y, \hat{y}) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$$

---

## Backpropagation

Backpropagation computes the gradients of the loss function with respect to all parameters.

### Output Layer Gradient

For the output layer $L$ with softmax activation and categorical cross-entropy loss:

$$\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial z^{[L]}} = a^{[L]} - y$$

This elegant form arises from the combination of softmax and cross-entropy derivatives.

### Hidden Layer Gradients

For hidden layers $l = L-1, L-2, ..., 1$:

$$\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \odot g'^{[l]}(z^{[l]})$$

where:
- $(W^{[l+1]})^T$ is the transpose of the weight matrix
- $\odot$ denotes element-wise (Hadamard) product
- $g'^{[l]}$ is the derivative of the activation function at layer $l$

### Weight Gradients

The gradient with respect to weights:

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{m} \delta^{[l]} (a^{[l-1]})^T$$

### Bias Gradients

The gradient with respect to biases:

$$\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} \delta^{[l]}_i$$

In matrix form:
$$\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \frac{1}{m} \sum_{\text{axis}=1} \delta^{[l]}$$

---

## Parameter Updates

### 1. Standard Gradient Descent

$$W^{[l]} := W^{[l]} - \alpha \frac{\partial \mathcal{L}}{\partial W^{[l]}}$$

$$b^{[l]} := b^{[l]} - \alpha \frac{\partial \mathcal{L}}{\partial b^{[l]}}$$

where $\alpha$ is the learning rate (hyperparameter).

### 2. Momentum

Momentum accelerates gradient descent by accumulating a velocity vector:

$$v_{dW} = \beta v_{dW} + (1-\beta) dW$$
$$v_{db} = \beta v_{db} + (1-\beta) db$$

$$W := W - \alpha v_{dW}$$
$$b := b - \alpha v_{db}$$

where:
- $\beta$ is the momentum coefficient (typically 0.9)
- $v_{dW}$ and $v_{db}$ are velocity terms initialized to 0

### 3. Adam Optimizer

Adam (Adaptive Moment Estimation) combines momentum with adaptive learning rates:

**First moment estimate (mean):**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

**Second moment estimate (variance):**
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

**Bias correction:**
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**Parameter update:**
$$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

where:
- $\beta_1 = 0.9$ (default)
- $\beta_2 = 0.999$ (default)
- $\epsilon = 10^{-8}$ (for numerical stability)
- $g_t$ is the gradient at time $t$
- $\theta$ represents parameters (W or b)

---

## Signal Preprocessing

### 1. Min-Max Normalization

Scales data to range $[0, 1]$:

$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

**Use case:** When you want to preserve the shape of the original distribution.

### 2. Z-Score Normalization (Standardization)

Transforms data to have mean 0 and standard deviation 1:

$$x_{\text{std}} = \frac{x - \mu}{\sigma}$$

where:
- $\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$ (mean)
- $\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \mu)^2}$ (standard deviation)

**Use case:** Most common for neural network inputs.

### 3. Moving Average Filter

Smooths signal by averaging over a sliding window:

$$x_{\text{filtered}}[n] = \frac{1}{w} \sum_{k=0}^{w-1} x[n-k]$$

where $w$ is the window size.

**Matrix form:**
$$x_{\text{filtered}} = \frac{1}{w} \mathbf{1}_w * x$$

where $*$ denotes convolution and $\mathbf{1}_w$ is a vector of ones with length $w$.

### 4. Butterworth Bandpass Filter

For EKG signals, typical bandpass filter parameters:
- Low cutoff: $f_{\text{low}} = 0.5$ Hz (removes baseline wander)
- High cutoff: $f_{\text{high}} = 40$ Hz (removes high-frequency noise)
- Sampling frequency: $f_s = 250$ Hz

**Transfer function:**
$$H(s) = \frac{(s/\omega_0)^n}{(s/\omega_0)^n + \sum_{k=1}^{n} a_k (s/\omega_0)^{n-k} + 1}$$

where $n$ is the filter order and $\omega_0$ is the center frequency.

---

## Feature Extraction

### Heart Rate Variability (HRV) Metrics

#### 1. SDNN (Standard Deviation of NN intervals)

Measures overall HRV:

$$\text{SDNN} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (RR_i - \overline{RR})^2}$$

where:
- $RR_i$ is the $i$-th R-R interval
- $\overline{RR} = \frac{1}{N} \sum_{i=1}^{N} RR_i$ (mean R-R interval)
- $N$ is the number of R-R intervals

#### 2. RMSSD (Root Mean Square of Successive Differences)

Measures short-term HRV:

$$\text{RMSSD} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N-1} (RR_{i+1} - RR_i)^2}$$

#### 3. pNN50

Percentage of successive R-R intervals that differ by more than 50 ms:

$$\text{pNN50} = \frac{\text{NN50}}{N-1} \times 100\%$$

where $\text{NN50} = \sum_{i=1}^{N-1} \mathbb{1}_{|RR_{i+1} - RR_i| > 50}$

### Window-based Segmentation

For creating input segments for the neural network:

**Number of windows:**
$$N_{\text{windows}} = \left\lfloor \frac{T - w}{s} \right\rfloor + 1$$

where:
- $T$ is the total signal length
- $w$ is the window size
- $s$ is the stride (step size)

**Window extraction:**
For window $i$ (where $i = 0, 1, ..., N_{\text{windows}}-1$):

$$\text{Window}_i = \text{signal}[i \cdot s : i \cdot s + w]$$

---

## Implementation Notes

### 1. Weight Initialization

**He Initialization (for ReLU):**
$$W^{[l]} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n^{[l-1]}}}\right)$$

**Xavier Initialization (for sigmoid/tanh):**
$$W^{[l]} \sim \mathcal{N}\left(0, \sqrt{\frac{1}{n^{[l-1]}}}\right)$$

**Bias Initialization:**
$$b^{[l]} = \mathbf{0}$$

### 2. Batch Processing

For mini-batch gradient descent with batch size $B$:

$$\text{Number of batches} = \left\lceil \frac{m}{B} \right\rceil$$

For each batch $i$:
- Extract samples: $X_{\text{batch}} = X[:, i \cdot B : (i+1) \cdot B]$
- Compute forward propagation
- Compute loss
- Compute backward propagation
- Update parameters

### 3. Learning Rate Scheduling

**Step Decay:**
$$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t/T \rfloor}$$

where:
- $\alpha_0$ is the initial learning rate
- $\gamma$ is the decay factor (e.g., 0.1)
- $T$ is the decay step (e.g., every 100 epochs)

**Exponential Decay:**
$$\alpha_t = \alpha_0 \cdot e^{-kt}$$

where $k$ is the decay rate.

### 4. Regularization

**L2 Regularization (Weight Decay):**

Modified loss:
$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2m} \sum_{l=1}^{L} ||W^{[l]}||_F^2$$

where:
- $\lambda$ is the regularization parameter
- $||W||_F^2 = \sum_{i,j} W_{ij}^2$ (Frobenius norm)

Modified gradient:
$$\frac{\partial \mathcal{L}_{\text{reg}}}{\partial W^{[l]}} = \frac{\partial \mathcal{L}}{\partial W^{[l]}} + \frac{\lambda}{m} W^{[l]}$$

**Dropout:**

During training, randomly set activations to zero with probability $p$:
$$a^{[l]}_{\text{dropout}} = a^{[l]} \odot \text{mask}$$

where $\text{mask} \sim \text{Bernoulli}(1-p)$

Scale activations during training:
$$a^{[l]}_{\text{dropout}} = \frac{a^{[l]} \odot \text{mask}}{1-p}$$

---

## EKG Synthesis Mathematical Model

### PQRST Complex Generation

The EKG signal is modeled as a sum of Gaussian functions:

$$\text{ECG}(t) = \sum_{i \in \{P, Q, R, S, T\}} A_i \exp\left(-\frac{(t - t_i)^2}{2\sigma_i^2}\right) + \text{baseline}(t) + \text{noise}(t)$$

where:
- $A_i$ is the amplitude of wave $i$
- $t_i$ is the time position of wave $i$
- $\sigma_i$ is the width (standard deviation) of wave $i$

**Typical values:**
| Wave | Time offset (s) | Amplitude | Width (s) |
|------|----------------|-----------|-----------|
| P    | 0.16           | 0.25      | 0.02      |
| Q    | 0.20           | -0.15     | 0.01      |
| R    | 0.22           | 1.5       | 0.015     |
| S    | 0.24           | -0.4      | 0.01      |
| T    | 0.38           | 0.35      | 0.04      |

**Baseline wander:**
$$\text{baseline}(t) = A_b \sin(2\pi f_b t)$$

where $A_b = 0.1$ and $f_b = 0.2$ Hz.

**Noise:**
$$\text{noise}(t) \sim \mathcal{N}(0, \sigma_n^2)$$

where $\sigma_n$ is the noise level (typically 0.05).

---

## Summary

This document provides all the mathematical formulations necessary for implementing and understanding the BPNN model for EKG classification. The implementation includes:

1. **Forward propagation**: Computing network outputs layer by layer
2. **Activation functions**: Sigmoid, tanh, ReLU, and softmax
3. **Loss computation**: Categorical cross-entropy for multi-class classification
4. **Backpropagation**: Computing gradients efficiently using the chain rule
5. **Parameter updates**: Gradient descent and advanced optimizers (momentum, Adam)
6. **Signal preprocessing**: Normalization and filtering techniques
7. **Feature extraction**: HRV metrics and window-based segmentation

All formulations have been implemented in the accompanying Jupyter notebooks (`dataloading.ipynb` and `randomdatasetgenerator.ipynb`).
