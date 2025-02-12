
## Weights and Biases in Neural Networks

In a neural network, **weights** and **biases** are the primary learnable parameters. They play a crucial role in determining how the network processes input data and generates output. These parameters are updated during training to optimize the network's performance.

---

### 1. **Weights**

- **Definition:**
  - Weights represent the strength of the connection between two neurons.
  - Each input to a neuron is multiplied by its corresponding weight before being passed to the next layer.

- **Role:**
  - Weights determine the importance of each input feature.
  - A higher weight means the input has a more significant impact on the neuron's output.

- **Mathematical Representation:**
  - For a neuron with inputs \( x_1, x_2, \dots, x_n \) and weights \( w_1, w_2, \dots, w_n \), the weighted sum is:
    \[
    z = \sum_{i=1}^n w_i \cdot x_i
    \]

- **Initialization:**
  - Weights are usually initialized with small random values.
  - Proper initialization is critical to ensure efficient training and prevent issues like vanishing or exploding gradients.

---

### 2. **Biases**

- **Definition:**
  - Bias is an additional parameter added to the weighted sum before applying the activation function.
  - It allows the neuron to shift the activation function's output, even when all inputs are zero.

- **Role:**
  - Biases enable the network to model more complex patterns and relationships.
  - Without biases, the network's representation capability would be limited.

- **Mathematical Representation:**
  - Including the bias term \( b \), the weighted sum becomes:
    \[
    z = \sum_{i=1}^n w_i \cdot x_i + b
    \]

---

### 3. **How They Work Together**

1. **Weighted Sum:**
   - The neuron computes the sum of the weighted inputs and bias:
     \[
     z = \sum_{i=1}^n w_i \cdot x_i + b
     \]

2. **Activation Function:**
   - The output is then passed through an activation function \( f(z) \), which introduces non-linearity:
     \[
     a = f(z)
     \]

3. **Output:**
   - The neuron's output \( a \) is passed to the next layer or returned as the final result.

---

### 4. **Training Weights and Biases**

- During training, weights and biases are adjusted using optimization algorithms like **Gradient Descent** to minimize the loss function.
- The adjustment is based on the gradient of the loss function with respect to each weight and bias.

---

### 5. **Example Calculation**

Consider a neuron with:

- Inputs: \( x_1 = 0.5, x_2 = 0.8 \)
- Weights: \( w_1 = 0.2, w_2 = 0.4 \)
- Bias: \( b = 0.1 \)
- Activation function: ReLU (Rectified Linear Unit)

#### Step 1: Compute Weighted Sum
\[
z = (0.2 \cdot 0.5) + (0.4 \cdot 0.8) + 0.1 = 0.1 + 0.32 + 0.1 = 0.52
\]

#### Step 2: Apply Activation Function
ReLU activation function:
\[
f(z) = \max(0, z)
\]
\[
a = \max(0, 0.52) = 0.52
\]

The neuron's output is \( a = 0.52 \).

---

### 6. **Importance of Weights and Biases**

1. **Weights:**
   - Define the strength of each input feature.
   - Influence how data flows through the network.

2. **Biases:**
   - Provide flexibility to the network.
   - Ensure the activation function does not always pass through the origin.

---

Understanding weights and biases is essential for grasping how neural networks learn and generalize from data.
