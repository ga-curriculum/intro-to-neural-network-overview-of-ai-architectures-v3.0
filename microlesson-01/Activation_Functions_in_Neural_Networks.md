
## Activation Functions 

An **activation function** introduces non-linearity into a neural network. It decides whether a neuron should be activated or not by transforming the weighted sum of inputs and bias into an output. This enables the network to learn complex patterns and relationships in the data.

---

### Why Activation Functions are Important

1. **Non-Linearity:**
   - Without activation functions, the entire neural network would behave like a linear transformation, which limits its ability to model complex data.

2. **Feature Learning:**
   - Activation functions allow the network to detect non-linear patterns, like edges in images or sentiment in text.

3. **Enabling Deep Networks:**
   - They help propagate non-linear transformations through multiple layers, making deep learning possible.

---

### Types of Activation Functions

1. **Linear Activation Function**
   - **Formula:** \( f(x) = x \)
   - **Advantages:**
     - Simple to compute.
   - **Disadvantages:**
     - Cannot model non-linear patterns.
     - All layers collapse into a single equivalent layer.

---

2. **Sigmoid Function**
   - **Formula:** 
     \[
     f(x) = \frac{1}{1 + e^{-x}}
     \]
   - **Output Range:** \( (0, 1) \)
   - **Advantages:**
     - Useful for binary classification.
   - **Disadvantages:**
     - Prone to vanishing gradient problem.
     - Slow convergence.

---

3. **Tanh (Hyperbolic Tangent) Function**
   - **Formula:**
     \[
     f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
     \]
   - **Output Range:** \( (-1, 1) \)
   - **Advantages:**
     - Zero-centered output.
     - Better than sigmoid for deep networks.
   - **Disadvantages:**
     - Suffers from vanishing gradient problem.

---

4. **ReLU (Rectified Linear Unit)**
   - **Formula:**
     \[
     f(x) = \max(0, x)
     \]
   - **Output Range:** \( [0, \infty) \)
   - **Advantages:**
     - Computationally efficient.
     - Helps mitigate the vanishing gradient problem.
   - **Disadvantages:**
     - Can lead to "dead neurons" (outputs stuck at 0).

---

5. **Leaky ReLU**
   - **Formula:**
     \[
     f(x) = \begin{cases} 
     x, & x > 0 \\
     \alpha x, & x \leq 0 
     \end{cases}
     \]
     - \( \alpha \): Small slope for negative inputs (e.g., 0.01).
   - **Advantages:**
     - Solves the "dead neuron" problem in ReLU.
   - **Disadvantages:**
     - Still not zero-centered.

---

6. **Softmax Function**
   - **Formula:**
     \[
     f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
     \]
     - \( x_i \): Input to the \( i \)-th neuron.
   - **Output Range:** \( (0, 1) \), with outputs summing to 1.
   - **Advantages:**
     - Used for multi-class classification.
   - **Disadvantages:**
     - Computationally expensive for large output layers.

---

### Choosing the Right Activation Function

1. **Binary Classification:**
   - Use **Sigmoid** for the output layer.

2. **Multi-Class Classification:**
   - Use **Softmax** for the output layer.

3. **Hidden Layers:**
   - Use **ReLU** for faster training and to handle vanishing gradients.
   - Use **Leaky ReLU** or other variants to avoid "dead neurons."

4. **Regression Tasks:**
   - Use a **Linear Activation Function** for the output layer.

---

### Example of ReLU in a Neuron

Consider a neuron with:
- Weighted sum \( z = -0.5 \).

ReLU activation:
\[
f(z) = \max(0, z)
\]
\[
f(-0.5) = 0
\]

Output: \( 0 \).

---

### Summary of Common Activation Functions

| Activation Function | Formula                  | Output Range | Advantages                       | Disadvantages                |
|---------------------|--------------------------|--------------|-----------------------------------|------------------------------|
| Linear              | \( f(x) = x \)          | \( (-\infty, \infty) \) | Simple                        | Limited to linear mappings   |
| Sigmoid             | \( \frac{1}{1 + e^{-x}} \) | \( (0, 1) \) | Binary classification           | Vanishing gradient           |
| Tanh                | \( \tanh(x) \)          | \( (-1, 1) \) | Zero-centered output            | Vanishing gradient           |
| ReLU                | \( \max(0, x) \)        | \( [0, \infty) \) | Efficient, avoids vanishing gradients | Dead neurons                |
| Leaky ReLU          | Defined above           | \( (-\infty, \infty) \) | Solves dead neuron problem     | Still not zero-centered      |
| Softmax             | Defined above           | \( (0, 1) \) | Multi-class classification      | Computationally expensive    |

Activation functions are a critical component of neural networks, enabling them to model non-linear relationships and solve complex tasks effectively.
