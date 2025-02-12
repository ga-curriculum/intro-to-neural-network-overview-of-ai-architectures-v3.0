<h1>
  <span class="headline">Intro to Neural Networks + Overview of AI Architectures </span>
  <span class="subhead">Introduction to Neural Networks</span>
</h1>

**Learning objective:** By the end of this lesson, you'll be able to:
- **Understand** the structure and function of an artificial neuron.
- **Explain** the role of activation functions in introducing non-linearity.
- **Differentiate** between shallow and deep neural network architectures.
- **Engage** in a simple design activity applying these concepts.


## Understanding Artificial Neurons
- **What is an Artificial Neuron?**  
  - **Analogy:** Think of it like a recipe. Ingredients (inputs) are measured (weighted), adjusted (bias), and then processed (activation function) to produce a dish (output).
- **Key Components:**
  - **Inputs:** The raw data or features.
  - **Weights:** Determine the importance of each input.
  - **Bias:** Adjusts the output threshold.
  - **Activation Function:** Decides if the neuron “fires.”

## Activation Functions
- **Why They Matter:**  
  - Introduce non-linearity, allowing neural networks to model complex patterns.
- **Popular Activation Functions:**
  - **ReLU:** Outputs the input if positive; otherwise, zero. Ideal for hidden layers.
  - **Sigmoid:** Squashes values between 0 and 1, useful for probability outputs.
  
Imagine an activation function as a gate: it decides if the signal should pass through or not.

## Shallow vs. Deep Neural Networks
- **Shallow Networks:**  
  - **Definition:** A network with just one hidden layer.
  - **Example Use-Cases:** Predicting house prices, simple binary classification (e.g., spam detection).
- **Deep Networks:**  
  - **Definition:** Networks with multiple hidden layers.
  - **Advantages:** Capable of learning hierarchical features (e.g., edges to objects in images).
  - **Challenges:** Overfitting, computational cost, and training complexities.

## Forward Propagation and Backpropagation
- **Forward Propagation:**  
  - **Process:** Data flows from input through each layer to produce an output.

<div class="mermaid">
graph TD;
    A[Input Layer: x] -->|W1 * x + b1| B[Hidden Layer 1: z1]
    B -->|Activation: f(z1)| C[Hidden Layer 2: a1]
    C -->|W2 * a1 + b2| D[Hidden Layer 3: z2]
    D -->|Activation: f(z2)| E[Output Layer: ŷ]

    subgraph Activation_Functions[Activation Functions]
        f1[ReLU, Sigmoid, etc.]
        f2[ReLU, Sigmoid, etc.]
    end

    %% Referencing activation functions outside the subgraph
    B -.-> f1
    D -.-> f2
</div>


- **Backpropagation:**  
  - **Purpose:** Adjust weights and biases based on error feedback.
  - **Simplification:** Compare it to adjusting a recipe based on taste tests.
  
<div class="mermaid">
graph TD;
    E[Output Layer] -->|Compute Loss: L(y, ŷ)| D[Hidden Layer 3]
    D -->|∂L/∂W3 (Weight Update)| C[Hidden Layer 2]
    C -->|∂L/∂W2 (Weight Update)| B[Hidden Layer 1]
    B -->|∂L/∂W1 (Weight Update)| A[Input Layer]
    
    subgraph Weight_Updates[Weight Updates]
        W1[W1 = W1 - η * ∂L/∂W1]
        W2[W2 = W2 - η * ∂L/∂W2]
        W3[W3 = W3 - η * ∂L/∂W3]
    end

    E -->|Gradient of Loss ∂L/∂ŷ| W3
    D -->|∂L/∂a3 * f'(z3)| W2
    C -->|∂L/∂a2 * f'(z2)| W1
</div>

- **Think about it 🤔**  
  - Why do you think backpropagation is essential for learning?

## **Activity**: Designing a Neural Network for Traffic Congestion Prediction

1. **Define the Task:**
   - **Inputs:** Vehicle counts, time of day, weather conditions.
   - **Output:** Traffic light durations (e.g., green, yellow, red).
2. **Design the Neuron:**
   - **Weights:** Determine which input (e.g., vehicle count) is most influential.
   - **Bias:** Adjust the threshold for the neuron’s activation.
   - **Activation Function:**  
     - Consider using **ReLU** for handling traffic intensity.
     - Think about **Sigmoid** for binary decisions like congestion vs. no congestion.
3. **Select the Network Architecture:**
   - **Shallow Network:** Might suffice for simple traffic patterns.
   - **Deep Network:** Better for capturing complex relationships (e.g., interplay of weather and time).
4. **Discussion Points:**
   - Which activation function would work best for this scenario and why?"
   - How might noisy data or overfitting impact this design?

- **Activity Instructions:**
  - In pairs, quickly sketch a simple diagram of your neural network design.
  - Be prepared to talk through your design and proccess!
