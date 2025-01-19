<h1>
  <span class="headline">[Intro to Nural Network +Overview of AI Architecture]</span>
  <span class="subhead"></span>
</h1>

**Learning objective:** 

By the end of this lesson, participants will be able to understand:

- The basics and key components of neural networks.
- Different types of neural networks and their functions.
- Core concepts like feedforward, backpropagation, and activation functions.
- Real-world applications of neural networks in AI, NLP, and autonomous systems.
- How data flows through a neural network from input to output.

![Nural Network](https://git.generalassemb.ly/modular-courses/ai-solution-architect-deloitte-ENT/blob/main/_images/Screenshot%202025-01-17%20173208.png)

[source](https://www.researchgate.net/publication/316736515_Deep_Neural_Networks_for_Text_A_Review)


---

## 1 Key Concepts

 **1.1 [Neuron (or Node)](https://git.generalassemb.ly/modular-curriculum-all-courses/intro-to-neural-network-overview-of-ai-architectures/blob/main/microlesson-01/Nuron%20README.md)**
   - The basic unit of a neural network.
   - A neuron takes inputs, applies a weight, adds a bias, and passes the result through an activation function to produce an output.

 **1.2 [Layers](https://git.generalassemb.ly/modular-curriculum-all-courses/intro-to-neural-network-overview-of-ai-architectures/blob/main/microlesson-01/Layers_in_Neural_Network.md)**
    - **Input Layer:** Receives raw data for the network.
   
   - **Hidden Layers:** Perform computations and extract features.
   
   - **Output Layer:** Produces the final output, such as a classification or regression result.

 **1.3 [ Weights and Biases](https://git.generalassemb.ly/modular-curriculum-all-courses/intro-to-neural-network-overview-of-ai-architectures/blob/main/microlesson-01/Weights_and_Biases_in_Neural_Networks.md)**
   - **Weights:** Represent the strength of connections between neurons.
   - **Biases:** Adjust the output along with the weighted input.

 **1.4 [Activation Function](https://git.generalassemb.ly/modular-curriculum-all-courses/intro-to-neural-network-overview-of-ai-architectures/blob/main/microlesson-01/Activation_Functions_in_Neural_Networks.md)**
   - Introduces non-linearity to the network, allowing it to learn complex patterns.
   - Examples: Sigmoid, ReLU, Tanh, Softmax.

 **1.5 Feedforward:**
   - The process where input data flows through the network from the input layer to the output layer.

 **1.6 Backpropagation:**
   - A learning algorithm used to train the network.
   - It adjusts the weights and biases by calculating the gradient of the loss function with respect to each parameter.

 **1.7 Loss Function:**
   - Measures the difference between the network's output and the actual target value.
   - Examples: Mean Squared Error (MSE), Cross-Entropy Loss.

 **1.8 Learning Rate:**
   - Determines the step size at which the weights are updated during training.

---

## 2 Key Deep Learning Architectures

 **2.1 Feedforward Neural Networks (FNN):**
 ![Layers](https://git.generalassemb.ly/modular-courses/ai-solution-architect-deloitte-ENT/blob/main/_images/Screenshot%202025-01-18%20164706.png)
     [source](https://www.researchgate.net/publication/299474560_Deep_Learning_for_Population_Genetic_Inference)
### The image is an example of a deep neural network with two hidden layers The first layer is the input data (each dataset has 5 statistics). The last layer predicts the 2 response variables. The last node in each input layer (+1) represents the bias term.

# Feedforward Neural Network (FNN)

A **Feedforward Neural Network (FNN)** is one of the simplest types of artificial neural networks. Information in an FNN flows only in one direction: from input nodes, through hidden nodes (if any), to output nodes. It does not form cycles or loops, unlike recurrent neural networks (RNNs).

---

## Structure of a Feedforward Neural Network

1. **Input Layer**:
   - This layer receives the input data. Each node corresponds to a feature in the input data.
   - Example: For a dataset containing customer information such as `Age`, `Income`, and `Purchase History`, the input layer will have three nodes (one for each feature).

2. **Hidden Layers** (Optional):
   - These layers process the inputs using weights and biases and apply an activation function to introduce non-linearity.
   - Each hidden layer comprises multiple neurons, allowing the network to learn complex patterns.

3. **Output Layer**:
   - This layer produces the final output of the network.
   - The number of neurons in the output layer depends on the problem:
     - Regression: Single neuron (e.g., predicting a continuous variable like `Annual Spending`).
     - Classification: Number of classes (e.g., two neurons for `Will Purchase` vs. `Will Not Purchase`).

---

## Mathematics Behind FNN

For a simple FNN with one hidden layer:

1. **Input to Hidden Layer**:
   z = W * x + b  
   where:
   - W is the weight matrix connecting input to the hidden layer.
   - x is the input vector (e.g., `Age`, `Income`, `Purchase History`).
   - b is the bias vector.

2. **Activation Function**:
   a = activation_function(z)  
   where activation_function can be ReLU, Sigmoid, or Tanh.

3. **Hidden Layer to Output Layer**:
   y = W_o * a + b_o  
   where:
   - W_o is the weight matrix connecting the hidden layer to the output layer.
   - b_o is the bias vector for the output layer.

4. **Loss Function**:
   - Measures the difference between the predicted output and the actual target.
   - Examples:
     - Mean Squared Error (MSE) for regression.
     - Cross-Entropy Loss for classification.

5. **Backpropagation**:
   - The error is propagated backward through the network to adjust weights and biases using gradient descent or similar optimization algorithms.

---

## Activation Functions

Common activation functions include:

1. **ReLU (Rectified Linear Unit)**:  
   f(x) = max(0, x)

2. **Sigmoid**:  
   f(x) = 1 / (1 + e^(-x))

3. **Tanh**:  
   f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

---

## Example: Customer Purchase Prediction

1. **Input**: Customer data with the following features:
   - Age: 30 years.
   - Income: $50,000.
   - Purchase History: 5 previous purchases.

   This data forms the input layer with three nodes.

2. **Hidden Layer**: Three neurons process the input data with a ReLU activation function.

3. **Output**: One neuron with a Sigmoid activation function predicts the probability of whether the customer will make a purchase.

---

## Advantages

1. Simple and easy to implement.
2. Suitable for basic tasks like regression and classification.
3. Provides a foundation for understanding more complex architectures.

---

## Limitations

1. Cannot handle sequential data effectively (use RNNs or transformers for this).
2. May require deep architectures (many layers) for complex problems.
3. Prone to overfitting if not regularized.


 **2.2 Convolutional Neural Networks (CNN):**
   - Specialized for image data.
   - Extract features using convolutional layers.

 **2.3 Recurrent Neural Networks (RNN):**
   - Designed for sequential data, such as time series or text.
   - Maintains a memory of previous inputs.


 **2.4 Transformer Networks:**
   - Used for natural language processing (NLP) tasks.
   - Examples: BERT, GPT.

---

## 3 Applications of Neural Networks

 **3.1 Image Recognition:**
   - Identifying objects, faces, or scenes in images.
   
 **3.2 Natural Language Processing (NLP):**
   - Sentiment analysis, machine translation, text generation.
   
 **3.2 Speech Recognition:**
   - Converting speech into text.
   
 **3.4 Autonomous Vehicles:**
   - Object detection and decision-making.
   
 **3.5 Healthcare:**
   - Diagnosing diseases and predicting patient outcomes.


