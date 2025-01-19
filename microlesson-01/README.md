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
 ![CNN](https://git.generalassemb.ly/modular-courses/ai-solution-architect-deloitte-ENT/blob/main/_images/Screenshot%202025-01-19%20171849.png)
 
 [source](https://www.researchgate.net/publication/336805909_A_High-Accuracy_Model_Average_Ensemble_of_Convolutional_Neural_Networks_for_Classification_of_Cloud_Image_Patches_on_Small_Datasets)
 
  # Convolutional Neural Network (CNN)

A **Convolutional Neural Network (CNN)** is a specialized type of neural network primarily designed for processing structured grid-like data, such as images or time series. CNNs are highly effective for tasks involving spatial hierarchies and patterns, making them widely used in computer vision and image-related problems.

---

## Key Features of CNNs

1. **Convolutional Layers**:
   - Perform feature extraction by applying filters (kernels) to the input data.
   - Filters slide over the input to detect local patterns, such as edges, textures, or colors.

2. **Pooling Layers**:
   - Downsample the feature maps to reduce dimensionality while retaining essential information.
   - Types of pooling:
     - **Max Pooling**: Selects the maximum value in a region.
     - **Average Pooling**: Computes the average value in a region.

3. **Fully Connected Layers**:
   - Connect all neurons in one layer to all neurons in the next.
   - These layers are used for final classification or regression tasks.

4. **Activation Functions**:
   - Introduce non-linearity into the model to learn complex patterns.
   - Common choices include ReLU (Rectified Linear Unit).

---

## How CNN Works

1. **Input**:
   - CNNs take structured data as input, like an image represented by pixel values (e.g., a 3D matrix for RGB images).

2. **Convolution**:
   - Filters (kernels) slide across the input, performing element-wise multiplication and summation to produce a **feature map**.
   - Captures spatial relationships in the data.

3. **Pooling**:
   - Reduces the size of feature maps while preserving critical features, helping to prevent overfitting and speeding up computation.

4. **Flattening**:
   - Converts the pooled feature maps into a 1D vector to feed into the fully connected layers.

5. **Output**:
   - The final output layer predicts the target (e.g., image classification, object detection).

---

## Applications of CNNs

1. **Image Classification**:
   - Recognizing objects in an image (e.g., detecting whether an image contains a dog or a cat).

2. **Object Detection**:
   - Identifying and localizing multiple objects in an image.

3. **Image Segmentation**:
   - Dividing an image into meaningful regions for tasks like medical imaging.

4. **Video Processing**:
   - Action recognition or video classification.

5. **Natural Language Processing**:
   - Analyzing text data in certain tasks, such as sentence classification or sentiment analysis.

---

## Advantages

1. **Efficient Feature Extraction**:
   - Automatically detects hierarchical patterns (edges, shapes, and objects).

2. **Reduced Parameters**:
   - Weight sharing in convolutional layers reduces the number of parameters compared to traditional fully connected networks.

3. **Scalability**:
   - Effective for large datasets and complex tasks.

---

## Limitations

1. **Computationally Expensive**:
   - Requires significant computational resources, especially for deep architectures.

2. **Data Dependency**:
   - Requires large amounts of labeled data to perform well.

3. **Interpretability**:
   - Difficult to interpret or explain the learned features and predictions.

---

Would you like an example implementation of a CNN in Python using TensorFlow or PyTorch? Let me know!


 # 2.3 Recurrent Neural Networks (RNN)
 
 ![RNN](https://git.generalassemb.ly/modular-courses/ai-solution-architect-deloitte-ENT/blob/main/_images/Screenshot%202025-01-19%20173152.png)
 [RNN](https://www.researchgate.net/publication/361681838_Sentiment_Analysis_of_Public_Social_Media_as_a_Tool_for_Health_-Related_Topics)
 

A **Recurrent Neural Network (RNN)** is a type of neural network designed to handle sequential data by introducing the concept of **memory**. RNNs use loops to pass information from one step of the sequence to the next, allowing the network to retain context and model dependencies across time steps.

---

## Key Features of RNNs

1. **Sequential Data Handling**:
   - RNNs are designed to process sequential data, such as text, speech, or time-series data.
   - Each time step in the input sequence influences the current computation and is passed forward as context for the next time step.

2. **Hidden State**:
   - The hidden state acts as the "memory" of the network, carrying information from previous time steps.
   - At each time step, the hidden state is updated based on:
     - The input at the current time step.
     - The hidden state from the previous time step.

3. **Recurrent Connections**:
   - Neurons in the hidden layer have recurrent connections that loop back to themselves, enabling the network to retain context across the sequence.

4. **Output at Each Time Step**:
   - RNNs can produce:
     - An output at each time step (e.g., predicting the next word in a sentence).
     - A single output after processing the entire sequence (e.g., sentiment analysis).

---

## How RNNs Work

1. **Hidden State Update**:
   - The hidden state at the current time step is calculated using:
     - The input at the current time step.
     - The hidden state from the previous time step.
     - A set of learnable weights and biases.

2. **Output at Each Time Step**:
   - The output at a given time step is computed using the updated hidden state and another set of weights and biases.

3. **Recurrent Process**:
   - The RNN processes data one step at a time, using the context from previous steps to influence the current computation.

---

## Applications of RNNs

1. **Natural Language Processing (NLP)**:
   - Text generation, machine translation, sentiment analysis, and language modeling.
2. **Speech Recognition**:
   - Recognizing spoken words or commands in audio sequences.
3. **Time-Series Prediction**:
   - Stock market forecasting, weather prediction, or anomaly detection in sequential data.
4. **Video Analysis**:
   - Action recognition, video captioning, or event detection.
5. **Music Generation**:
   - Composing melodies based on patterns in musical sequences.

---

## Challenges of RNNs

1. **Vanishing/Exploding Gradients**:
   - During training, gradients can become too small (vanishing) or too large (exploding), making it difficult to capture long-term dependencies.
   - This problem can be mitigated using advanced architectures like **LSTMs (Long Short-Term Memory)** and **GRUs (Gated Recurrent Units)**.

2. **Sequential Computation**:
   - RNNs process data step by step, making them computationally slower compared to other architectures like CNNs.

---

## Example Workflow

Suppose we are training an RNN for text generation:

1. **Input**:
   - A sequence of words or characters, such as "I love machine learning".
2. **Hidden State**:
   - The RNN processes one word at a time, updating its hidden state at each step.
3. **Output**:
   - At each step, the network predicts the next word (e.g., predicting "learning" after "machine").
4. **Training**:
   - Use a loss function, such as Cross-Entropy Loss, to compare predictions with the actual sequence and optimize the weights using backpropagation through time (BPTT).

---

Would you like an example implementation of an RNN in Python? Let me know!



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


