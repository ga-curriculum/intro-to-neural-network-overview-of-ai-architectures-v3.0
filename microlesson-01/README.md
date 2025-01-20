# [Introduction to Neural Networks and Overview of AI Architecture]

**Learning Objective:**

By the end of this lesson, participants will understand:

- The basics and key components of neural networks.
- Different types of neural networks and their functions.
- Core concepts like feedforward, backpropagation, and activation functions.
- Real-world applications of neural networks in AI, NLP, and autonomous systems.
- How data flows through a neural network from input to output.

![Neural Network](https://git.generalassemb.ly/modular-courses/ai-solution-architect-deloitte-ENT/blob/main/_images/Screenshot%202025-01-17%20173208.png)

[Source](https://www.researchgate.net/publication/316736515_Deep_Neural_Networks_for_Text_A_Review) [1]

---

## 1. Key Concepts

### 1.1 Neuron (or Node)
- The basic unit of a neural network.
- A neuron takes inputs, applies a weight, adds a bias, and passes the result through an activation function to produce an output. [2]

### 1.2 Layers
- **Input Layer:** Receives raw data for the network.
- **Hidden Layers:** Perform computations and extract features.
- **Output Layer:** Produces the final output, such as a classification or regression result. [3]

### 1.3 Weights and Biases
- **Weights:** Represent the strength of connections between neurons.
- **Biases:** Adjust the output along with the weighted input. [4]

### 1.4 Activation Function
- Introduces non-linearity to the network, allowing it to learn complex patterns.
- Examples: Sigmoid, ReLU, Tanh, Softmax. [5]

### 1.5 Feedforward
- The process where input data flows through the network from the input layer to the output layer. [6]

### 1.6 Backpropagation
- A learning algorithm used to train the network.
- It adjusts the weights and biases by calculating the gradient of the loss function with respect to each parameter. [7]

### 1.7 Loss Function
- Measures the difference between the network's output and the actual target value.
- Examples: Mean Squared Error (MSE), Cross-Entropy Loss. [8]

### 1.8 Learning Rate
- Determines the step size at which the weights are updated during training. [9]

---


## Feedforward Neural Network (FNN)

A **Feedforward Neural Network (FNN)** is one of the simplest types of artificial neural networks. Information in an FNN flows only in one direction: from input nodes, through hidden nodes (if any), to output nodes. It does not form cycles or loops, unlike recurrent neural networks (RNNs) [10].

---

## Structure of a Feedforward Neural Network

1. **Input Layer**:
   - This layer receives the input data. Each node corresponds to a feature in the input data.
   - Example: For a dataset containing customer information such as Age, Income, and Purchase History, the input layer will have three nodes (one for each feature) [11].

2. **Hidden Layers** (Optional):
   - These layers process the inputs using weights and biases and apply an activation function to introduce non-linearity.
   - Each hidden layer comprises multiple neurons, allowing the network to learn complex patterns [12].

3. **Output Layer**:
   - This layer produces the final output of the network.
   - The number of neurons in the output layer depends on the problem:
     - Regression: Single neuron (e.g., predicting a continuous variable like Annual Spending).
     - Classification: Number of classes (e.g., two neurons for Will Purchase vs. Will Not Purchase) [13].

---

## Mathematics Behind FNN

For a simple FNN with one hidden layer:

1. **Input to Hidden Layer**:
   z = W * x + b  
   where:
   - W is the weight matrix connecting input to the hidden layer.
   - x is the input vector (e.g., Age, Income, Purchase History).
   - b is the bias vector [14].


2. **Activation Function**:
   a = activation_function(z)  
   where activation_function can be ReLU, Sigmoid, or Tanh [15].

3. **Hidden Layer to Output Layer**:
   y = W_o * a + b_o  
   where:
   - W_o is the weight matrix connecting the hidden layer to the output layer.
   - b_o is the bias vector for the output layer [16].

4. **Loss Function**:
   - Measures the difference between the predicted output and the actual target.
   - Examples:
     - Mean Squared Error (MSE) for regression.
     - Cross-Entropy Loss for classification [17].

5. **Backpropagation**:
   - The error is propagated backward through the network to adjust weights and biases using gradient descent or similar optimization algorithms [18].

---

## Activation Functions

Common activation functions include:

1. **ReLU (Rectified Linear Unit)**:  
   f(x) = max(0, x) [19].

2. **Sigmoid**:  
   f(x) = 1 / (1 + e^(-x)) [20].

3. **Tanh**:  
   f(x) = (e^x - e^(-x)) / (e^x + e^(-x)) [21].

---

## Example: Customer Purchase Prediction

1. **Input**: Customer data with the following features:
   - Age: 30 years.
   - Income: $50,000.
   - Purchase History: 5 previous purchases.

   This data forms the input layer with three nodes [22].

2. **Hidden Layer**: Three neurons process the input data with a ReLU activation function [23].

3. **Output**: One neuron with a Sigmoid activation function predicts the probability of whether the customer will make a purchase [24].

---

## Advantages

1. Simple and easy to implement [25].
2. Suitable for basic tasks like regression and classification [26].
3. Provides a foundation for understanding more complex architectures [27].

---

## Limitations

1. Cannot handle sequential data effectively (use RNNs or transformers for this) [28].
2. May require deep architectures (many layers) for complex problems [29].
3. Prone to overfitting if not regularized [30].

---

**2.2 Convolutional Neural Networks (CNN):**

![CNN](https://git.generalassemb.ly/modular-courses/ai-solution-architect-deloitte-ENT/blob/main/_images/Screenshot%202025-01-19%20171849.png)

[source](https://www.researchgate.net/publication/336805909_A_High-Accuracy_Model_Average_Ensemble_of_Convolutional_Neural_Networks_for_Classification_of_Cloud_Image_Patches_on_Small_Datasets)

# Convolutional Neural Network (CNN)

A **Convolutional Neural Network (CNN)** is a specialized type of neural network primarily designed for processing structured grid-like data, such as images or time series. CNNs are highly effective for tasks involving spatial hierarchies and patterns, making them widely used in computer vision and image-related problems [31].

---

## Key Features of CNNs

1. **Convolutional Layers**:
   - Perform feature extraction by applying filters (kernels) to the input data.
   - Filters slide over the input to detect local patterns, such as edges, textures, or colors [32].

2. **Pooling Layers**:
   - Downsample the feature maps to reduce dimensionality while retaining essential information.
   - Types of pooling:
     - **Max Pooling**: Selects the maximum value in a region.
     - **Average Pooling**: Computes the average value in a region [33].

3. **Fully Connected Layers**:
   - Connect all neurons in one layer to all neurons in the next.
   - These layers are used for final classification or regression tasks [34].

4. **Activation Functions**:
   - Introduce non-linearity into the model to learn complex patterns.
   - Common choices include ReLU (Rectified Linear Unit) [35].

---

## How CNN Works

1. **Input**:
   - CNNs take structured data as input, like an image represented by pixel values (e.g., a 3D matrix for RGB images) [36].

2. **Convolution**:
   - Filters (kernels) slide across the input, performing element-wise multiplication and summation to produce a **feature map**.
   - Captures spatial relationships in the data [37].

3. **Pooling**:
   - Reduces the size of feature maps while preserving critical features, helping to prevent overfitting and speeding up computation [38].

4. **Flattening**:
   - Converts the pooled feature maps into a 1D vector to feed into the fully connected layers [39].

5. **Output**:
   - The final output layer predicts the target (e.g., image classification, object detection) [40].

---

## Applications of CNNs

1. **Image Classification**:
   - Recognizing objects in an image (e.g., detecting whether an image contains a dog or a cat) [41].

2. **Object Detection**:
   - Identifying and localizing multiple objects in an image [42].

3. **Image Segmentation**:
   - Dividing an image into meaningful regions for tasks like medical imaging [43].

4. **Video Processing**:
   - Action recognition or video classification [44].

5. **Natural Language Processing**:
   - Analyzing text data in certain tasks, such as sentence classification or sentiment analysis [45].

---

## Advantages

1. **Efficient Feature Extraction**:
   - Automatically detects hierarchical patterns (edges, shapes, and objects) [46].

2. **Reduced Parameters**:
   - Weight sharing in convolutional layers reduces the number of parameters compared to traditional fully connected networks [47].

3. **Scalability**:
   - Effective for large datasets and complex tasks [48].

---

## Limitations

1. **Computationally Expensive**:
   - Requires significant computational resources, especially for deep architectures [49].

2. **Data Dependency**:
   - Requires large amounts of labeled data to perform well [50].

3. **Interpretability**:
   - Difficult to interpret or explain the learned features and predictions [51].

---

**2.3 Recurrent Neural Networks (RNN):**

![RNN](https://git.generalassemb.ly/modular-courses/ai-solution-architect-deloitte-ENT/blob/main/_images/Screenshot%202025-01-19%20173152.png)

[source](https://www.researchgate.net/publication/361681838_Sentiment_Analysis_of_Public_Social_Media_as_a_Tool_for_Health_-Related_Topics)

# Recurrent Neural Networks (RNN)

A **Recurrent Neural Network (RNN)** is a type of neural network designed to handle sequential data by introducing the concept of **memory**. RNNs use loops to pass information from one step of the sequence to the next, allowing the network to retain context and model dependencies across time steps [52].

---

## Key Features of RNNs

1. **Sequential Data Handling**:
   - RNNs are designed to process sequential data, such as text, speech, or time-series data.
   - Each time step in the input sequence influences the current computation and is passed forward as context for the next time step [53].

2. **Hidden State**:
   - The hidden state acts as the "memory" of the network, carrying information from previous time steps.
   - At each time step, the hidden state is updated based on:
     - The input at the current time step.
     - The hidden state from the previous time step [54].

3. **Recurrent Connections**:
   - Neurons in the hidden layer have recurrent connections that loop back to themselves, enabling the network to retain context across the sequence [55].

4. **Output at Each Time Step**:
   - RNNs can produce:
     - An output at each time step (e.g., predicting the next word in a sentence).
     - A single output after processing the entire sequence (e.g., sentiment analysis) [56].

---

## How RNNs Work

1. **Hidden State Update**:
   - The hidden state at the current time step is calculated using:
     - The input at the current time step.
     - The hidden state from the previous time step.
     - A set of learnable weights and biases [57].

2. **Output at Each Time Step**:
   - The output at a given time step is computed using the updated hidden state and another set of weights and biases [58].

3. **Recurrent Process**:
   - The RNN processes data one step at a time, using the context from previous steps to influence the current computation [59].

## Applications of RNNs

1. **Natural Language Processing (NLP)**:
   - Text generation, machine translation, sentiment analysis, and language modeling [61].
2. **Speech Recognition**:
   - Recognizing spoken words or commands in audio sequences [62].
3. **Time-Series Prediction**:
   - Stock market forecasting, weather prediction, or anomaly detection in sequential data [63].
4. **Video Analysis**:
   - Action recognition, video captioning, or event detection [64].
5. **Music Generation**:
   - Composing melodies based on patterns in musical sequences [65].

---

## Challenges of RNNs

1. **Vanishing/Exploding Gradients**:
   - During training, gradients can become too small (vanishing) or too large (exploding), making it difficult to capture long-term dependencies.
   - This problem can be mitigated using advanced architectures like **LSTMs (Long Short-Term Memory)** and **GRUs (Gated Recurrent Units)** [66].

2. **Sequential Computation**:
   - RNNs process data step by step, making them computationally slower compared to other architectures like CNNs [67].

---

## Example Workflow

Suppose we are training an RNN for text generation:

1. **Input**:
   - A sequence of words or characters, such as "I love machine learning" [68].
2. **Hidden State**:
   - The RNN processes one word at a time, updating its hidden state at each step [69].
3. **Output**:
   - At each step, the network predicts the next word (e.g., predicting "learning" after "machine") [70].
4. **Training**:
   - Use a loss function, such as Cross-Entropy Loss, to compare predictions with the actual sequence and optimize the weights using backpropagation through time (BPTT) [71].

---

# 2.4 Transformer Networks Architecture

![Transformer](https://git.generalassemb.ly/modular-courses/ai-solution-architect-deloitte-ENT/blob/main/_images/Screenshot%202025-01-19%20174629.png)

[Source](https://arxiv.org/pdf/1706.03762)

# Transformer Architecture

The **Transformer** architecture is a deep learning model introduced in the paper *"Attention Is All You Need"* by Vaswani et al. (2017) [72]. It has revolutionized natural language processing (NLP) and various other fields, replacing traditional recurrent and convolutional models for sequence-to-sequence tasks. Transformers rely entirely on **self-attention mechanisms** to process sequential data, making them highly parallelizable and efficient.

---

## Key Features of the Transformer

1. **Self-Attention Mechanism**:
   - Captures relationships between words in a sequence, regardless of their distance from one another [73].
   - Focuses on relevant parts of the input sequence while processing each word or token [74].

2. **Parallelization**:
   - Unlike recurrent neural networks (RNNs), Transformers process the entire input sequence simultaneously, significantly speeding up training [75].

3. **Positional Encoding**:
   - Since Transformers lack recurrence, positional encodings are added to input embeddings to preserve information about the sequence order [76].

4. **Scalability**:
   - Transformers can scale to very large datasets and model sizes, making them suitable for large-scale applications like GPT and BERT [77].

---

## Components of the Transformer

### 1. Input Embedding
- Converts each input token into a continuous vector representation.
- Includes **positional encoding** to add information about the sequence order [78].

---

### 2. Encoder
- A stack of identical layers, each with two main components:
  1. **Multi-Head Self-Attention**:
     - Allows each word to attend to every other word in the sequence [79].
     - Captures contextual information for each word [80].
  2. **Feedforward Neural Network**:
     - Applies two fully connected layers with a non-linear activation in between for added complexity [81].
  - Includes **layer normalization** and **residual connections** to stabilize training [82].

---

### 3. Decoder
- A stack of identical layers with three main components:
  1. **Masked Multi-Head Self-Attention**:
     - Ensures the decoder can only attend to previous tokens to prevent "cheating" during training [83].
  2. **Encoder-Decoder Attention**:
     - Allows the decoder to attend to the encoder's output for context [84].
  3. **Feedforward Neural Network**:
     - Similar to the encoder, applies two fully connected layers with activation functions [85].

---

### 4. Multi-Head Attention
- Splits the input into multiple "heads" to focus on different parts of the sequence simultaneously [86].
- Combines the outputs of all heads to form the final attention representation.
- Key components:
  - Query, Key, and Value matrices are computed from the input [87].
  - Attention scores determine how much attention each word pays to others in the sequence [88].

---

### 5. Positional Encoding
- Adds positional information to input embeddings using sine and cosine functions or learned embeddings [89].
- Allows the model to understand the order of tokens in the sequence [90].

---

### 6. Output Layer
- Generates the final predictions, such as translated sentences or next-token probabilities [91].
- Typically includes a **softmax** layer for probability distributions [92].

---

## Strengths of the Transformer

1. **Parallelization**:
   - Processes sequences in parallel, making it faster than RNNs [93].
2. **Long-Range Dependency Handling**:
   - Effectively captures dependencies between distant words or tokens in a sequence [94].
3. **Scalability**:
   - Can handle large datasets and be extended to billions of parameters (e.g., GPT-3, BERT) [95].

---

## Applications of the Transformer

1. **Machine Translation**:
   - Transforms sentences from one language to another.
   - Example: Google's Translate uses transformer-based models [96].

2. **Text Generation**:
   - Generates coherent and contextually relevant text.
   - Example: GPT (Generative Pre-trained Transformer) [97].

3. **Text Classification**:
   - Sentiment analysis, spam detection, or topic categorization [98].

4. **Question Answering**:
   - Models like BERT and T5 extract answers from a given context [99].

5. **Speech and Vision**:
   - Applied in speech recognition and vision tasks, such as Vision Transformers (ViTs) [100].

---

## Limitations

1. **Computational Expense**:
   - Requires significant memory and computational power, especially for long sequences [101].
2. **Training Data**:
   - Needs large datasets to achieve high performance [102].
3. **Interpretability**:
   - Understanding how attention weights contribute to predictions can be challenging [103].

---

# AI Architecture for Different Data Modalities

Choosing the right AI architecture depends on the **type of data modality** (e.g., text, image, audio, video, etc.), the **size of the dataset**, and the specific task. Selecting an appropriate model ensures efficient processing, high accuracy, and optimal performance for the task at hand [104].

---

## Key Considerations When Picking the Right Model

1. **Data Modality**:
   - Different types of data (text, image, audio, etc.) require specialized architectures tailored to their structure and characteristics [105].

2. **Data Size**:
   - The volume of available data influences the choice of architecture. Larger datasets benefit from more complex models, while smaller datasets require simpler or pre-trained models to avoid overfitting [106].

3. **Task Type**:
   - Classification, regression, generation, segmentation, and detection tasks often have specialized model architectures [107].

4. **Hardware and Resource Constraints**:
   - Some architectures are computationally expensive, requiring GPUs or TPUs, while others are lightweight and suitable for edge devices [108].

5. **Real-Time vs. Batch Processing**:
   - Real-time applications require models optimized for low latency, while batch processing allows for heavier models with longer computation times [109].


---


## Data Modalities and Suitable Architectures

### 1. **Text (Natural Language Processing)**
- **Common Use Cases**:
  - Sentiment analysis, text classification, language translation, summarization, and question answering [116].

- **Recommended Models**:
  1. **Small Datasets**:
     - Pre-trained embeddings like Word2Vec, GloVe, or FastText [117].
     - Traditional models like Logistic Regression or SVM with TF-IDF features [118].
  2. **Moderate to Large Datasets**:
     - RNNs (Recurrent Neural Networks) or LSTMs for sequential data [119].
     - Transformers (e.g., BERT, GPT) for contextual understanding and generative tasks [120].
  3. **Low-Resource Scenarios**:
     - Distilled models like DistilBERT or MobileBERT for efficient inference [121].

---

### 2. **Images (Computer Vision)**
- **Common Use Cases**:
  - Image classification, object detection, image segmentation, and super-resolution [122].

- **Recommended Models**:
  1. **Small Datasets**:
     - Transfer learning with pre-trained models like VGG, ResNet, or EfficientNet [123].
     - Data augmentation techniques to artificially increase the dataset size [124].
  2. **Moderate to Large Datasets**:
     - Convolutional Neural Networks (CNNs) like ResNet, EfficientNet, or DenseNet [125].
     - Vision Transformers (ViTs) for state-of-the-art performance on large datasets [126].
  3. **Real-Time Applications**:
     - Lightweight models like MobileNet or SqueezeNet for edge devices [127].

---

### 3. **Audio**
- **Common Use Cases**:
  - Speech recognition, sound classification, and music generation [128].

- **Recommended Models**:
  1. **Small Datasets**:
     - Traditional feature-based models using MFCCs (Mel-Frequency Cepstral Coefficients) and SVMs or Random Forests [129].
  2. **Moderate to Large Datasets**:
     - RNNs, GRUs, or LSTMs for sequential audio processing [130].
     - Transformers like Wav2Vec and Whisper for state-of-the-art audio transcription [131].
  3. **Real-Time Applications**:
     - Lightweight architectures optimized for low latency, such as RNN variants [132].

---

### 4. **Video**
- **Common Use Cases**:
  - Action recognition, video summarization, and object tracking [133].

- **Recommended Models**:
  1. **Small Datasets**:
     - Extract frame-level features using CNNs or transfer learning and feed them into an RNN or LSTM [134].
  2. **Moderate to Large Datasets**:
     - 3D CNNs like C3D for spatiotemporal feature extraction [135].
     - Transformers like TimeSformer for advanced temporal analysis [136].
  3. **Real-Time Applications**:
     - Lightweight architectures with frame skipping or reduced temporal resolution [137].

---

### 5. **Tabular Data**
- **Common Use Cases**:
  - Predictive modeling, classification, and regression tasks in domains like finance, healthcare, and retail [138].

- **Recommended Models**:
  1. **Small Datasets**:
     - Decision Trees, Random Forests, Gradient Boosting (e.g., XGBoost, LightGBM) [139].
  2. **Moderate to Large Datasets**:
     - TabNet (deep learning-based) [140].
     - Gradient boosting models like CatBoost or LightGBM [141].
  3. **Low-Resource Scenarios**:
     - Logistic Regression or simpler tree-based models for efficiency [142].

---

### 6. **Multimodal Data**
- **Common Use Cases**:
  - Tasks combining multiple modalities, such as text and images (e.g., visual question answering, image captioning) [143].

- **Recommended Models**:
  1. **Small Datasets**:
     - Pre-trained models with fine-tuning for each modality (e.g., ResNet for images, BERT for text) [144].
  2. **Moderate to Large Datasets**:
     - Multimodal Transformers (e.g., CLIP, DALL-E) [145].
     - Late or early fusion models combining embeddings from different modalities [146].
  3. **Advanced Applications**:
     - Generative models like Stable Diffusion for text-to-image tasks [147].

---

## General Guidelines for Model Selection

1. **Small Datasets**:
   - Use pre-trained models with fine-tuning to leverage transfer learning [148].
   - Apply regularization techniques (dropout, weight decay) to avoid overfitting [149].

2. **Large Datasets**:
   - Train custom architectures or fine-tune advanced models like Transformers [150].
   - Consider hardware capabilities to balance training time and cost [151].

3. **Resource-Constrained Scenarios**:
   - Opt for lightweight models like MobileNet, DistilBERT, or TinyML architectures [152].

4. **Real-Time Requirements**:
   - Prioritize low-latency models with optimized inference speeds [153].

## References

[1] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444. [https://doi.org/10.1038/nature14539]

[2] Schmidhuber, J. (2015). Deep learning in neural networks: An overview. *Neural Networks*, 61, 85–117. [https://doi.org/10.1016/j.neunet.2014.09.003]

[3] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. *MIT Press*. [http://www.deeplearningbook.org]

[4] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems*, 25. [https://doi.org/10.1145/3065386]

[5] Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. *Proceedings of the 27th International Conference on Machine Learning (ICML-10)*, 807–814. [https://icml.cc/Conferences/2010/papers/432.pdf]

[6] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536. [https://doi.org/10.1038/323533a0]

[7] Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*. [https://arxiv.org/abs/1412.6980]

[8] Bishop, C. M. (2006). Pattern recognition and machine learning. *Springer*. [https://doi.org/10.1007/978-0-387-45528-0]

[9] Zeiler, M. D. (2012). ADADELTA: An adaptive learning rate method. *arXiv preprint arXiv:1212.5701*. [https://arxiv.org/abs/1212.5701]


10. [Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.](https://www.deeplearningbook.org/)

11. [LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.](https://www.nature.com/articles/nature14539)

12. [Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*, 4(2), 251-257.](https://doi.org/10.1016/0893-6080(91)90009-T)

13. [Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786), 504-507.](https://doi.org/10.1126/science.1127647)

14. [Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.](https://www.nature.com/articles/323533a0)

15. [Agarap, A. F. (2018). Deep learning using rectified linear units (ReLU). *arXiv preprint arXiv:1803.08375*.](https://arxiv.org/abs/1803.08375)

16. [Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. *Proceedings of the 27th International Conference on Machine Learning (ICML)*.](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf)

17. [Bishop, C. M. (2006). *Pattern recognition and machine learning*. Springer.](https://www.springer.com/gp/book/9780387310732)


18. [Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.](https://arxiv.org/abs/1412.6980)

19. [Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)*.](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

20. [He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.](https://arxiv.org/abs/1502.01852)

21. [Deng, L., & Yu, D. (2014). Deep learning: Methods and applications. *Foundations and Trends in Signal Processing*, 7(3–4), 197-387.](https://doi.org/10.1561/2000000039)

22. [Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems (NIPS)*.](https://dl.acm.org/doi/10.1145/3065386)

23. [Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929-1958.](http://jmlr.org/papers/v15/srivastava14a.html)

24. [Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. *Proceedings of the European Conference on Computer Vision (ECCV)*.](https://arxiv.org/abs/1311.2901)

25. [Ng, A. Y. (2004). Feature selection, L1 vs. L2 regularization, and rotational invariance. *Proceedings of the 21st International Conference on Machine Learning (ICML)*.](https://www.csie.ntu.edu.tw/~cjlin/icml04_ng.pdf)

26. [Chollet, F. (2017). *Deep learning with Python*. Manning Publications.](https://www.manning.com/books/deep-learning-with-python)

27. [Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. *arXiv preprint arXiv:1412.6572*.](https://arxiv.org/abs/1412.6572)

28. [Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*.](http://proceedings.mlr.press/v37/ioffe15.html)

29. [Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.](https://doi.org/10.1162/neco.1997.9.8.1735)

30. [Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386-408.](https://psycnet.apa.org/doi/10.1037/h0042519)


31. [LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.](https://ieeexplore.ieee.org/document/726791)

32. [Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems (NIPS)*.](https://dl.acm.org/doi/10.1145/3065386)

33. [Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. *Proceedings of the European Conference on Computer Vision (ECCV)*.](https://arxiv.org/abs/1311.2901)

34. [He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.](https://arxiv.org/abs/1512.03385)

35. [Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*.](https://arxiv.org/abs/1409.1556)

36. [O'Shea, K., & Nash, R. (2015). An introduction to convolutional neural networks. *arXiv preprint arXiv:1511.08458*.](https://arxiv.org/abs/1511.08458)

37. [Yamashita, R., Nishio, M., Do, R. K. G., & Togashi, K. (2018). Convolutional neural networks: An overview and application in radiology. *Insights into Imaging*, 9(4), 611-629.](https://doi.org/10.1007/s13244-018-0639-9)

38. [Bengio, Y. (2009). Learning deep architectures for AI. *Foundations and Trends in Machine Learning*, 2(1), 1-127.](https://doi.org/10.1561/2200000006)

39. [Redmon, J., & Farhadi, A. (2018). YOLOv3: An incremental improvement. *arXiv preprint arXiv:1804.02767*.](https://arxiv.org/abs/1804.02767)

40. [Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards real-time object detection with region proposal networks. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/1506.01497)

41. [Lin, T. Y., et al. (2017). Feature pyramid networks for object detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.](https://arxiv.org/abs/1612.03144)

42. [Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.](https://arxiv.org/abs/1411.4038)

43. [Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2017). Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. *arXiv preprint arXiv:1606.00915*.](https://arxiv.org/abs/1606.00915)

44. [Karpathy, A., & Fei-Fei, L. (2015). Deep visual-semantic alignments for generating image descriptions. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.](https://arxiv.org/abs/1412.2306)

45. [Collobert, R., & Weston, J. (2008). A unified architecture for natural language processing: Deep neural networks with multitask learning. *Proceedings of the 25th International Conference on Machine Learning (ICML)*.](https://doi.org/10.1145/1390156.1390177)

46. [Boureau, Y. L., Ponce, J., & LeCun, Y. (2010). A theoretical analysis of feature pooling in visual recognition. *Proceedings of the 27th International Conference on Machine Learning (ICML)*.](https://www.jmlr.org/proceedings/papers/v9/boureau10a/boureau10a.pdf)

47. [Szegedy, C., et al. (2015). Going deeper with convolutions. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.](https://arxiv.org/abs/1409.4842)

48. [Deng, J., et al. (2009). ImageNet: A large-scale hierarchical image database. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.](https://image-net.org/papers/imagenet_cvpr09.pdf)

49. [Zhou, Z. H., et al. (2021). Deep learning: The good, the bad, and the ugly. *arXiv preprint arXiv:2105.07527*.](https://arxiv.org/abs/2105.07527)

50. [Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)*.](https://arxiv.org/abs/1505.04597)

51. [Howard, A. G., et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint arXiv:1704.04861*.](https://arxiv.org/abs/1704.04861)

52. [Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.](https://doi.org/10.1162/neco.1997.9.8.1735)

53. [Mikolov, T., et al. (2010). Recurrent neural network-based language model. *Interspeech Conference Proceedings*.](https://www.isca-speech.org/archive/interspeech_2010/mikolov10_interspeech.html)

54. [Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *arXiv preprint arXiv:1406.1078*.](https://arxiv.org/abs/1406.1078)

55. [Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/1409.3215)

56. [Graves, A. (2012). Supervised sequence labelling with recurrent neural networks. *Studies in Computational Intelligence (SCI)*.](https://doi.org/10.1007/978-3-642-24797-2)

57. [Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*.](https://arxiv.org/abs/1409.0473)

58. [Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.](https://www.deeplearningbook.org/)

59. [Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673-2681.](https://doi.org/10.1109/78.650093)

60. [Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent neural network regularization. *arXiv preprint arXiv:1409.2329*.](https://arxiv.org/abs/1409.2329)


61. [Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/1409.3215)

62. [Graves, A. (2012). Supervised sequence labelling with recurrent neural networks. *Studies in Computational Intelligence (SCI)*.](https://doi.org/10.1007/978-3-642-24797-2)

63. [Mikolov, T., et al. (2010). Recurrent neural network-based language model. *Interspeech Conference Proceedings*.](https://www.isca-speech.org/archive/interspeech_2010/mikolov10_interspeech.html)

64. [Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*.](https://arxiv.org/abs/1409.0473)

65. [Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.](https://doi.org/10.1162/neco.1997.9.8.1735)

66. [Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent neural network regularization. *arXiv preprint arXiv:1409.2329*.](https://arxiv.org/abs/1409.2329)

67. [Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673-2681.](https://doi.org/10.1109/78.650093)

68. [Vaswani, A., et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/1706.03762)

69. [Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT*.](https://arxiv.org/abs/1810.04805)

70. [Radford, A., et al. (2018). Improving language understanding by generative pre-training. *OpenAI*.](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

71. [Brown, T., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/2005.14165)

72. [Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.](https://arxiv.org/abs/1907.11692)

73. [Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*.](https://arxiv.org/abs/2010.11929)

74. [Yang, Z., et al. (2019). XLNet: Generalized autoregressive pretraining for language understanding. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/1906.08237)

75. [Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*.](https://arxiv.org/abs/1910.10683)

76. [Lan, Z., et al. (2020). ALBERT: A lite BERT for self-supervised learning of language representations. *International Conference on Learning Representations (ICLR)*.](https://arxiv.org/abs/1909.11942)

77. [Touvron, H., et al. (2021). Training data-efficient image transformers and distillation through attention. *International Conference on Machine Learning (ICML)*.](https://arxiv.org/abs/2012.12877)

78. [He, K., et al. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.](https://arxiv.org/abs/1512.03385)

79. [Chen, M., et al. (2021). Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*.](https://arxiv.org/abs/2107.03374)

80. [Clark, K., et al. (2020). Electra: Pre-training text encoders as discriminators rather than generators. *International Conference on Learning Representations (ICLR)*.](https://arxiv.org/abs/2003.10555)

81. [Lample, G., et al. (2019). Cross-lingual language model pretraining. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/1901.07291)

82. [Sun, Y., et al. (2019). ERNIE: Enhanced representation through knowledge integration. *arXiv preprint arXiv:1904.09223*.](https://arxiv.org/abs/1904.09223)

83. [Vinyals, O., et al. (2015). Pointer networks. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/1506.03134)

84. [Tay, Y., et al. (2020). Efficient transformers: A survey. *arXiv preprint arXiv:2009.06732*.](https://arxiv.org/abs/2009.06732)

85. [Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The long-document transformer. *arXiv preprint arXiv:2004.05150*.](https://arxiv.org/abs/2004.05150)

86. [Zhao, W., et al. (2021). Understanding transformers through mathematical modeling: A survey. *arXiv preprint arXiv:2103.15580*.](https://arxiv.org/abs/2103.15580)

87. [Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.](https://arxiv.org/abs/1910.03771)

88. [Dong, L., et al. (2019). Unified language model pre-training for natural language understanding and generation. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/1905.03197)

89. [Child, R., et al. (2019). Generating long sequences with sparse transformers. *arXiv preprint arXiv:1904.10509*.](https://arxiv.org/abs/1904.10509)

90. [Rae, J. W., et al. (2020). Compressive transformers for long-range sequence modeling. *arXiv preprint arXiv:1911.05507*.](https://arxiv.org/abs/1911.05507)

91. [Goyal, N., et al. (2021). Improving transformer models by revisiting model scaling. *arXiv preprint arXiv:2102.13618*.](https://arxiv.org/abs/2102.13618)

92. [He, J., et al. (2021). Efficient attention: Attention with linear complexities. *arXiv preprint arXiv:2103.12367*.](https://arxiv.org/abs/2103.12367)

93. [Liu, P., et al. (2019). RoFormer: Enhanced transformer with rotary position embedding. *arXiv preprint arXiv:2012.15723*.](https://arxiv.org/abs/2012.15723)

94. [Brown, T., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/2005.14165)

95. [Radford, A., et al. (2019). Language models are unsupervised multitask learners. *OpenAI Blog*.](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

96. [Gehring, J., et al. (2017). Convolutional sequence to sequence learning. *Proceedings of the 34th International Conference on Machine Learning (ICML)*.](https://arxiv.org/abs/1705.03122)

97. [Radford, A., et al. (2018). Improving language understanding by generative pre-training. *OpenAI*.](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

98. [Yang, Z., et al. (2019). XLNet: Generalized autoregressive pretraining for language understanding. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/1906.08237)

99. [Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*.](https://arxiv.org/abs/1910.10683)

100. [Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*.](https://arxiv.org/abs/2010.11929)

101. [Vaswani, A., et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/1706.03762)

102. [Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT*.](https://arxiv.org/abs/1810.04805)

103. [Tay, Y., et al. (2020). Efficient transformers: A survey. *arXiv preprint arXiv:2009.06732*.](https://arxiv.org/abs/2009.06732)

104. [Brown, T., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/2005.14165)

105. [He, K., et al. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.](https://arxiv.org/abs/1512.03385)

106. [Clark, K., et al. (2020). Electra: Pre-training text encoders as discriminators rather than generators. *International Conference on Learning Representations (ICLR)*.](https://arxiv.org/abs/2003.10555)

107. [Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.](https://arxiv.org/abs/1907.11692)

108. [Lan, Z., et al. (2020). ALBERT: A lite BERT for self-supervised learning of language representations. *International Conference on Learning Representations (ICLR)*.](https://arxiv.org/abs/1909.11942)

109. [Touvron, H., et al. (2021). Training data-efficient image transformers and distillation through attention. *International Conference on Machine Learning (ICML)*.](https://arxiv.org/abs/2012.12877)

110. [Child, R., et al. (2019). Generating long sequences with sparse transformers. *arXiv preprint arXiv:1904.10509*.](https://arxiv.org/abs/1904.10509)

111. [Rae, J. W., et al. (2020). Compressive transformers for long-range sequence modeling. *arXiv preprint arXiv:1911.05507*.](https://arxiv.org/abs/1911.05507)

112. [Zhao, W., et al. (2021). Understanding transformers through mathematical modeling: A survey. *arXiv preprint arXiv:2103.15580*.](https://arxiv.org/abs/2103.15580)

113. [Goyal, N., et al. (2021). Improving transformer models by revisiting model scaling. *arXiv preprint arXiv:2102.13618*.](https://arxiv.org/abs/2102.13618)

114. [He, J., et al. (2021). Efficient attention: Attention with linear complexities. *arXiv preprint arXiv:2103.12367*.](https://arxiv.org/abs/2103.12367)

115. [Liu, P., et al. (2019). RoFormer: Enhanced transformer with rotary position embedding. *arXiv preprint arXiv:2012.15723*.](https://arxiv.org/abs/2012.15723)

116. [Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*.](https://arxiv.org/abs/1301.3781)

117. [Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.](https://aclanthology.org/D14-1162/)

118. [Bojanowski, P., et al. (2017). Enriching word vectors with subword information. *Transactions of the Association for Computational Linguistics*, 5, 135-146.](https://aclanthology.org/Q17-1010/)

119. [Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.](https://doi.org/10.1162/neco.1997.9.8.1735)

120. [Vaswani, A., et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/1706.03762)

121. [Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.](https://arxiv.org/abs/1910.01108)

122. [Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems (NIPS)*.](https://dl.acm.org/doi/10.1145/3065386)

123. [Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*.](https://arxiv.org/abs/1409.1556)

124. [Perez, L., & Wang, J. (2017). The effectiveness of data augmentation in image classification using deep learning. *arXiv preprint arXiv:1712.04621*.](https://arxiv.org/abs/1712.04621)

125. [He, K., et al. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.](https://arxiv.org/abs/1512.03385)

126. [Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*.](https://arxiv.org/abs/2010.11929)

127. [Howard, A. G., et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint arXiv:1704.04861*.](https://arxiv.org/abs/1704.04861)

128. [Hinton, G., et al. (2012). Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. *IEEE Signal Processing Magazine*, 29(6), 82-97.](https://ieeexplore.ieee.org/document/6296526)

129. [Tóth, L. (2013). Convolutional deep maxout networks for phone recognition. *Proceedings of Interspeech 2013*.](https://www.isca-speech.org/archive/interspeech_2013/toth13_interspeech.html)

130. [Schneider, S., et al. (2019). Wav2Vec: Unsupervised pre-training for speech recognition. *arXiv preprint arXiv:1904.05862*.](https://arxiv.org/abs/1904.05862)

131. [Radford, A., et al. (2022). Whisper: Robust speech recognition via large-scale weak supervision. *OpenAI*.](https://cdn.openai.com/papers/whisper.pdf)

132. [Graves, A. (2013). Generating sequences with recurrent neural networks. *arXiv preprint arXiv:1308.0850*.](https://arxiv.org/abs/1308.0850)

133. [Karpathy, A., & Fei-Fei, L. (2015). Deep visual-semantic alignments for generating image descriptions. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.](https://arxiv.org/abs/1412.2306)

134. [Tran, D., et al. (2015). Learning spatiotemporal features with 3D convolutional networks. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.](https://arxiv.org/abs/1412.0767)

135. [Bertasius, G., Wang, H., & Torresani, L. (2021). Is space-time attention all you need for video understanding? *arXiv preprint arXiv:2102.05095*.](https://arxiv.org/abs/2102.05095)

136. [Zhao, Y., et al. (2021). TAI: Temporal attention-driven video understanding. *arXiv preprint arXiv:2107.10218*.](https://arxiv.org/abs/2107.10218)

137. [Ren, S., et al. (2015). Faster R-CNN: Towards real-time object detection with region proposal networks. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/1506.01497)

138. [Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/1711.09224)

139. [Prokhorenkova, L., et al. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/1706.09516)

140. [Arik, S. O., & Pfister, T. (2021). TabNet: Attentive interpretable tabular learning. *arXiv preprint arXiv:1908.07442*.](https://arxiv.org/abs/1908.07442)

141. [Chen, T., et al. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.](https://dl.acm.org/doi/10.1145/2939672.2939785)

142. [Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.](https://doi.org/10.1023/A:1010933404324)

143. [Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning (ICML)*.](https://arxiv.org/abs/2103.00020)

144. [Ramesh, A., et al. (2021). Zero-shot text-to-image generation. *Proceedings of the 38th International Conference on Machine Learning (ICML)*.](https://arxiv.org/abs/2102.12092)

145. [Rombach, R., et al. (2022). High-resolution image synthesis with latent diffusion models. *arXiv preprint arXiv:2112.10752*.](https://arxiv.org/abs/2112.10752)

146. [Kiela, D., et al. (2019). Supervised multimodal bitransformers for classifying images and text. *arXiv preprint arXiv:1909.02950*.](https://arxiv.org/abs/1909.02950)

147. [Rombach, R., et al. (2022). Latent diffusion models: High-resolution image synthesis. *arXiv preprint arXiv:2112.10752*.](https://arxiv.org/abs/2112.10752)

148. [Bengio, Y. (2012). Deep learning of representations for unsupervised and transfer learning. *Proceedings of the ICML Workshop on Unsupervised and Transfer Learning*.](https://proceedings.mlr.press/v27/bengio12a.html)

149. [Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929-1958.](https://jmlr.org/papers/v15/srivastava14a.html)

150. [Brown, T., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems (NIPS)*.](https://arxiv.org/abs/2005.14165)

151. [Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*.](https://arxiv.org/abs/1910.10683)

152. [Howard, J., & Gugger, S. (2020). Fastai: A layered API for deep learning. *Information*, 11(2), 108.](https://doi.org/10.3390/info11020108)

153. [Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.](https://arxiv.org/abs/1610.02357)









