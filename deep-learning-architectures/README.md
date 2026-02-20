<h1>
  <span class="headline">Intro to Neural Networks + Overview of AI Architectures </span>
  <span class="subhead">Overview of Deep Learning Architectures</span>
</h1>


## **Lesson Objectives**
By the end of this lesson, students will be able to:
- Identify key deep learning architectures and their applications.
- Compare and contrast the differences between FNNs, CNNs, RNNs, and Transformers.
- Implement a basic CNN model for image classification.
- Load and fine-tune a pre-trained Transformer model for text classification.

---

## **What is a Deep Learning Architecture?**
Deep learning architectures define **how neural networks are structured to process and learn from data.**. Different architectures are optimized for different types of input data, such as structured numbers, images, or text.

> **Key Takeaway:** The choice of architecture impacts model performance, efficiency, and suitability for specific AI tasks.

---

## **Traditional Deep Learning Architectures**

<div class="mermaid">
graph TD;
    A[Feedforward Neural Network - FNN] -->|Structured Data| B[Credit Risk Prediction];
    A -->|Tabular Data| C[Fraud Detection];
    D[Convolutional Neural Network - CNN] -->|Images| E[Object Recognition];
    D -->|Medical Images| F[Medical Image Analysis];
    G[Recurrent Neural Network - RNN] -->|Time-Series| H[Stock Price Prediction];
    G -->|Speech Data| I[Speech Recognition];
</div>

### **Feedforward Neural Networks (FNNs)**
Used primarily for structured/tabular data. Information flows in a single direction, from input to output, without loops or memory.

### **Convolutional Neural Networks (CNNs)**
Designed for **image processing**, CNNs are designed for image and spatial data processing, using convolutional layers to automatically extract meaningful spatial patterns such as edges and textures.


### **Recurrent Neural Networks (RNNs)**
Specialized for **sequential data**, where the order of inputs matters, such as time-series data, speech, or text.


---

## **Coding Walkthrough: Implementing a Simple CNN**

**In this walkthrough we are going to achieve two things:**
1. **Train** a basic CNN to classify images from the **MNIST dataset**.
2. **Understand** how convolutional layers extract spatial features.

 
**About the MNIST Dataset:**
> The MNIST dataset is a collection of **70,000 grayscale images of handwritten digits** (0-9), commonly used for training image classification models. Each image is 28x28 pixels and labeled with its corresponding digit. MNIST serves as a benchmark dataset for experimenting with and evaluating image classification models, and deep learning techniques, making it an ideal starting point for working with Convolutional Neural Networks (CNNs).

### **Python Code:**
```python
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values

# Define a simple CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train[..., None], y_train, epochs=3, validation_data=(x_test[..., None], y_test))
```

### **Discussion Questions:**
- What does each CNN layer do in this model?
- How does this architecture differ from a simple feedforward network?

---

## **The Rise of Transformers**

### **Why Transformers?**
- Overcame the limitations of RNNs in processing sequential data.
- Introduced the concept of **self-attention**, enabling models to focus on the most relevant parts of an input sequence.


#### Sequential Processing in RNNs
<div class="mermaid">
graph LR;
    A[Input Sequence] -->|Token 1| B[RNN Step 1];
    B -->|Token 2| C[RNN Step 2];
    C -->|Token 3| D[RNN Step 3];
    D -->|Final Output| E[Prediction];
</div>

#### Parallel Processing in Transformers
<div class="mermaid">
graph LR;
    A[Input Sequence] -->|Tokenized| B[Self-Attention Layer];
    B -->|Simultaneous Computation| C[Multi-Head Attention];
    C -->|Processed Features| D[Final Output Predictions];
</div>

> **Key Takeaway**: Transformers are fasterand more scalable than RNNs because they process all input tokens in parallel using self-attention, whereas RNNs process tokens sequentially, making them slower and less efficient for long sequences.


**Example Applications:**
- **Text Processing:** GPT, BERT, T5
- **Image Analysis:** Vision Transformers (ViTs)

---

## **Coding Walkthrough: Fine-Tuning a Transformer for Text Classification**

### **Objective:**
- Load a pre-trained Transformer model (**DistilBERT**) and fine-tune it for sentiment classification.

### **Python Code:**
```python
from transformers import pipeline

# Load a pre-trained Transformer model for text classification
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Test with sample sentences
print(classifier("This movie was amazing!"))
print(classifier("I did not enjoy this product at all."))
```

### **Discussion Questions:**
- What advantages do Transformers have over RNNs?
- Why are pre-trained models important in deep learning?

---

## **From Transformers to Large Language Models (LLMs)**

### **What is an LLM?**
Large Language Models (LLMs) are Transformers trained at **massive scale** — on billions or trillions of tokens of text — giving them broad, general-purpose language capabilities. Where earlier Transformer models like BERT were trained for a specific task (e.g., classification), LLMs like GPT are trained to understand and generate language across virtually any topic or task.

> **Key Takeaway:** An LLM is not a fundamentally different architecture — it is the Transformer architecture pushed to scale, with the emergent ability to generalize far beyond any single task.

### **The Shift: From Task-Specific to General-Purpose Models**

<div class="mermaid">
graph LR;
    A[Early Transformers] -->|Trained for one task| B[BERT: Classification];
    A -->|Trained for one task| C[T5: Translation];
    D[Large Language Models] -->|Trained on broad data| E[GPT: Generation];
    D -->|One model, many tasks| F[Summarization, Q&A, Code, Chat...];
</div>

| Model Type | Example | Training Approach | Primary Use |
|---|---|---|---|
| Task-Specific Transformer | BERT | Labeled task data | Classification, NER |
| General-Purpose LLM | GPT-4, Claude, Gemini | Self-supervised on massive text | Generation, reasoning, summarization |
| Domain-Specific LLM | Med-PaLM | Fine-tuned on domain data | Healthcare Q&A, clinical notes |

### **Core LLM Capabilities**
- **Text Generation** — drafting emails, reports, and code.
- **Summarization** — condensing long documents into key points.
- **Question Answering** — retrieving and reasoning over information.
- **Classification & Extraction** — categorizing content, pulling structured data from unstructured text.
- **Conversation** — maintaining context across multi-turn dialogue.

### **Why This Matters for You**
LLMs are the foundation of the generative AI tools transforming business workflows today — from customer support chatbots to document automation to code assistants. Understanding that they are built on the same Transformer architecture you just learned means you already have the conceptual foundation.

In **Day 5**, we will go deeper: how to fine-tune LLMs for specific tasks, when to use Retrieval-Augmented Generation (RAG) instead of fine-tuning, and how LLMs power autonomous AI agent workflows.

### **Discussion Questions:**
- Can you think of a business process in your organization where an LLM's text generation or summarization capability could save time?
- What risks might come with using a general-purpose LLM versus a task-specific model?

---

## **Summary & Key Takeaways**
- Different deep learning architectures are optimized for different data types and tasks
- CNNs excel at **image and spatial data processing**, while Transformers dominate **text and multimodal applications**.
- **Selecting** the right architecture is critical for **model performance and efficiency**.
- Pre-trained models can **reduce training time and improve results**.
- LLMs are **Transformers at scale** — the same architecture, trained on vastly more data to achieve general-purpose language understanding and generation.


