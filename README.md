# **Product Review Sentiment Analyzer**

This project demonstrates the process of building a machine learning model to perform **sentiment analysis** on product reviews, classifying them as either **positive** or **negative**. It covers fundamental natural language processing (NLP) techniques and supervised learning to achieve this goal. 📊

***

### **Key Objectives**

* **NLP Basics**: Understand and apply foundational NLP concepts, including tokenization, stop word removal, and stemming, to prepare text data for analysis.
* **Text Vectorization**: Convert raw text data into numerical features that a machine learning model can process. This project uses techniques like **CountVectorizer** or **TF-IDF** to transform text into a feature matrix.
* **Binary Classification**: Train a machine learning model to perform binary classification, which means it learns to predict one of two outcomes: "positive" or "negative" sentiment.

***

### **Core Deliverables**

* **Preprocessing Pipeline**: A complete script or function that handles the cleaning and preparation of the text data, from raw reviews to a usable format.
* **Sentiment Classifier**: A trained machine learning model, such as a **Logistic Regression** or **Support Vector Machine (SVM)**, capable of predicting the sentiment of a given review.
* **F1-Score Report**: A report evaluating the model's performance using the **F1-score**, which provides a balanced measure of the model's precision and recall.

***

### **Project Workflow**

1.  **Data Loading**: The dataset of product reviews and their corresponding sentiment labels is loaded.
2.  **Preprocessing**: The raw text reviews are cleaned and preprocessed using the defined pipeline.
3.  **Vectorization**: The cleaned text is converted into a numerical feature matrix.
4.  **Model Training**: A classification model is trained on the vectorized data.
5.  **Model Evaluation**: The trained model's performance is evaluated on a test set, and the F1-score report is generated.

***

### **Technologies Used**

* **Python**: The primary programming language for the project.
* **NLTK (Natural Language Toolkit)**: A library for various text preprocessing tasks.
* **scikit-learn**: A machine learning library used for text vectorization, model training, and performance evaluation.
