# Fake News Detection using Logistic Regression

This project involves the development of a machine learning model to detect fake news using logistic regression. It leverages natural language processing (NLP) techniques to preprocess news text data, convert it into numerical format using the TF-IDF method, and then classify the news articles as "real" or "fake."

## Table of Contents
1. [Objective](#objective)
2. [Dependencies](#dependencies)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Development](#model-development)
6. [Model Evaluation](#model-evaluation)
7. [Building a Predictive System](#building-a-predictive-system)
8. [How to Use](#how-to-use)
9. [License](#license)

---

## Objective

The goal of this project is to build a logistic regression model that can predict whether a given news article is real or fake based on its content. The model uses text data preprocessing, including tokenization, stemming, and stopword removal, to transform raw news articles into a format that can be fed into a machine learning model.

---

## Dependencies

The following libraries and modules are required to run this project:

- `numpy` - For numerical operations
- `pandas` - For data manipulation and analysis
- `re` - For regular expression operations
- `nltk` - For natural language processing tasks
- `sklearn` - For machine learning functionalities

You can install the required libraries using:

```bash
pip install numpy pandas scikit-learn nltk
```

Additionally, you may need to download the NLTK stopwords by running the following code before using them:

```python
import nltk
nltk.download('stopwords')
```

---

## Dataset

The dataset used in this project is a CSV file containing news articles labeled as "real" (0) or "fake" (1). The dataset has the following columns:

- `id`: Unique identifier for the article
- `title`: Title of the news article
- `author`: Author of the article
- `text`: The text content of the article
- `label`: Binary label indicating whether the news is real (0) or fake (1)

The dataset is loaded into a pandas DataFrame for easy manipulation and analysis.

---

## Data Preprocessing

The data preprocessing steps include:

1. **Handling Missing Values**: 
   - Missing values in the `author` and `title` columns are replaced with empty strings.

2. **Merging Text Data**: 
   - The `author` and `title` columns are merged to create a `content` column, which is used as input for the model.

3. **Text Cleaning**:
   - Non-alphabetical characters (including numbers and punctuation) are removed from the text.
   - All text is converted to lowercase.
   - Stopwords (common words like "and", "the", etc.) are removed.
   - Stemming is applied to reduce words to their root form.

4. **Vectorization**:
   - TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the cleaned text into numerical vectors.

---

## Model Development

A logistic regression model is used to classify news articles as real or fake. The process includes:

1. **Splitting the Dataset**: 
   - The dataset is split into training and testing sets using an 80-20 split ratio with stratified sampling to maintain the class balance.

2. **Training the Model**: 
   - The logistic regression model is trained on the training data.

3. **Prediction**:
   - After training, the model predicts the labels of the testing set and evaluates its performance.

---

## Model Evaluation

The performance of the logistic regression model is evaluated using the accuracy score:

- **Training Accuracy**: The model's accuracy on the training data.
- **Testing Accuracy**: The model's accuracy on the testing data.

For example, the model achieved the following accuracy scores:
- Training Accuracy: `98.64%`
- Testing Accuracy: `97.91%`

These accuracy scores show that the logistic regression model performs well at classifying real and fake news articles.

---

## Building a Predictive System

After training the logistic regression model, it can be used to classify new, unseen news articles.

To predict whether a given news article is real or fake, you can provide an input text (from the `x_test` set or new data) to the model:

```python
# Example usage
x_new = x_test[120]  # or new unseen text
prediction = model.predict(x_new)
if prediction[0] == 0:
    print('The news is Real')
else:
    print('The news is Fake')
```

This system outputs whether the news article is real (0) or fake (1).

---

## How to Use

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Download the dataset (train.csv) and place it in the appropriate directory.
4. Run the script to train the model and evaluate its performance on the testing data.
5. Use the trained model to classify new articles by passing them through the prediction function.

---
