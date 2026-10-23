# SMS Spam Classification using NLP

A Natural Language Processing (NLP) project exploring different text representation techniques for classifying SMS messages as **Spam** or **Ham (Legitimate)** using machine learning.

This project compares three approaches to text representation: **Bag of Words (BoW), TF-IDF, and pretrained Word2Vec embeddings**. The goal is to understand how different representations of text influence classification performance.

## Project Overview

Text data cannot be directly processed by most traditional machine learning algorithms. It must first be transformed into numerical representations.

In this project, I explored how different NLP techniques convert text into numerical features and how these features can be used to build a spam classification model.

The project covers text preprocessing, feature extraction, model training, and performance evaluation.

## Objectives

- Understand the fundamentals of text preprocessing in NLP.
- Implement Bag of Words and TF-IDF for text vectorization.
- Explore pretrained Word2Vec embeddings and their application to text classification.
- Train and evaluate machine learning classification models.
- Compare model performance using accuracy, precision, recall, and F1-score.
- Understand the strengths and limitations of different text representation techniques.

## Technologies Used

- **Python** — Core programming language
- **Pandas & NumPy** — Data manipulation and numerical operations
- **NLTK** — Text preprocessing, stop-word removal, and stemming
- **Scikit-learn** — Text vectorization, model training, and evaluation
- **Gensim** — Working with Word2Vec embeddings
- **Jupyter Notebook** — Experimentation, implementation, and documentation

## Project Workflow

### 1. Data Preprocessing

The text data was prepared before being passed to the machine learning models. The preprocessing workflow included:

- Converting text to lowercase
- Removing unwanted characters using regular expressions
- Tokenizing text into individual words
- Removing English stop words
- Applying Porter stemming

These steps help standardize the text and reduce unnecessary variation.

### 2. Bag of Words (BoW)

Bag of Words represents a document using the occurrence or frequency of words, without considering their order or deeper semantic meaning.

In this project, `CountVectorizer` was used with a vocabulary limit of 2,500 features and binary word-occurrence features.

A **Multinomial Naive Bayes** classifier was used for classification.

### 3. TF-IDF

TF-IDF stands for *Term Frequency–Inverse Document Frequency*. It assigns weights to words based on their frequency in a document and how common or rare they are across the document collection.

The implementation used `TfidfVectorizer` with a maximum of 2,500 features and both unigrams and bigrams.

A **Multinomial Naive Bayes** classifier was used to classify the resulting feature vectors.

### 4. Pretrained Word2Vec

Word2Vec represents words as dense numerical vectors learned from their surrounding contexts. Unlike count-based representations, these embeddings can capture relationships between words.

A pretrained Word2Vec model was used to obtain word embeddings. These word vectors were then used to represent SMS messages for classification.

This experiment explored whether semantic word representations could outperform count-based methods on the SMS spam classification task.

## Results

The following results were obtained during experimentation:

| Technique | Accuracy | Spam Precision | Spam Recall | Spam F1-Score |
|---|---:|---:|---:|---:|
| Bag of Words | 98% | 90% | 93% | 92% |
| TF-IDF | 98% | 99% | 87% | 93% |
| Pretrained Word2Vec | 95.87% | 89% | 79% | 84% |

*Note: These are the reported results from the experiments. A controlled comparison requires the same train/test split and evaluation procedure for every model.*

### Key Observations

- **Bag of Words** achieved strong overall accuracy and detected a high proportion of spam messages.
- **TF-IDF** achieved a strong spam F1-score, balancing precision and recall.
- **Pretrained Word2Vec** achieved 95.87% accuracy, but its spam recall was lower than that of the count-based approaches.
- More sophisticated text representations do not automatically lead to better classification performance.
- Evaluating precision, recall, and F1-score is particularly important when the dataset contains more legitimate messages than spam messages.

## Key Learnings

Through this project, I developed a better understanding of:

- Text preprocessing and feature engineering for NLP.
- The differences between count-based and embedding-based text representations.
- How vectorization techniques affect downstream machine learning models.
- Model evaluation using accuracy, precision, recall, and F1-score.
- The practical trade-offs between different NLP approaches.

The main takeaway was that the best text representation depends on the problem and dataset. For spam detection, word frequencies and distinctive phrases can be highly informative, making BoW and TF-IDF effective baselines.

## How to Run the Project

1. Clone or download this repository.
2. Install the required Python libraries.
3. Open the Jupyter Notebook.
4. Run the cells in sequence to reproduce the preprocessing, vectorization, training, and evaluation steps.

Install the main dependencies with:

```bash
pip install pandas numpy nltk scikit-learn gensim jupyter
```

**Note:** The pretrained Word2Vec model may require a separate download and additional storage, depending on the model used in the notebook. The dataset must also be available at the path expected by the notebook.

## Future Improvements

- Train Word2Vec embeddings on a larger text corpus and compare them with pretrained embeddings.
- Evaluate all approaches using an identical train/test split.
- Investigate techniques to improve spam recall.
- Experiment with alternative classifiers and additional NLP representations.
- Explore more advanced approaches such as FastText or transformer-based text embeddings.

## Conclusion

This project provided hands-on experience with fundamental NLP techniques and their application to a real-world text classification problem. By implementing and comparing Bag of Words, TF-IDF, and pretrained Word2Vec, I explored how text representation influences model performance.

It strengthened my understanding of NLP preprocessing, feature extraction, classification, and evaluation, while highlighting the importance of selecting techniques based on empirical results rather than complexity alone.

---

**Skills:** Python · Natural Language Processing · Text Preprocessing · Feature Engineering · Word Embeddings · Scikit-learn · Machine Learning

