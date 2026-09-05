# Amazon Reviews Sentiment Analysis

A deep learning project for binary sentiment classification of Amazon customer reviews using CNN, LSTM, and CNN-LSTM models.

## Dataset

**Amazon Reviews for Sentiment Analysis**

[Kaggle Dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)

The dataset contains **4 million Amazon reviews**:

- **3.6M training reviews**
- **400K test reviews**
- **2 classes:** Positive and Negative
- **1.8M positive + 1.8M negative** reviews in the training data
- `__label__1` → Negative
- `__label__2` → Positive

The dataset was downloaded using **KaggleHub** and analyzed before model training.

## Project Workflow

Dataset → EDA → Text Preprocessing → Tokenization → TF-IDF Models → Deep Learning Models → Model Comparison → Final Training → Evaluation

## EDA

EDA was performed in **Jupyter Notebook using Anaconda**.

The analysis included:

- Dataset structure and class distribution
- Missing values and duplicate reviews
- Review length and word count
- Common words and text patterns
- HTML tags and sentiment-related text features

## Text Preprocessing

- Converted text to lowercase
- Removed HTML tags
- Removed extra whitespace
- Preserved punctuation and negation words such as `not`
- Tokenized using Keras Tokenizer
- Vocabulary size: **30,000**
- Maximum sequence length: **200**

## Model Comparison

A **balanced subset of 500,000 training reviews (250,000 per class)** was used for computationally efficient model comparison.

| Model | Accuracy |
|---|---:|
| Logistic Regression | 93.00% |
| XGBoost | 88.08% |
| CNN | 93.19% |
| LSTM | 93.04% |
| CNN-LSTM | **93.38%** |

CNN-LSTM achieved the highest accuracy and was selected for final training.

## Model Training

Model training was performed in **Google Colab using GPU acceleration**.

The selected CNN-LSTM model was retrained using **3.24M training reviews** and evaluated on **300K previously untouched test reviews**.

### CNN-LSTM Architecture

Embedding → Conv1D → MaxPooling → LSTM → Dense → Dropout → Output

## Final Results

**Test Accuracy: 95.35%**

| Metric | Score |
|---|---:|
| Accuracy | **95.35%** |
| Precision | 95–96% |
| Recall | 95–96% |
| F1-Score | 95% |

## Model and Tokenizer

- `cnn_lstm_final.keras` — trained CNN-LSTM sentiment classification model.
- `amazon_tokenizer.json.gz` — compressed Keras tokenizer used to convert review text into sequences.
- `amazon_config.json` — stores model configuration such as vocabulary size and maximum sequence length.

The tokenizer is stored in compressed `.json.gz` format to reduce file size and make it suitable for Git LFS storage.

## Repository Structure

```text
Sentiment_Analysis/
├── Reviews_EDA.ipynb
├── model_Train.ipynb
├── cnn_lstm_final.keras
├── amazon_tokenizer.json.gz
├── amazon_config.json
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **XGBoost**
- **TensorFlow / Keras**
- **Jupyter Notebook**
- **Anaconda**
- **Google Colab**
- **Git**
- **Git LFS**
- **GitHub**
