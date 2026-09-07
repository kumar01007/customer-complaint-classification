# Consumer Complaint Classification

An NLP-based machine learning and deep learning project for classifying consumer complaints into financial product categories using complaint narratives.

The project compares a traditional **TF-IDF + Logistic Regression** baseline with **GloVe-based RNN and LSTM models**, and investigates the effect of **Focal Loss** under severe class imbalance.

---

## Project Overview

Consumer complaints submitted to financial institutions can belong to different product categories such as credit reporting, debt collection, mortgages, credit cards, and student loans.

Manually routing a large number of complaints to the appropriate category can be time-consuming. This project explores how NLP and deep learning can be used to automatically classify complaints based on their written narratives.

The main focus of the project is not just building a single model, but comparing different approaches and understanding how model choice and class imbalance affect performance.

---

## Dataset

The project uses the **CFPB Consumer Complaint Database**.

The original dataset contains approximately:

* **1.28 million complaints**
* **18 product categories**
* **18 columns**

Only complaints containing a **consumer complaint narrative** are used for the text-based modeling tasks.

After filtering and grouping the original product categories, the target variable was consolidated into **11 classes**:

* Credit reporting
* Debt collection
* Mortgage
* Credit card
* Bank account
* Student loan
* Loan
* Money transfer
* Vehicle loan
* Prepaid card
* Other

Approximately **383,564 complaints** contain usable complaint narratives.

### Class Imbalance

The dataset is highly imbalanced. The ratio between the largest and smallest classes is approximately **424:1**.

Because of this imbalance, accuracy alone is not sufficient for evaluating the models. Macro-level Precision, Recall, and F1-score are also considered.

---

## Project Workflow

```text
Consumer Complaint Dataset
          ↓
Data Understanding & EDA
          ↓
Select Complaints with Narratives
          ↓
Group Product Categories
          ↓
Text Preprocessing
          ↓
Train / Validation / Test Split
          ↓
TF-IDF + Logistic Regression
          ↓
GloVe Word Embeddings
          ↓
RNN
          ↓
LSTM
          ↓
LSTM + Focal Loss
          ↓
Validation-based Model Comparison
```

---

## Models

### 1. TF-IDF + Logistic Regression

A traditional machine learning baseline was developed using:

* TF-IDF vectorization
* Logistic Regression

This provides a reference point for comparing the performance of neural network-based approaches.

The trained TF-IDF vectorizer and Logistic Regression model are saved in the `models/` directory.

### 2. GloVe + RNN

Pre-trained **GloVe 50-dimensional word embeddings** were used to represent the complaint text.

A recurrent neural network was then trained to capture sequential information from the complaint narratives.

### 3. GloVe + LSTM

An LSTM-based model was implemented to better capture longer-term dependencies in complaint narratives.

The LSTM model showed a substantial improvement over the vanilla RNN on the validation set.

### 4. LSTM + Focal Loss

Because of the severe class imbalance, **Focal Loss** was investigated as an alternative training objective.

The results were compared against the standard Cross Entropy loss to determine whether focusing more strongly on difficult examples improved overall macro-level performance.

---

## Evaluation

The models were evaluated using:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1-score
* Confusion Matrix

Macro F1 was given particular importance because of the large imbalance between product categories.

### Validation Results

The validation experiments showed that:

* The **vanilla RNN** performed substantially worse than the LSTM.
* The **LSTM** achieved around **81% validation accuracy** with approximately **63% Macro F1**.
* Adding **Focal Loss** did not provide a meaningful improvement over standard Cross Entropy on the validation set.
* Based on validation performance, the **LSTM with Cross Entropy** was selected as the preferred deep learning model.

The final test set was intentionally kept untouched and was not used for the model-selection process.

---

## Repository Structure

```text
customer-complaint-classification/
│
├── models/
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── rnn_best.pth
│   ├── lstm_best.pth
│   └── lstm_focal_best.pth
│
├── outputs/
│   └── label_encoder.pkl
│
├── 01_data_understanding.ipynb
├── 02_ml_baseline.ipynb
├── 03_rnn_lstm.ipynb
│
└── README.md
```

---

## Notebooks

### `01_data_understanding.ipynb`

Covers:

* Dataset inspection
* Missing-value analysis
* Duplicate analysis
* Complaint narrative availability
* Product-category distribution
* Class imbalance analysis
* Text statistics
* Product-category grouping
* Dataset preparation

### `02_ml_baseline.ipynb`

Covers:

* Train/validation/test preparation
* Text vectorization using TF-IDF
* Logistic Regression baseline
* Model evaluation
* Saving the trained baseline model and vectorizer

### `03_rnn_lstm.ipynb`

Covers:

* GloVe embeddings
* Text sequence preparation
* RNN model
* LSTM model
* Focal Loss
* Validation-based comparison of deep learning models

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* NLTK
* GloVe
* Matplotlib
* Jupyter Notebook

---

## Key Takeaways

1. Traditional TF-IDF-based machine learning provides a useful baseline for large-scale complaint text.
2. LSTM was substantially more effective than the vanilla RNN for this dataset.
3. Severe class imbalance makes macro-level evaluation important.
4. Focal Loss did not provide a meaningful overall improvement over Cross Entropy in these experiments.
5. Model selection was based on validation performance while keeping the final test set untouched.

---

## Dataset Source

The dataset is based on the **Consumer Complaint Database** maintained by the Consumer Financial Protection Bureau (CFPB).

The raw dataset is not included in this repository.

---

## Author

**Bobbadi Kumar**

B.Tech. Electrical Engineering
Indian Institute of Technology, Bhubaneswar
