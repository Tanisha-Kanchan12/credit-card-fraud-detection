#  Credit Card Fraud Detection System

This is a Machine Learning project where I built a model to detect
fraudulent credit card transactions.  
The goal of the project is to correctly identify fraud cases from highly
imbalanced transaction data.

The project also includes a simple web application that demonstrates
model predictions on real (anonymized) transaction samples.


## Problem Statement

Credit card fraud causes significant financial loss for customers and banks.
Since fraudulent transactions are very rare compared to legitimate ones,
traditional accuracy-based models can be misleading.

This project focuses on detecting fraud while minimizing missed fraud cases
(false negatives).

## Tools & Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Streamlit  


## Dataset Information

The dataset contains credit card transactions made by European cardholders.

- Total transactions: **284,807**
- Fraud cases: **492 (0.172%)**
- The dataset is **highly imbalanced**

### Columns:
- `V1` – `V28`: Anonymized features obtained using **PCA transformation**
- `Time`: Seconds elapsed since the first transaction
- `Amount`: Transaction amount
- `Class`:
  - `0` → Normal transaction  
  - `1` → Fraud transaction  

Due to confidentiality, original feature details are not available.


## Exploratory Data Analysis (EDA)

During EDA, I:
- Analyzed the severe class imbalance between fraud and normal transactions
- Visualized transaction amount distribution for fraud vs normal cases
- Checked data types and missing values
- Compared statistical measures across classes

This helped in understanding fraud patterns and dataset limitations.

## Machine Learning Model

- Algorithm used: **Logistic Regression**
- Target variable: **`Class`**
- Train-test split: **80% training, 20% testing**
- Class imbalance handled using:
  - Undersampling
  - `class_weight = 'balanced'`

Logistic Regression was chosen for its simplicity, interpretability, and
effectiveness for binary classification problems.


## Model Evaluation

Since the dataset is highly imbalanced, accuracy alone is not reliable.
The model was evaluated using:

- Confusion Matrix
- Precision, Recall, and F1-score
- **Precision–Recall Curve**
- ROC Curve and ROC-AUC Score

Special focus was given to **Recall for the fraud class**, as missing a fraud
is more costly than flagging a normal transaction.


## Web Application

A demo web application was built using **Streamlit**.

- Real user input is avoided because features are anonymized (PCA-based)
- The app randomly selects a real transaction from the dataset
- Displays:
  - Fraud prediction
  - Fraud probability

This demonstrates the complete ML pipeline from training to deployment.


## Project Structure

- `app.py` → Streamlit demo application  
- `creditcard.csv` → Dataset  
- `fraud_detection.ipynb` → EDA & model training  
- `fraud_model.pkl` → Trained model  
- `requirements.txt` → Required libraries  


## Key Insights

- Fraud cases are extremely rare compared to normal transactions
- Accuracy is misleading for fraud detection problems
- Recall is the most important metric for identifying fraud
- Logistic Regression provides a strong and interpretable baseline model

## “Dataset not included due to size and confidentiality constraints.”
