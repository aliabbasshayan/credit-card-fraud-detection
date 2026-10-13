# Credit Card Fraud Detection

A Machine Learning project that detects potentially fraudulent
credit card transactions using Logistic Regression.

## Features
- Fraud and normal transaction classification
- Fraud probability prediction
- Gradio web interface
- Feature scaling before prediction
- Model evaluation using Accuracy and ROC-AUC

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Gradio
- Joblib
- Jupyter Notebook

## Model Performance
- Accuracy: 97.53%
- ROC-AUC: 92.46%

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Make sure `model.pkl` and `scaler.pkl` are
present in the project directory.

## Disclaimer
This project is for educational purposes only.
Predictions should not be used as the sole basis
for real-world financial decisions.
