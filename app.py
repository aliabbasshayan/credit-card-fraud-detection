
import gradio as gr
import pandas as pd
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

feature_names = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17",
    "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28",
    "Amount"
]


def predict_fraud(*values):
    input_data = pd.DataFrame(
        [values],
        columns=feature_names
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        result = "🚨 FRAUDULENT TRANSACTION"
    else:
        result = "✅ NORMAL TRANSACTION"

    return result, f"{probability:.2%}"


inputs = [
    gr.Number(label=feature, value=0)
    for feature in feature_names
]

with gr.Blocks(title="Credit Card Fraud Detection") as demo:

    gr.Markdown("# 💳 Credit Card Fraud Detection")
    gr.Markdown(
        "Enter transaction details to predict whether "
        "a transaction is normal or fraudulent."
    )

    with gr.Group():
        gr.Markdown("### Transaction Details")

        with gr.Row():
            with gr.Column():
                for i in range(0, 10):
                    inputs[i].render()

            with gr.Column():
                for i in range(10, 20):
                    inputs[i].render()

            with gr.Column():
                for i in range(20, 30):
                    inputs[i].render()

    predict_btn = gr.Button(
        "🔍 Predict Transaction",
        variant="primary"
    )

    gr.Markdown("### Prediction Result")

    prediction_output = gr.Textbox(
        label="Transaction Status",
        interactive=False
    )

    probability_output = gr.Textbox(
        label="Fraud Probability",
        interactive=False
    )

    predict_btn.click(
        fn=predict_fraud,
        inputs=inputs,
        outputs=[
            prediction_output,
            probability_output
        ]
    )

    gr.Markdown(
        """
        ---
        ### 📊 Model Information

        - **Model:** Logistic Regression
        - **Accuracy:** 97.53%
        - **ROC-AUC:** 92.46%

        *V1–V28 are anonymized features from the dataset.*
        """
    )

demo.launch()
