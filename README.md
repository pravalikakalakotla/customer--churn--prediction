
# Customer Churn Prediction

🚀 **Live Demo:** [Try the Customer Churn Predictor](https://pravalika-churn-predictor.streamlit.app/)


## 1. Project Overview

Customer Churn Prediction is a machine learning project that predicts whether a telecom customer is likely to leave a service.

The project analyzes customer information, service subscriptions, contract details, tenure, and billing information to estimate churn probability.

## 2. Problem Statement

Customer churn can affect the revenue and growth of telecom companies. Identifying customers who may leave can help businesses understand churn patterns and plan customer-retention strategies.

This project uses machine learning to classify customers as likely to stay or churn.

## 3. Objectives

- Analyze customer churn data.
- Clean and prepare the dataset.
- Explore factors associated with churn.
- Train and compare machine learning models.
- Build an interactive web application for predictions.

## 4. Dataset

**Dataset:** IBM Telco Customer Churn

The dataset contains 7,043 customer records and 21 columns, including customer demographics, subscribed services, contract types, charges, tenure, and churn status.

The target column is `Churn`:
- `Yes` — customer churned.
- `No` — customer stayed.

## 5. Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## 6. Project Workflow

1. Load the customer churn dataset.
2. Inspect and clean missing or invalid values.
3. Perform exploratory data analysis (EDA).
4. Encode categorical features and scale numerical features.
5. Train a Logistic Regression model.
6. Compare Logistic Regression with Random Forest.
7. Evaluate model performance.
8. Save the trained model.
9. Build an interactive Streamlit application.

## 7. Model Comparison

Two classification models were evaluated on the same test split.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 80.55% | 78.28% |
| Churn Precision | 65.72% | 61.89% |
| Churn Recall | 55.88% | 47.33% |
| Churn F1-score | 60.40% | 53.64% |
| ROC-AUC | 84.21% | 82.32% |

Logistic Regression performed better across the reported metrics and was selected for the prediction application.

## 8. Application Features

- Customer information input form.
- Service, contract, and billing selections.
- Churn prediction.
- Estimated churn probability.
- Model performance information.

## 9. How to Run the Project

### Install dependencies

```bash
python -m pip install pandas scikit-learn matplotlib seaborn joblib streamlit
```

### Run the data preparation scripts

Run these from the project folder, in order, if you need to reproduce the workflow:

```bash
python inspect_data.py
python clean_data.py
python eda.py
python train_model.py
```

### Start the application

```bash
python -m streamlit run app.py
```

The app should open in your browser at:

`http://localhost:8501`

## 10. Limitations

- Predictions are estimates, not guarantees of future customer behavior.
- The model was evaluated on a held-out test set from the available dataset.
- Churn recall is approximately 55.88%, so some customers who churned were missed.
- Performance on new or different customer populations may vary.

## 11. Future Enhancements

- Improve churn recall through threshold tuning and model experiments.
- Add visual summaries of customer churn patterns.
- Deploy the application online.
- Explore methods for explaining individual predictions.

## 12. Conclusion

This project demonstrates a complete beginner-level machine learning workflow, from data cleaning and exploratory analysis to model comparison and an interactive prediction application.
