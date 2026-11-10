# E-Commerce Purchase Prediction System

## 🚀 Live Demo

Try the deployed application here:

**[Open E-Commerce Purchase Prediction App](https://ecommerce-purchase-prediction-hh2lvreghmhfuqsnvafhm5.streamlit.app/)**

## 📌 Project Overview

The E-Commerce Purchase Prediction System is a Machine Learning project that predicts whether a website visitor is likely to make a purchase based on their browsing session activities.

The system analyzes customer session information, such as the number of pages visited, time spent on product pages, bounce rate, exit rate, and visitor type.

A Random Forest Classification model is used to predict the purchase outcome. The project also includes an interactive web application developed using Streamlit.

## 🎯 Project Objectives

- Analyze e-commerce customer browsing behaviour.
- Understand the factors related to purchasing decisions.
- Clean and prepare the dataset for machine learning.
- Train and compare different classification algorithms.
- Predict whether a visitor session will result in a purchase.
- Develop an interactive web application using Streamlit.

## 📊 Dataset

**Dataset Name:** Online Shoppers Purchasing Intention Dataset

**Source:** UCI Machine Learning Repository

[View Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

Dataset Details:

- Original records: 12,330
- Duplicate records removed: 125
- Cleaned records: 12,205
- Input features: 17
- Target variable: Revenue

The target variable `Revenue` indicates whether a shopping session resulted in a purchase.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- VS Code

## ⚙️ Project Workflow

1. **Data Collection:** Downloaded the Online Shoppers Purchasing Intention Dataset from the UCI Machine Learning Repository.

2. **Data Exploration:** Analyzed the dataset, checked data types, and explored customer purchasing behaviour.

3. **Data Cleaning:** Checked for missing values and removed 125 duplicate records.

4. **Feature Analysis:** Studied the relationship between customer browsing activities and purchase outcomes.

5. **Data Preparation:** Separated numerical and categorical features and divided the dataset into training and testing sets.

6. **Model Training:** Trained Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest classification models.

7. **Model Evaluation:** Compared the models using accuracy, precision, recall, F1-score, and confusion matrices.

8. **Model Saving:** Saved the selected Random Forest model using Joblib.

9. **Web Application:** Developed an interactive prediction application using Streamlit.

10. **Deployment:** Deployed the application using Streamlit Community Cloud.

## 🤖 Machine Learning Models

Three classification algorithms were trained and evaluated.

| Model | Accuracy |
|---|---:|
| Logistic Regression | 88.69% |
| K-Nearest Neighbors (KNN) | 87.34% |
| Random Forest | 89.59% |

### Best Performing Model: Random Forest

The Random Forest model was selected based on its overall performance.

Evaluation results:

- Accuracy: 89.59%
- Precision (Purchase): 65%
- Recall (Purchase): 72%
- F1-Score (Purchase): 68%

The model uses class weighting to help handle the imbalance between purchase and non-purchase sessions.

## 🌐 Streamlit Web Application

The project includes an interactive web application where users can enter shopping session details and receive a purchase prediction.

### Application Features

- User-friendly interface.
- Input fields for customer browsing activities.
- Numerical and categorical input options.
- Purchase prediction.
- Purchase probability display.
- Interactive prediction results.

### Application Screenshots

**1. Application Home Page**

![Application Home Page](screenshots/app-home.png)

**2. Prediction Result**

![Prediction Result](screenshots/prediction-result.png)

## 📁 Project Structure

```text
Ecommerce-Purchase-Prediction/
│
├── dataset/
│   ├── online_shoppers_intention.csv
│   └── cleaned_online_shoppers.csv
│
├── models/
│   └── best_purchase_model.pkl
│
├── screenshots/
│   ├── app-home.png
│   └── prediction-result.png
│
├── data_exploration.py
├── data_cleaning.py
├── feature_analysis.py
├── data_preparation.py
├── model_preprocessing.py
├── logistic_regression.py
├── knn_model.py
├── random_forest.py
├── random_forest_tuning.py
├── save_best_model.py
├── test_saved_model.py
├── inspect_app_inputs.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 💻 How to Run the Project Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/farzinmansoor/Ecommerce-Purchase-Prediction.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Ecommerce-Purchase-Prediction
```

### Step 3: Create a Virtual Environment

```bash
python3 -m venv .venv
```

### Step 4: Activate the Virtual Environment

For macOS and Linux:

```bash
source .venv/bin/activate
```

### Step 5: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 6: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your web browser.

## 📚 References

Sakar, C. O., & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository.

Dataset License: CC BY 4.0.

[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

## ⚠️ Disclaimer

This project is developed for educational and demonstration purposes. Predictions are based on a trained machine learning model and are not guaranteed to represent actual customer purchasing decisions.

## 👨‍💻 Author

**Farzin Mansoor**

- GitHub: [farzinmansoor](https://github.com/farzinmansoor)
- LinkedIn: [Farzin Mansoor](https://www.linkedin.com/in/farzin-mansoor-8045b3361/)

---

⭐ If you find this project interesting, feel free to explore the repository and try the live application!