# E-Commerce Purchase Prediction System

## 🚀 Live Demo

Try the deployed application here:

[**Open E-Commerce Purchase Prediction App**](https://ecommerce-purchase-prediction-hh2lvreghmhfuqsnvafhm5.streamlit.app/)

## 📌 Project Overview

The E-Commerce Purchase Prediction System is a Machine Learning project that predicts whether an e-commerce browsing session is likely to result in a purchase based on visitor activities.

The system analyzes customer session information, such as the number of pages visited, time spent on product pages, bounce rate, exit rate, page values, and visitor type.

A Random Forest Classification model is used to predict the purchase outcome. The project also includes an interactive web application developed using Streamlit.

## 🎯 Project Objectives

- Analyze e-commerce customer browsing behaviour.
- Understand the factors associated with purchasing decisions.
- Clean and prepare the dataset for machine learning.
- Train and compare different classification algorithms.
- Evaluate model performance using suitable classification metrics.
- Predict whether a visitor session will result in a purchase.
- Develop and deploy an interactive web application using Streamlit.

## 📊 Dataset

**Dataset Name:** Online Shoppers Purchasing Intention Dataset

**Source:** UCI Machine Learning Repository

[View Original Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

### Dataset Details

| Description               | Value                 |
| ------------------------- | --------------------- |
| Original records          | 12,330                |
| Duplicate records removed | 125                   |
| Cleaned records           | 12,205                |
| Input features            | 17                    |
| Target variable           | Revenue               |
| File format               | CSV                   |
| Project type              | Binary Classification |

### Dataset Description

The dataset contains information about visitor browsing activities on an e-commerce website.

Each record represents an individual browsing session. The dataset includes numerical and categorical features describing page visits, browsing duration, visitor characteristics, and session information.

The dataset was collected from the UCI Machine Learning Repository and organized using Python and Pandas.

### Input Features

The dataset contains 17 input features.

| No. | Feature                 | Description                                                          |
| --- | ----------------------- | -------------------------------------------------------------------- |
| 1   | Administrative          | Number of administrative pages visited during a session              |
| 2   | Administrative_Duration | Time spent on administrative pages in seconds                        |
| 3   | Informational           | Number of informational pages visited                                |
| 4   | Informational_Duration  | Time spent on informational pages in seconds                         |
| 5   | ProductRelated          | Number of product-related pages visited                              |
| 6   | ProductRelated_Duration | Time spent on product-related pages in seconds                       |
| 7   | BounceRates             | Average bounce-rate metric for pages visited during the session      |
| 8   | ExitRates               | Average exit-rate metric for pages visited during the session        |
| 9   | PageValues              | Average value of visited pages in relation to completed transactions |
| 10  | SpecialDay              | Closeness of the session date to a special shopping day              |
| 11  | Month                   | Month in which the browsing session occurred                         |
| 12  | OperatingSystems        | Encoded category representing the visitor's operating system         |
| 13  | Browser                 | Encoded category representing the visitor's browser                  |
| 14  | Region                  | Encoded category representing the visitor's geographical region      |
| 15  | TrafficType             | Encoded category representing the source or type of website traffic  |
| 16  | VisitorType             | Visitor category: New Visitor, Returning Visitor, or Other           |
| 17  | Weekend                 | Indicates whether the session occurred on a weekend                  |

The OperatingSystems, Browser, Region, and TrafficType features contain numerical category identifiers. These identifiers are treated as categorical variables rather than continuous numerical measurements.

### Target Variable

**Revenue** is the target variable used for prediction.

It contains two possible values:

- **True:** The browsing session resulted in a purchase.
- **False:** The browsing session did not result in a purchase.

The objective of the machine learning models is to predict the Revenue value from the 17 input features.

### Label Generation Method

The target labels were provided in the original UCI dataset.

The Revenue variable indicates whether a transaction occurred during the recorded browsing session.

The project uses these existing labels for supervised machine learning. No artificial purchase labels were generated.

### Dataset Collection and Storage

The original dataset was downloaded from the UCI Machine Learning Repository and stored as:

`dataset/online_shoppers_intention.csv`

After preprocessing and duplicate removal, the cleaned dataset was saved as:

`dataset/cleaned_online_shoppers.csv`

### Data Cleaning

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Inspected the number of rows and columns.
3. Checked feature names and data types.
4. Checked for missing values.
5. Identified duplicate records.
6. Removed 125 duplicate records.
7. Saved the cleaned dataset in CSV format.

No missing values were found in the original dataset.

After cleaning, the dataset contained 12,205 records.

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
- Git and GitHub

## ⚙️ Project Workflow

1. **Data Collection:** Downloaded the Online Shoppers Purchasing Intention Dataset from the UCI Machine Learning Repository.

2. **Data Exploration:** Inspected the dataset, checked data types, and analyzed customer purchasing behaviour.

3. **Data Cleaning:** Checked for missing values and removed 125 duplicate records.

4. **Feature Analysis:** Studied relationships between browsing activities and purchase outcomes.

5. **Data Preparation:** Separated numerical and categorical features, applied preprocessing, and divided the dataset into training and testing sets.

6. **Model Training:** Trained Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest classification models.

7. **Model Evaluation:** Compared models using accuracy, precision, recall, F1-score, and confusion matrices.

8. **Hyperparameter Tuning:** Applied RandomizedSearchCV to optimize the Random Forest model.

9. **Model Saving:** Saved the selected Random Forest pipeline using Joblib.

10. **Web Application:** Developed an interactive prediction application using Streamlit.

11. **Deployment:** Deployed the application using Streamlit Community Cloud.

## 🔍 Data Preprocessing

The following preprocessing techniques were applied before model training.

### Numerical Features

Numerical features were scaled using StandardScaler where required.

### Categorical Features

Categorical features were converted into numerical representations using OneHotEncoder.

### Training and Testing Split

The cleaned dataset was divided into:

- Training records: 9,764
- Testing records: 2,441
- Training percentage: 80%
- Testing percentage: 20%

Stratified sampling was used to maintain a similar distribution of purchase and non-purchase records across the training and testing sets.

Preprocessing was implemented using Scikit-learn pipelines to help prevent data leakage.

## 📈 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand customer browsing behaviour and identify patterns related to purchasing decisions.

The analysis included:

- Distribution of purchase and non-purchase sessions.
- Comparison of purchasing behaviour across visitor types.
- Analysis of time spent on product-related pages.
- Correlation analysis of numerical features.
- Examination of PageValues and purchasing outcomes.

### Important Observations

**1. Class Imbalance**

The dataset contains substantially more non-purchase sessions than purchase sessions.

After cleaning:

- No Purchase: 10,297 sessions
- Purchase: 1,908 sessions

This imbalance makes precision, recall, and F1-score important evaluation metrics alongside accuracy.

**2. PageValues**

Sessions resulting in purchases generally had higher PageValues.

Approximate average PageValues:

- No Purchase: 2.00
- Purchase: 27.26

**3. Product-Related Duration**

Sessions resulting in purchases generally involved more time spent on product-related pages.

Approximate average ProductRelated_Duration:

- No Purchase: 1,082.98 seconds
- Purchase: 1,876.21 seconds

These observations describe associations in the dataset and do not establish that a feature directly causes a purchase.

## 🤖 Machine Learning Models

Three traditional machine learning classification algorithms were trained and evaluated.

| Model                     |   Accuracy |
| ------------------------- | ---------: |
| Logistic Regression       |     88.69% |
| K-Nearest Neighbors (KNN) |     87.34% |
| Random Forest             | **89.59%** |

### Best Performing Model: Random Forest

The Random Forest model was selected based on its overall performance, particularly its ability to identify purchase sessions.

Evaluation results:

- Accuracy: 89.59%
- Precision (Purchase): 65%
- Recall (Purchase): 72%
- F1-Score (Purchase): 68%

The model uses class weighting to help handle the imbalance between purchase and non-purchase sessions.

## 🔧 Hyperparameter Tuning

RandomizedSearchCV was used to tune the Random Forest model.

The tuning process used 3-fold cross-validation and explored the following hyperparameters:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

### Best Hyperparameters

| Hyperparameter    | Value |
| ----------------- | ----- |
| n_estimators      | 300   |
| max_depth         | 30    |
| min_samples_split | 2     |
| min_samples_leaf  | 1     |

### Baseline vs Tuned Random Forest

| Metric               | Baseline Random Forest | Tuned Random Forest |
| -------------------- | ---------------------: | ------------------: |
| Accuracy             |                 89.59% |              89.43% |
| Precision (Purchase) |                    65% |                 65% |
| Recall (Purchase)    |                    72% |                 71% |
| F1-Score (Purchase)  |                    68% |                 68% |

The tuned model did not improve the test-set results.

Therefore, the original Random Forest model was retained as the selected model for deployment.

## 💾 Saved Machine Learning Model

The selected Random Forest model was saved using Joblib.

Saved model path:

`models/best_purchase_model.pkl`

The saved object includes the preprocessing pipeline and trained classifier, allowing new input data to be transformed and classified consistently.

The saved model was loaded and tested to verify that its predictions matched the original model.

## 🌐 Streamlit Web Application

The project includes an interactive web application where users can enter shopping session details and receive a purchase prediction.

### Application Features

- User-friendly interface.
- Input fields for customer browsing activities.
- Numerical and categorical input options.
- Helpful descriptions for input fields.
- Purchase prediction.
- Estimated purchase probability.
- Interactive prediction results.

### Application Screenshots

**1. Application Home Page**



**2. Prediction Result**



### Live Application

[**Try the E-Commerce Purchase Prediction App**](https://ecommerce-purchase-prediction-hh2lvreghmhfuqsnvafhm5.streamlit.app/)

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

## ⚠️ Limitations and Responsible Use

- The model was trained on a specific historical e-commerce dataset and may not generalize to all online stores.
- The dataset contains more non-purchase sessions than purchase sessions, which can affect model performance.
- Predictions depend on the quality and relevance of the input features.
- The application does not automatically collect live website visitor data.
- Purchase probabilities are model estimates and do not guarantee an actual purchase.
- The project is intended for educational and demonstration purposes.

## 📚 References

Sakar, C. O., & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository.

Dataset License: CC BY 4.0.

[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

## 👨‍💻 Author

**Farzin Mansoor**

- GitHub: [https://github.com/farzinmansoor]
- LinkedIn: [https://www.linkedin.com/in/farzin-mansoor-8045b3361/]

---

⭐ If you find this project interesting, feel free to explore the repository and try the live application!