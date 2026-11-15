
# ==========================================
# E-COMMERCE PURCHASE PREDICTION SYSTEM
# MACHINE LEARNING + DEEP LEARNING DASHBOARD
# ==========================================

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="E-Commerce Purchase Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_purchase_model.pkl"
VISUALIZATION_DIR = BASE_DIR / "visualizations"
EXPERIMENT_CSV = BASE_DIR / "models" / "ann_experiment_results.csv"


# ==========================================
# 2. LOAD RANDOM FOREST MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def show_image(filename, caption=None):
    image_path = VISUALIZATION_DIR / filename

    if image_path.exists():
        st.image(
            str(image_path),
            caption=caption,
            use_container_width=True
        )
    else:
        st.info(
            f"Visualization not found: {filename}"
        )


def show_metric_cards(metrics):
    columns = st.columns(len(metrics))

    for column, (label, value) in zip(columns, metrics):
        with column:
            st.metric(label, value)


# ==========================================
# 4. HEADER
# ==========================================

st.title("🛒 E-Commerce Purchase Prediction System")

st.write(
    "An interactive Machine Learning and Deep Learning "
    "project for analyzing online shopping behaviour "
    "and predicting customer purchases."
)

st.caption(
    "Dataset: UCI Online Shoppers Purchasing Intention | "
    "12,205 cleaned sessions | 17 input features"
)

st.divider()


# ==========================================
# 5. MAIN NAVIGATION
# ==========================================

tab_prediction, tab_eda, tab_models, tab_ann = st.tabs([
    "🛒 Purchase Prediction",
    "📊 Data Analysis",
    "🤖 Model Comparison",
    "🧠 ANN Experiments"
])


# ==========================================
# TAB 1: PURCHASE PREDICTION
# ==========================================

with tab_prediction:

    st.header("Customer Purchase Prediction")

    st.write(
        "Enter customer browsing information to predict "
        "whether the shopping session is likely to "
        "result in a purchase."
    )

    st.info(
        "How to use: Enter the customer's actual browsing "
        "information and click Predict Purchase. "
        "The Random Forest model will estimate the result."
    )

    st.caption(
        "Prediction model: Random Forest | "
        "Test accuracy: 89.59% | "
        "Purchase-class F1-score: approximately 0.68"
    )

    # --------------------------------------
    # INPUT FIELD EXPLANATIONS
    # --------------------------------------

    with st.expander(
        "📖 Understand the input fields before starting"
    ):

        st.markdown("""
        **Page Information**
        - Administrative: Number of account-related pages visited.
        - Informational: Number of help or information pages visited.
        - Product Related: Number of product pages visited.
        - Duration fields: Total time spent on those pages, in seconds.

        **Session Information**
        - Bounce Rate: Dataset metric describing how frequently pages are associated with single-page sessions.
        - Exit Rate: Dataset metric describing how frequently pages are the final page in a session.
        - Page Values: Dataset metric related to the contribution of pages to purchase goals. It is not the product price.
        - Special Day: Dataset measure of closeness to a special day.

        **Category Information**
        - Region, Operating System, Browser and Traffic Type use numerical category codes from the original dataset.
        - Their exact real-world names are not provided.
        - Do not change the codes or guess their meanings.
        """)

    st.divider()

    # --------------------------------------
    # CUSTOMER INPUT FORM
    # --------------------------------------

    with st.form("purchase_prediction_form"):

        st.subheader("Customer Browsing Information")

        col1, col2 = st.columns(2)

        # ----------------------------------
        # LEFT COLUMN: PAGE INFORMATION
        # ----------------------------------

        with col1:

            st.markdown("### 📄 Page Information")

            administrative = st.number_input(
                "Administrative Pages",
                min_value=0,
                max_value=27,
                value=0,
                step=1,
                help=(
                    "Number of account-related pages visited, "
                    "such as login or account pages."
                )
            )

            administrative_duration = st.number_input(
                "Administrative Duration (seconds)",
                min_value=0.0,
                max_value=3398.75,
                value=0.0,
                step=1.0,
                help=(
                    "Total time spent on administrative pages, "
                    "measured in seconds."
                )
            )

            informational = st.number_input(
                "Informational Pages",
                min_value=0,
                max_value=24,
                value=0,
                step=1,
                help=(
                    "Number of information pages visited, "
                    "such as FAQ or contact pages."
                )
            )

            informational_duration = st.number_input(
                "Informational Duration (seconds)",
                min_value=0.0,
                max_value=2549.375,
                value=0.0,
                step=1.0,
                help=(
                    "Total time spent on informational pages, "
                    "measured in seconds."
                )
            )

            product_related = st.number_input(
                "Product-Related Pages",
                min_value=0,
                max_value=705,
                value=1,
                step=1,
                help=(
                    "Number of product pages visited, "
                    "such as shoes, shirts or mobiles."
                )
            )

            product_related_duration = st.number_input(
                "Product-Related Duration (seconds)",
                min_value=0.0,
                max_value=63973.52223,
                value=60.0,
                step=1.0,
                help=(
                    "Total time spent viewing product pages, "
                    "measured in seconds."
                )
            )

            page_values = st.number_input(
                "Page Values",
                min_value=0.0,
                max_value=361.7637419,
                value=0.0,
                step=0.1,
                help=(
                    "A dataset metric related to the "
                    "contribution of pages to purchase goals. "
                    "It is not the product price."
                )
            )

        # ----------------------------------
        # RIGHT COLUMN: SESSION INFORMATION
        # ----------------------------------

        with col2:

            st.markdown("### 📊 Session Information")

            bounce_rates = st.number_input(
                "Bounce Rate",
                min_value=0.0,
                max_value=0.2,
                value=0.02,
                step=0.01,
                help=(
                    "Dataset bounce-rate metric. "
                    "Example: 0.20 represents 20%."
                )
            )

            exit_rates = st.number_input(
                "Exit Rate",
                min_value=0.0,
                max_value=0.2,
                value=0.03,
                step=0.01,
                help=(
                    "Dataset metric describing how frequently "
                    "a page is the final page in a session."
                )
            )

            special_day = st.number_input(
                "Special Day",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help=(
                    "Dataset measure of how close the session "
                    "is to a special shopping day. Range: 0 to 1."
                )
            )

            month = st.selectbox(
                "Month",
                [
                    "Aug", "Dec", "Feb", "Jul", "June",
                    "Mar", "May", "Nov", "Oct", "Sep"
                ],
                help="Month in which the shopping session occurred."
            )

            visitor_type = st.selectbox(
                "Visitor Type",
                [
                    "Returning_Visitor",
                    "New_Visitor",
                    "Other"
                ],
                help=(
                    "Select whether the visitor is new, "
                    "returning or in another category."
                )
            )

            weekend = st.selectbox(
                "Weekend",
                [False, True],
                format_func=lambda value: (
                    "Yes" if value else "No"
                ),
                help=(
                    "Was the shopping session on "
                    "Saturday or Sunday?"
                )
            )

            operating_system = st.selectbox(
                "Operating System",
                list(range(1, 9)),
                format_func=lambda value: (
                    f"OS Category {value}"
                ),
                help=(
                    "Numerical category identifying the "
                    "visitor's operating system. "
                    "Exact names are not provided."
                )
            )

            browser = st.selectbox(
                "Browser",
                list(range(1, 14)),
                format_func=lambda value: (
                    f"Browser Category {value}"
                ),
                help=(
                    "Numerical category identifying the "
                    "visitor's browser. Exact names "
                    "are not provided."
                )
            )

            region = st.selectbox(
                "Region",
                list(range(1, 10)),
                format_func=lambda value: (
                    f"Region Category {value}"
                ),
                help=(
                    "Geographical category code from the "
                    "original dataset. City and state names "
                    "are not provided."
                )
            )

            traffic_type = st.selectbox(
                "Traffic Type",
                list(range(1, 21)),
                format_func=lambda value: (
                    f"Traffic Category {value}"
                ),
                help=(
                    "Category identifying how the visitor "
                    "reached the website. Exact code "
                    "descriptions are not provided."
                )
            )

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Predict Purchase",
            use_container_width=True
        )

    # --------------------------------------
    # PREDICTION RESULT
    # --------------------------------------

    if submitted:

        input_data = {
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_system,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend
        }

        input_df = pd.DataFrame([input_data])

        try:
            model = load_model()

            prediction = model.predict(input_df)[0]

            probabilities = model.predict_proba(
                input_df
            )[0]

            purchase_index = list(
                model.classes_
            ).index(True)

            purchase_probability = float(
                probabilities[purchase_index]
            )

            st.divider()
            st.subheader("🎯 Prediction Result")

            if prediction:
                st.success(
                    "The customer is likely to make "
                    "a purchase! 🛍️"
                )
            else:
                st.warning(
                    "The customer is unlikely to "
                    "make a purchase."
                )

            st.metric(
                "Estimated Purchase Probability",
                f"{purchase_probability * 100:.2f}%"
            )

            st.progress(purchase_probability)

            st.caption(
                "This is a machine learning estimate "
                "based on the entered session details. "
                "It is not a guarantee of purchase."
            )

        except Exception as error:
            st.error(
                "The prediction could not be completed. "
                "Please check the saved model and input data."
            )
            st.exception(error)


# ==========================================
# TAB 2: DATA ANALYSIS
# ==========================================

with tab_eda:

    st.header("📊 Exploratory Data Analysis")

    st.write(
        "Explore customer browsing patterns and "
        "the factors associated with purchase behaviour."
    )

    show_metric_cards([
        ("Original Sessions", "12,330"),
        ("Cleaned Sessions", "12,205"),
        ("Input Features", "17")
    ])

    st.divider()

    st.subheader("1. Purchase Distribution")

    show_image(
        "purchase_distribution.png",
        "Purchase and non-purchase session distribution"
    )

    st.write(
        "The dataset contains substantially more "
        "non-purchase sessions than purchase sessions. "
        "This class imbalance makes recall and F1-score "
        "important evaluation metrics."
    )

    st.divider()

    st.subheader("2. Visitor Purchase Behaviour")

    show_image(
        "visitor_purchase_behavior.png",
        "Purchase behaviour across visitor categories"
    )

    st.write(
        "The chart compares purchase and non-purchase "
        "session counts across different visitor types."
    )

    st.divider()

    st.subheader("3. Product Browsing Duration")

    show_image(
        "product_browsing_duration.png",
        "Product-related browsing duration"
    )

    st.write(
        "Purchase sessions generally involve more "
        "time spent browsing product-related pages."
    )

    st.divider()

    st.subheader("4. Feature Correlation Heatmap")

    show_image(
        "feature_correlation_heatmap.png",
        "Correlations among numerical browsing features"
    )

    st.write(
        "The heatmap helps identify relationships "
        "between numerical browsing features."
    )

    st.divider()

    st.subheader("5. Purchase Percentage by Visitor Type")

    show_image(
        "visitor_purchase_percentage.png",
        "Purchase percentages by visitor category"
    )

    st.write(
        "This chart uses percentages instead of "
        "raw session counts, making it easier to "
        "compare visitor categories."
    )

    st.divider()

    st.subheader("Key EDA Findings")

    finding1, finding2 = st.columns(2)

    with finding1:
        st.metric(
            "Average PageValues — Purchase",
            "27.26"
        )

        st.metric(
            "Average PageValues — No Purchase",
            "2.00"
        )

    with finding2:
        st.metric(
            "Product Duration — Purchase",
            "1,876.21 sec"
        )

        st.metric(
            "Product Duration — No Purchase",
            "1,082.98 sec"
        )

    st.info(
        "These findings describe associations in "
        "the dataset. They do not prove that a "
        "particular feature directly causes a purchase."
    )


# ==========================================
# TAB 3: MODEL COMPARISON
# ==========================================

with tab_models:

    st.header("🤖 Machine Learning Model Comparison")

    st.write(
        "Three traditional machine learning models "
        "and two ANN configurations were evaluated "
        "on the held-out test set."
    )

    st.caption(
        "Precision, recall, and F1-score below "
        "refer to the purchase class."
    )

    model_results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "KNN",
            "Random Forest (Baseline)",
            "Random Forest (Tuned)",
            "Initial ANN",
            "Best Experimental ANN"
        ],
        "Accuracy (%)": [
            88.69,
            87.34,
            89.59,
            89.43,
            89.80,
            89.80
        ],
        "Precision": [
            0.75,
            0.66,
            0.65,
            0.65,
            0.6997,
            0.6895
        ],
        "Recall": [
            0.41,
            0.39,
            0.72,
            0.71,
            0.6099,
            0.6335
        ],
        "F1 Score": [
            0.53,
            0.49,
            0.68,
            0.68,
            0.6517,
            0.6603
        ]
    })

    st.subheader("Performance Comparison")

    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------
    # IMPROVED HORIZONTAL ACCURACY CHART
    # --------------------------------------

    st.subheader("Model Accuracy")

    accuracy_chart = model_results.set_index(
        "Model"
    )[["Accuracy (%)"]]

    st.bar_chart(
        accuracy_chart,
        horizontal=True,
        sort=False,
        x_label="Accuracy (%)",
        y_label="Model",
        use_container_width=True
    )

    # --------------------------------------
    # IMPROVED HORIZONTAL F1-SCORE CHART
    # --------------------------------------

    st.subheader("Purchase-Class F1-score")

    f1_chart = model_results.set_index(
        "Model"
    )[["F1 Score"]]

    st.bar_chart(
        f1_chart,
        horizontal=True,
        sort=False,
        x_label="F1-score",
        y_label="Model",
        use_container_width=True
    )

    st.divider()

    st.subheader("Model Selection")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Highest Test Accuracy",
            "89.80%",
            help="Achieved by both evaluated ANN models."
        )

    with col2:
        st.metric(
            "Baseline Random Forest Recall",
            "72%"
        )

    st.success(
        "Random Forest was retained for deployment "
        "because it achieved stronger purchase-class "
        "F1-score and recall than the ANN."
    )

    st.divider()

    st.subheader("Random Forest Hyperparameter Tuning")

    tuning_results = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Purchase Precision",
            "Purchase Recall",
            "Purchase F1-score"
        ],
        "Baseline": [
            "89.59%",
            "0.65",
            "0.72",
            "0.68"
        ],
        "Tuned": [
            "89.43%",
            "0.65",
            "0.71",
            "0.68"
        ]
    })

    st.dataframe(
        tuning_results,
        hide_index=True,
        use_container_width=True
    )

    with st.expander("View Best Hyperparameters"):

        st.code(
            """n_estimators = 300
max_depth = 30
min_samples_split = 2
min_samples_leaf = 1""",
            language="python"
        )

        st.write(
            "RandomizedSearchCV used 3-fold "
            "cross-validation and achieved a "
            "best cross-validation F1-score "
            "of approximately 0.6794."
        )

    st.divider()

    st.subheader("Random Forest Overfitting Analysis")

    show_image(
        "random_forest_overfitting_analysis.png",
        "Training and testing performance of tuned Random Forest"
    )

    overfitting_results = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score"
        ],
        "Training": [
            0.9995,
            0.9967,
            1.0000,
            0.9984
        ],
        "Testing": [
            0.8943,
            0.6483,
            0.7094,
            0.6775
        ]
    })

    st.dataframe(
        overfitting_results,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "The tuned Random Forest shows significant "
        "overfitting. Its training F1-score was "
        "approximately 0.9984, compared with "
        "0.6775 on the test set."
    )


# ==========================================
# TAB 4: ANN EXPERIMENTS
# ==========================================

with tab_ann:

    st.header("🧠 Artificial Neural Network Experiments")

    st.write(
        "Artificial Neural Networks were trained "
        "using TensorFlow and Keras to compare "
        "deep learning with traditional machine learning."
    )

    show_metric_cards([
        ("Training Records", "7,811"),
        ("Validation Records", "1,953"),
        ("Testing Records", "2,441"),
        ("Processed Features", "74")
    ])

    st.divider()

    st.subheader("Initial ANN Architecture")

    architecture = pd.DataFrame({
        "Component": [
            "Input Features",
            "Hidden Layer 1",
            "Hidden Layer 2",
            "Output Layer",
            "Activation",
            "Optimizer",
            "Loss Function",
            "Batch Size",
            "Maximum Epochs",
            "Epochs Trained",
            "Trainable Parameters"
        ],
        "Configuration": [
            "74",
            "64 neurons",
            "32 neurons",
            "1 neuron",
            "ReLU / Sigmoid",
            "Adam",
            "Binary Cross-Entropy",
            "32",
            "50",
            "21",
            "6,913"
        ]
    })

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("ANN Training History")

    col1, col2 = st.columns(2)

    with col1:
        show_image(
            "ann_training_loss.png",
            "Training vs validation loss"
        )

    with col2:
        show_image(
            "ann_training_accuracy.png",
            "Training vs validation accuracy"
        )

    st.write(
        "The initial ANN used early stopping "
        "to monitor validation loss and restore "
        "the best weights."
    )

    st.divider()

    st.subheader("ANN Hyperparameter Experiments")

    st.write(
        "Four ANN configurations were tested "
        "using different numbers of layers, "
        "neurons, batch sizes, and optimizers."
    )

    if EXPERIMENT_CSV.exists():

        experiments_df = pd.read_csv(
            EXPERIMENT_CSV
        )

        st.dataframe(
            experiments_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        experiments_df = pd.DataFrame({
            "Model": [
                "ANN 1",
                "ANN 2",
                "ANN 3",
                "ANN 4"
            ],
            "Layers": [
                "[32]",
                "[64, 32]",
                "[128, 64, 32]",
                "[64, 32]"
            ],
            "Batch Size": [
                32, 32, 64, 64
            ],
            "Optimizer": [
                "adam",
                "adam",
                "adam",
                "rmsprop"
            ],
            "Epochs Trained": [
                28, 15, 13, 17
            ],
            "F1 Score": [
                0.623239,
                0.609319,
                0.613757,
                0.605405
            ]
        })

        st.dataframe(
            experiments_df,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "The ANN experiment results above "
        "were measured on the validation set."
    )

    st.subheader("Experiment Comparison")

    show_image(
        "ann_experiment_comparison.png",
        "Validation F1-score across four ANN configurations"
    )

    st.divider()

    st.subheader("Best ANN Configuration")

    st.success(
        "ANN 1 achieved the highest validation "
        "F1-score among the four experiments."
    )

    best_config = pd.DataFrame({
        "Setting": [
            "Hidden Layers",
            "Neurons",
            "Optimizer",
            "Batch Size",
            "Epochs Trained",
            "Validation F1-score"
        ],
        "Value": [
            "1",
            "32",
            "Adam",
            "32",
            "28",
            "0.6232"
        ]
    })

    st.dataframe(
        best_config,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Final ANN Test Results")

    show_metric_cards([
        ("Accuracy", "89.80%"),
        ("Precision", "68.95%"),
        ("Recall", "63.35%"),
        ("F1-score", "66.03%")
    ])

    st.subheader("Final ANN Confusion Matrix")

    confusion_df = pd.DataFrame(
        [
            [1950, 109],
            [140, 242]
        ],
        index=[
            "Actual No Purchase",
            "Actual Purchase"
        ],
        columns=[
            "Predicted No Purchase",
            "Predicted Purchase"
        ]
    )

    st.dataframe(
        confusion_df,
        use_container_width=True
    )

    st.info(
        "The best experimental ANN correctly "
        "identified 242 purchase sessions "
        "and 1,950 non-purchase sessions."
    )

    st.divider()

    st.subheader("ANN Experiment Findings")

    st.markdown("""
    - The simpler ANN with 32 neurons achieved the best validation F1-score.
    - Adding more layers did not improve validation performance in these experiments.
    - Adam performed better than RMSprop in the tested configurations.
    - Early stopping helped limit unnecessary training.
    - The final ANN achieved 89.80% test accuracy.
    - Random Forest retained a stronger purchase-class F1-score.
    """)

    st.caption(
        "The experimental ANN was evaluated on "
        "the held-out test set after configuration "
        "selection using validation F1-score."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "E-Commerce Purchase Prediction System | "
    "Machine Learning & Deep Learning Project"
)

st.markdown(
    "[GitHub Repository]"
    "(https://github.com/farzinmansoor/"
    "Ecommerce-Purchase-Prediction)"
)
