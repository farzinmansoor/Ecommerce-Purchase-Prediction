
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ShopSense AI | Purchase Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_purchase_model.pkl"
VISUALIZATION_DIR = BASE_DIR / "visualizations"
EXPERIMENT_CSV = BASE_DIR / "models" / "ann_experiment_results.csv"

GREEN = "#0F3D3A"
CREAM = "#F5DABF"
LIGHT = "#FFF9F1"
MINT = "#E3EFE8"
WHITE = "#FFFFFF"

# ============================================================
# 2. GREEN AND CREAM THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --green: #0F3D3A;
        --cream: #F5DABF;
        --light: #FFF9F1;
        --mint: #E3EFE8;
    }

    html, body,
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"] {
        background-color: #FFF9F1 !important;
        color: #0F3D3A;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #0F3D3A;
    }

    h1 {
        font-weight: 800;
        letter-spacing: -0.8px;
    }

    /* SIDEBAR */

    [data-testid="stSidebar"] {
        background-color: #0F3D3A !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] strong {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"]
    [data-testid="stCaptionContainer"] p {
        color: #E8D9C5 !important;
    }

    [data-testid="stSidebar"]
    [role="radiogroup"] label {
        border-radius: 12px !important;
        padding: 0.65rem 0.8rem !important;
        margin-bottom: 0.35rem !important;
    }

    [data-testid="stSidebar"]
    [role="radiogroup"] label p {
        color: #FFFFFF !important;
        font-weight: 650 !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"]
    [role="radiogroup"] label:hover {
        background-color: rgba(
            245, 218, 191, 0.18
        ) !important;
    }

    [data-testid="stSidebar"]
    [role="radiogroup"]
    label:has(input:checked) {
        background-color: #F5DABF !important;
        border-radius: 12px !important;
    }

    [data-testid="stSidebar"]
    [role="radiogroup"]
    label:has(input:checked) p {
        color: #0F3D3A !important;
        font-weight: 750 !important;
    }

    [data-testid="stSidebar"]
    [role="radiogroup"]
    label:has(input:checked) * {
        color: #0F3D3A !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(
            245, 218, 191, 0.25
        ) !important;
    }

    /* METRIC CARDS */

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE8DF;
        border-radius: 16px;
        padding: 1.1rem;
        box-shadow: 0 3px 12px
            rgba(15, 61, 58, 0.04);
    }

    [data-testid="stMetricLabel"] * {
        color: #54716A !important;
    }

    [data-testid="stMetricValue"] * {
        color: #0F3D3A !important;
    }

    /* FORMS */

    div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE8DF !important;
        border-radius: 18px !important;
        padding: 1.4rem !important;
    }

    div[data-testid="stForm"] label p {
        color: #0F3D3A !important;
        font-weight: 600 !important;
    }

    /* INPUT AND DROPDOWN LABELS */

    [data-testid="stMain"] label p,
    [data-testid="stMain"]
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMain"]
    [data-testid="stSelectbox"] label * {
        color: #0F3D3A !important;
        font-weight: 650 !important;
        opacity: 1 !important;
    }

    /* NUMBER INPUTS */

    [data-testid="stMain"] [data-baseweb="input"],
    [data-testid="stMain"] [data-baseweb="input"] input,
    [data-testid="stMain"]
    [data-testid="stNumberInput"]
    [data-baseweb="input"] > div,
    [data-testid="stMain"]
    [data-testid="stNumberInput"]
    [data-baseweb="input"] > div > div {
        background-color: #FFFFFF !important;
        color: #0F3D3A !important;
    }

    [data-testid="stMain"]
    [data-baseweb="input"] {
        border: 1px solid #B8CEC2 !important;
        border-radius: 9px !important;
    }

    [data-testid="stMain"]
    [data-baseweb="input"] input {
        color: #0F3D3A !important;
        -webkit-text-fill-color: #0F3D3A !important;
    }

    /* NUMBER INPUT BUTTONS */

    [data-testid="stMain"]
    [data-testid="stNumberInput"] button {
        background-color: #E3EFE8 !important;
        color: #0F3D3A !important;
    }

    [data-testid="stMain"]
    [data-testid="stNumberInput"] button * {
        color: #0F3D3A !important;
        stroke: #0F3D3A !important;
    }

    /* SELECTBOXES */

    [data-testid="stMain"]
    [data-baseweb="select"] > div,
    [data-testid="stMain"]
    [data-baseweb="select"] > div > div {
        background-color: #FFFFFF !important;
        color: #0F3D3A !important;
    }

    [data-testid="stMain"]
    [data-baseweb="select"] > div {
        border: 1px solid #B8CEC2 !important;
        border-radius: 9px !important;
    }

    [data-testid="stMain"]
    [data-baseweb="select"] span,
    [data-testid="stMain"]
    [data-baseweb="select"] svg {
        color: #0F3D3A !important;
        fill: #0F3D3A !important;
    }

    /* PREDICTION BUTTON */

    div.stFormSubmitButton > button,
    div.stButton > button[kind="primary"] {
        background-color: #F5DABF !important;
        color: #0F3D3A !important;
        border: 2px solid #0F3D3A !important;
        border-radius: 12px !important;
        font-weight: 750 !important;
        min-height: 50px !important;
        opacity: 1 !important;
    }

    div.stFormSubmitButton > button *,
    div.stButton > button[kind="primary"] * {
        color: #0F3D3A !important;
        fill: #0F3D3A !important;
        opacity: 1 !important;
    }

    div.stFormSubmitButton > button:hover,
    div.stButton > button[kind="primary"]:hover {
        background-color: #EAC9A5 !important;
        color: #0F3D3A !important;
        border-color: #0F3D3A !important;
    }

    /* TABLES */

    .shopsense-table-wrapper {
        width: 100%;
        overflow-x: auto;
        border: 1px solid #DCE8DF;
        border-radius: 12px;
        margin: 0.5rem 0 1.2rem;
    }

    .shopsense-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95rem;
    }

    .shopsense-table thead th {
        background-color: #0F3D3A;
        color: #FFFFFF;
        padding: 13px 14px;
        text-align: left;
        white-space: nowrap;
    }

    .shopsense-table tbody td {
        background-color: #FFFFFF;
        color: #0F3D3A;
        padding: 12px 14px;
        border-bottom: 1px solid #DCE8DF;
    }

    .shopsense-table tbody tr:hover td {
        background-color: #E3EFE8;
    }

    .shopsense-table tbody tr:last-child td {
        border-bottom: none;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #DCE8DF;
        border-radius: 12px;
        overflow: hidden;
    }

    /* EXPANDERS */

    div[data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #DCE8DF;
        border-radius: 12px;
    }

    /* IMAGES AND ALERTS */

    [data-testid="stImage"] img {
        border-radius: 12px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #DCE8DF;
    }

    /* OVERVIEW HERO */

    .hero {
        background: #0F3D3A;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: #FFF9F1 !important;
        margin: 0 0 0.8rem;
        font-size: clamp(1.8rem, 4vw, 2.7rem);
    }

    .hero p {
        color: #F5DABF !important;
        margin: 0;
        line-height: 1.7;
    }

    /* SECTION HEADINGS */

    .section-heading {
        background: #E3EFE8;
        color: #0F3D3A;
        border-left: 5px solid #0F3D3A;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem;
        font-weight: 750;
        font-size: 1.15rem;
    }

    /* OVERVIEW CARDS */

    .soft-card {
        background: #FFFFFF;
        border: 1px solid #DCE8DF;
        border-radius: 15px;
        padding: 1.3rem;
        margin: 0.6rem 0;
        min-height: 145px;
    }

    .soft-card h3 {
        color: #0F3D3A !important;
        margin-top: 0;
        font-size: 1.15rem;
    }

    .soft-card p {
        color: #48645E !important;
        margin-bottom: 0;
    }

    .footer {
        color: #54716A;
        text-align: center;
        padding: 1rem;
        font-size: 0.85rem;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 1.4rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def section(title):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )


def page_header(title, description):
    st.title(title)
    st.write(description)
    st.divider()


def show_metrics(metrics):
    columns = st.columns(len(metrics))

    for column, (label, value) in zip(columns, metrics):
        with column:
            st.metric(label, value)


def show_image(filename, caption=None):
    path = VISUALIZATION_DIR / filename

    if path.exists():
        st.image(
            str(path),
            caption=caption,
            use_container_width=True,
        )
    else:
        st.info(f"Visualization not found: {filename}")


def styled_table(data, height=None):
    """
    Full-width green-and-cream table
    with clean numeric formatting.
    """

    formatted_data = data.copy()

    for column in formatted_data.columns:
        if pd.api.types.is_float_dtype(
            formatted_data[column]
        ):
            if "Accuracy (%)" in column:
                formatted_data[column] = (
                    formatted_data[column].map(
                        lambda value: f"{value:.2f}"
                    )
                )
            else:
                formatted_data[column] = (
                    formatted_data[column].map(
                        lambda value: f"{value:.4f}"
                    )
                )

    table_html = formatted_data.to_html(
        index=False,
        escape=True,
        classes="shopsense-table",
        border=0,
    )

    st.markdown(
        '<div class="shopsense-table-wrapper">'
        + table_html
        + "</div>",
        unsafe_allow_html=True,
    )


def comparison_chart(data, metric):
    """
    Horizontal comparison chart
    with full model names.
    """

    fig, ax = plt.subplots(figsize=(11, 5.5))

    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    labels = data["Model"].tolist()
    values = data[metric].tolist()

    colors = [
        GREEN if "Random Forest" in label
        else "#4E8576" if "ANN" in label
        else "#A6C7B9"
        for label in labels
    ]

    bars = ax.barh(
        labels,
        values,
        color=colors,
        height=0.6,
    )

    ax.invert_yaxis()

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.spines["bottom"].set_color("#DCE8DF")

    ax.tick_params(
        axis="y",
        length=0,
        labelsize=10,
        colors=GREEN,
        pad=10,
    )

    ax.tick_params(
        axis="x",
        colors="#54716A",
    )

    ax.grid(
        axis="x",
        alpha=0.18,
        color=GREEN,
    )

    ax.set_axisbelow(True)

    max_value = max(values)

    if metric == "Accuracy (%)":
        ax.set_xlim(0, 105)
        value_format = "{:.2f}%"
    else:
        ax.set_xlim(0, max(1.0, max_value * 1.15))
        value_format = "{:.3f}"

    for bar, value in zip(bars, values):
        ax.text(
            value + (
                1.0 if metric == "Accuracy (%)"
                else 0.015
            ),
            bar.get_y() + bar.get_height() / 2,
            value_format.format(value),
            va="center",
            ha="left",
            fontsize=10,
            color=GREEN,
            fontweight="bold",
        )

    ax.set_xlabel(
        metric,
        color=GREEN,
        fontweight="bold",
    )

    fig.subplots_adjust(
        left=0.32,
        right=0.95,
        top=0.96,
        bottom=0.13,
    )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("## 🛒 ShopSense AI")

    st.caption("Purchase Prediction Dashboard")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🛒 Purchase Prediction",
            "📊 Data Analysis",
            "🤖 Model Comparison",
            "🧠 ANN Experiments",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("**Project information**")

    st.caption("Dataset: UCI Online Shoppers")
    st.caption("12,205 cleaned sessions")
    st.caption("17 original input features")
    st.caption("Random Forest + ANN")

    st.divider()

    st.caption("Machine Learning & Deep Learning")


# ============================================================
# 5. OVERVIEW PAGE
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        """
        <div class="hero">
            <h1>🛒 E-Commerce Purchase Prediction</h1>
            <p>
                Explore shopping behaviour, compare machine
                learning models, and estimate the likelihood
                of a customer completing a purchase.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_metrics([
        ("Original Sessions", "12,330"),
        ("Cleaned Sessions", "12,205"),
        ("Input Features", "17"),
        ("Best Test Accuracy", "89.80%"),
    ])

    section("Explore the dashboard")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="soft-card">
                <h3>🛒 Purchase Prediction</h3>
                <p>
                    Enter browsing information and receive
                    a purchase prediction from the deployed
                    Random Forest model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="soft-card">
                <h3>🤖 Model Comparison</h3>
                <p>
                    Compare accuracy, precision, recall,
                    F1-score, hyperparameter tuning,
                    and overfitting.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="soft-card">
                <h3>📊 Data Analysis</h3>
                <p>
                    Explore purchase patterns, visitor
                    behaviour, product browsing duration,
                    and feature correlations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="soft-card">
                <h3>🧠 ANN Experiments</h3>
                <p>
                    Review neural network architecture,
                    training curves, hyperparameter
                    experiments, and test performance.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section("Project highlights")

    st.markdown(
        """
        - **Prediction model:** Random Forest
        - **Random Forest test accuracy:** 89.59%
        - **Best experimental ANN accuracy:** 89.80%
        - **Model selection:** Random Forest retained
          because of stronger purchase-class recall
          and F1-score.
        - **Evaluation:** Separate training, validation,
          and held-out testing where appropriate.
        """
    )

    st.info(
        "Use the navigation menu on the left "
        "to explore each part of the project."
    )


# ============================================================
# 6. PURCHASE PREDICTION PAGE
# ============================================================

elif page == "🛒 Purchase Prediction":

    page_header(
        "🛒 Customer Purchase Prediction",
        "Enter browsing details to estimate whether "
        "a shopping session will result in a purchase.",
    )

    show_metrics([
        ("Prediction Model", "Random Forest"),
        ("Test Accuracy", "89.59%"),
        ("Purchase F1-score", "0.68"),
    ])

    with st.expander("📖 Understand the input fields"):

        st.markdown(
            """
            **Page information**

            - Administrative: Account-related pages visited.
            - Informational: Help or information pages visited.
            - Product Related: Product pages visited.
            - Duration: Time spent on pages in seconds.

            **Session information**

            - Bounce Rate: Dataset bounce-rate metric.
            - Exit Rate: Dataset exit-rate metric.
            - Page Values: Contribution of pages to purchase
              goals, not product price.
            - Special Day: Closeness to a special shopping day.

            **Category information**

            - Operating System, Browser, Region, and
              Traffic Type are numerical dataset codes.
            - Their exact real-world names are not
              supplied by the dataset.
            """
        )

    st.info(
        "Complete the browsing details below and click "
        "Predict Purchase to view the result."
    )

    with st.form("purchase_prediction_form"):

        section("Step 1 — Page activity")

        col1, col2 = st.columns(2)

        with col1:

            administrative = st.number_input(
                "Administrative Pages",
                min_value=0,
                max_value=27,
                value=0,
                step=1,
            )

            administrative_duration = st.number_input(
                "Administrative Duration (seconds)",
                min_value=0.0,
                max_value=3398.75,
                value=0.0,
                step=1.0,
            )

            informational = st.number_input(
                "Informational Pages",
                min_value=0,
                max_value=24,
                value=0,
                step=1,
            )

            informational_duration = st.number_input(
                "Informational Duration (seconds)",
                min_value=0.0,
                max_value=2549.375,
                value=0.0,
                step=1.0,
            )

        with col2:

            product_related = st.number_input(
                "Product-Related Pages",
                min_value=0,
                max_value=705,
                value=1,
                step=1,
            )

            product_related_duration = st.number_input(
                "Product-Related Duration (seconds)",
                min_value=0.0,
                max_value=63973.52223,
                value=60.0,
                step=1.0,
            )

            page_values = st.number_input(
                "Page Values",
                min_value=0.0,
                max_value=361.7637419,
                value=0.0,
                step=0.1,
                help="This is not the product price.",
            )

        section("Step 2 — Session behaviour")

        col1, col2 = st.columns(2)

        with col1:

            bounce_rates = st.number_input(
                "Bounce Rate",
                min_value=0.0,
                max_value=0.2,
                value=0.02,
                step=0.01,
            )

            exit_rates = st.number_input(
                "Exit Rate",
                min_value=0.0,
                max_value=0.2,
                value=0.03,
                step=0.01,
            )

        with col2:

            special_day = st.number_input(
                "Special Day",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
            )

            month = st.selectbox(
                "Month",
                [
                    "Aug", "Dec", "Feb", "Jul", "June",
                    "Mar", "May", "Nov", "Oct", "Sep",
                ],
            )

        section("Step 3 — Visitor information")

        col1, col2 = st.columns(2)

        with col1:

            visitor_type = st.selectbox(
                "Visitor Type",
                [
                    "Returning_Visitor",
                    "New_Visitor",
                    "Other",
                ],
            )

            weekend = st.selectbox(
                "Weekend",
                [False, True],
                format_func=lambda value:
                    "Yes" if value else "No",
            )

            operating_system = st.selectbox(
                "Operating System",
                list(range(1, 9)),
                format_func=lambda value:
                    f"OS Category {value}",
            )

        with col2:

            browser = st.selectbox(
                "Browser",
                list(range(1, 14)),
                format_func=lambda value:
                    f"Browser Category {value}",
            )

            region = st.selectbox(
                "Region",
                list(range(1, 10)),
                format_func=lambda value:
                    f"Region Category {value}",
            )

            traffic_type = st.selectbox(
                "Traffic Type",
                list(range(1, 21)),
                format_func=lambda value:
                    f"Traffic Category {value}",
            )

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Predict Purchase",
            use_container_width=True,
        )

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
            "Weekend": weekend,
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

            section("Step 4 — Prediction result")

            if prediction:
                st.success(
                    "The customer is likely "
                    "to make a purchase! 🛍️"
                )
            else:
                st.warning(
                    "The customer is unlikely "
                    "to make a purchase."
                )

            st.metric(
                "Estimated Purchase Probability",
                f"{purchase_probability * 100:.2f}%",
            )

            st.progress(purchase_probability)

            st.caption(
                "This is a machine learning estimate "
                "based on the entered session details. "
                "It is not a guarantee of purchase."
            )

        except Exception as error:

            st.error(
                "Prediction could not be completed. "
                "Check the saved model and input data."
            )

            st.exception(error)


# ============================================================
# 7. DATA ANALYSIS PAGE
# ============================================================

elif page == "📊 Data Analysis":

    page_header(
        "📊 Exploratory Data Analysis",
        "Explore customer behaviour and discover "
        "patterns associated with online purchases.",
    )

    show_metrics([
        ("Original Sessions", "12,330"),
        ("Cleaned Sessions", "12,205"),
        ("Duplicates Removed", "125"),
        ("Input Features", "17"),
    ])

    section("Explore visualizations")

    chart_options = {
        "Purchase Distribution": (
            "purchase_distribution.png",
            "The dataset has substantially more "
            "non-purchase sessions than purchase sessions.",
        ),
        "Visitor Purchase Behaviour": (
            "visitor_purchase_behavior.png",
            "Compare purchase and non-purchase counts "
            "across visitor categories.",
        ),
        "Product Browsing Duration": (
            "product_browsing_duration.png",
            "Purchase sessions generally involve "
            "longer product browsing durations.",
        ),
        "Feature Correlation Heatmap": (
            "feature_correlation_heatmap.png",
            "Explore correlations among numerical features.",
        ),
        "Visitor Purchase Percentage": (
            "visitor_purchase_percentage.png",
            "Compare purchase percentages rather than "
            "raw counts across visitor categories.",
        ),
    }

    selected_chart = st.selectbox(
        "Choose an analysis",
        list(chart_options.keys()),
    )

    filename, explanation = chart_options[selected_chart]

    st.subheader(selected_chart)

    show_image(filename, selected_chart)

    st.write(explanation)

    section("Key EDA findings")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Average PageValues — Purchase",
            "27.26",
        )

        st.metric(
            "Average PageValues — No Purchase",
            "2.00",
        )

    with col2:

        st.metric(
            "Product Duration — Purchase",
            "1,876.21 sec",
        )

        st.metric(
            "Product Duration — No Purchase",
            "1,082.98 sec",
        )

    st.info(
        "These findings describe associations, "
        "not proof that a feature causes a purchase."
    )


# ============================================================
# 8. MODEL COMPARISON PAGE
# ============================================================

elif page == "🤖 Model Comparison":

    page_header(
        "🤖 Machine Learning Model Comparison",
        "Compare traditional machine learning and "
        "deep learning performance on the test set.",
    )

    model_results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "KNN",
            "Random Forest (Baseline)",
            "Random Forest (Tuned)",
            "Initial ANN",
            "Best Experimental ANN",
        ],
        "Accuracy (%)": [
            88.69, 87.34, 89.59,
            89.43, 89.80, 89.80,
        ],
        "Precision": [
            0.75, 0.66, 0.65,
            0.65, 0.6997, 0.6895,
        ],
        "Recall": [
            0.41, 0.39, 0.72,
            0.71, 0.6099, 0.6335,
        ],
        "F1 Score": [
            0.53, 0.49, 0.68,
            0.68, 0.6517, 0.6603,
        ],
    })

    section("Performance summary")

    show_metrics([
        ("Highest Test Accuracy", "89.80%"),
        ("Random Forest Recall", "72%"),
        ("Random Forest F1", "0.68"),
    ])

    st.caption(
        "Precision, recall, and F1-score "
        "refer to the purchase class."
    )

    styled_table(model_results)

    section("Visual model comparison")

    metric = st.selectbox(
        "Choose performance metric",
        [
            "Accuracy (%)",
            "Precision",
            "Recall",
            "F1 Score",
        ],
    )

    comparison_chart(model_results, metric)

    st.caption(
        "Full model names are displayed. "
        "Higher values indicate better performance."
    )

    section("Model selection")

    st.success(
        "Random Forest was retained for deployment "
        "because it achieved stronger purchase-class "
        "recall and F1-score than the ANN models."
    )

    section("Random Forest hyperparameter tuning")

    tuning_results = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Purchase Precision",
            "Purchase Recall",
            "Purchase F1-score",
        ],
        "Baseline": [
            "89.59%", "0.65", "0.72", "0.68",
        ],
        "Tuned": [
            "89.43%", "0.65", "0.71", "0.68",
        ],
    })

    styled_table(tuning_results)

    with st.expander("View best hyperparameters"):

        st.code(
            """n_estimators = 300
max_depth = 30
min_samples_split = 2
min_samples_leaf = 1""",
            language="python",
        )

        st.write(
            "RandomizedSearchCV used 3-fold "
            "cross-validation. Best cross-validation "
            "F1-score: 0.6794."
        )

    section("Overfitting analysis")

    show_image(
        "random_forest_overfitting_analysis.png",
        "Tuned Random Forest training "
        "vs testing performance",
    )

    overfitting_results = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
        ],
        "Training": [
            0.9995,
            0.9967,
            1.0000,
            0.9984,
        ],
        "Testing": [
            0.8943,
            0.6483,
            0.7094,
            0.6775,
        ],
    })

    styled_table(overfitting_results)

    st.warning(
        "The tuned Random Forest shows substantial "
        "overfitting: training F1-score was "
        "approximately 0.9984, while testing "
        "F1-score was 0.6775."
    )


# ============================================================
# 9. ANN EXPERIMENTS PAGE
# ============================================================

elif page == "🧠 ANN Experiments":

    page_header(
        "🧠 Artificial Neural Network Experiments",
        "Review neural network training, architecture, "
        "experiments, and final evaluation.",
    )

    show_metrics([
        ("Training Records", "7,811"),
        ("Validation Records", "1,953"),
        ("Testing Records", "2,441"),
        ("Processed Features", "74"),
    ])

    section("Initial ANN architecture")

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
            "Trainable Parameters",
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
            "6,913",
        ],
    })

    styled_table(architecture)

    section("ANN training history")

    col1, col2 = st.columns(2)

    with col1:

        show_image(
            "ann_training_loss.png",
            "Training vs validation loss",
        )

    with col2:

        show_image(
            "ann_training_accuracy.png",
            "Training vs validation accuracy",
        )

    st.caption(
        "Early stopping monitored validation loss "
        "and restored the best weights."
    )

    section("Hyperparameter experiments")

    st.write(
        "Four ANN configurations were tested using "
        "different layer sizes, batch sizes, "
        "and optimizers."
    )

    if EXPERIMENT_CSV.exists():

        experiments_df = pd.read_csv(EXPERIMENT_CSV)

    else:

        experiments_df = pd.DataFrame({
            "Model": [
                "ANN 1", "ANN 2", "ANN 3", "ANN 4",
            ],
            "Layers": [
                "[32]", "[64, 32]",
                "[128, 64, 32]", "[64, 32]",
            ],
            "Batch Size": [
                32, 32, 64, 64,
            ],
            "Optimizer": [
                "adam", "adam", "adam", "rmsprop",
            ],
            "Epochs Trained": [
                28, 15, 13, 17,
            ],
            "F1 Score": [
                0.623239,
                0.609319,
                0.613757,
                0.605405,
            ],
        })

    styled_table(experiments_df)

    st.caption(
        "Experiment results were measured "
        "on the validation set."
    )

    show_image(
        "ann_experiment_comparison.png",
        "Validation F1-score comparison",
    )

    section("Best ANN configuration")

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
            "Validation F1-score",
        ],
        "Value": [
            "1",
            "32",
            "Adam",
            "32",
            "28",
            "0.6232",
        ],
    })

    styled_table(best_config)

    section("Final ANN test results")

    show_metrics([
        ("Accuracy", "89.80%"),
        ("Precision", "68.95%"),
        ("Recall", "63.35%"),
        ("F1-score", "66.03%"),
    ])

    st.subheader("Confusion matrix")

    confusion_df = pd.DataFrame(
        [
            [1950, 109],
            [140, 242],
        ],
        index=[
            "Actual No Purchase",
            "Actual Purchase",
        ],
        columns=[
            "Predicted No Purchase",
            "Predicted Purchase",
        ],
    )

    st.dataframe(
        confusion_df,
        use_container_width=True,
    )

    st.info(
        "The best experimental ANN correctly "
        "identified 242 purchase sessions "
        "and 1,950 non-purchase sessions."
    )

    section("ANN experiment findings")

    st.markdown(
        """
        - The simpler ANN with 32 neurons achieved
          the highest validation F1-score.
        - Adding more layers did not improve validation
          performance in these experiments.
        - Adam performed better than RMSprop in
          the tested configurations.
        - Early stopping helped limit unnecessary training.
        - The final ANN achieved 89.80% test accuracy.
        - Random Forest retained a stronger
          purchase-class F1-score.
        """
    )

    st.caption(
        "The experimental ANN was evaluated on the "
        "held-out test set after selecting its "
        "configuration using validation F1-score."
    )


# ============================================================
# 10. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        E-Commerce Purchase Prediction System
        <br>
        Machine Learning & Deep Learning Project
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "[GitHub Repository]"
    "(https://github.com/farzinmansoor/"
    "Ecommerce-Purchase-Prediction)"
)
