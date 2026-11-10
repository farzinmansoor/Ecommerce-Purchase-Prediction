
# Improved Streamlit Web Application

import streamlit as st
import pandas as pd
import joblib

# Configure the application
st.set_page_config(
    page_title="E-Commerce Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

# Load the saved machine learning model
@st.cache_resource
def load_model():
    return joblib.load("models/best_purchase_model.pkl")

model = load_model()

# Application heading
st.title("🛒 E-Commerce Purchase Prediction System")

st.write(
    "Predict whether an online shopping session is likely "
    "to result in a purchase using customer browsing details."
)

st.info(
    "How to use: Enter the customer's actual browsing information "
    "and click Predict Purchase. The model will estimate the result."
)

# Help section
with st.expander("📖 Understand the input fields before starting"):

    st.markdown("""
    **Page Information**
    - Administrative: Number of account-related pages visited.
    - Informational: Number of help or information pages visited.
    - Product Related: Number of product pages visited.
    - Duration fields: Total time spent on those pages, in seconds.

    **Session Information**
    - Bounce Rate: Proportion of visits that leave after one page.
    - Exit Rate: Proportion of pageviews that are the final page
      in a session.
    - Page Values: Dataset metric related to the contribution
      of pages to purchase goals. It is not the product price.
    - Special Day: Dataset measure of closeness to a special day.

    **Category Information**
    - Region, Operating System, Browser and Traffic Type
      use numerical category codes from the original dataset.
    - These codes are not errors. Their exact real-world names
      are not provided in the dataset.
    - Do not change the codes or guess their meanings.
    """)

st.divider()

# Create the customer input form
with st.form("purchase_prediction_form"):

    st.subheader("Customer Browsing Information")

    col1, col2 = st.columns(2)

    # LEFT COLUMN
    with col1:

        st.markdown("### 📄 Page Information")

        administrative = st.number_input(
            "Administrative Pages",
            min_value=0,
            max_value=27,
            value=0,
            step=1,
            help="Number of account-related pages visited, such as login or account pages."
        )

        administrative_duration = st.number_input(
            "Administrative Duration (seconds)",
            min_value=0.0,
            max_value=3398.75,
            value=0.0,
            step=1.0,
            help="Total time spent on administrative pages, measured in seconds."
        )

        informational = st.number_input(
            "Informational Pages",
            min_value=0,
            max_value=24,
            value=0,
            step=1,
            help="Number of information pages visited, such as FAQ or contact pages."
        )

        informational_duration = st.number_input(
            "Informational Duration (seconds)",
            min_value=0.0,
            max_value=2549.375,
            value=0.0,
            step=1.0,
            help="Total time spent on informational pages, measured in seconds."
        )

        product_related = st.number_input(
            "Product-Related Pages",
            min_value=0,
            max_value=705,
            value=1,
            step=1,
            help="Number of product pages visited, such as shoes, shirts or mobiles."
        )

        product_related_duration = st.number_input(
            "Product-Related Duration (seconds)",
            min_value=0.0,
            max_value=63973.52223,
            value=60.0,
            step=1.0,
            help="Total time spent viewing product pages, measured in seconds."
        )

        page_values = st.number_input(
            "Page Values",
            min_value=0.0,
            max_value=361.7637419,
            value=0.0,
            step=0.1,
            help="A dataset metric related to the contribution of pages to purchase goals. It is not the product price."
        )

    # RIGHT COLUMN
    with col2:

        st.markdown("### 📊 Session Information")

        bounce_rates = st.number_input(
            "Bounce Rate",
            min_value=0.0,
            max_value=0.2,
            value=0.02,
            step=0.01,
            help="Proportion of visits that leave after viewing only one page. Example: 0.20 means 20%."
        )

        exit_rates = st.number_input(
            "Exit Rate",
            min_value=0.0,
            max_value=0.2,
            value=0.03,
            step=0.01,
            help="Proportion of pageviews that are the final page in a session."
        )

        special_day = st.number_input(
            "Special Day",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Dataset measure of how close the session is to a special shopping day. Range: 0 to 1."
        )

        month = st.selectbox(
            "Month",
            ["Aug", "Dec", "Feb", "Jul", "June",
             "Mar", "May", "Nov", "Oct", "Sep"],
            help="Month in which the shopping session occurred."
        )

        visitor_type = st.selectbox(
            "Visitor Type",
            ["Returning_Visitor", "New_Visitor", "Other"],
            help="Select whether the visitor is new, returning or in another category."
        )

        weekend = st.selectbox(
            "Weekend",
            [False, True],
            format_func=lambda value: "Yes" if value else "No",
            help="Was the shopping session on Saturday or Sunday?"
        )

        operating_system = st.selectbox(
            "Operating System",
            list(range(1, 9)),
            format_func=lambda value: f"OS Category {value}",
            help="Numerical category identifying the visitor's operating system. The original dataset does not provide the exact names."
        )

        browser = st.selectbox(
            "Browser",
            list(range(1, 14)),
            format_func=lambda value: f"Browser Category {value}",
            help="Numerical category identifying the visitor's browser. The original dataset does not provide the exact names."
        )

        region = st.selectbox(
            "Region",
            list(range(1, 10)),
            format_func=lambda value: f"Region Category {value}",
            help="Geographical category code from the original dataset. It does not provide city or state names."
        )

        traffic_type = st.selectbox(
            "Traffic Type",
            list(range(1, 21)),
            format_func=lambda value: f"Traffic Category {value}",
            help="Category identifying how the visitor reached the website. Exact code descriptions are not provided."
        )

    st.divider()

    # Prediction button
    submitted = st.form_submit_button(
        "🔍 Predict Purchase",
        use_container_width=True
    )

# Make the prediction
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

    # Convert input details into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict purchase or no purchase
    prediction = model.predict(input_df)[0]

    # Calculate purchase probability
    probabilities = model.predict_proba(input_df)[0]

    purchase_index = list(model.classes_).index(True)

    purchase_probability = probabilities[purchase_index]

    # STEP 8: Display the result
    st.divider()
    st.subheader("🎯 Prediction Result")

    if prediction:
        st.success("The customer is likely to make a purchase! 🛍️")
    else:
        st.warning("The customer is unlikely to make a purchase.")

    st.metric(
        "Estimated Purchase Probability",
        f"{purchase_probability * 100:.2f}%"
    )

    st.progress(float(purchase_probability))

    st.caption(
        "This is a machine learning estimate based on the entered "
        "session details. It is not a guarantee of purchase."
    )
