
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("SuperKart Sales Revenue Forecasting")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for product features
product_weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
product_sugar_content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)
product_allocated_area = st.number_input(
    "Product Allocated Area Ratio",
    min_value=0.000,
    max_value=1.000,
    step=0.001,
    value=0.027
)
product_mrp = st.number_input(
    "Product Maximum Retail Price (MRP)",
    min_value=0.0,
    value=117.08
)

# Collect engineered product features
product_id_char = st.selectbox(
    "Product ID Character",
    ["FD", "DR", "NC"]
)

product_type_category = st.selectbox(
    "Product Type Category",
    ["Perishables", "Non Perishables"]
)

# Collect engineered store feature
store_age_years = st.number_input(
    "Store Age (Years)",
    min_value=0,
    value=16
)

# Collect store features
store_size = st.selectbox(
    "Store Size Tier",
    ["Medium", "High", "Small"]
)

store_location_city_type = st.selectbox(
    "Store Location City Type",
    ["Tier 2", "Tier 1", "Tier 3"]
)

store_type = st.selectbox(
    "Store Type Classification",
    [
        "Supermarket Type2",
        "Supermarket Type1",
        "Departmental Store",
        "Food Mart"
    ]
)

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    'Product_Weight': product_weight,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Allocated_Area': product_allocated_area,
    'Product_MRP': product_mrp,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type,
    'Product_Id_char': product_id_char,
    'Store_Age_Years': int(store_age_years),
    'Product_Type_Category': product_type_category
}])

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):

    # Send the input data to the Flask API
    response = requests.post(
        f"{BACKEND_URL}/v1/predict",
        json=input_data.to_dict(orient='records')[0]
    )

    if response.status_code == 200:
        prediction = response.json()['Predicted Sales Total (in dollars)']
        st.success(
            f"Predicted Sales Total (in dollars): {prediction}"
        )
    else:
        st.error("Unable to connect to the prediction API.")


# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader(
    "Upload CSV file for batch prediction",
    type=["csv"]
)

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:

    if st.button("Predict Batch", type="primary"):

        # Send the uploaded file to the Flask API
        response = requests.post(
            f"{BACKEND_URL}/v1/predictbatch",
            files={"file": uploaded_file}
        )

        if response.status_code == 200:
            predictions = response.json()

            st.success("Batch predictions completed!")

            # Display the predictions in a table
            st.write(predictions)

        else:
            st.error("Unable to connect to the prediction API.")
