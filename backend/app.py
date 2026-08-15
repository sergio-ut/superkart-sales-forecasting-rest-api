# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_sales_predictor_api = Flask("SuperKart Sales Predictor")

# Load the trained machine learning model
model = joblib.load("superkart_sales_forecasting_model_v1_0.joblib")

# Define a route for the home page (GET request)
@superkart_sales_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Revenue Forecasting API!"

# Define an endpoint for single product sales prediction (POST request)
@superkart_sales_predictor_api.post('/v1/predict')
def predict_sales():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing product and store details and returns
    the forecasted sales revenue as a JSON response.
    """
    # Get the JSON data from the request body
    sales_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': sales_data['Product_Weight'],
        'Product_Sugar_Content': sales_data['Product_Sugar_Content'],
        'Product_Allocated_Area': sales_data['Product_Allocated_Area'],
        'Product_Type': sales_data['Product_Type'],
        'Product_MRP': sales_data['Product_MRP'],
        'Store_Id': sales_data['Store_Id'],
        'Store_Establishment_Year': sales_data['Store_Establishment_Year'],
        'Store_Size': sales_data['Store_Size'],
        'Store_Location_City_Type': sales_data['Store_Location_City_Type'],
        'Store_Type': sales_data['Store_Type']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction directly using the pipeline
    predicted_sales = model.predict(input_data)[0]

    # Convert predicted_sales to Python float to prevent JSON encoding errors
    predicted_sales = round(float(predicted_sales), 2)

    # Return the predicted sales total
    return jsonify({'Predicted Sales Total (in dollars)': predicted_sales})


# Define an endpoint for batch prediction (POST request)
@superkart_sales_predictor_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing details for multiple products/stores
    and returns the predicted sales totals as a dictionary.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all rows in the DataFrame using the pipeline
    predicted_raw_sales = model.predict(input_data).tolist()

    # Round the sales predictions cleanly
    predicted_sales_totals = [round(float(sales), 2) for sales in predicted_raw_sales]

    # Create a dictionary of predictions with Product IDs as keys matching sample template workflow
    product_ids = input_data['Product_Id'].tolist()
    output_dict = dict(zip(product_ids, predicted_sales_totals))

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_sales_predictor_api.run(debug=True)
