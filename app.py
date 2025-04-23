# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('model2.pkl', 'rb') as file:
    model = pickle.load(file)

# Encoding mappings
item_mapping = {'Baking Goods': 0, 'Breads': 1, 'Breakfast': 2, 'Canned': 3, 'Dairy': 4,
                'Frozen Foods': 5, 'Fruits and Vegetables': 6, 'Hard Drinks': 7, 'Health and Hygiene': 12,
                'Household': 11, 'Meat': 10, 'Others': 9, 'Seafood': 8, 'Snack Foods': 13,
                'Soft Drinks': 14, 'Starchy Foods': 15}

outlet_mapping = {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}
location_mapping = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
fat_content_mapping = {'Low Fat': 0, 'Regular': 1}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    item_weight = float(request.form['item_weight'])
    item_fat = fat_content_mapping.get(request.form['item_fat_content'], 0)
    item_vis = float(request.form['item_visibility'])
    item_type = item_mapping.get(request.form['item_type'], 0)
    item_mrp = float(request.form['item_mrp'])
    location = location_mapping.get(request.form['location_type'], 0)
    outlet = outlet_mapping.get(request.form['outlet_type'], 0)
    year = int(request.form['year'])

    # Derived feature
    outlet_age = year - 2021

    input_df = pd.DataFrame([[item_weight, item_fat, item_vis, item_type, item_mrp,
                               location, outlet, outlet_age]],
                             columns=['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP',
                                      'Outlet_Location_Type', 'Outlet_Type', 'Age_Outlet'])

    prediction = model.predict(input_df)[0]
    predicted_growth = prediction * ((1 + 0.05) ** outlet_age)

    return render_template('index.html',
                           prediction_text=f"Predicted Sales: ₹{round(prediction, 2)}",
                           future_prediction_text=f"Estimated Sales for {year}: ₹{round(predicted_growth, 2)}")

if __name__ == "__main__":
    app.run(debug=True)
