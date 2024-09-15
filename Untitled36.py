#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle

# Load the Ridge model from the pickle file
with open('ridge_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Example data processing function
def preprocess_data(df):
    # Convert categorical features to numeric using Label Encoding
    label_encoders = {}
    categorical_features = ['Product_Name', 'Category', 'Brand', 'Processor_Type', 'Operating_System']
    
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        label_encoders[feature] = le

    # Feature scaling
    scaler = StandardScaler()
    features_to_scale = ['Price', 'RAM', 'Battery_Life', 'Price_per_GB']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    return df

# Process the data
df1 = pd.read_csv('product_data3.csv')
df1 = preprocess_data(df1)

# Example user input
user_input = {
    'Product_Name': 'ProTab 11',
    'Category': 'Tablet',
    'Brand': 'Innovatech',
    'Price': 499,
    #'Units_Sold': 5000,
    'Customer_Rating': 4.3,
    'Processor_Type': 'Apple A14',
    'RAM': 6,
    'Battery_Life': 10,
    'Operating_System': 'iOS',
    'Warranty_Period': 1,
    'Price_per_GB': 1.949219
}

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])

# Process the user input
user_input_df = preprocess_data(user_input_df)

# Make prediction
user_input_processed = user_input_df[['Product_Name', 'Category', 'Brand', 'Price','Customer_Rating','Processor_Type', 'RAM', 'Battery_Life', 'Price_per_GB', 'Processor_Type', 'Operating_System']]
prediction = model.predict(user_input_processed)

print(f'Predicted value: {prediction[0]}')


# In[ ]:




