#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and encoders
model = joblib.load("flight_price_model.pkl")
oe = joblib.load("ordinal_encoder.pkl")         # Ordinal encoder: Class, Dep_Period, Arr_Period, Stop
ohe = joblib.load("onehot_encoder.pkl")         # OneHot encoder: Airline, From, To
feature_columns = joblib.load("feature_columns.pkl")  # Column order after encoding

# App Title & Image
st.title("Flight Price Predictor")
st.image(r"C:\Users\Dharmik\Downloads\85843c0f-7513-4868-9917-3dd877925a03.jpeg", width=300)

# --- User Inputs ---
airline = st.selectbox("Airline", ["Vistara", "Air India", "Indigo", "GO FIRST", "AirAsia", "SpiceJet", "StarAir", "Trujet"])
source_city = st.selectbox("Source City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
departure_time = st.selectbox("Departure Time", ["Morning", "Afternoon", "Evening", "Night"])
arrival_time = st.selectbox("Arrival Time", ["Morning", "Afternoon", "Evening", "Night"])
stops = st.selectbox("Number of Stops", ['0', '1', '2'])   # Keep as string for encoder compatibility
destination_city = st.selectbox("Destination City", ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
flight_class = st.radio("Class", ["economic", "business"])
day = st.slider("Day of the Month", 1, 31)

# --- Prediction Logic ---
if st.button("Predict Price"):
    try:
        # Step 1: Raw Input Dictionary
        input_dict = {
            'Airline': [airline],
            'From': [source_city],
            'To': [destination_city],
            'Class': [flight_class],
            'Stop': [stops],   # Keep as string
            'Dep_Period': [departure_time],
            'Arr_Period': [arrival_time],
            'Day': [int(day)]
        }

        df_input = pd.DataFrame(input_dict)

        # Step 2: Column Separation
        ordinal_cols = ['Dep_Period', 'Arr_Period', 'Class', 'Stop']
        onehot_cols = ['Airline', 'From', 'To']

        # Step 3: Ordinal Encoding
        encoded_ordinal = oe.transform(df_input[ordinal_cols])
        df_input[ordinal_cols] = encoded_ordinal

        # Step 4: OneHot Encoding
        df_encoded = pd.DataFrame(
            ohe.transform(df_input[onehot_cols]),
            columns=ohe.get_feature_names_out(onehot_cols)
        )



        # Step 5: Combine all features
        final_input = pd.concat([
            df_encoded.reset_index(drop=True),
            df_input[['Day']].reset_index(drop=True),
            df_input[ordinal_cols].reset_index(drop=True)
        ], axis=1)

        # Step 6: Align columns to model input order
        final_input = final_input.reindex(columns=feature_columns, fill_value=0)

        # Step 7: Predict
        prediction_log = model.predict(final_input)
        prediction = np.expm1(prediction_log)


        # Step 8: Display result
        st.success(f"Predicted Ticket Price: ₹{prediction[0]:,.2f}")
        
    except Exception as e:
        import traceback
        st.error("Prediction failed. See error below:")
        st.code(traceback.format_exc())


# In[ ]:





# In[ ]:




