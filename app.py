import streamlit as st
import pickle
import pandas as pd
st.title("Inventory Demand Forecast")


# Load model
with open("inventory_demand_model.pkl", "rb") as f:
    model = pickle.load(f)

st.write("### Enter details for prediction")

# User input
store_id = st.number_input("Select Store ID", min_value=1,max_value=5,value=1,step=1)
product_id = st.number_input("Select Product ID",min_value=1,max_value=20,value=1,step=1 )

year = st.number_input("Year", min_value=2023, max_value=2026, value=2023, step=1)

month = st.number_input("Month", min_value=1, max_value=12, value=1, step=1)


if st.button("Predict Inventory Rate"):
    try:
        # Input for prediction (ensure all required features are present)
        input_data = [[store_id, product_id, year, month]]

        # Predict using the trained model
        prediction = model.predict(input_data)

        # Flatten the prediction (in case it's 2D)
        preds = prediction.flatten()

        # Define labels in the same order as your training targets
        labels = ["Inventory Level", "Units Ordered", "Units Sold", "Price"]

        # Loop and display each result
        for i in range(len(preds)):
            if labels[i] == "Price":
                st.success(f"Predicted {labels[i]}: ₹{preds[i]:.2f}")
            else:
                st.success(f"Predicted {labels[i]}: {preds[i]:.2f}")

    except Exception as e:
        st.error(f" Prediction failed: {e}")
