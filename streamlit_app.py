import streamlit as st
import requests

st.set_page_config(page_title="Olist Churn Predictor", layout="centered")

st.title("Olist Customer Churn Prediction")
st.write("Enter the order details below to calculate the probability of customer churn.")

# Use columns for a cleaner input layout
col1, col2, col3 = st.columns(3)
with col1:
    price = st.number_input("Price ($)", min_value=0.0, value=400.0)
with col2:
    freight_value = st.number_input("Freight ($)", min_value=0.0, value=20.0)
with col3:
    delivery_time = st.number_input("Delivery (Days)", min_value=0.0, value=50.0)

if st.button("Analyze Churn Risk"):
    data = {
        "price": price,
        "freight_value": freight_value,
        "delivery_time": delivery_time
    }

    try:
        response = requests.post("http://api:8000/predict", json=data)
        result = response.json()
        
        prob = result['probability']
        prediction = result['prediction']

        st.divider()
        st.subheader("Results")

        # 1. Visual Gauge/Progress Bar
        st.write(f"**Churn Probability:** {prob:.2%}")
        st.progress(prob)

        # 2. Conditional Formatting based on Risk
        if prob > 0.7:
            st.error(f"### HIGH RISK: Potential Churn Detected")
            st.write("Action: Consider offering a high-value discount or priority support.")
        elif prob > 0.4:
            st.warning(f"### MEDIUM RISK: Monitoring Recommended")
            st.write("Action: Send a follow-up satisfaction survey.")
        else:
            st.success(f"### LOW RISK: Loyal Customer")
            st.write("Action: No immediate intervention needed.")

    except Exception as e:
        st.error(f"Error connecting to API: {e}")