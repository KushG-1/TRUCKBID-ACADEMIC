
import streamlit as st
import pandas as pd
import pickle
import os
import sys

# Add maal to Python path
sys.path.append('maal')
st.title("🚚 TRUCKBID - Bid Prediction Model")
st.image("model_performance.png", caption="Your Trained Model")

# Check if model exists
model_path = 'maal/models/truck_pricing_model.pkl'  # Update exact filename
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    st.success(f"✅ Model loaded: {model_path}")
else:
    st.error(f"❌ Model not found: {model_path}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload test data CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    
    if st.button("🔮 Predict Truck Bids", type="primary"):
        try:
            predictions = model.predict(df)
            st.success(f"✅ Predicted {len(predictions)} bids!")
            st.dataframe(pd.DataFrame({'Predicted_Bid': predictions}))
            st.balloons()
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
