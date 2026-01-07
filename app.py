"""
Streamlit App for Kathmandu Truck Pricing Model
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from src.ktm_distances import get_all_areas, get_distance

# Set page configuration
st.set_page_config(
    page_title="Kathmandu Truck Pricing Model",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #F0F9FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    .price-display {
        font-size: 2rem;
        font-weight: bold;
        color: #10B981;
        text-align: center;
    }
    .metric-box {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and encoders
@st.cache_resource
def load_model():
    """Load the trained model and encoders"""
    try:
        model = joblib.load('models/truck_pricing_model.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, label_encoders, feature_names
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Please ensure you've trained the model and saved it in the 'models' directory")
        return None, None, None

# Main app
def main():
    # App title and description
    st.markdown('<h1 class="main-header">🚚 Kathmandu Truck Pricing Model</h1>', unsafe_allow_html=True)
    st.markdown("""
    This app predicts truck transportation prices in Kathmandu using machine learning. 
    Enter the trip details below to get an estimated price.
    """)
    
    # Load model
    model, label_encoders, feature_names = load_model()
    
    if model is None:
        st.stop()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="sub-header">📋 Trip Details</h3>', unsafe_allow_html=True)
        
        # Input fields
        
        # distance = st.number_input(
        #     "Distance (km)",
        #     min_value=1.0,
        #     max_value=100.0,
        #     value=10.0,
        #     step=0.5,
        #     help="Enter the distance of the trip in kilometers"
        # )

        # Pickup location
        pickup_location = st.selectbox("📍 Pickup Location", options=get_all_areas(), index=0)

    # Drop location  
        drop_location = st.selectbox("📦 Delivery Location", options=get_all_areas(), index=4)

    # Auto-calculate distance using YOUR function
        distance = get_distance(pickup_location, drop_location)
        st.success(f"📏 **Distance: {distance:.1f} km**")
        st.info(f"**Route:** {pickup_location} → {drop_location}")

        
        truck_category = st.selectbox(
            "Truck Category",
            options=label_encoders['truck_category'].classes_,
            help="Select the size/category of the truck"
        )
        
        traffic_level = st.selectbox(
            "Traffic Level",
            options=label_encoders['traffic_level'].classes_,
            help="Select the expected traffic level"
        )
        
        time_of_day = st.selectbox(
            "Time of Day",
            options=label_encoders['time_of_day'].classes_,
            help="Select when the trip will occur"
        )
        
        is_peak_hour = st.checkbox(
            "Is Peak Hour?",
            help="Check if the trip is during peak hours (6-9 AM or 5-8 PM)"
        )
        
        # Calculate price button
        calculate_button = st.button("🚀 Calculate Estimated Price")
    
    with col2:
        st.markdown('<h3 class="sub-header">💰 Price Prediction</h3>', unsafe_allow_html=True)
        
        if calculate_button:
            # Prepare input data
            input_data = pd.DataFrame({
                'distance_km': [distance],
                'truck_category': [truck_category],
                'traffic_level': [traffic_level],
                'time_of_day': [time_of_day],
                'is_peak_hour': [int(is_peak_hour)]
            })
            
            # Encode categorical variables
            input_encoded = input_data.copy()
            for col in ['truck_category', 'traffic_level', 'time_of_day']:
                le = label_encoders[col]
                input_encoded[col] = le.transform(input_data[col])
            
            # Make prediction
            predicted_price = model.predict(input_encoded[feature_names])[0]
            
            # Calculate price range (±15%)
            lower_bound = predicted_price * 0.85
            upper_bound = predicted_price * 1.15
            
            # Display prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown('<p class="price-display">NPR {:,.0f}</p>'.format(predicted_price), unsafe_allow_html=True)
            st.markdown(f"**Price Range:** NPR {lower_bound:,.0f} - {upper_bound:,.0f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display trip details
            st.markdown("**Trip Summary:**")
            st.info(f"""
            - **Distance:** {distance} km
            - **Truck Category:** {truck_category}
            - **Traffic Level:** {traffic_level}
            - **Time of Day:** {time_of_day}
            - **Peak Hour:** {'Yes' if is_peak_hour else 'No'}
            """)
            
            # Manual calculation for comparison
            st.markdown("---")
            st.markdown("**📊 Manual Estimate Comparison**")
            
            # Manual calculation logic
            if truck_category == 'SMALL':
                manual_base = max(distance * 18, 400)
                base_rate = 18
                min_charge = 400
            elif truck_category == 'MEDIUM':
                manual_base = max(distance * 25, 700)
                base_rate = 25
                min_charge = 700
            else:  # LARGE
                manual_base = max(distance * 35, 1200)
                base_rate = 35
                min_charge = 1200
            
            # Apply traffic factor
            traffic_factors = {'Light': 1.0, 'Medium': 1.1, 'Heavy': 1.3, 'Very Heavy': 1.5}
            traffic_factor = traffic_factors.get(traffic_level, 1.1)
            
            # Apply time factor
            if is_peak_hour:
                time_factor = 1.2
                time_note = "Peak hour surcharge applied"
            elif time_of_day == 'Night':
                time_factor = 0.9
                time_note = "Night discount applied"
            else:
                time_factor = 1.0
                time_note = "Standard rate"
            
            manual_price = manual_base * traffic_factor * time_factor
            
            # Display manual calculation breakdown
            with st.expander("View Manual Calculation Breakdown"):
                st.write(f"**Base Calculation:**")
                st.write(f"- Distance: {distance} km × NPR {base_rate}/km = NPR {distance * base_rate:,.0f}")
                st.write(f"- Minimum charge: NPR {min_charge}")
                st.write(f"- Base price (higher of above): NPR {manual_base:,.0f}")
                
                st.write(f"**Traffic Factor:** {traffic_factor}× ({traffic_level} traffic)")
                st.write(f"**Time Factor:** {time_factor}× ({time_note})")
                st.write(f"**Manual Estimate:** NPR {manual_price:,.0f}")
            
            # Compare ML vs Manual
            difference = predicted_price - manual_price
            difference_percent = (difference / manual_price) * 100
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("ML Prediction", f"NPR {predicted_price:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_b:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Manual Estimate", f"NPR {manual_price:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_c:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Difference", f"NPR {difference:,.0f}", 
                         f"{difference_percent:+.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization
            st.markdown("---")
            st.markdown("**📈 Price Components Visualization**")
            
            # Create a bar chart comparing price components
            fig = go.Figure(data=[
                go.Bar(
                    name='ML Prediction',
                    x=['ML Prediction'],
                    y=[predicted_price],
                    text=[f'NPR {predicted_price:,.0f}'],
                    textposition='auto',
                    marker_color='#3B82F6'
                ),
                go.Bar(
                    name='Manual Estimate',
                    x=['Manual Estimate'],
                    y=[manual_price],
                    text=[f'NPR {manual_price:,.0f}'],
                    textposition='auto',
                    marker_color='#10B981'
                )
            ])
            
            fig.update_layout(
                title='ML vs Manual Price Comparison',
                yaxis_title='Price (NPR)',
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Display placeholder before calculation
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown('<p class="price-display">Enter trip details and click "Calculate"</p>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show example predictions
            st.markdown("**💡 Example Predictions:**")
            examples = [
                {"Distance": "5 km", "Truck": "SMALL", "Traffic": "Light", "Price": "NPR 400-500"},
                {"Distance": "15 km", "Truck": "MEDIUM", "Traffic": "Heavy", "Price": "NPR 1,200-1,500"},
                {"Distance": "25 km", "Truck": "LARGE", "Traffic": "Medium", "Price": "NPR 2,000-2,500"},
            ]
            
            for example in examples:
                st.write(f"• {example['Distance']}, {example['Truck']} truck, {example['Traffic']} traffic: {example['Price']}")
    
    # Model Information Section
    st.markdown("---")
    st.markdown('<h3 class="sub-header">🤖 Model Information</h3>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Model Details:**")
        st.info("""
        - **Algorithm:** Random Forest Regressor
        - **Trees:** 100 decision trees
        - **Max Depth:** 10 levels
        - **Training Data:** Synthetic Kathmandu truck data
        - **Target Variable:** Accepted Price (NPR)
        """)
    
    with col4:
        st.markdown("**Features Used:**")
        st.success("""
        1. Distance (km)
        2. Truck Category
        3. Traffic Level
        4. Time of Day
        5. Peak Hour Indicator
        """)
    
    # Feature Importance Visualization
    st.markdown("---")
    st.markdown('<h3 class="sub-header">📊 Feature Importance</h3>', unsafe_allow_html=True)
    
    # Get feature importance from model
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # Create horizontal bar chart
        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance in Price Prediction',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Batch Prediction Section
    st.markdown("---")
    st.markdown('<h3 class="sub-header">📁 Batch Prediction</h3>', unsafe_allow_html=True)
    
    with st.expander("Upload CSV for multiple predictions"):
        uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                batch_data = pd.read_csv(uploaded_file)
                
                # Check if required columns exist
                required_cols = ['distance_km', 'truck_category', 'traffic_level', 'time_of_day', 'is_peak_hour']
                missing_cols = [col for col in required_cols if col not in batch_data.columns]
                
                if missing_cols:
                    st.error(f"Missing columns in uploaded file: {missing_cols}")
                else:
                    # Encode categorical variables
                    batch_encoded = batch_data.copy()
                    for col in ['truck_category', 'traffic_level', 'time_of_day']:
                        le = label_encoders[col]
                        # Handle unseen categories
                        batch_encoded[col] = batch_encoded[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else 0
                        )
                    
                    # Make predictions
                    predictions = model.predict(batch_encoded[feature_names])
                    batch_data['predicted_price_npr'] = predictions
                    batch_data['price_range_lower'] = predictions * 0.85
                    batch_data['price_range_upper'] = predictions * 1.15
                    
                    # Display results
                    st.success(f"✅ Successfully predicted prices for {len(batch_data)} trips!")
                    
                    # Show preview
                    st.dataframe(batch_data.head(10))
                    
                    # Download button
                    csv = batch_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="truck_price_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Please ensure your CSV has the correct columns: distance_km, truck_category, traffic_level, time_of_day, is_peak_hour")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
        <p>🚚 Kathmandu Truck Pricing Model | ML-Powered Price Estimation</p>
        <p>Note: Prices are estimates. Actual prices may vary based on additional factors.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()