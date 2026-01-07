"""
Train ML Pricing Model for Kathmandu Trucks
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD YOUR GENERATED DATA
try:
    df = pd.read_csv('data/data.csv')
except FileNotFoundError:
    print("❌ Dataset not found! Check filename.")

# 2. EXPLORE THE DATA
print("\n📊 Dataset Preview:")
print(df.head())
print(f"\nColumns: {list(df.columns)}")
print(f"\nData Types:")
print(df.dtypes)

# Features (X) - what we use to predict
features = ['distance_km', 'truck_category', 'traffic_level', 'time_of_day', 'is_peak_hour']
X = df[features].copy()

# Target (y) - what we want to predict
y = df['accepted_price_npr']

print(f"Features: {features}")
print(f"Target: Price (NPR)")
print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")


# Save encoders for later use
label_encoders = {}

categorical_cols = ['truck_category', 'traffic_level', 'time_of_day']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"  Encoded '{col}' with {len(le.classes_)} classes: {list(le.classes_)}")

# SPLIT DATA INTO TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Create and train model
model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth of trees
    min_samples_split=5,   # Minimum samples to split
    min_samples_leaf=2,    # Minimum samples in leaf
    random_state=42,       # For reproducibility
    n_jobs=-1              # Use all CPU cores
)

model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("🎯 Model Performance Metrics:")
print(f"  Training R² Score:  {train_r2:.3f}")
print(f"  Testing R² Score:   {test_r2:.3f}")
print(f"  Training MAE:       NPR {train_mae:,.0f}")
print(f"  Testing MAE:        NPR {test_mae:,.0f}")
print(f"  Training RMSE:      NPR {train_rmse:,.0f}")
print(f"  Testing RMSE:       NPR {test_rmse:,.0f}")

# 8. ANALYZE FEATURE IMPORTANCE
print("\n🎯 Feature Importance Analysis:")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance Ranking:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']:20}: {row['importance']:.3f} ({row['importance']*100:.1f}%)")

# 9. SAVE MODEL AND ENCODERS
print("\n💾 Saving model and encoders...")

# Create models directory if it doesn't exist
import os
if not os.path.exists('models'):
    os.makedirs('models')

# Save model
joblib.dump(model, 'models/truck_pricing_model.pkl')
print("✅ Model saved: models/truck_pricing_model.pkl")

# Save label encoders
joblib.dump(label_encoders, 'models/label_encoders.pkl')
print("✅ Label encoders saved: models/label_encoders.pkl")

# Save feature names
joblib.dump(features, 'models/feature_names.pkl')
print("✅ Feature names saved: models/feature_names.pkl")

# 10. CREATE VISUALIZATIONS
print("\n📊 Creating visualizations...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ML Model Performance Analysis', fontsize=16)

# 1. Actual vs Predicted (Test set)
ax1 = axes[0, 0]
ax1.scatter(y_test, y_test_pred, alpha=0.6)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Price (NPR)')
ax1.set_ylabel('Predicted Price (NPR)')
ax1.set_title('Actual vs Predicted Prices (Test Set)')
ax1.grid(True, alpha=0.3)

# 2. Feature Importance
ax2 = axes[0, 1]
bars = ax2.barh(feature_importance['feature'], feature_importance['importance'])
ax2.set_xlabel('Importance')
ax2.set_title('Feature Importance in Price Prediction')
ax2.invert_yaxis()  # Highest importance at top

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', va='center')

# 3. Error Distribution
ax3 = axes[1, 0]
errors = y_test - y_test_pred
ax3.hist(errors, bins=30, edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='r', linestyle='--')
ax3.set_xlabel('Prediction Error (NPR)')
ax3.set_ylabel('Frequency')
ax3.set_title('Prediction Error Distribution')
ax3.grid(True, alpha=0.3)

# 4. Price vs Distance with Predictions
ax4 = axes[1, 1]
sample_idx = np.random.choice(len(X_test), min(100, len(X_test)), replace=False)
ax4.scatter(X_test.iloc[sample_idx]['distance_km'], 
           y_test.iloc[sample_idx], alpha=0.6, label='Actual')
ax4.scatter(X_test.iloc[sample_idx]['distance_km'], 
           y_test_pred[sample_idx], alpha=0.6, label='Predicted', marker='x')
ax4.set_xlabel('Distance (km)')
ax4.set_ylabel('Price (NPR)')
ax4.set_title('Price vs Distance (Sample)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved: model_performance.png")

# 11. TEST WITH SAMPLE PREDICTIONS
print("\n🧪 Testing sample predictions...")

# Get original feature names (before encoding)
def decode_features(encoded_features, encoders):
    """Decode encoded features back to original labels"""
    decoded = {}
    for feature, value in encoded_features.items():
        if feature in encoders:
            # Find the label for this encoded value
            le = encoders[feature]
            decoded_value = le.inverse_transform([value])[0]
            decoded[feature] = decoded_value
        else:
            decoded[feature] = value
    return decoded

# Test with a few samples
test_samples = [
    # Sample 1: Short trip, small truck, light traffic
    {
        'distance_km': 4.2,
        'truck_category': 'SMALL',
        'traffic_level': 'Light',
        'time_of_day': 'Afternoon',
        'is_peak_hour': 0
    },
    # Sample 2: Medium trip, medium truck, heavy traffic (peak)
    {
        'distance_km': 12.5,
        'truck_category': 'MEDIUM',
        'traffic_level': 'Heavy',
        'time_of_day': 'Evening',
        'is_peak_hour': 1
    },
    # Sample 3: Long trip, large truck, night
    {
        'distance_km': 18.7,
        'truck_category': 'LARGE',
        'traffic_level': 'Medium',
        'time_of_day': 'Night',
        'is_peak_hour': 0
    }
]

print("\n📋 Sample Predictions:")
print("-" * 80)

for i, sample in enumerate(test_samples, 1):
    # Prepare input
    input_df = pd.DataFrame([sample])
    
    # Encode categorical features
    input_encoded = input_df.copy()
    for col in categorical_cols:
        le = label_encoders[col]
        # Handle unseen categories gracefully
        try:
            input_encoded[col] = le.transform([sample[col]])[0]
        except ValueError:
            # Use most common category as fallback
            input_encoded[col] = 0
    
    # Make prediction
    predicted_price = model.predict(input_encoded[features])[0]
    
    # Calculate price range (±15%)
    lower_bound = predicted_price * 0.85
    upper_bound = predicted_price * 1.15
    
    print(f"\nSample {i}:")
    print(f"  Route: {sample['distance_km']}km | Truck: {sample['truck_category']}")
    print(f"  Traffic: {sample['traffic_level']} | Time: {sample['time_of_day']}")
    print(f"  Peak Hour: {'Yes' if sample['is_peak_hour'] else 'No'}")
    print(f"  Predicted Price: NPR {predicted_price:,.0f}")
    print(f"  Price Range: NPR {lower_bound:,.0f} - {upper_bound:,.0f}")
    
    # Compare with manual calculation
    if sample['truck_category'] == 'SMALL':
        manual_base = max(4.2 * 18, 400)
    elif sample['truck_category'] == 'MEDIUM':
        manual_base = max(12.5 * 25, 700)
    else:
        manual_base = max(18.7 * 35, 1200)
    
    # Apply traffic factor
    traffic_factors = {'Light': 1.0, 'Medium': 1.1, 'Heavy': 1.3, 'Very Heavy': 1.5}
    traffic_factor = traffic_factors.get(sample['traffic_level'], 1.1)
    
    if sample['is_peak_hour']:
        manual_price = manual_base * traffic_factor * 1.2
    elif sample['time_of_day'] == 'Night':
        manual_price = manual_base * traffic_factor * 0.9
    else:
        manual_price = manual_base * traffic_factor
    
    print(f"  Manual Estimate: NPR {manual_price:,.0f}")
    print(f"  Difference: {((predicted_price - manual_price)/manual_price*100):+.1f}%")

