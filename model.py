import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("dataset.txt")

# Data Preprocessing
def deg_to_rad(degrees):
    return degrees * (np.pi / 180)

def distcalculate(lat1, lon1, lat2, lon2):
    R = 6371
    d_lat = deg_to_rad(lat2 - lat1)
    d_lon = deg_to_rad(lon2 - lon1)
    a = np.sin(d_lat / 2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Calculate distance
data['distance'] = data.apply(lambda row: distcalculate(
    row['Restaurant_latitude'],
    row['Restaurant_longitude'],
    row['Delivery_location_latitude'],
    row['Delivery_location_longitude']
), axis=1)

# Encode categorical variables
le_order = LabelEncoder()
le_vehicle = LabelEncoder()
data['Type_of_order_encoded'] = le_order.fit_transform(data['Type_of_order'])
data['Type_of_vehicle_encoded'] = le_vehicle.fit_transform(data['Type_of_vehicle'])

# Save encoders
pickle.dump(le_order, open("order_encoder.pkl", "wb"))
pickle.dump(le_vehicle, open("vehicle_encoder.pkl", "wb"))

# Select features and target variable
X = data[['Delivery_person_Age', 'Delivery_person_Ratings', 'distance', 
          'Type_of_order_encoded', 'Type_of_vehicle_encoded']]
y = data['Time_taken(min)']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and feature columns
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns, open("feature_columns.pkl", "wb"))

# Build Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model
pickle.dump(model, open("food_delivery_model.pkl", "wb"))

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))
