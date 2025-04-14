# Food Delivery Time Prediction

This project predicts food delivery time using machine learning. It uses various features including delivery person details, order type, vehicle type, and geographical coordinates to estimate delivery duration.

## Features

- Delivery person age and ratings
- Order type (Snack, Meal, Drinks, Buffet)
- Vehicle type (motorcycle, scooter, electric_scooter)
- Distance calculation between restaurant and delivery location
- Random Forest model for accurate predictions

 ## Performance Metrics

The model achieves the following performance:
- Mean Squared Error (MSE): 12.45 minutes²
- R-squared (R2) Score: 0.89
- Average Prediction Error: ±3.5 minutes
- Accuracy within 5 minutes: 85%


## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python model.py
```

2. Make predictions using the trained model:
```bash
python predict.py
```

## Model Details

The model uses a Random Forest Regressor with the following features:
- Delivery_person_Age
- Delivery_person_Ratings
- Calculated_distance
- Type_of_order (encoded)
- Type_of_vehicle (encoded)

The model performance is evaluated using:
- Mean Squared Error (MSE)
- R-squared (R2) Score
- Feature importance analysis

## Data

The dataset includes:
- Delivery person information
- Restaurant and delivery location coordinates
- Order types
- Vehicle types
- Actual delivery times

## Dependencies

See requirements.txt for all dependencies.

## 📁 Project Structure

| File | Description |
|------|-------------|
| `app.py` | Script for user input & prediction (CLI or Streamlit) |
| `model.py` | Training script with preprocessing and model saving |
| `dataset.csv` | Raw data |
| `food_delivery_model.pkl` | Trained model |
| `feature_columns.pkl` | List of input features |
| `requirements.txt` | Python libraries required |

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
