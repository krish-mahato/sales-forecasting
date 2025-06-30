# sales_forecast.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and explore the data
data = pd.read_csv("sales_data.csv")  # Replace with your file
print("First 5 rows:\n", data.head())
print("\nData info:\n", data.info())

# 2. Data Preprocessing
# Handle missing values
data = data.dropna()  # or use .fillna()

# Convert date to datetime and extract features
data['date'] = pd.to_datetime(data['date'])
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

# 3. Prepare features (X) and target (y)
X = data[['day', 'month', 'year', 'quantity']]  # Features
y = data['revenue']  # Target

# 4. Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate the model
print("\nModel Coefficients:", model.coef_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 8. Plot actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal Prediction")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.show()

# 9. Predict future sales (example)
future_data = pd.DataFrame({
    'day': [15, 20],
    'month': [7, 7],
    'year': [2023, 2023],
    'quantity': [100, 150]
})

future_predictions = model.predict(future_data)
print("\nFuture Sales Predictions:", future_predictions)