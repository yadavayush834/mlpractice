# Beginner-friendly Linear Regression in Python
# This example predicts house prices based on house size

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create sample data (house sizes and prices)
# In real projects, you'd load this from a file
house_sizes = np.array([500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 
                       1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400])

# Prices in thousands (with some realistic variation)
house_prices = np.array([50, 60, 65, 80, 85, 95, 110, 115, 125, 135,
                        140, 155, 165, 175, 185, 195, 210, 220, 230, 245])

# Step 2: Prepare the data
# Reshape data for sklearn (it expects 2D arrays)
X = house_sizes.reshape(-1, 1)  # Features (house sizes)
y = house_prices                # Target (house prices)

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))

# Step 4: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Model Equation: Price = {model.coef_[0]:.2f} * Size + {model.intercept_:.2f}")

# Step 7: Visualize the results
plt.figure(figsize=(12, 5))

# Plot 1: Training data and model line
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', alpha=0.7, label='Training Data')
plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Best Fit Line')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($1000s)')
plt.title('Linear Regression: Training Data')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Test predictions vs actual
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, color='green', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Prices ($1000s)')
plt.ylabel('Predicted Prices ($1000s)')
plt.title('Predictions vs Actual Values')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Step 8: Make a prediction for a new house
new_house_size = 1750  # sq ft
predicted_price = model.predict([[new_house_size]])
print(f"\nPrediction: A {new_house_size} sq ft house would cost approximately ${predicted_price[0]:.0f},000")

# Step 9: Show model details
print(f"\nModel Details:")
print(f"Slope (coefficient): {model.coef_[0]:.2f} - This means each additional sq ft adds ${model.coef_[0]*1000:.0f} to the price")
print(f"Intercept: {model.intercept_:.2f} - This is the base price in thousands")
