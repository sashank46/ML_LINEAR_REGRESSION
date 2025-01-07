import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Data Preparation
data = {
    'X': [1, 2, 3, 4, 5],
    'y': [2, 3, 5, 7, 11]
}

df = pd.DataFrame(data)
X = df[['X']]
y = df['y']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial Feature Transformation
poly = PolynomialFeatures(degree=10)  # Change degree for more flexibility
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Model Training
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predictions
y_pred = model.predict(X_test_poly)

# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test_poly.shape[1] - 1)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R^2 Score: {r2}')
print(f'Adjusted R^2 Score: {adj_r2}')

# Visualization
plt.scatter(X, y, color='blue', label='Data Points')
plt.scatter(X_train, y_train, color='green', label='Training Data')
plt.scatter(X_test, y_test, color='orange', label='Testing Data')

X_poly = poly.transform(X)
plt.plot(X, model.predict(X_poly), color='red', label='Polynomial Regression Line')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()
