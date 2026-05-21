import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("Advertising.csv")

# Input data
X = data[['TV', 'Radio', 'Newspaper']]

# Output data
y = data['Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict sales
y_pred = model.predict(X_test)

# Calculate error
error = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error:", error)