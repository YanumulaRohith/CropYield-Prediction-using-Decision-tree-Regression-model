import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data set
df = pd.read_csv("crop_yield_data.csv",encoding= 'unicode_escape')
df = df[:100]
# Normalize the data set
attributes = ['Rain Fall (mm)', 'Fertilizer(urea) (kg/acre)', 'Temperature (°C)', 'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
for attr in attributes:
    mean = df[attr].mean()
    std = df[attr].std()
    df[attr] = (df[attr] - mean) / std
# Display the normalized data set
print(df.head(100))


# Split the data into training and testing sets
X = df[attributes]
y = df['Yeild (Q/acre)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=38)

# Train the decision tree regressor
regressor = DecisionTreeRegressor(random_state=41)  #41
regressor.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = regressor.predict(X_test)
# Measure R2 score
r2 = r2_score(y_test, y_pred)

# Measure MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display the results
print('R Squared(R2) Score:', r2)
print('Mean Square Error(MSE):', mse)
print('Root Mean Square Error(RMSE):', rmse)


# Plot the R2 score over a range of values
values = np.linspace(0, 1, 100)
plt.plot(values, [r2_score(y_test,y_pred * v) for v in values])
plt.xlabel("Prediction multiplier")
plt.ylabel("R2 score")
#plt.show()