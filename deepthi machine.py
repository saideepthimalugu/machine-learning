#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Import libraries required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load the California Housing Dataset directly loaded
housing = fetch_california_housing(as_frame=True)
df = housing.frame  # Convert to DataFrame

# basic info about the dataset
print(df.head())

# Spliting the  dataset into training and testing sets
X = df.drop(columns=['MedHouseVal'])  # Features
y = df['MedHouseVal']  # Target variable (house price)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Plot actual vs predicted prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
# Plot diagonal line (ideal predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black', linestyle='--')
plt.show()


# In[8]:


df.isna().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


df.info()


# In[12]:


df.describe()


# In[3]:


# Print features(attributes) importance (coefficients)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_})
print(feature_importance.sort_values(by='Importance', ascending=False))


# In[4]:


#heat map 
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# In[5]:


from sklearn.ensemble import RandomForestRegressor

# Train the  Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest MSE: {mse_rf:.2f}")
print(f"Random Forest R² Score: {r2_rf:.2f}")


# In[6]:


from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best R² Score:", grid_search.best_score_)


# In[7]:


import joblib

# Save the model in pc
joblib.dump(rf_model, "california_housing_model.pkl")

# Load the model for later use
loaded_model = joblib.load("california_housing_model.pkl")


# In[ ]:




