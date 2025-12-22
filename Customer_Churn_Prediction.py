# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('Telco_Customer_Churn.csv')

# Show the first 5 rows to verify
df.head()

# 1. Drop customerID (it's unique for everyone and useless for prediction)
df.drop('customerID', axis=1, inplace=True)

# 2. Fix 'TotalCharges': Convert to numeric and force errors to NaN (Not a Number)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 3. Check for null values created by the conversion
print("Missing values in TotalCharges:", df['TotalCharges'].isnull().sum())

# 4. Fill missing values with 0 (assuming tenure is 0 for these) or drop them
df.dropna(inplace=True)

# 5. Verify data types are now correct
df.info()

# Set plot style
sns.set(style="whitegrid")

# 1. Visualize the Target Variable (Churn)
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution (Target Variable)')
plt.show()

# 2. Visualize Numerical Distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['tenure'], kde=True, ax=axes[0])
axes[0].set_title('Distribution of Tenure (Months)')

sns.histplot(df['MonthlyCharges'], kde=True, ax=axes[1])
axes[1].set_title('Distribution of Monthly Charges')
plt.show()

# 3. Visualize Categorical features vs Churn 
plt.figure(figsize=(8, 5))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract Type')
plt.show()

# 1. Encode the Target Variable 'Churn' 
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 2. Convert other categorical columns using "One-Hot Encoding"
df_encoded = pd.get_dummies(df, drop_first=True)

# 3. Define Features (X) and Target (y)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# 4. Scale the features 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data successfully encoded and scaled.")
print("Feature shape:", X_scaled.shape)

# 1. Split data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Train Model A: Logistic Regression (Baseline)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# 3. Train Model B: Random Forest (Tree-based)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Models trained successfully.")

# Function to print results nicely
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"--- {name} Results ---")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    print("\n")

# Evaluate both models
evaluate_model(log_model, "Logistic Regression")
evaluate_model(rf_model, "Random Forest")

# Get feature importances from Random Forest
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

# Plot top 10 important features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Top 10 Factors Driving Customer Churn')
plt.show()