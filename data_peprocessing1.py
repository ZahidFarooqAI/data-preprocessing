import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Example dataset
data = {
    'Age': [25, np.nan, 35, 40, np.nan],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Salary': [50000, 60000, np.nan, 80000, 75000]
}
df = pd.DataFrame(data)

# 1. Handle missing values
imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

# 2. Encode categorical data
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])

# 3. Normalize salary
scaler = MinMaxScaler()
df[['Salary']] = scaler.fit_transform(df[['Salary']])

# 4. Split data
X = df[['Age', 'Gender', 'Salary']]
y = [0, 1, 0, 1, 0]  # Example target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(df)
#output
    Age         Gender   Salary
0  25.000000       1  0.000000
1  33.333333       0  0.333333
2  35.000000       0  0.541667
3  40.000000       1  1.000000
4  33.333333       0  0.833333

