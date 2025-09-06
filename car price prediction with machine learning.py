import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# ---- data (same as yours) ----
data = {
    'car_name': ['Toyota', 'Honda', 'Ford', 'BMW', 'Audi', 'Toyota', 'Honda', 'BMW', 'Ford', 'Audi'],
    'year': [2015, 2017, 2016, 2018, 2019, 2014, 2017, 2018, 2015, 2019],
    'present_price': [10.5, 9.0, 7.5, 35.0, 40.0, 11.0, 9.5, 33.0, 8.0, 39.0],
    'selling_price': [8.0, 7.0, 6.0, 30.0, 35.0, 7.5, 6.5, 31.0, 6.0, 34.0],
    'driven_km': [50000, 30000, 40000, 20000, 15000, 60000, 35000, 22000, 45000, 14000],
    'owner': [1, 0, 0, 0, 1, 1, 0, 0, 1, 0],
    'fuel_type': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Petrol', 'Diesel', 'Diesel', 'Petrol', 'Petrol'],
    'seller_type': ['Dealer', 'Individual', 'Dealer', 'Dealer', 'Individual', 'Dealer', 'Individual', 'Dealer', 'Individual', 'Dealer'],
    'transmission': ['Manual', 'Manual', 'Automatic', 'Automatic', 'Manual', 'Manual', 'Automatic', 'Automatic', 'Manual', 'Manual']
}
df = pd.DataFrame(data)

# ---- simple feature tweak: use 'age' instead of raw 'year' ----
current_year = 2025
df['age'] = current_year - df['year']

# ---- features/target ----
target = 'selling_price'
features = ['car_name','fuel_type','seller_type','transmission','age','present_price','driven_km','owner']
X = df[features].copy()
y = df[target].copy()

categorical_features = ['car_name','fuel_type','seller_type','transmission']
numeric_features = ['age','present_price','driven_km','owner']

preprocessor = ColumnTransformer(
    transformers=[
        # handle_unknown='ignore' avoids errors when a category is missing in train/test folds
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False), categorical_features),
        ('num', StandardScaler(), numeric_features),
    ]
)

# ---- regularized model to prevent extreme coeffs on tiny data ----
model = Lasso(alpha=0.1, max_iter=10000, tol=1e-4, selection='cyclic')

pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', model)
])

# ---- split & fit ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
pipeline.fit(X_train, y_train)

# ---- evaluate ----
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R^2 Score: {r2:.3f}")

# ---- quick 5-fold CV for stability check (still tiny data!) ----
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(pipeline, X, y, scoring='r2', cv=cv)
cv_mse = -cross_val_score(pipeline, X, y, scoring='neg_mean_squared_error', cv=cv)
print(f"CV R^2 (mean ± std): {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
print(f"CV MSE (mean): {cv_mse.mean():.3f}")

# ---- plot: actual vs predicted ----
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Selling Price (lakhs)')
plt.ylabel('Predicted Selling Price (lakhs)')
plt.title('Actual vs Predicted (Lasso)')
plt.tight_layout()
plt.show()

