import sklearn
import numpy as np
import pandas as pd
import seaborn as sns

penguim = sns.load_dataset("penguins")

with sns.axes_style('whitegrid'):

  grafico = sns.pairplot(data=penguim, hue="sex", palette="pastel")

with sns.axes_style('whitegrid'):

  grafico = sns.pairplot(data=penguim, hue="species", palette="pastel")

with sns.axes_style('whitegrid'):

  grafico = sns.pairplot(data=penguim, hue="island", palette="pastel")

penguim_cleaned = penguim.dropna()
display(penguim_cleaned.head())

penguim[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]

from sklearn.preprocessing import StandardScaler

numerical_cols = penguim_cleaned.select_dtypes(include=np.number).columns

scaler = StandardScaler()

penguim_cleaned[numerical_cols] = scaler.fit_transform(penguim_cleaned[numerical_cols])

for col in numerical_cols:
    penquim_cleaned = penguim_cleaned.rename(columns={col: col + "_std"})

display(penguim_cleaned.head())

nominal_cols = ['species', 'island', 'sex']

for col in nominal_cols:
    if col in penguim.columns:
        one_hot_encoded = pd.get_dummies(penguim[col], prefix=col)
        penguim = pd.concat([penguim, one_hot_encoded], axis=1)

        new_col_names = {c: f'{col}_nom_{c.split("_")[-1]}' for c in one_hot_encoded.columns}
        penguim.rename(columns=new_col_names, inplace=True)


display(penguim.head())

target_variable = penguim_cleaned[['sex']]

scaled_features = penquim_cleaned.filter(like='_std')

nominal_features = penguim.filter(like='_nom')

final_df = pd.concat([target_variable, scaled_features, nominal_features], axis=1)

display(final_df.head())

from sklearn.model_selection import train_test_split
import pandas as pd

# Remove rows with missing values in the target variable ('sex')
final_df_cleaned = final_df.dropna(subset=['sex'])

# One-hot encode the target variable 'sex'
y = pd.get_dummies(final_df_cleaned['sex'], prefix='sex', drop_first=True) # Drop one column to avoid multicollinearity

# Separate features (X)
X = final_df_cleaned.drop('sex', axis=1)

# Split the data into training and testing sets (2/3 train, 1/3 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Linear Regression model trained successfully.")

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import numpy as np


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

model = LinearRegression()

model.fit(X_train_imputed, y_train)

print("Linear Regression model trained successfully with imputed data.")

from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(X_train)
X_test_imputed = imputer.transform(X_test)


X_test_imputed_df = pd.DataFrame(X_test_imputed, columns=X_test.columns)

y_pred = model.predict(X_test_imputed_df)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"RMSE para o modelo de Regressão Linear: {rmse}")

"""| species	| island | bill_length_mm | bill_depth_mm | flipper_length_mm | sex |
| --- | --- | --- | --- | --- | --- |
| Adelie | Biscoe | 38.2 | 18.1 | 185.0 | Male |
"""

new_penguin_data = {
    'species': ['Adelie'],
    'island': ['Biscoe'],
    'bill_length_mm': [38.2],
    'bill_depth_mm': [18.1],
    'flipper_length_mm': [185.0],
    'sex': ['Male']
}
new_penguin_df = pd.DataFrame(new_penguin_data)

numerical_cols_new_prediction = new_penguin_df.select_dtypes(include=np.number).columns
new_penguin_numerical = new_penguin_df[numerical_cols_new_prediction]

nominal_cols_new_prediction = new_penguin_df.select_dtypes(include='object').columns
new_penguin_nominal = new_penguin_df[nominal_cols_new_prediction]

new_penguin_numerical_scaled = scaler_retraining.transform(new_penguin_numerical)
new_penguin_numerical_scaled_df = pd.DataFrame(new_penguin_numerical_scaled, columns=numerical_cols_new_prediction)


new_penguin_nominal_encoded = pd.get_dummies(new_penguin_nominal)

new_penguin_nominal_encoded = new_penguin_nominal_encoded.reindex(columns=nominal_feature_cols_retrained_X, fill_value=0)

retrained_feature_cols_order = X_retrained.columns

new_penguin_processed_features = pd.concat([new_penguin_numerical_scaled_df, new_penguin_nominal_encoded], axis=1)

new_penguin_processed_features = new_penguin_processed_features.reindex(columns=retrained_feature_cols_order, fill_value=0)

predicted_weight = model_retrained.predict(new_penguin_processed_features)

print(f"Massa corporal prevista para o novo pinguim: {predicted_weight[0].item():.2f}g")
