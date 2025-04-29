# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv('car_prices.csv', on_bad_lines='skip') 

# Clean variables
#Find main body types
body_types = df.groupby(["body"])
with open("body_type_counts.txt", "w") as f:    
    for item in body_types:
        f.write(str(item[0]) + "\n")

# df['transmission'] = df['transmission'].map({'manual': 0, 'automatic': 1})
# df = df[df['transmission'].notna()]

df = df.dropna(subset=['year', 'model', "odometer", "sellingprice"])
df = df[df['sellingprice'] < 100000]
df = df.reset_index(drop=True)

# def clean_body_type(body):
#     body = body.lower() 
#     if "convertible" in body:
#         return "convertible"
#     elif "coupe" in body:
#         return "coupe"
#     elif "wagon" in body:
#         return "wagon"
#     elif any(word in body for word in ["cab", "max", "crew"]):
#         return "pickup"
#     elif "minivan" in body:
#         return "minivan"
#     elif "van" in body:
#         return "van"
#     elif "sedan" in body:
#         return "sedan"
#     elif "hatchback" in body:
#         return "hatchback"
#     elif "suv" in body:
#         return "suv"
#     else:
#         return "other"  

# df["body"] = df["body"].apply(clean_body_type)

df["condition"] = pd.to_numeric(df["condition"])
df['condition'].fillna(df['condition'].median(), inplace=True)

#Create dummy for car body and make
model_df = pd.get_dummies(df['model'], prefix='model', drop_first=True)
# make_df = pd.get_dummies(df['make'], prefix='make', drop_first=True)

# Concatenate the dummy variables to the original dataframe
df = pd.concat([df, model_df], axis=1)

# Split the data into training and testing sets
X = df[['year', 'odometer', 'condition'] + list(model_df.columns)]

y = df['sellingprice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

#Linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict prices for the test set
y_pred = model.predict(X_test)

#Add predicted prices to the dataframe
df['predicted_price'] = model.predict(X)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Root Mean Squared Error: {mse**.5}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')
# print("Coefficients with Feature Names:")
# for feature, coef in zip(X.columns, model.coef_):
#     print(f"{feature}: {coef}")

# Visualize the results
plt.scatter(y_test, y_pred)
plt.axline((0, 0), (100000, 100000), linewidth=1, color='r')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# Display the first few rows of the dataframe
print(df.head())

#Sample 5 poor predictions
test_indices = X_test.index
test = df.loc[test_indices]
poor_predictions = test[abs(test['sellingprice'] - test["predicted_price"])  > 30000].sample(5)
print(poor_predictions[['sellingprice', 'predicted_price', 'make', "model", "trim", 'year', 'odometer', "condition"]])

# Print coefficients and intercept
for feature, coef in zip(X.columns, model.coef_[:3]):
    print(f"{feature}: {round(coef, 5)}")
print("Intercept:", model.intercept_)

coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
coefficients.to_csv("coefficients.csv", index=False)