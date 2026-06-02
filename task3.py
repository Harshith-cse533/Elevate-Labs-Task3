#To Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#To Upload Dataset
from google.colab import files
uploaded = files.upload()
#To Load Dataset
df = pd.read_csv('house_price.csv')
#View Dataset
df.head()
#To Select Input and Output Columns
X = df[['Area']]
y = df['Price']
#To Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# To Create Model
model = LinearRegression()
#To Train Model
model.fit(X_train, y_train)
#To Predict Prices
y_pred = model.predict(X_test)
#To Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print("MAE =", mae)
#To Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print("MSE =", mse)
#To Calculate R² Score
r2 = r2_score(y_test, y_pred)
print("R2 Score =", r2)
#To Plot Grap
plt.scatter(X, y)

plt.plot(X, model.predict(X),
         color='red')

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Linear Regression")

plt.show()
#To Check Coefficients
print("Coefficient =", model.coef_)
print("Intercept =", model.intercept_)
