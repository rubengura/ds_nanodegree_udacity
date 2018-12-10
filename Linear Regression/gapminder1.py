# TODO: Add import statements
from sklearn.linear_model import LinearRegression
import pandas as pd

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv') 
y_values = bmi_life_data.loc[:, ['Life expectancy']]
x_values = bmi_life_data.loc[:, ['BMI']]

print(x_values.shape)

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(x_values, y_values)

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)