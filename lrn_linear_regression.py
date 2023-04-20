import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diseases = datasets.load_diabetes()
#print(diseases)
diseases_X = diseases.data[:,np.newaxis,2]

#Splitting the data
diseases_X_train = diseases_X[:-30]
diseases_X_test = diseases_X[-20:]

diseases_Y_train = diseases.target[:-30]
diseases_Y_test = diseases.target[-20:]

reg = linear_model.LinearRegression()
reg.fit(diseases_X_train,diseases_Y_train)

y_predict = reg.predict(diseases_X_test)

accuracy = mean_squared_error(diseases_Y_test,y_predict)

print(accuracy)

weight = reg.coef_
intercept = reg.intercept_
print(weight,intercept) 

plt.scatter(diseases_X_test,diseases_Y_test)
plt.plot(diseases_X_test,y_predict)
plt.show()

#If you have a dataset in a csv file 
'''
Example the dataset named 'Cars'

cars = pd.read_csv("cars.csv")

#All printing options 
print("cars.head")
print("cars.columns")


plt.figure(figsize(16,9))
plt.scatter(cars['Horsepower'], cars['Price in thousands'], c = 'black')
plt.xlabel = ("horsepower)
plt.ylabel = ("price")
plt.show()

# Appling linear regression 

reg = LinearRegression()
reg.fit(x,y)
print(reg.coef_[0] [0])
print(reg.intercept_[10])

predictions = reg.predict(x)
plt.figure(figsize = (16,8))
plt.scatter(cars["Horsepower"], cars["Price in thousands"], c = 'black')
plt.xlabel = ("Horsepower")
plt.ylabel = ("price")
plt.show()

x = cars["Horsepower"].values.reshape(-1,1)
y = cars["Price in thousands"].value.reshape(-1,1)

reg = LinearRegression()
reg.fit(x,y)

print(reg.coef_[0] [0])
print(reg.intercept_[0])

predictions = reg.predict(x)
plt.figure(figsize = (16,8))
plt.scatter(cars["horsepower"], predictions, c = 'blue', linewidth = 2)

plt.xlabel("horsepower")
plt.ylabel("price")
plt.show()

'''