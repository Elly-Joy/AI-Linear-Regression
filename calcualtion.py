import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load dataset
data = pd.read_csv("Nairobi_Office_Price_Ex.csv")
x = data['SIZE'].values
y = data['PRICEgi'].values

def mean_squared_error(y_true, y_prediction):
    return np.mean((y_true- y_prediction)**2)

def gradient_descent(x, y, m, c, learning_rate):
    n = len(y)
    y_prediction = m*x + c

    #calculate gradients
    d_m = (-2/n)*sum(x*(y-y_prediction))
    d_c = (-2/n)*sum(y-y_prediction)

    #update weights
    m -= learning_rate*d_m
    c -= learning_rate*d_c
    return m, c

#Set initial parameters
np.random.seed(42)
m, c = np.random.randn(2)
learning_rate = 0.01
epochs = 10

#Train model
for epoch in range(epochs):
   #predict the current values
    y_prediction = m*x + c
    #calculate the error
    error= mean_squared_error(y, y_prediction)
    print(f'Epoch: {epoch}, Error: {error}')
     # Update m and c using gradient descent
    m, c = gradient_descent(x, y, m, c, learning_rate)

    # Plotting
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, m * x + c, color='red', label='Line of Best Fit')
plt.xlabel('Office Size')
plt.ylabel('Office Price')
plt.legend()
plt.title('Linear Regression Line of Best Fit')
plt.show()


office_size = 100
predicted_price = m * office_size + c
print(f"Predicted Office Price for size 100 sq. ft: {predicted_price}")
