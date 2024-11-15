import numpy as np
from sklearn.neighbors import KNeighborsRegressor

def main():
    # ask for n (number of points) and validate
    n = int(input("Enter N (positive integer for number of points): "))
    if n <= 0:
        print("N must be a positive integer.")
        return
    
    # ask for k (number of neighbors) and validate
    k = int(input("Enter k (positive integer for number of neighbors): "))
    if k <= 0:
        print("k must be a positive integer.")
        return
    
    # initialize arrays to store the x and y coordinates
    x_values = np.zeros(n)
    y_values = np.zeros(n)
    
    # read n (x, y) points
    for i in range(n):
        x = float(input(f"Enter x value for point {i + 1}: "))
        y = float(input(f"Enter y value for point {i + 1}: "))
        x_values[i] = x
        y_values[i] = y
    
    # calculate and print the variance of y values
    variance_y = np.var(y_values)
    print(f"Variance of labels (y values): {variance_y}")
    
    # ask for x (the input to predict y for)
    x_input = float(input("Enter X (value to predict Y): "))
    
    # perform k-nn regression if k <= n
    if k <= n:
        # reshape data for scikit-learn
        x_values = x_values.reshape(-1, 1)
        model = KNeighborsRegressor(n_neighbors=k)
        
        # fit the model on the provided points
        model.fit(x_values, y_values)
        
        # predict the value of y for the input x
        y_output = model.predict(np.array([[x_input]]))[0]
        print(f"Predicted Y for X={x_input} with k={k} is: {y_output}")
    else:
        print("Error! k cannot be greater than N.")
        
if __name__ == "__main__":
    main()
