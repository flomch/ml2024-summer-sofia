import numpy as np

def knn_regression(N, k, points, X):
    # Check if k is greater than N
    if k > N:
        return "Error: k should be less than or equal to the number of points (N)."
    
    # Convert list of points to a Numpy array for efficient processing
    points_array = np.array(points)
    x_values = points_array[:, 0]  # Extract all x values
    y_values = points_array[:, 1]  # Extract all y values

    # Calculate absolute distances between each x value and the input X
    distances = np.abs(x_values - X)

    # Find indices of the k nearest neighbors
    nearest_indices = np.argsort(distances)[:k]
    nearest_y_values = y_values[nearest_indices]

    # Calculate the mean of the nearest y values as the predicted Y
    predicted_y = np.mean(nearest_y_values)
    return predicted_y

# Main function to handle sequential CLI inputs and output the result
def main():
    # Step 1: Read N
    while True:
        try:
            N = int(input("Enter the number of points N (positive integer): "))
            if N <= 0:
                raise ValueError("N must be a positive integer.")
            break
        except ValueError as e:
            print("Invalid input for N:", e)

    # Step 2: Read k
    while True:
        try:
            k = int(input("Enter the value of k (positive integer): "))
            if k <= 0:
                raise ValueError("k must be a positive integer.")
            break
        except ValueError as e:
            print("Invalid input for k:", e)

    if k > N:
        print('k must be less than or equal to N.')

    # Step 3: Read N (x, y) points
    points = []
    print(f"Enter {N} (x, y) points one by one:")
    for i in range(N):
        while True:
            try:
                x = float(input(f"Enter x value for point {i + 1}: "))
                y = float(input(f"Enter y value for point {i + 1}: "))
                points.append((x, y))
                break
            except ValueError:
                print("Invalid input. x and y values must be real numbers.")

    # Step 4: Read the X value for prediction
    while True:
        try:
            X = float(input("Enter the X value for prediction: "))
            break
        except ValueError:
            print("Invalid input. X must be a real number.")

    # Step 5: Perform k-NN regression and display the result
    result = knn_regression(N, k, points, X)
    print("Predicted Y value:", result)

if __name__ == "__main__":
    main()
