import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def main():
    # input N 
    n = int(input("Enter N (positive integer for the training set size): "))
    if n <= 0:
        print("N must be a positive integer.")
        return

    # initialize arrays for training data
    x_train = np.zeros(n)
    y_train = np.zeros(n, dtype=int)

    # read N training points
    print("Enter training data (x, y) pairs:")
    for i in range(n):
        x_train[i] = float(input(f"Enter x value for training point {i + 1}: "))
        y_train[i] = int(input(f"Enter y value for training point {i + 1} (non-negative integer): "))
        if y_train[i] < 0:
            print("Error! y must be a non-negative integer.")
            return

    # input M
    m = int(input("Enter M (positive integer for the test set size): "))
    if m <= 0:
        print("M must be a positive integer.")
        return

    # initialize arrays for test data
    x_test = np.zeros(m)
    y_test = np.zeros(m, dtype=int)

    # read M test points
    print("Enter test data (x, y) pairs:")
    for i in range(m):
        x_test[i] = float(input(f"Enter x value for test point {i + 1}: "))
        y_test[i] = int(input(f"Enter y value for test point {i + 1} (non-negative integer): "))
        if y_test[i] < 0:
            print("Error! y must be a non-negative integer.")
            return

    # reshape data for sklearn
    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    # hyperparam search for k (1 <= k <= 10)
    best_k = 0
    best_accuracy = 0.0

    for k in range(1, 11):
        # knn classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)

        # test the classifier on the test set
        y_pred = knn.predict(x_test)
        acc = accuracy_score(y_test, y_pred)

        # update the best k and accuracy if current is better
        if acc > best_accuracy:
            best_k = k
            best_accuracy = acc

    # output the best k and corresponding test accuracy
    print(f"Best k: {best_k}, Test Accuracy: {best_accuracy:.2f}")


if __name__ == "__main__":
    main()
