import numpy as np
from sklearn.metrics import precision_score, recall_score

def main():
    # ask for n (number of points) and validate
    n = int(input("Enter N (positive integer for number of points): "))
    if n <= 0:
        print("N must be a positive integer.")
        return
    
    # initialize arrays to store the x and y coordinates
    y = np.zeros(n)
    pred = np.zeros(n)
    
    # read n (x, y) points
    for i in range(n):
        y1 = float(input(f"Enter y value for point {i + 1}: "))
        y2 = float(input(f"Enter pred value for point {i + 1}: "))
        y[i] = y1
        pred[i] = y2
    
    precision = precision_score(y, pred, average='binary')
    recall = recall_score(y, pred, average='binary')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}')

if __name__ == "__main__":
    main()
