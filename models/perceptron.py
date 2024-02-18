import numpy as np
import time


def activation_func(z,labels):
    if z > 0:
        return float(labels[0])
    else:
        return float(labels[1])


def predict(X, labels,w):
    m, n = X.shape
    predicted_labels = []
    for idx, x_i in enumerate(X):
        # Insering 1 for bias, X0 = 1.
        x_i = np.insert(x_i, 0, 1).reshape(-1, 1)
        # Calculating prediction/hypothesis.
        y_hat = activation_func(np.dot(x_i.T, w),labels=labels)
        predicted_labels.append(y_hat)
    return predicted_labels


def perceptron(X, y,labels,lr,epochs,init="zeros"):

    # features of X
    m, n = X.shape
    # initialize weights
    if init == "zeros":
        w = np.zeros((n + 1, 1))
    elif init == "uniform":
        w = np.random.uniform(-1, 1, size=(n + 1, 1))
    elif init == "random":
        w = np.random.random(size=(n + 1, 1)) * 2 - 1  # Random values between -1 and 1
    else:
        raise ValueError("Invalid initialization method. Please choose 'zeros', 'uniform', or 'random'.")
    # empty list to store misclassified at every iteration
    n_miss_list = []

    # Training
    for epoch in range(epochs):
        # log training updates
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # variable to store #misclassified
        n_miss = 0

        for idx, x_i in enumerate(X):
            # insert 1 for bias
            x_i = np.insert(x_i, 0, 1).reshape(-1, 1)

            # calculating prediction
            y_hat = np.squeeze(activation_func(np.dot(x_i.T, w),labels=labels))
            error = y[idx] - y_hat
            # Updating if the example is misclassified
            if error != 0:
                # incrementing by 1
                n_miss = n_miss + 1
                w = w + lr * error * x_i

        n_miss_list.append(n_miss)
        print("number of misclassified examples: ", n_miss)
        print("Time taken: %.2fs" % (time.time() - start_time))

    return w, n_miss_list

