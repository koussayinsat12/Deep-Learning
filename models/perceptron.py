import numpy as np
import time



def activation_func(z):
    if z > 0:
        return z
    else:
        return 0


def perceptron(X, y, lr, epochs):
    # features of X
    m, n = X.shape

    # initialize weights
    w = np.zeros((n + 1, 1))

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
            y_hat = np.squeeze(activation_func(np.dot(x_i.T, w)))
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



