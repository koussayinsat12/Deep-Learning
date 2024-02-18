import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def plot_decision_boundary(X, w):
    x1 = [min(X[:, 0]), max(X[:, 0])]
    m = -w[1] / w[2]
    c = -w[0] / w[2]
    x2 = m * x1 + c

    plt.plot(x1, x2, 'g-', label="Decision Boundary")  # Changed x2 to x1 for the plot
    plt.legend()  # Moved the legend to this function
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")






def generate_data(mu1, mu2, variance1, variance2, test_size):
    mu1 = mu1
    mu2 = mu2
    sigma1 = np.sqrt(variance1)  # Écart-type pour la classe -1
    sigma2 = np.sqrt(variance2)  # Écart-type pour la classe 0
    num_samples_per_class = 125

    class_minus1 = np.random.normal(loc=mu1, scale=sigma1, size=(num_samples_per_class, 2))
    class_0 = np.random.normal(loc=mu2, scale=sigma2, size=(num_samples_per_class, 2))

    labels_minus1 = np.full(num_samples_per_class, -1)
    labels_0 = np.full(num_samples_per_class, 0)

    data = np.vstack((class_minus1, class_0))
    labels = np.concatenate((labels_minus1, labels_0))

    indices = np.arange(len(data))
    np.random.shuffle(indices)
    shuffled_data = data[indices]
    shuffled_labels = labels[indices]

    train_data, test_data, train_labels, test_labels = train_test_split(shuffled_data, shuffled_labels,
                                                                        test_size=test_size, random_state=42)
    return {"train_data": train_data, "test_data": test_data, "train_labels": train_labels, "test_labels": test_labels}
