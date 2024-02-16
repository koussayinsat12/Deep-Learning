import matplotlib.pyplot as plt


def plot_decision_boundary(X, w):
    x1 = [min(X[:, 0]), max(X[:, 0])]
    m = -w[1]/w[2]
    c = -w[0]/w[2]
    x2 = m * x1 + c

    # Plotting
    plt.scatter(X[:, 0],X[:,1])
    plt.plot(x1,x2, 'b-')
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
