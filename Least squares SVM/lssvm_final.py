import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


def load_data(file_name):
    df = pd.read_csv(file_name, header=None)
    y = None

    y = df[[0]]
    x = df.drop([0], axis=1)

    def translate(v):
        return 1 if v == 3 else -1

    y = y.applymap(translate)

    #   translate to np.array
    x = np.array(x)
    y = np.array(y)

    #   insert bias
    x = np.insert(x, 0, values=1, axis=1)
    y = y.T[0]

    return x, y

class LSSVM:
    def __init__(self, kernel='linear', C=1.0, gamma=1.0, d=1.0):
        self.kernel_type = 'poly'
        self.C = C
        self.gamma = gamma
        self.d = d

    def kernel(self, x1, x2):
        if x1.ndim == 1:
            x1 = np.array([x1]).T

        if x2.ndim == 1:
            x2 = np.array([x2]).T

        m1 = x1.shape[0]
        m2 = x2.shape[0]
        k = np.zeros((m1, m2))

        type = self.kernel_type

        if type == 'linear':
            k = np.dot(x1, x2.T)
        elif type == 'poly':
            k = np.power(np.dot(x1, x2.T) + 1, self.d)
        elif type == 'rbf':
            k = np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

        return k


    def fit(self, X_train, y_train):
        N = X_train.shape[0]
        K = self.kernel(X_train, X_train)

        # Compute omega
        omega = np.zeros((N, N))
        for k in range(K.shape[0]):
            for l in range(K.shape[1]):
                omega[k, l] = y_train[k] * y_train[l] * K[k, l]

        # Matrix A
        I = np.eye(omega.shape[0])
        ZZCI = omega + self.C ** -1 * I

        A11 = np.zeros((1, 1))
        y_train = y_train.reshape(len(y_train), 1)
        A1 = np.hstack((A11, -y_train.T))  # Row 1
        A2 = np.hstack((y_train, ZZCI))  # Row 2
        A = np.vstack((A1, A2))

        # vector b
        b = np.vstack((np.zeros((1, 1)), np.ones((N, 1))))

        # Solve the linear equation Ax = b
        x = np.linalg.solve(A, b)

        self.bias = x[0]
        self.alpha = x[1:len(x)]

    def predict(self, X_test, X_train, y_train):
        self.alpha = self.alpha.flatten()
        y_hat = np.sign(np.dot(np.multiply(self.alpha, y_train), self.kernel(X_train, X_test)) + self.bias)

        return y_hat



X_train, y_train = load_data("pa2_train.csv")
X_test, y_test = load_data("pa2_valid.csv")
d_list = [i for i in range(1,6)]
# print(X_test.shape)
# print(y_test.shape)

test_acc = []

for i in range(len(d_list)):
    clf = LSSVM(kernel='poly', C=1.0, gamma=0.01, d=d_list[i])
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test, X_train, y_train)
    accuracy = (y_predict == y_test).mean()
    test_acc.append(accuracy)

with open('validation_accuracy.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerow(test_acc)

plt.plot(d_list, test_acc, marker='o', label="Test Accuracy")
plt.xlabel('polynomial kernel degree (d)')
plt.ylabel('Accuracy')
plt.title('Test Accuracies')
plt.grid()
plt.legend()
plt.show()

print("Test Accuracy:", test_acc)
