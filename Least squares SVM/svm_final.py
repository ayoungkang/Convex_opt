import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import cvxopt
#np.random.seed(1)

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

class SVM:
    def __init__(self, kernel='linear', C=1.0, gamma=1.0, d=1.0):
        self.kernel_type = 'poly'
        self.C = C
        self.gamma = gamma
        self.d = d
        self.SV = None

    def kernel(self, x1, x2):
        m1 = x1.shape[0]
        m2 = x2.shape[0]
        k = np.zeros((m1, m2))

        type = self.kernel_type

        if type == 'linear':
            k = np.dot(x1, x2.T)
        elif type == 'poly':
            k = np.power(np.dot(x1, x2.T), self.d)
        elif type == 'rbf':
            k = np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

        return k


    def fit(self, X_train, y_train):

        N, m = X_train.shape
        C = self.C
        # Compute the Gram matrix
        k = self.kernel(X_train, X_train)
        # Construct P, q, A, b, G, h matrices for CVXOPT
        P = cvxopt.matrix(np.outer(y_train, y_train) * k, tc='d')
        q = cvxopt.matrix(np.ones(N) * -1, tc='d')
        A = cvxopt.matrix(y_train, (1, N), tc='d')
        b = cvxopt.matrix(0.0, tc='d')
        # hard-margin SVM
        #if C is None or C == 0:
        G = cvxopt.matrix(np.diag(np.ones(N) * -1), tc='d')
        h = cvxopt.matrix(np.zeros(N), tc='d')
        # soft-margin SVM
        # G = cvxopt.matrix(np.vstack((np.diag(np.ones(N) * -1), np.eye(N))), tc='d')
        # h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * C)), tc='d')

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        a = np.ravel(solution['x'])

        sv = a > 0
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X_train[sv]
        self.sv_y = y_train[sv]

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * k[ind[n], sv])
        self.b /= len(self.a)



    def predict(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                #print("X[i]", np.shape(X[i]))
                #print("sv", np.shape(sv))
                s += a * sv_y * self.kernel(X[i], sv)

            y_predict[i] = s

        return np.sign(y_predict + self.b)



X_train, y_train = load_data("pa2_train.csv")
X_test, y_test = load_data("pa2_valid.csv")

d_list = [i for i in range(1,6)]
#d_list = [1]
# print(X_test.shape)
# print(y_test.shape)
test_acc = []

for i in range(len(d_list)):
    clf = SVM(kernel='poly', C=0, gamma=0.01, d=d_list[i])
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    accuracy = (y_predict == y_test).mean()
    test_acc.append(accuracy)


with open('svm_test_accuracy.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerow(test_acc)

plt.plot(d_list, test_acc, marker='o', label="Test Accuracy")
plt.xlabel('polynomial kernel degree (d)')
plt.ylabel('Accuracy')
plt.title('Test accuracies')
plt.grid()
plt.legend()
plt.show()

