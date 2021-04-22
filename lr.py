import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LR:
    def fit(self, X,y,lr=0.001,n_iter=1000):
        X = np.insert(X,0,1,axis=1)
        self.unique_y=np.unique(y)
        self.w = np.zeros((len(self.unique_y), X.shape[1]))
        y = self.one_hot(y)
        for i in range(n_iter):
            predictions = self.probabilities(X)
            error = predictions - y
            gradient = np.dot(error.T,X)
            self.w -= (lr * gradient)
        return self
    def probabilities(self, X):
        scores = np.dot(X, self.w.T)
        return self.softmax(scores)
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(-1, 1)
    def predict(self, X):
        X = np.insert(X , 0,1, axis = 1)
        argmax = np.argmax(self.probabilities(X), axis=1)
        for i, c in enumerate(argmax):
            argmax[i] = self.unique_y[c]
        return argmax
    def score(self, X, y):
        return np.mean(self.predict(X)==y)
    def one_hot(self,y):
        uy = list(np.unique(y))
        encoded = np.zeros((len(y), len(uy)))
        for i,c in enumerate(y):
            encoded[i][uy.index(c)] = 1
        return encoded

cleanup_nums = {"buying": {"low": 1, "med": 2, "high": 3, "vhigh": 4},
                "maint": {"low": 1, "med": 2, "high": 3, "vhigh": 4},
                "doors": {"2": 2, "3": 3, "4": 4, "5more": 5},
                "persons": {"2": 2, "4": 4, "more": 6},
                "lug_boot": {"small": 1, "med": 2, "big": 3},
                "safety": {"low": 1, "med": 2, "high": 3},
                "class": {"unacc": 1,"acc": 2, "good": 3, "vgood": 4}
               }

data = pd.read_csv("data/iris_X_train.csv").values
ans = pd.read_csv("data/iris_Y_train.csv").values.ravel()
datat = pd.read_csv("data/iris_X_test.csv").values
anst = pd.read_csv("data/iris_Y_test.csv").values.ravel()

cardata = pd.read_csv("data/car_X_train.csv").replace(cleanup_nums).values
carans = pd.read_csv("data/car_Y_train.csv").replace(cleanup_nums).values.ravel()
cardatat = pd.read_csv("data/car_X_test.csv").replace(cleanup_nums).values
caranst = pd.read_csv("data/car_Y_test.csv").replace(cleanup_nums).values.ravel()

variables = [0.0000001,0.0000005,0.000001,0.000005,0.00001,0.00005,0.0001,0.0005,0.001,0.005]

car_accuracy = []
iris_accuracy = []

for i, var in enumerate(variables):
    print("lr = ",var)
    lr = LR()
    lr.fit(data,ans,lr = var)
    a = lr.score(datat,anst)
    print("iris accuracy", a)
    iris_accuracy.append(a)

    lr2 = LR()
    lr2.fit(cardata,carans,n_iter=10000, lr = var)
    b = lr2.score(cardatat,caranst)
    print("car accuracy", b)
    car_accuracy.append(b)

fig = plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0))
ax1.plot(variables,car_accuracy,label='car')
ax1.plot(variables,iris_accuracy, label='iris')
ax1.set_xscale('log')
plt.legend()
plt.xlabel('Learning rate')
plt.ylabel('Accuracy')
plt.title('Accuracy under different learning rates')
plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.2)
plt.show()
