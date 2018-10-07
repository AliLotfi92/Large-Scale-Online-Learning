
### Author: Ali Lotfi Rezaabad ###



import numpy as np
import random
import time
import matplotlib.pyplot as plt

####Main Data####
Variable_Size = 20



def Data_Generator (Data_Size):
    mu = np.ones((Variable_Size,1))
    X_1 = np.random.normal(mu, mu, size=[Variable_Size, Data_Size])
    Y_1 = np.ones((Data_Size, 1))

    mu_1 = np.transpose([np.append([-1], -mu[1:])])
    X_2 = np.random.normal(mu_1, mu, size=[Variable_Size, Data_Size])
    Y_2 = -np.ones((Data_Size, 1))


    X = []
    Y = []
    for i in range(Data_Size):
        A = np.random.randint(2)

        if A:
            X.append([X_1[:, i]])
            Y.append([Y_1[i]])

        else:
            X.append([X_2[:, i]])
            Y.append([Y_2[i]])

    X = np.asarray(X)
    Y = np.asarray(Y)

    X = np.reshape(X, (Data_Size, Variable_Size))
    Y = np.reshape(Y, (Data_Size, 1))
    return X, Y

def Gradient(theta, x, y):
    y = np.asarray([y]).T
    f = np.tanh((2 / 3) * np.dot(x.T, theta))
    return -1 * 1.7159 * (2 / 3) * x * (1.5 * y - 1.7159 * f) * (1 - f ** 2)

def Hessian(theta, x):
    f = np.tanh((2 / 3) * np.dot(x.T, theta))**2
    return (1.7159 * (2 / 3)*(1- f ))** 2 * np.dot(x, x.T)

def error(theta, x, y):
    y = np.asarray([y]).T
    f = 1.7159*np.tanh((2 / 3) * np.dot(x.T, theta))
    return (1.5*y - f)**2

def Batch(X, Y, Data_Size):

    theta = np.random.uniform(-0.1, 0.1, size=(Variable_Size, 1))
    norm = 1e6
    while norm >= (0.01/Data_Size):
        g = 0
        h = 0
        e = 0
        for j in range(Data_Size):
            Xa = np.reshape(np.asarray(X[j, :]), (Variable_Size, 1))
            g = g + Gradient(theta, Xa, Y[j])
            h = h + Hessian(theta, Xa)
            e = e + error(theta, Xa, Y[j])/Data_Size

        hinv = np.linalg.inv(h)
        theta = theta - np.matmul(hinv, g)
        norm = np.linalg.norm((np.matmul(hinv, g)))
        #print(norm)
    return theta



def online(X, Y, Data_Size):
    phi = np.identity(Variable_Size)
    theta = np.random.uniform(-0.1, 0.1, size=(Variable_Size, 1))

    for i in range(Data_Size):
        Xa = np.reshape(np.asarray(X[i, :]), (Variable_Size, 1))
        tau = max(20, i - 40)
        a = 1 - (2 / tau)
        AA = np.tanh(0.66 * np.dot(Xa.T, theta))

        f = 1.71 * 0.66 * (1 - (AA ** 2))

        b = (2 / tau) * f ** 2
        Au = np.matmul(phi, Xa)

        uTAu = np.dot(Xa.T, Au)
        AuAuT = np.matmul(Au, Au.T)
        g = Gradient(theta, Xa, Y[i])
        phi = (1 / (a)) * (phi - (AuAuT) / ((a / b) + uTAu))

        theta = theta - (1 / tau) * np.matmul(phi, g)

    return theta


def Testerror(theta, XTest, YTest):
    e = 0
    for j in range(Data_Size_test):
        x = np.reshape(np.asarray(XTest[j, :]), (Variable_Size, 1))
        y = np.asarray([YTest[j]]).T
        f = 1.7159*np.tanh((2 / 3) * np.dot(x.T, theta))
        e += (1.5*y - f)**2/Data_Size_test

    return e




Data_Size_test = np.int_(1e5)
X_test, Y_test = Data_Generator(Data_Size_test)

# Optimal_theta_test = Batch(X_test, Y_test, Data_Size_test)
Optimal_theta_test = np.array([[0.11728777],
 [0.1180399],
 [0.11867964],
 [0.11414399],
 [0.11343874],
 [0.11880157],
 [0.12226298],
 [0.11567674],
 [0.11457017],
 [0.11745554],
 [0.11836563],
 [0.11642935],
 [0.11803515],
 [0.11609518],
 [0.1158847 ],
 [0.11768414],
 [0.12223212],
 [0.11876327],
 [0.11679625],
 [0.11327589]])

Iterations =6
Data_Size_Permutated = [1000, 2000, 5000]
Data_Size_Length = len(Data_Size_Permutated)

theta_mat_batch = np.zeros((Data_Size_Length, Iterations))
error_mat_batch = np.zeros((Data_Size_Length, Iterations))

theta_mat_online = np.zeros((Data_Size_Length, Iterations))
error_mat_online = np.zeros((Data_Size_Length, Iterations))


Data_Size_train = np.int_(1e5)
X, Y = Data_Generator(Data_Size_train)

cnt = 0
for i in Data_Size_Permutated:
     for j in range(Iterations):
         print('Data Size:', i, 'Iteration:', j)
         indx = random.sample(range(Data_Size_train), i)

         X_train, Y_train = X[indx], Y[indx]

         BatchTheta= Batch(X_train, Y_train, i)
         theta_mat_batch[cnt, j] = np.linalg.norm(BatchTheta - Optimal_theta_test)**2
         error_mat_batch[cnt, j] = Testerror(BatchTheta , X_test, Y_test)

         OnlineTheta = online(X_train, Y_train, i)
         theta_mat_online[cnt, j] = np.linalg.norm(OnlineTheta - Optimal_theta_test)**2
         error_mat_online[cnt, j] = Testerror(OnlineTheta,X_test,Y_test)



     cnt +=1


plt.figure(0)
meanBatch = np.mean(theta_mat_batch, axis=1)
errorBatch = np.std(theta_mat_batch, axis = 1)
plt.errorbar(Data_Size_Permutated, meanBatch, yerr=errorBatch, fmt='-o', color='r')



meanOnline = np.mean(theta_mat_online, axis=1)
errorOnline = np.std(theta_mat_online, axis = 1)
plt.errorbar(Data_Size_Permutated, meanOnline, yerr=errorOnline, fmt='-s', color='b')
plt.ylabel('Theta Error')
plt.xlabel('Training Sample Size')
plt.title("Fig1: avg error vs. training size")
plt.show()



meanBatch = np.mean(error_mat_batch, axis=1)
errorBatch = np.std(error_mat_batch, axis = 1)
plt.errorbar(Data_Size_Permutated, meanBatch, yerr=errorBatch, fmt='-o', color='r')


meanOnline = np.mean(error_mat_online, axis=1)
errorOnline = np.std(error_mat_online, axis = 1)
plt.errorbar(Data_Size_Permutated, meanOnline, yerr=errorOnline, fmt='-s', color='b')
plt.ylabel('MSE')
plt.xlabel('Training Sample Size')
plt.title("Fig2: MSE vs. training size")

plt.show()

