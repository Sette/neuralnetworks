
import numpy as np

e = np.array([[1,-1,-1,-1],[-1,-1,-1,1]])
w = np.transpose(e* 1/2)

N = e.shape[1]
M = e.shape[0]


b = [N/2 for i in range(M)]

mx = np.array([[1,1,-1,-1],
     [1,-1,-1,-1],
     [-1,-1,-1,1],
     [-1,-1,1,1]])


y = []
for x in mx:
    test = []
    for j in range(M):
        sum = 0
        for i in range(len(x)):
            sum += x[i] * w[i][j]
        sum += b[j]
        test.append(sum)
    y.append(test)

#--------------------------------MaxNet ---------------------------- #

a = []
geral = []
E = 1 / (M-1)
for j in range(N):
    a = []
    for i in range(2):

        if (i == 0):
            k = 1
        else:
            i = 1
            k = 0

        x = y[j][i] - E * y[j][k]
        if (x > 0):
            a.append(x)
        else:
            a.append(0)

    geral.append(a)

print(geral)
