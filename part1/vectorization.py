import time

import numpy as np

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i] * x2[i]

toc = time.process_time()
print("dot = ", str(dot), "\n ------ Computation time = " + str(1000 * (toc - tic)) + " ms")

tic = time.process_time()
outer = np.zeros((len(x1), len(x2)))

for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i, j] = x1[i] * x2[j]

toc = time.process_time()
print("outer = ", str(outer), "\n ------ Computation time = " + str(1000 * (toc - tic)) + " ms")

tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i] * x2[i]

toc = time.process_time()
print("elementwise multiplication = ", str(mul), "\n ------ Computation time = " + str(1000 * (toc - tic)) + " ms")

W = np.random.rand(3, len(x1))
tic = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i, j] * x1[j]

toc = time.process_time()
print("gdot = ", str(gdot), "\n ------ Computation time = " + str(1000 * (toc - tic)) + " ms")

tic = time.process_time()
dot = np.dot(x1, x2)
toc = time.process_time()
print("dot = ", str(dot), "\n ------ Computation time = " + str(1000 * (toc - tic)) + " ms")

tic = time.process_time()
outer = np.outer(x1, x2)
toc = time.process_time()
print("outer = ", str(outer), "\n ------ Computation time = " + str(1000 * (toc - tic)) + " ms")

tic = time.process_time()
mul = np.multiply(x1, x2)
toc = time.process_time()
print("elementwise multiplication = ", str(mul), "\n ------ Computation time = " + str(1000 * (toc - tic)) + " ms")

tic = time.process_time()
dot = np.dot(W, x1)
toc = time.process_time()
print("gdot = ", str(gdot), "\n ------ Computation time = " + str(1000 * (toc - tic)) + " ms")
