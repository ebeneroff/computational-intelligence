# Evan Beneroff EE509 Computational Intelligence
# single perceptron NAND gate
import numpy as np

# hard lim function
def hardlim(a):
   if(a >= 0):
      return 1
   return 0

# adaptation of weight vectors
def adapt(w, n, d, y, x):
   return w + n*(d - y)*x

# compute of actual response
def compute(w, x):
   return np.dot(w.T, x)

# input data set
# format is [x1, x2, x3, 1]
X = np.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1],
               [0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 1, 1], 
               [1, 1, 0, 1], [1, 1, 1, 1]])

# output data set
D = np.array([1, 1, 1, 1, 1, 1, 1, 0])

# starting weights
# format is [w1, w2, w3, bias]
W1 = np.array([0, 0, 0, 0])

# train for p iterations of n training sets
for p in range(50):
   for n in range(8):
      # view weights as they are adapted
      print(W1)

      a = compute(W1, X[n])
      #if(hardlim(a) != D[n]):
      W1 = adapt(W1, .25, D[n], a, X[n])

# output of weights
print("Output After Training:")
print(W1)

# some user i/o
while(1):
   x1 = input("enter x1 input:")
   x2 = input("enter x2 input:")
   x3 = input("enter x3 input:")
   g = np.array([float(x1), float(x2), float(x3), 1])
   print(compute(W1, g))
