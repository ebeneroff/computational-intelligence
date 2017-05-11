# Evan Beneroff - EE509 Computational Intelligence - Homework 3
# Feedforward neural network with one hidden layer (5 neurons) and
# one output layer, this will train the net to realize the function
# f(x) = x * e^(-x)
import numpy as np
import random
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

# hidden layer activation function
def tangsig(x):
   return np.tanh(x);

# output neuron activation function
def purelin(x):
   if(x > 0):
      return 1
   return 0;

# desired output function f(x) = x * e^(-x)
def nonlin_func(x):
   return x * np.exp(-x)

# neural net function to call
def neural_net():
   # 1 input, 5 hidden w/ tanh(x) activation, 1 output w/ linear activation
   net = buildNetwork(1, 5, 1, hiddenclass=TanhLayer)
   dataset = SupervisedDataSet(1, 1)
   results = np.zeros((4, 50))
   
   # generate 50 evenly distributed inputs and get output before training
   count = 0
   for n in range(50):
      count += .1
      results[0, n] = count
      results[1, n] = net.activate([results[0, n]])

   # add 500 random samples
   for n in range(500):
      rand = np.random.rand() * 20
      # samples consist of random input and desired output
      dataset.addSample((rand), nonlin_func(rand))
   
   # train using backpropagation
   trainer = BackpropTrainer(net, dataset)
   
   # train for 100 epochs
   for p in range(100):
      trainer.train()
   
   # get output after training and desired output for comparison
   for n in range(50):
      results[2, n] = net.activate([results[0, n]])
      results[3, n] = nonlin_func(results[0, n])

   np.savetxt("data.txt", results, fmt='%10.5f', delimiter=",")

neural_net()
