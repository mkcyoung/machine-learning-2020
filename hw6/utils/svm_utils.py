import numpy as np
from copy import copy
import pandas as pd
import math
import random
import matplotlib.pyplot as plt

class Data:
  def __init__(self,file_path=None):
    if file_path != None:
      self.raw_data,\
      self.y,\
      self.X,\
      self.num_examples,\
      self.num_features = self.load_data_from_path(file_path)

  def load_data_from_path(self,file_path):
    data = np.loadtxt(file_path, delimiter = ",")
    labels = data[:,0]
    instances = data[:,1:]

    # Add a 1 to the end of each instance
    bias = np.ones((data.shape[0],1))
    instances = np.append(instances,bias,axis=1)

    num_examples = data.shape[0]
    num_features = instances.shape[1]
    return data,labels,instances,num_examples,num_features

  def load_data(self,raw_data):
    self.raw_data = raw_data
    self.y = raw_data[:,0]
    instances = raw_data[:,1:]
    # Add a 1 to the end of each instance
    bias = np.ones((raw_data.shape[0],1))
    self.X = np.append(instances,bias,axis=1)
    self.num_examples = raw_data.shape[0]
    self.num_features = self.X.shape[1]

  def add_bias_to_features(self):
    # Add a 1 to the end of each instance
    bias = np.ones((self.num_examples,1))
    self.X = np.append(self.X,bias,axis=1)

  def add_data(self,data):
    # takes as input another data object and adds that data to this object
    self.raw_data = np.vstack((self.raw_data,data.raw_data))
    self.X = np.vstack((self.X,data.X))
    self.y = np.hstack((self.y,data.y))
    self.num_examples += data.num_examples

  # returns shuffled labels and instances
  def shuffle_data(self):
    shuffled_raw_data = np.copy(self.raw_data)
    np.random.shuffle(shuffled_raw_data)
    shuffled_labels = shuffled_raw_data[:,0]
    shuffled_instances = shuffled_raw_data[:,1:]
    # add in bias
    bias = np.ones((shuffled_raw_data.shape[0],1))
    shuffled_instances = np.append(shuffled_instances,bias,axis=1)

    return shuffled_instances,shuffled_labels

def get_majority_baseline(data):
  labels,counts = np.unique(data.y,return_counts=True)
  print("labels: ",labels,"counts: ",counts)
  max_index = np.argmax(counts)
  max_label = labels[max_index]
  majority_baseline = counts[max_index] / data.num_examples
  return majority_baseline,max_label

# SVM Class

class SVM:
  def __init__(self):
    self.W = None
    self.Weights = {} # init empty dict of Weights, add to this for each epoch
    self.accuracies = {} # init empty dict of accuracies, which I store at end of each epoch
    self.loss = {} # dictionary contatining loss at each step
    self.num_updates = 0 # records number of updates made 

  def initialize_weights(self,num_features):
    self.W = np.array([np.random.uniform(-0.01,0.01) for _ in range(num_features)])
    # self.W = np.zeros((num_features)) # init to zeros
  
  def train(self,data,epochs=1,learning_rate=1,reg_strength=1):
    C = reg_strength
    epochs = epochs
    N = data.num_examples
    D = data.num_features
    # print("N:",N,"D (including b):",D)
    # initialize weights
    self.initialize_weights(D)
    
    for t in range(epochs):
      lr = learning_rate / (1 + t) # we use a decaying learning rate
      # shuffle data
      X,y = data.shuffle_data()
      # loop over each example in the training set
      for i in range(N):
        v = y[i]*(self.W.T.dot(X[i]))
        # print(v)
        if v <= 1.0: 
          self.W = (1.0-lr)*self.W + (lr*C*y[i])*X[i]
        else:
          self.W = (1.0-lr)*self.W
      # store this iteration of weights 
      self.Weights[t] = self.W
      # store the accuracy of these weights
      self.accuracies[t] = self.get_accuracy_own_weights(data,self.W)
      # Compute and store the loss 
      self.loss[t] = self.compute_loss(data,self.W,C)


  # Helper methods for predicting and accuracy
  def get_best_weights_and_bias(self):
    # print(self.accuracies.items())
    best_epoch = max(self.accuracies,key=self.accuracies.get)
    # print("best epoch: ",best_epoch)
    return self.Weights[best_epoch],best_epoch

  def predict(self,data):
    predictions = np.sign(data.dot(self.W))
    return predictions

  def get_predict_accuracy(self,data):
    predictions = self.predict(data.X)
    equal = np.equal(predictions,data.y)
    return np.sum(equal)/data.num_examples

  def get_accuracy_own_weights(self,data,W):
    predictions = np.sign(data.X.dot(W)) # Should the prediction have a margin? No, I don't think so
    equal = np.equal(predictions,data.y)
    return np.sum(equal)/data.num_examples

  def compute_loss(self,data,W,C):
    # "Loss" of the entire dataset
    X = data.X
    y = data.y
    loss = 0.5*(W.T.dot(W)) 
    a = 1 - y*W.dot(X.T)
    a[a<0] = 0
    return loss + C*np.sum(a)



# plot learning curve
def plot_learning(x,y,title,x_label,y_label):
  # Let's plot
  plt.style.use('default')
  plt.rcParams['font.family'] = 'Avenir'
  plt.figure(figsize = (6,4.5))
  # My PCA
  plt.plot(x,y)
  plt.title(title,fontsize=15)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  [i.set_linewidth(0.4) for i in plt.gca().spines.values()]
  plt.show()







