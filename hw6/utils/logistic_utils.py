import numpy as np

class LOGREG():
  def __init__(self):
    self.W = None
    self.Weights = {} # init empty dict of Weights, add to this for each epoch
    self.accuracies = {} # init empty dict of accuracies, which I store at end of each epoch
    self.loss = {} # dictionary contatining loss at each step
    self.num_updates = 0 # records number of updates made 

  def initialize_weights(self,num_features):
    # self.W = np.array([np.random.uniform(-0.01,0.01) for _ in range(num_features)])
    self.W = np.zeros((num_features)) # init to zeros
  
  def train(self,data,epochs=1,learning_rate=1,reg_strength=1):
    C = reg_strength
    epochs = epochs
    N = data.num_examples
    D = data.num_features
    # print("N:",N,"D (including b):",D)
    # initialize weights
    self.initialize_weights(D)
    
    for t in range(epochs):
      lr = learning_rate #/ (1 + t) # we use a decaying learning rate
      # shuffle data - doing this instead of random sampling, essentially same 
      # thing but is easier to keep track of epochs this way
      X,y = data.shuffle_data()
      # loop over each example in the training set
      for i in range(N):
        # compute gradient
        z = -y[i]*self.W.T.dot(X[i])
        dW = (np.exp(z)/(1.0 + np.exp(z)))*(-y[i]*X[i]) + (2.0/C)*self.W
        # dW = (sigmoid(self.W.T.dot(X[i]))-y[i])*X[i]
        # dW = (1-sigmoid(z))*(-y[i]*X[i]) + (2.0/C)*self.W
        # print("grad",dW)
        # update weights by stepping along gradient
        # Maybe this is my issue......
        # print(dW.shape)
        self.W = self.W - lr*(dW) 
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
    z = -y*W.dot(X.T)
    loss = np.sum(np.log(1+np.exp(z))) + (1/C)*W.T.dot(W)
    # print(loss)
    return loss


# sigmoid function
def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))