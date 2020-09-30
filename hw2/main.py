# Imports
import numpy as np
import math
import matplotlib.pyplot as plt

########################## Initializing ############################

# Initializing

np.random.seed(42)
TRAINING_PATH = 'data/csv-format/train.csv'
TESTING_PATH = 'data/csv-format/test.csv'
TEST_NAMES = np.loadtxt('data/raw-data/test_names.txt', delimiter = ",", dtype = str)
TRAIN_NAMES = np.loadtxt('data/raw-data/train_names.txt', delimiter = ",", dtype = str)

########################## Data Class #############################

# Defining Data class
# In hw1, I didn't like the way I loaded data,
# Will define new data class using csv and numpy

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
    num_examples = data.shape[0]
    num_features = instances.shape[1]
    return data,labels,instances,num_examples,num_features

  def load_data(self,raw_data):
    self.raw_data = raw_data
    self.y = raw_data[:,0]
    self.X = raw_data[:,1:]
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
    return shuffled_instances,shuffled_labels

####################### Most frequent element - Majority baseline #########################

def get_majority_baseline(data):
  labels,counts = np.unique(data.y,return_counts=True)
#   print("labels: ",labels,"counts: ",counts)
  max_index = np.argmax(counts)
  max_label = labels[max_index]
  majority_baseline = counts[max_index] / data.num_examples
  return majority_baseline,max_label



###########################  Perceptron class #########################################

class Perceptron:
  def __init__(self):
    self.W = None
    self.b = None
    self.W_a = None # averaged weights
    self.b_a = None # Averaged bias
    self.Weights = {} # init empty dict of Weights, add to this for each epoch
    self.accuracies = {} # init empty dict of accuracies, which I store at end of each epoch
    self.bias = {} # init emmpty dictionary that stores biases 
    self.num_updates = 0 # records number of updates made 

  def initialize_weights(self,num_features):
    self.W = np.array([np.random.uniform(-0.01,0.01) for _ in range(num_features)])
    self.W_a = np.array([np.random.uniform(-0.01,0.01) for _ in range(num_features)])

  def initialize_bias(self):
    self.b = np.random.uniform(-0.01,0.01)
    self.b_a = np.random.uniform(-0.01,0.01)

  # 1 - Simple Perceptron
  def train_simple(self,data,epochs,learning_rate):
    lr = learning_rate
    num_examples = data.num_examples
    num_features = data.num_features
    # Initialize my weights and bias
    self.initialize_weights(num_features)
    self.initialize_bias()

    # Begin epochs
    for epoch in range(epochs):
      # shuffle the data around 
      X,y = data.shuffle_data()
      # Iterate through examples, performing updates if criteria not met
      for i in range(num_examples):
        a = self.W.T.dot(X[i]) + self.b
        if y[i]*a < 0:
          self.W += lr*y[i]*X[i]
          self.b += lr*y[i]
          # iterate update
          self.num_updates += 1
      # store this iteration of weights 
      self.Weights[epoch] = self.W
      self.bias[epoch] = self.b
      # store the accuracy of these weights and biases
      self.accuracies[epoch] = self.get_accuracy_own_weights(data,self.W,self.b)

  # 2 - Decay Perceptron
  def train_decay(self,data,epochs,learning_rate):
    lr = learning_rate
    t = 0
    num_examples = data.num_examples
    num_features = data.num_features
    # Initialize my weights and bias
    self.initialize_weights(num_features)
    self.initialize_bias()

    # Begin epochs
    for epoch in range(epochs):
      # update learning rate
      lr = learning_rate / (1 + t)
      # shuffle the data around 
      X,y = data.shuffle_data()
      # Iterate through examples, performing updates if criteria not met
      for i in range(num_examples):
        a = self.W.T.dot(X[i]) + self.b
        if y[i]*a < 0:
          self.W += lr*y[i]*X[i]
          self.b += lr*y[i]
          # iterate update
          self.num_updates += 1
      # store this iteration of weights 
      self.Weights[epoch] = self.W
      self.bias[epoch] = self.b
      # store the accuracy of these weights and biases
      self.accuracies[epoch] = self.get_accuracy_own_weights(data,self.W,self.b)
      #increment t
      t += 1

  # 3 - Averaged Perceptron
  def train_averaged(self,data,epochs,learning_rate):
    lr = learning_rate
    num_examples = data.num_examples
    num_features = data.num_features
    # Initialize my weights and bias
    self.initialize_weights(num_features)
    self.initialize_bias()

    # Begin epochs
    for epoch in range(epochs):
      # shuffle the data around 
      X,y = data.shuffle_data()
      # Iterate through examples, performing updates if criteria not met
      for i in range(num_examples):
        a = self.W.T.dot(X[i]) + self.b
        if y[i]*a < 0:
          self.W += lr*y[i]*X[i]
          self.b += lr*y[i]
          # iterate update
          self.num_updates += 1
        # Add to the averaged weights and bias 
        self.W_a += self.W
        self.b_a += self.b
      # store this iteration of weights 
      self.Weights[epoch] = self.W_a
      self.bias[epoch] = self.b_a
      # store the accuracy of these weights and biases
      self.accuracies[epoch] = self.get_accuracy_own_weights(data,self.W_a,self.b_a)

  # 4 - Margin Perceptron
  def train_margin(self,data,epochs,learning_rate,margin):
    lr = learning_rate
    t = 0
    num_examples = data.num_examples
    num_features = data.num_features
    # Initialize my weights and bias
    self.initialize_weights(num_features)
    self.initialize_bias()

    # Begin epochs
    for epoch in range(epochs):
      # update learning rate
      lr = learning_rate / (1 + t)
      # shuffle the data around 
      X,y = data.shuffle_data()
      # Iterate through examples, performing updates if criteria not met
      for i in range(num_examples):
        a = self.W.T.dot(X[i]) + self.b
        if y[i]*a < margin:
          self.W += lr*y[i]*X[i]
          self.b += lr*y[i]
          # iterate update
          self.num_updates += 1
      # store this iteration of weights 
      self.Weights[epoch] = self.W
      self.bias[epoch] = self.b
      # store the accuracy of these weights and biases
      self.accuracies[epoch] = self.get_accuracy_own_weights(data,self.W,self.b)
      #increment t
      t += 1

  def get_best_weights_and_bias(self):
    # print(self.accuracies.items())
    best_epoch = max(self.accuracies,key=self.accuracies.get)
    # print("best epoch: ",best_epoch)
    return self.Weights[best_epoch],self.bias[best_epoch],best_epoch

  def predict(self,data):
    predictions = np.sign(data.dot(self.W) + self.b)
    return predictions

  def get_predict_accuracy(self,data):
    predictions = self.predict(data.X)
    equal = np.equal(predictions,data.y)
    return np.sum(equal)/data.num_examples

  def get_accuracy_own_weights(self,data,W,b):
    predictions = np.sign(data.X.dot(W) + b)
    equal = np.equal(predictions,data.y)
    return np.sum(equal)/data.num_examples
    


##############################  Plotting function ##############################
# plot learning curve
def plot_learning(x,y,title):
  # Let's plot
  plt.style.use('default')
  plt.rcParams['font.family'] = 'Avenir'
  plt.figure(figsize = (6,4.5))
  # My PCA
  plt.plot(x,y)
  plt.title(title,fontsize=15)
  plt.xlabel("epochs")
  plt.ylabel("training accuracy")
  [i.set_linewidth(0.4) for i in plt.gca().spines.values()]



############################ Cross validation functions #########################
# cross validation - make a function of the perceptron eventually maybe

def cross_validate(epochs,learning_rates,p_type,new_features=None):
  if new_features == None:
    fold_ids = ['fold1','fold2','fold3','fold4','fold5']
    path_names = ['data/csv-format/CVfolds/{}.csv'.format(i) for i in fold_ids]
    # folds = { index : Data(path_name) for index,path_name in enumerate(path_names) }
    # list of each fold 
    folds = [np.loadtxt(file_path, delimiter = ",") for file_path in path_names]
    
  else:
    folds = [ data[1].raw_data for data in new_features.items()]

  k = len(folds)
  # dictionaries storing accuracies corresponding to certain hyper parameter combinations
  mean_accuracies = {} 
  standard_deviations = {}
  
  for lr in learning_rates:
    accuracies = []
    # Need to concatenate 4 of the folds into one training set and leave out one as my test set
    for i in range(k):
      # Initialize new data objects
      val_data = Data()
      train_data = Data()
      folds_copy = list.copy(folds)
      # Set validation data
      val_data.load_data(np.array(folds_copy.pop(i)))
      # set training data
      train_data.load_data(np.concatenate(folds_copy,axis=0))

      # train on validation and training folds
      perceptron = Perceptron()
      if p_type == "simple":
        perceptron.train_simple(train_data,epochs,lr)

        weights,bias,best_epoch = perceptron.get_best_weights_and_bias()
        # weights = perceptron.W_a
        # bias = perceptron.b_a
        # train_accuracy = perceptron.get_accuracy_own_weights(train_data)
        val_accuracy = perceptron.get_accuracy_own_weights(val_data,weights,bias)
        accuracies.append(val_accuracy)

        # # just using the final set of weights and bias to test accuracy here
        # # train_accuracy = perceptron.get_predict_accuracy(train_data)
        # val_accuracy = perceptron.get_predict_accuracy(val_data)
        # accuracies.append(val_accuracy)
      if p_type == "decay":
        perceptron.train_decay(train_data,epochs,lr)

        weights,bias,best_epoch = perceptron.get_best_weights_and_bias()
        # weights = perceptron.W_a
        # bias = perceptron.b_a
        # train_accuracy = perceptron.get_accuracy_own_weights(train_data)
        val_accuracy = perceptron.get_accuracy_own_weights(val_data,weights,bias)
        accuracies.append(val_accuracy)

        # # just using the final set of weights and bias to test accuracy here
        # # train_accuracy = perceptron.get_predict_accuracy(train_data)
        # val_accuracy = perceptron.get_predict_accuracy(val_data)
        # accuracies.append(val_accuracy)

      if p_type == "averaged":
        perceptron.train_averaged(train_data,epochs,lr)
        # just using the final set of weights and bias to test accuracy here
        # Get best weights and bias 
        weights,bias,best_epoch = perceptron.get_best_weights_and_bias()
        # weights = perceptron.W_a
        # bias = perceptron.b_a
        # train_accuracy = perceptron.get_accuracy_own_weights(train_data)
        val_accuracy = perceptron.get_accuracy_own_weights(val_data,weights,bias)
        accuracies.append(val_accuracy)
    mean_accuracies[lr] = np.mean(accuracies)
    standard_deviations[lr] = np.std(accuracies)
  
#   print(mean_accuracies.items())
#   print(standard_deviations.items())
  best_lr = max(mean_accuracies,key=mean_accuracies.get)
  print("best lr: ",best_lr)
  print("cross-val accuracy: ",mean_accuracies[best_lr])
  return best_lr
    

def cross_validate_margins(epochs,learning_rates,margins):
  fold_ids = ['fold1','fold2','fold3','fold4','fold5']
  path_names = ['data/csv-format/CVfolds/{}.csv'.format(i) for i in fold_ids]
  # folds = { index : Data(path_name) for index,path_name in enumerate(path_names) }
  # list of each fold 
  folds = [np.loadtxt(file_path, delimiter = ",") for file_path in path_names]
  k = len(folds)

  # dictionaries storing accuracies corresponding to certain hyper parameter combinations
  mean_accuracies = {} 
  standard_deviations = {}
  
  for lr in learning_rates:
    accuracies = []
    for margin in margins:
      # Need to concatenate 4 of the folds into one training set and leave out one as my test set
      for i in range(k):
        # Initialize new data objects
        val_data = Data()
        train_data = Data()
        folds_copy = list.copy(folds)
        # Set validation data
        val_data.load_data(np.array(folds_copy.pop(i)))
        # set training data
        train_data.load_data(np.concatenate(folds_copy,axis=0))

        # train on validation and training folds
        perceptron = Perceptron()
        
        perceptron.train_margin(train_data,epochs,lr,margin)

        weights,bias,best_epoch = perceptron.get_best_weights_and_bias()
        # weights = perceptron.W_a
        # bias = perceptron.b_a
        # train_accuracy = perceptron.get_accuracy_own_weights(train_data)
        val_accuracy = perceptron.get_accuracy_own_weights(val_data,weights,bias)
        accuracies.append(val_accuracy)

        # # just using the final set of weights and bias to test accuracy here
        # # train_accuracy = perceptron.get_predict_accuracy(train_data)
        # val_accuracy = perceptron.get_predict_accuracy(val_data)
        # accuracies.append(val_accuracy)

      mean_accuracies[(lr,margin)] = np.mean(accuracies)
      standard_deviations[(lr,margin)] = np.std(accuracies)
  
#   print(mean_accuracies.items())
#   print(standard_deviations.items())
  best_vals = max(mean_accuracies,key=mean_accuracies.get)
  print("best lr: ",best_vals[0])
  print("best margin: ",best_vals[1])
  print("cross-val accuracy: ",mean_accuracies[best_vals])
  return best_vals



###################################  Feature transformation ################################
def feature_transformation(data):
  # create data object
  full_data = Data()
  full_features = []
#   print(data.num_examples)
  # outermost loop loops over example in dataset
  for i in range(data.num_examples):
    new_features = []
    # second loop loops over full feature vector
    for index,j in enumerate(data.X[i]):
      # third loop loops over truncated feature vector
      for k in data.X[i][index:]:
        new_features.append(j*k)
    full_features.append(np.array(new_features))
  X = np.vstack(full_features)
  # now, attach the labels to the beginning
  full_data.load_data(np.insert(X,0,data.y,axis=1))
  return full_data




############################### MAIN FUNCTION ###############################################
def main():
    
    # loading data
    training_data = Data(TRAINING_PATH)
    testing_data = Data(TESTING_PATH)

    ######## Reporting on majority baseline #############

    most_frequent_training_label = get_majority_baseline(training_data)
    most_frequent_testing_label = get_majority_baseline(testing_data)
    print("===================== Majority baselines =========================")
    print("training majority baseline: ",most_frequent_training_label)
    print("testing majority baseline: ",most_frequent_testing_label)

    print('\n')
    ############## 1. Simple perceptron #####################
    print("===================== 1. Simple Perceptron =========================")
    learning_rates = [1,0.1,0.01]
    epochs = 10
    best_lr = cross_validate(epochs,learning_rates,'simple')
    training_data = Data(TRAINING_PATH)
    test_data = Data(TESTING_PATH)
    epochs = 20
    learning_rate = best_lr
    simple_perceptron = Perceptron()
    simple_perceptron.train_simple(training_data,epochs,learning_rate)
    print("number of updates: ",simple_perceptron.num_updates)
    # Get the best weights and bias from this training
    W,b,best_epoch = simple_perceptron.get_best_weights_and_bias()
    # training set accuracy:
    print("best training set accuracy: ", simple_perceptron.accuracies[best_epoch] )
    # Use these weights and bias to evaluate on the test set
    test_accuracy = simple_perceptron.get_accuracy_own_weights(test_data,W,b)
    print("final test accuracy: ",test_accuracy)

    y = list(simple_perceptron.accuracies.values())
    x = [i for i in range(epochs)]
    title = '1 - simple perceptron learning curve'
    plot_learning(x,y,title)
    plt.show()
    print('\n')

    ####################### 2. Decaying the learning rate #######################
    print("================== 2. Decaying the learning rate ==============")
    # 2- Decaying the learning rate
    # clear old network
    simple_perceptron = None

    # decay perceptron cross validation
    learning_rates = [1,0.1,0.01]
    epochs = 10
    best_lr = cross_validate(epochs,learning_rates,'decay')

    training_data = Data(TRAINING_PATH)
    test_data = Data(TESTING_PATH)
    epochs = 20
    learning_rate = best_lr
    decay_perceptron = Perceptron()
    decay_perceptron.train_decay(training_data,epochs,learning_rate)
    print("number of updates: ",decay_perceptron.num_updates)
    # Get the best weights and bias from this training
    W,b,best_epoch = decay_perceptron.get_best_weights_and_bias()
    # training set accuracy:
    print("best training set accuracy: ", decay_perceptron.accuracies[best_epoch] )
    # Use these weights and bias to evaluate on the test set
    test_accuracy = decay_perceptron.get_accuracy_own_weights(test_data,W,b)
    print("final test accuracy: ",test_accuracy)

    y = list(decay_perceptron.accuracies.values())
    x = [i for i in range(epochs)]
    title = '2 - decay perceptron learning curve'
    plot_learning(x,y,title)
    plt.show()
    print('\n')

    ################# 3. Averaged Perceptron #########################
    print("============== 3. Averaged Perceptron ==============")
    # 3 - Averaged Perceptron
    # clear old network
    decay_perceptron = None

    # average perceptron cross validation

    learning_rates = [1,0.1,0.01]
    epochs = 10
    best_lr = cross_validate(epochs,learning_rates,'averaged')

    training_data = Data(TRAINING_PATH)
    test_data = Data(TESTING_PATH)
    epochs = 20
    learning_rate = best_lr
    average_perceptron = Perceptron()
    average_perceptron.train_averaged(training_data,epochs,learning_rate)
    print("number of updates: ",average_perceptron.num_updates)
    # Get the best weights and bias from this training
    W,b,best_epoch = average_perceptron.get_best_weights_and_bias()
    # training set accuracy:
    print("best training set accuracy: ", average_perceptron.accuracies[best_epoch] )
    # Use these weights and bias to evaluate on the test set
    test_accuracy = average_perceptron.get_accuracy_own_weights(test_data,W,b)
    print("final test accuracy: ",test_accuracy)

    y = list(average_perceptron.accuracies.values())
    x = [i for i in range(epochs)]
    title = '3 - averaged perceptron learning curve'
    plot_learning(x,y,title)
    plt.show()
    print('\n')


    ################# 4. Margin Perceptron #########################
    print("============== 4. Margin Perceptron ==============")
    # 4 - Margin Perceptron
    # clear old network
    average_perceptron = None

    # margin perceptron cross validation
    learning_rates = [1,0.1,0.01]
    margins = [1, 0.1, 0.01]
    epochs = 10
    best_vals = cross_validate_margins(epochs,learning_rates,margins)
    best_lr = best_vals[0]
    best_margin = best_vals[1]

    training_data = Data(TRAINING_PATH)
    test_data = Data(TESTING_PATH)
    epochs = 20
    learning_rate = best_lr
    margin = best_margin
    margin_perceptron = Perceptron()
    margin_perceptron.train_margin(training_data,epochs,learning_rate,margin)
    print("number of updates: ",margin_perceptron.num_updates)
    # Get the best weights and bias from this training
    W,b,best_epoch = margin_perceptron.get_best_weights_and_bias()
    # training set accuracy:
    print("best training set accuracy: ", margin_perceptron.accuracies[best_epoch] )
    # Use these weights and bias to evaluate on the test set
    test_accuracy = margin_perceptron.get_accuracy_own_weights(test_data,W,b)
    print("final test accuracy: ",test_accuracy)

    y = list(margin_perceptron.accuracies.values())
    x = [i for i in range(epochs)]
    title = '4 - margin perceptron learning curve'
    plot_learning(x,y,title)
    plt.show()

    print("\n")

    ######################## 5. Extra Cred #############################
    print("============== 5. Extra Credit ==============")

    # Rework the feature space 
    training_features_orig = Data(TRAINING_PATH)
    testing_features_orig = Data(TESTING_PATH)
    fold_ids = ['fold1','fold2','fold3','fold4','fold5']
    path_names = ['data/csv-format/CVfolds/{}.csv'.format(i) for i in fold_ids]
    # folds = { index : Data(path_name) for index,path_name in enumerate(path_names) }
    # list of each fold 
    fold_features_orig = [Data(file_path) for file_path in path_names]

    # transforming the data
    print("Tranforming features...")
    transformed_test_data = feature_transformation(testing_features_orig)
    print("...1/7")
    transformed_training_data = feature_transformation(training_features_orig)
    print("...2/7")
    transformed_folds = {}
    for index,i in enumerate(fold_features_orig):
        transformed_folds[index] = feature_transformation(i)
        print("...{}/7".format(index+3))
    print("DONE")

    # running these tranformed features on average 

    # 3 - Averaged Perceptron w/ transformed features
    # clear old network
    margin_perceptron = None

    learning_rates = [1,0.1,0.01]
    epochs = 10
    print("Cross Validating...")
    best_lr = cross_validate(epochs,learning_rates,'averaged',transformed_folds)
    print("DONE")
    print("Training...")
    training_data = transformed_training_data
    test_data = transformed_test_data
    epochs = 20
    learning_rate = best_lr
    average_perceptron = Perceptron()
    average_perceptron.train_averaged(training_data,epochs,learning_rate)
    print("number of updates: ",average_perceptron.num_updates)
    # Get the best weights and bias from this training
    W,b,best_epoch = average_perceptron.get_best_weights_and_bias()
    # training set accuracy:
    print("best training set accuracy: ", average_perceptron.accuracies[best_epoch] )
    # Use these weights and bias to evaluate on the test set
    test_accuracy = average_perceptron.get_accuracy_own_weights(test_data,W,b)
    print("final test accuracy: ",test_accuracy)

    y = list(average_perceptron.accuracies.values())
    x = [i for i in range(epochs)]
    title = '5 - average perceptron w/ new features '
    plot_learning(x,y,title)
    plt.show()




if __name__ == "__main__":
    main()
