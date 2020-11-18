# Tree classes

import numpy as np
from copy import copy
import pandas as pd
import math


class treeData:
  def __init__(self,path=None):
    self.path = path
    self.data = {}
    self.length = 0
    self.classes = None
    self.num_features = 206
    self.np_data = np.zeros((0,self.num_features+1))
    if self.path != None:
      self.data = self.load_data(self.path)
      self.length = len(self.data)
      label_arr = get_labels(self.data)
      self.classes = np.unique(label_arr)

  def load_data(self,path):
    # pandas load csv
    # raw_data = pd.read_csv(path)
    # data = raw_data.to_numpy()
    data = np.loadtxt(path, delimiter = ",")
    self.np_data = data
    self.num_features = data.shape[1]-1
    # print(raw_data)
    # print(data)
    # Will store as dictionary with label and features items with index as key
    data_dict = {}
    for index,line in enumerate(data):
      features = []
      for i,feat in enumerate(line[1:]):
        if feat != 0:
          features.append(str(i))
      if line[0] == 1:
        label = "+1"
      else:
        label = "-1"
      data_dict[index] = {
          'label': label,
          'features': features
      }
    return data_dict

  def add_data(self,path_to_add):
    # with open(path_to_add) as f:
    #   raw_data = [line.split() for line in f]
    data = np.loadtxt(path_to_add, delimiter = ",")
    # concat to np_data by stacking on bottom
    self.np_data = np.concatenate((self.np_data,data),axis=0)
    new_index = self.length
    data_dict = {}
    for index,line in enumerate(data):
      features = []
      for i,feat in enumerate(line[1:]):
        if feat != 0:
          features.append(str(i))
      if line[0] == 1:
        label = "+1"
      else:
        label = "-1"
      self.data[index + new_index] = {
          'label': label,
          'features': features
      }
    # Update length, and classes
    self.length = len(self.data)
    label_arr = get_labels(self.data)
    self.classes = np.unique(label_arr)

  def sample_data(self,p):
    # return a subset of the data according to the input percentage
    # maybe best to use raw numpy array to sample, then convert to my tree
    # data structure
    num_samples = int(self.length * p)
    # print("num samples",num_samples)
    choices = np.random.randint(self.length-1, size=num_samples)
    # print(choices[0:5])
    data = self.np_data[choices,:]
    # print("new data shape",data.shape)
    data_dict = {}
    for index,line in enumerate(data):
      features = []
      for i,feat in enumerate(line[1:]):
        if feat != 0:
          features.append(str(i))
      if line[0] == 1:
        label = "+1"
      else:
        label = "-1"
      data_dict[index] = {
          'label': label,
          'features': features
      }
    return data_dict



def get_labels(data):
  # Returns list of labels given a data dict input
  labels = []
  for item in data.items():
    labels.append(item[1]['label'])
  return labels

def count_labels(data,classes):
  # receives data and classes and returns label count
  label_arr = get_labels(data)
  label_counts = {}
  for label in classes:
    label_counts[label] = label_arr.count(label)
  return label_counts

def split_on_attr(attr,data):
  # returns two subsets of input data split on attribute
  if attr == None:
    return data
  has_attr = {}
  no_attr = {}
  for item in data.items():
    if attr in item[1]['features']:
      has_attr[item[0]] = item[1]
    else:
      no_attr[item[0]] = item[1]
  return has_attr,no_attr

def get_common_label(data):
  labels = get_labels(data)
  return max(set(labels), key = labels.count)



# impurity measures and information gain related functions

# Computing Gini 
def compute_gini(count_arr):
  a,b = count_arr
  total = a+b
  if total == 0:
    return 0
  frac_a = a/total
  frac_b = b/total
  arr = [frac_a,frac_b]
  gini = 0
  for i in range(len(arr)):
    gini += arr[i]*(1-arr[i])
  return gini

# Computing entropy 
def compute_entropy(count_arr):
  # input is array of respective counts 
  # Given two integers representing number of respective labels,
  # returns entropy 
  if len(count_arr) == 0:
    return 0
  a, b = count_arr
  if a==0 or b==0:
    return 0
  total = a+b
  frac_a = a/total
  frac_b = b/total
  return -frac_a*math.log2(frac_a) - frac_b*math.log2(frac_b)


# Lets have this information gain function combine everything 
def information_gain(data,attr_split,classes,purity_func,verbose=0):
  # Compute entropy of entire data input
  full_labels = list(count_labels(data,classes).values())
  full_purity = purity_func(full_labels)
  # Total instance count
  S = len(data)
  # print(data)
  # Split data across indicated attribute
  has_attr, no_attr = split_on_attr(attr_split,data)
  # print(has_attr,no_attr)
  # return attribute counts for each split
  label_counts = [count_labels(has_attr,classes), count_labels(no_attr,classes)]
  if verbose == 2: print("label counts of split on {}: {}".format(attr_split,label_counts))
  sub_purity = 0
  total_count = 0
  for label_count in label_counts:
    counts = list(label_count.values())
    total_count += sum(counts)
    sub_purity += (sum(counts)/(S))*purity_func(counts)
  if S != total_count:
    raise Exception(print("totals don't match up"))
  if verbose == 1 or verbose == 2: print("Information gain splitting on {}: {}".format(attr_split,full_purity - sub_purity))
  return full_purity - sub_purity


# Returns best attribute along with information gain given data and attributes to consider
def get_best_attr(data,attributes,classes,purity_func=compute_entropy):
  max_information_gain = 0
  best_attr = None
  for attr in attributes:
    if information_gain(data,str(attr),classes,purity_func) > max_information_gain:
      max_information_gain = information_gain(data,str(attr),classes,purity_func)
      best_attr = attr
  if best_attr == None:
    # just pick one 
    best_attr = attributes[0]

  return best_attr,max_information_gain



# Classes for Tree

class Node:
  def __init__(self,label=None):
    self.level = None
    self.attribute = None
    self.information_gain = None

    self.label = label
    self.left = None # reference to left child node
    self.right = None # reference to right child node



class DecisionTreeClassifier:
  # pass in data object along with full list of attributes at first
  def __init__(self,data,attrs,purity_measure=compute_entropy):
    self.data = data.data
    self.most_common_label = str(get_common_label(self.data))
    self.attrs = attrs
    self.classes = data.classes
    self.num_classes = len(data.classes)
    self.tree = None
    self.tree_depth = 0
    self.purity_measure = purity_measure

  def build_tree(self,depth_limit=float('inf')):
    self.tree = self.train_tree(self.data,self.attrs,0,depth_limit)
    return self.tree

  def train_tree(self,data,attrs,level,depth_limit):
    new_attrs = attrs.copy()
    new_data = data.copy()
    current_level = level # way of tracking depth
    # Using ID3 algorithm
    # Base case, when we have all of one label 
    if 0 in list(count_labels(new_data,self.classes).values()):
      labels = count_labels(new_data,self.classes)
      max_label = max(labels,key=labels.get)
      # Return a single node tree with the correct label
      node = Node(max_label)
      node.level = current_level
      return node
    # If we are at our max depth, return a Node with the most common label 
    if current_level == depth_limit:
      common_label_node = Node(str(get_common_label(data))) 
      common_label_node.level = current_level
      return common_label_node
    # recursive case
    else:
      # iterate to next level and store if it's bigger than current max
      next_level = current_level + 1
      if next_level > self.tree_depth:
        self.tree_depth = next_level

      # Make root node
      root = Node()
      A,_information_gain = get_best_attr(new_data,new_attrs,self.classes,self.purity_measure) # best A
      # make A the root node
      root.attribute = A
      root.information_gain = _information_gain
      root.level = next_level
      # Split on the attribute
      [split_1,split_0] = split_on_attr(str(A),new_data)
      splits = [split_0,split_1] # Put 0 value first the 1 value second
      # Remove attribute from list
      new_attrs.remove(A)
      if A in new_attrs: print("ALERT")
      for i in range(2): # Possible values are a 0 and a 1
        if len(splits[i]) == 0:  # If the set is empty 
          common_label_node = Node(str(get_common_label(data))) 
          common_label_node.level = next_level
          if i == 0:
          # Add to left of tree
            root.left = common_label_node
          # Add to the right of tree
          if i == 1:
            root.right = common_label_node
        # Add branch for A taking value v
        if i == 0:
          # Add to left of tree
          root.left = self.train_tree(splits[i],new_attrs,next_level,depth_limit)
        if i == 1:
          # Add to right of tree
          root.right = self.train_tree(splits[i],new_attrs,next_level,depth_limit)
      return root

  def get_prediction(self,instance):
    # Feed in data instance features and return label
    example = instance.copy()
    current_node = self.tree
    # traverse along the tree
    traversing = True
    while traversing:
      if current_node.label:
        # print("label",current_node.label)
        return current_node.label
      else:
        # print("splitting attr",current_node.attribute)
        splitting_Attr = str(current_node.attribute)
        if splitting_Attr in example:
          # Go right, b/c it has it
          # print("right")
          current_node = current_node.right
        else:
          # Go left, b/c it doesn't have it
          # print("left")
          current_node = current_node.left

  def get_predict_accuracy(self,data):
    myData = data.data.copy()
    correct_labels = 0
    total_examples = len(myData)
    for i in myData.items():
      label = i[1]['label']
      features = i[1]['features']
      index = i[0]
      predicted_label = self.get_prediction(features)
      # print(index,label,predicted_label,features)
      if label == predicted_label:
        correct_labels += 1
      # print(correct_labels)
    return correct_labels / total_examples

  # returns predicitons in numpy array
  def get_predictions(self,data):
    myData = data.data.copy()
    correct_labels = 0
    total_examples = len(myData)
    predictions = []
    for i in myData.items():
      label = i[1]['label']
      features = i[1]['features']
      index = i[0]
      predicted_label = int(self.get_prediction(features))
      # print(index,label,predicted_label,features)
      # if predicted_label == -1:
      #   predicted_label = 0
      predictions.append(predicted_label)
    return np.array(predictions)


  def get_predict_error(self,data):
    return 1 - self.get_predict_accuracy(data)

  def get_depth(self):
    return self.tree_depth


def trees_to_vec(training_data,eval_data,num_trees,depth):
  """
    Constructs N trees of depth d using training_data and returns
    2 N+1-dimensional matrices where each element is the prediction of that 
    particular tree on training data examples and eval_data examples
    N = num_trees
    D = # of validation examples
    T = # training examples
    d = chosen depth of trees

    Output: (T and D) x (N + 1) numpy array where each element is -1 or +1 corresponding with
    the tree prediction. (Also tack the true label on the front)
  """
  D = eval_data.length
  T = training_data.length
  # print("D",D)
  num_features = training_data.num_features
  # print("num_features",num_features)
  attributes = [i for i in range(1,num_features)]
  # Prep predictions matrix
  eval_vec = np.zeros((D,(num_trees+1)))
  train_vec = np.zeros((T,(num_trees+1)))
  # Add labels to output vectors
  train_vec[:,0] = training_data.np_data[:,0]
  eval_vec[:,0] = eval_data.np_data[:,0]
  # make new train_data variable so I don't alter original
  train_data = copy(training_data)
  for i in range(num_trees):
    # sample data
    sampled_data = train_data.sample_data(0.1)
    train_data.data = sampled_data
    # train tree w/ sampled data
    treeClass = DecisionTreeClassifier(train_data,attributes)
    tree = treeClass.build_tree(depth)
    # create feature transormation
    train_vec[:,i+1] = treeClass.get_predictions(training_data)
    eval_vec[:,i+1] = treeClass.get_predictions(eval_data)
  return train_vec,eval_vec