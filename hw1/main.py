import numpy as np
import math

###################### Data functions and class ################################

# Lets make a class for the data
# What operations do I need to perform on the data during the algorithm?
# Count labels
# split data based on certain attributes, here the attributes
# will be the columns ... should I unsparsify this data? 
# I think I can just return it to the original representation?


class Data:
  def __init__(self,path=None):
    self.path = path
    self.data = {}
    self.length = 0
    self.classes = None
    if self.path != None:
      self.data = self.load_data(self.path)
      self.length = len(self.data)
      label_arr = get_labels(self.data)
      self.classes = np.unique(label_arr)

  def load_data(self,path):
    with open(path) as f:
      raw_data = [line.split() for line in f]
    # Will store as dictionary with label and features items with index as key
    data_dict = {}
    for index,line in enumerate(raw_data):
      features = []
      for feat in line[1:]:
        features.append(feat[:-2])
      data_dict[index] = {
          'label': line[0],
          'features': features
      }
    return data_dict

  def add_data(self,path_to_add):
    with open(path_to_add) as f:
      raw_data = [line.split() for line in f]
    new_index = self.length
    for index,line in enumerate(raw_data):
      features = []
      for feat in line[1:]:
        features.append(feat[:-2])
      self.data[index + new_index] = {
          'label': line[0],
          'features': features
      }
    # Update length, and classes
    self.length = len(self.data)
    label_arr = get_labels(self.data)
    self.classes = np.unique(label_arr)


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


###############    impurity measures and information gain related functions #############################

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


########################     Classes for Tree and decision tree classifier    ##################################

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
      common_label_node = Node(str(get_common_label(new_data))) 
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
          common_label_node = Node(str(get_common_label(new_data))) 
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

  def get_predict_error(self,data):
    return 1 - self.get_predict_accuracy(data)

  def get_depth(self):
    return self.tree_depth



###################### Main function ##########################
######## Loads data, trains tree, and prints out all info necessary for report #################
def main():
    ########### 1. FULL TREES ###################
    # (a) [6 points] Implement the decision tree data structure and the ID3 algorithm for your decision tree 
    #     (Remember that the decision tree need not be a binary tree!). For debugging your implementation, 
    #     you can use the previous toy examples like the restaurant data from Table 1. Discuss what approaches 
    #     and design choices you had to make for your implementation and what data structures you used.

    # Load train & test data
    train_path = 'data/a1a.train'
    train_data = Data(train_path)
    test_path = 'data/a1a.test'
    test_data = Data(test_path)
    
    attributes = [i for i in range(1,124)] # features from 1-123
    decision_tree = DecisionTreeClassifier(train_data,attributes)
    decision_tree.build_tree() # no depth limit

    print('\n')
    print('######################## HW1 - Decision Trees ####################################')
    print('\n')
    # Most common label + entropy in training data
    
    print('MOST COMMON LABEL IN TRAINING SET: ',get_common_label(train_data.data))
    counts = list(count_labels(train_data.data,train_data.classes).values())
    print('LABEL COUNTS IN TRAINING SET: ',count_labels(train_data.data,train_data.classes))
    print('ENTROPY OF TRAINING SET: ', compute_entropy(counts))


    print('\n')
    print('######################## FULL TREE RESULTS #######################################')
    print('\n')

    # Decision tree results
    print("FULL TREE - root feature:",decision_tree.tree.attribute)
    print("FULL TREE - root information gain:", decision_tree.tree.information_gain)
   

    # (b) Report the error of your decision tree on all the examples data/a1a.train.
    print("FULL TREE - training accuracy: ",decision_tree.get_predict_accuracy(train_data))
    print("FULL TREE - training error: ",decision_tree.get_predict_error(train_data))


    # (c) Report the error of your decision tree on the examples in data/a1a.test.
    print("FULL TREE - test accuracy: ",decision_tree.get_predict_accuracy(test_data))
    print("FULL TREE - test error: ", decision_tree.get_predict_error(test_data))


    # (d) Report the maximum depth of your decision tree.
    print("FULL TREE - max depth: ",decision_tree.get_depth())
    print('\n')

    print('######################## FULL TREE RESULTS - GINI INDEX ########################')
    print('\n')

    # report same results using gini
    decision_tree_gini = DecisionTreeClassifier(train_data,attributes,compute_gini)
    decision_tree_gini.build_tree()

    # Decision tree results w/ gini
    print("GINI - training accuracy: ",decision_tree_gini.get_predict_accuracy(train_data))
    print("GINI - training error: ",decision_tree_gini.get_predict_error(train_data))
    print("GINI - test accuracy: ",decision_tree_gini.get_predict_accuracy(test_data))
    print("GINI - test error: ", decision_tree_gini.get_predict_error(test_data))
    print("GINI - root feature:",decision_tree_gini.tree.attribute)
    print("GINI - root information gain:", decision_tree_gini.tree.information_gain)
    print("GINI - max depth: ",decision_tree_gini.get_depth())

    print('\n')
    print('##################### LIMITING DEPTH - CROSS VALIDATION ########################')
    print('\n')

    # Performing 5 fold cross validation
    depths = [1, 2, 3, 4, 5]
    mean_accuracies = [] 
    standard_deviations = []
    k = 5
    fold_ids = ['fold1','fold2','fold3','fold4','fold5']
    path_names = ['data/CVfolds/{}'.format(i) for i in fold_ids]

    for current_depth in depths:
        accuracies = []
        # Need to concatenate 4 of the folds into one training set and leave out one as my test set
        for index,test_fold_name in enumerate(fold_ids):
            test_fold = Data(path_names[index])
            # print("test fold length: ",test_fold.length)
            # Create training data object
            training_fold = Data()
            # training_folds = [folds[train_fold_name] for train_fold_name in fold_names if train_fold_name != test_fold_name ]
            for idx,train_fold_name in enumerate(fold_ids):
                if train_fold_name != test_fold_name:
                    training_fold.add_data(path_names[idx])
            # print("training fold length: ", training_fold.length)
            # print("total length", test_fold.length + training_fold.length)

            # Now I have my test fold and training fold
            # Build tree using training fold
            attributes = [i for i in range(1,124)]
            decision_tree = DecisionTreeClassifier(training_fold,attributes)
            decision_tree.build_tree(current_depth)
            # Evaluate tree using the set aside test fold 
            # print("train error: ",decision_tree.get_predict_error(training_fold))
            # print("train accuracy: ",decision_tree.get_predict_accuracy(training_fold))
            # print("test error: ",decision_tree.get_predict_error(test_fold))
            # print("test accuracy: ",decision_tree.get_predict_accuracy(test_fold))
            accuracies.append(decision_tree.get_predict_accuracy(test_fold))
        mean_accuracies.append(np.mean(accuracies))
        standard_deviations.append(np.std(accuracies))

    for depth,accuracy in enumerate(zip(mean_accuracies,standard_deviations)):
        acc,std = accuracy
        depth += 1
        print("Depth: {} - Accuracy: {} +/- {}".format(depth,acc,std))
    
    print('\n')
    print('##################  RETRAINING USING BEST DEPTH (3)  ############################')
    print('\n')

    decision_tree_limited = DecisionTreeClassifier(train_data,attributes)
    decision_tree_limited.build_tree(3)

    # Decision tree results
    print("DEPTH LIMITED - training accuracy: ",decision_tree_limited.get_predict_accuracy(train_data))
    print("DEPTH LIMITED - training error: ",decision_tree_limited.get_predict_error(train_data))
    print("DEPTH LIMITED - test accuracy: ",decision_tree_limited.get_predict_accuracy(test_data))
    print("DEPTH LIMITED - test error: ", decision_tree_limited.get_predict_error(test_data))
    print("DEPTH LIMITED - root attribuite:",decision_tree_limited.tree.attribute)
    print("DEPTH LIMITED - root information gain:", decision_tree_limited.tree.information_gain)
    print("DEPTH LIMITED - max depth: ",decision_tree_limited.get_depth())

    print('\n')


if __name__ == "__main__":
    main()