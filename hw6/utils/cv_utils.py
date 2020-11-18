import numpy as np
from utils.svm_utils import *
from utils.tree_utils  import *

def cross_validate(epochs,learning_rates,regularizations,verbose=False,model=SVM()):
  fold_ids = ['fold1','fold2','fold3','fold4','fold5']
  path_names = ['data/csv-format/CVfolds/{}.csv'.format(i) for i in fold_ids]
  # folds = { index : Data(path_name) for index,path_name in enumerate(path_names) }
  # list of each fold 
  folds = [np.loadtxt(file_path, delimiter = ",") for file_path in path_names]
  k = len(folds)

  # dictionaries storing accuracies corresponding to certain hyper parameter combinations
  mean_accuracies = {} 
  standard_deviations = {}

  num_combs = len(learning_rates)*len(regularizations)
  progress = 0
  
  for lr in learning_rates:
    for C in regularizations:
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
        # train on folds
        svm = model
        svm.train(train_data,epochs,lr,C)
        weights,best_epoch = svm.get_best_weights_and_bias()
        # calculate validation accuracy
        val_accuracy = svm.get_accuracy_own_weights(val_data,weights)
        accuracies.append(val_accuracy)

      mean_accuracies[(lr,C)] = np.mean(accuracies)
      standard_deviations[(lr,C)] = np.std(accuracies)
      if verbose == True:
        print("accuracy: ",mean_accuracies[(lr,C)],"lr: ",lr,"C: ",C)
        # print("list",accuracies)
        progress += 1
        print("{:.4}% complete".format(100*progress/num_combs))

#   print(mean_accuracies.items())
#   print(standard_deviations.items())
  best_vals = max(mean_accuracies,key=mean_accuracies.get)
  print("best lr: ",best_vals[0],"best C: ",best_vals[1],"cross-val accuracy: ",mean_accuracies[best_vals])
  return best_vals



def cv_svm_over_trees(epochs,learning_rates,regularizations,depths,verbose=False,model=SVM()):
  fold_ids = ['fold1','fold2','fold3','fold4','fold5']
  path_names = ['data/csv-format/CVfolds/{}.csv'.format(i) for i in fold_ids]
  # folds = { index : Data(path_name) for index,path_name in enumerate(path_names) }
  # list of each fold 
  # folds = [np.loadtxt(file_path, delimiter = ",") for file_path in path_names]
  k = len(fold_ids)

  # dictionaries storing accuracies corresponding to certain hyper parameter combinations
  mean_accuracies = {} 
  standard_deviations = {}

  num_combs = len(learning_rates)*len(regularizations)*len(depths)
  progress = 0

  accuracies = {}

  for d in depths:
    for i in range(k):
      # Initialize new tree data objects
      val_data = treeData()
      train_data = treeData()
      # Set validation data
      val_data.add_data(path_names[i])
      # set training data
      [train_data.add_data(path_names[j]) for j in range(k) if j != i]
      # Now transform features
      train,eval = trees_to_vec(train_data,val_data,200,d)
      # Now load this into svm data objects
      trnData = Data()
      trnData.load_data(train)
      valData = Data()
      valData.load_data(eval)
      for lr in learning_rates:
        for C in regularizations:
          # train on folds
          svm = model
          svm.train(trnData,epochs,lr,C)
          weights,best_epoch = svm.get_best_weights_and_bias()
          # calculate validation accuracy
          val_accuracy = svm.get_accuracy_own_weights(valData,weights)
          if accuracies.get((lr,C,d))!= None:
            accuracies[(lr,C,d)].append(val_accuracy)
          else:
            accuracies[(lr,C,d)] = [val_accuracy]
          if i == 4:
            mean_accuracies[(lr,C,d)] = np.mean(accuracies[(lr,C,d)])
    if verbose == True:
      print("accuracy: ",mean_accuracies[(lr,C,d)],"lr: ",lr,"C: ",C,"depth:",d)
      print("accuracies",accuracies[(lr,C,d)])
      progress += 1
      print("{:.4}% complete".format(100*progress/num_combs))
        
  # this takes forever because I'm extracting the features for every single param combo
  # for lr in learning_rates:
  #   for C in regularizations:
  #     for d in depths:
  #       accuracies = []
  #       # Need to concatenate 4 of the folds into one training set and leave out one as my test set
  #       for i in range(k):
  #         # Initialize new tree data objects
  #         val_data = treeData()
  #         train_data = treeData()
  #         # Set validation data
  #         val_data.add_data(path_names[i])
  #         # set training data
  #         [train_data.add_data(path_names[j]) for j in range(k) if j != i]
  #         # Now transform features
  #         train,eval = trees_to_vec(train_data,val_data,200,d)
  #         # Now load this into svm data objects
  #         trnData = Data()
  #         trnData.load_data(train)
  #         valData = Data()
  #         valData.load_data(eval)
  #         # train on folds
  #         svm = model
  #         svm.train(trnData,epochs,lr,C)
  #         weights,best_epoch = svm.get_best_weights_and_bias()
  #         # calculate validation accuracy
  #         val_accuracy = svm.get_accuracy_own_weights(valData,weights)
  #         accuracies.append(val_accuracy)
  #       print(accuracies)
  #       mean_accuracies[(lr,C,d)] = np.mean(accuracies)
  #       standard_deviations[(lr,C,d)] = np.std(accuracies)
  #       if verbose == True:
  #         print("accuracy: ",mean_accuracies[(lr,C,d)],"lr: ",lr,"C: ",C,"depth:",d)
  #         # print("list",accuracies)
  #         progress += 1
  #         print("{:.4}% complete".format(100*progress/num_combs))

#   print(mean_accuracies.items())
  # print(standard_deviations.items())
  best_vals = max(mean_accuracies,key=mean_accuracies.get)
  print("best lr: ",best_vals[0],"best C: ",best_vals[1],"best depth: ",best_vals[2],"cross-val accuracy: ",mean_accuracies[best_vals])
  return best_vals