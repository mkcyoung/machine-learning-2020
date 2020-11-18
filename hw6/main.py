# Imports
import numpy as np
from copy import copy
import pandas as pd
import math
import random
import matplotlib.pyplot as plt

# Local imports
from utils.tree_utils import *
from utils.svm_utils import *
from utils.logistic_utils import *
from utils.cv_utils import *


########################## Initializing ############################

# Initializing random variable
np.random.seed(42)
# Loading file paths
TRAINING_PATH = 'data/csv-format/train.csv'
TESTING_PATH = 'data/csv-format/test.csv'

############################### MAIN FUNCTION ###############################################
def main():

    # cross validate svm
    epochs = 15 #chosen by visual inspection 
    learning_rates = [10**0, 10**-1, 10**-2, 10**-3, 10**-4]
    regularizations = [10**3, 10**2, 10**1, 10**0, 10**-1, 10**-2]
    print('\n')
    print("================= SVM Cross validation & Training ======================")
    best_vals = cross_validate(epochs,learning_rates,regularizations,verbose=False)
    
    # Train using best params
    learning_rate = best_vals[0]
    C = best_vals[1]

    training_data = Data(TRAINING_PATH)
    test_data = Data(TESTING_PATH)
    epochs = 40
    svm = SVM()
    svm.train(training_data,epochs,learning_rate,C)
    # test set accuracy
    # Get the best weights and bias from this training
    W,best_epoch = svm.get_best_weights_and_bias()
    # training set accuracy:
    print("best training set accuracy: ", svm.accuracies[best_epoch] )
    # Use these weights and bias to evaluate on the test set
    test_accuracy = svm.get_accuracy_own_weights(test_data,W)
    print("final test accuracy: ",test_accuracy)

    y = list(svm.accuracies.values())
    x = [i for i in range(epochs)]
    title = 'SVM training learning curve'
    plot_learning(x,y,title,'epochs','training accuracy')

    y = list(svm.loss.values())
    x = [i for i in range(epochs)]
    title = 'SVM training loss'
    plot_learning(x,y,title,'epochs','loss')

    print('\n')     
    print("================= Logistic Regression Cross validation & Training ======================")
    print('\n')

    # cross validate
    epochs = 15
    learning_rates = [10**0,10**-1,10**-2,10**-3,10**-4,10**-5]
    regularizations = [10**-1,10**0,10**1,10**2,10**3,10**4]
    best_params = cross_validate(epochs,learning_rates,regularizations,verbose=False,model=LOGREG())

    # Train with these:
    training_data = Data(TRAINING_PATH)
    test_data = Data(TESTING_PATH)

    epochs = 40
    learning_rate = best_params[0]
    C = best_params[1]
    log_reg = LOGREG()
    log_reg.train(training_data,epochs,learning_rate,C)

    # test set accuracy
    # Get the best weights and bias from this training
    W,best_epoch = log_reg.get_best_weights_and_bias()
    # training set accuracy:
    print("best training set accuracy: ", log_reg.accuracies[best_epoch] )
    # Use these weights and bias to evaluate on the test set
    test_accuracy = log_reg.get_accuracy_own_weights(test_data,W)
    print("final test accuracy: ",test_accuracy)


    y = list(log_reg.accuracies.values())
    x = [i for i in range(epochs)]
    title = 'Logistic regression training learning curve'
    plot_learning(x,y,title,'epochs','training accuracy')

    y = list(log_reg.loss.values())
    x = [i for i in range(epochs)]
    title = 'Logistic regression training loss'
    plot_learning(x,y,title,'epochs','loss')
    print('\n')
    print("================= SVM over trees Cross validation & Training ======================")
    print('\n')
    print('Sorry, takes a VERY long time (~10 min)')
    print('\n')
    # CV! will prob take forever
    learning_rates = [10**0,10**-1,10**-2,10**-3,10**-4,10**-5]
    regularizations = [10**3,10**2,10**1,10**0,10**-1,10**-2]
    depths = [1,2,4,8]
    epochs = 15

    best_params = cv_svm_over_trees(epochs,learning_rates,regularizations,depths,verbose=False)

    # Train and test using best params....
    trnData = treeData(TRAINING_PATH)
    valData = treeData(TESTING_PATH)

    num_trees = 200
    depth = best_params[2]
    # creating feature representations
    train, eval = trees_to_vec(trnData,valData,num_trees,depth)
    # loading into SVM data format
    trnData = Data()
    trnData.load_data(train)
    valData = Data()
    valData.load_data(eval)
    # setting params
    lr = best_params[0]
    C = best_params[1]
    epochs = 40
    # training
    svm = SVM()
    svm.train(trnData,epochs,lr,C)
    weights,best_epoch = svm.get_best_weights_and_bias()
    # training set accuracy:
    print("best training set accuracy: ", svm.accuracies[best_epoch] )
    # Use these weights and bias to evaluate on the test set
    test_accuracy = svm.get_accuracy_own_weights(valData,weights)
    print("final test accuracy: ",test_accuracy)

    y = list(svm.accuracies.values())
    x = [i for i in range(epochs)]
    title = 'SVM over trees training learning curve'
    plot_learning(x,y,title,'epochs','training accuracy')

    y = list(svm.loss.values())
    x = [i for i in range(epochs)]
    title = 'SVM over trees training loss'
    plot_learning(x,y,title,'epochs','loss')






if __name__ == "__main__":
    main()
