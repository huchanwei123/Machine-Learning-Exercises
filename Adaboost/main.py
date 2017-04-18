import csv
import pdb
import os, sys
import time
import math
import numpy as np
import argparse
import adaboost
import multiprocessing as mp

def cross_validation(iter_):
	data_num, data_size = train_data_.shape
	fold_size = int(data_num / N_fold)
	residual = data_num - fold_size * N_fold
	min_error = math.inf   
	train_data = np.zeros((data_num - fold_size, data_size))
	train_label = np.zeros(data_num - fold_size)
	test_data = np.zeros((fold_size, data_size))
	test_label = np.zeros(fold_size)
	total_error = 0
	for i in range(0, N_fold):
		if i == 0:
			train_data[:,:] = train_data_[fold_size:, :]
			train_label[:] = train_label_[fold_size:]
			test_data[:,:] = train_data_[0:fold_size, :]
			test_label[:] = train_label_[0:fold_size]
		else:
			train_data[:,:] = np.append(train_data_[0:i*fold_size, :], train_data_[(i+1)*fold_size:, :], axis=0)
			train_label[:] = np.append(train_label_[0:i*fold_size], train_label_[(i+1)*fold_size:])
			test_data[:,:] = train_data_[i*fold_size:(i+1)*fold_size, :]
			test_label[:] = train_label_[i*fold_size:(i+1)*fold_size]
			 
		best = adaboost.Adaboost(train_data, train_label, 'validation', iter_)
		error = adaboost.adaClassify(test_data, test_label, 'validation', best)
		total_error = total_error + error
	CV_error = total_error/N_fold
	print('[Result] Cross-Validation Error of T =', iter_, 'is', CV_error)

	return iter_, CV_error

###################################### Data and Parameter Setup ########################################

parser = argparse.ArgumentParser(description='Argument of the input')
parser.add_argument('--train_path', dest='train_path', type=str,
                    default = './../data/alphabet_DU_training.csv', help='=== training data path ===')
parser.add_argument('--test_path', dest='test_path', type=str,
                    default = './../data/alphabet_DU_testing.csv', help='=== testing data path ===')
parser.add_argument('--T_range', dest='validation_iter', type=int,
					default = 50, help='=== The range while finding the best T , default is 50===')
parser.add_argument('--N_fold', dest='N_fold', type=int,
                    default = 5, help='=== How many folds ? Default is 5 ===')
parser.add_argument('--core_num', dest='core_num', type=int,
                    default = mp.cpu_count(), help='=== How many CPU cores to use? Default is all of your CPU core ===')
args = parser.parse_args()

global N_fold, validation_iter, train_data_, train_label_
Train_DATA_PATH = args.train_path
Test_DATA_PATH = args.test_path
validation_iter = args.validation_iter
N_fold = args.N_fold

# Load the data
train_data_, train_label_ = adaboost.load_data(Train_DATA_PATH)
test_data, test_label = adaboost.load_data(Test_DATA_PATH)

# Find best T
pool = mp.Pool(args.core_num)
print('Using', args.core_num, 'CPU cores to run !!')
_iter = range(1, validation_iter + 1)
start_time = time.time()
CV_error_list = pool.map(cross_validation, _iter)
with open(str(N_fold)+'_fold_find_optimal_T.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(CV_error_list)

min_error = math.inf
best_T = 0
for i in range(1, validation_iter+1):
	if(min_error > CV_error_list[i-1][1]):
		min_error = CV_error_list[i-1][1]
		best_T = CV_error_list[i-1][0]

print('Choose optimal T =', best_T, '\n')
print('Time cost for cross-validation =', (time.time() - start_time)/60, 'mins')

# start our Adaboost 
best_hypothesis = adaboost.Adaboost(train_data_, train_label_, 'training', best_T)
out_hypothesis = []
for i in range(best_T):
    out_hypothesis.append([best_hypothesis[i]['iter'], best_hypothesis[i]['dim'], best_hypothesis[i]['thresh'], best_hypothesis[i]['inequal'] ,best_hypothesis[i]['alpha']])

with open(str(N_fold) + '_fold_output_AdaBoost_hypothesis_header.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['iteration_index','attribute_index','threshold', 'direction', 'boosting_parameter'])
    w.writerows(out_hypothesis)

train_accu = adaboost.adaClassify(train_data_, train_label_, 'testing', best_hypothesis)
test_accu, predict_output = adaboost.adaClassify(test_data, test_label, 'testing', best_hypothesis)
with open(str(N_fold) + '_fold_predict_output', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerows(predict_output)

print('Training Accuracy =', train_accu, '%')
print('Testing Accuracy =', test_accu, '%')
