# Useful starting lines
import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
from proj1_helpers import *
from implementations import *
from features import *
from cross_validation import *

# Evaluate a model

def accuracy(y, y_pred):
	""" Compute accuracy. """
	return np.mean(y_pred == y)

# Load the whole data

# Pickle dataset for fast reload
y_train, x_train, ids_train, headers = load_csv_data('data/train.csv')
y_test, x_test, ids_test, headers_test = load_csv_data('data/test.csv')
pickle.dump((y_train, x_train, ids_train, headers), open('train.pickle', 'wb'))
pickle.dump((y_test, x_test, ids_test, headers_test), open('test.pickle', 'wb'))

# Load dataset using pickle
def reload_dataset():
	global y_train, x_train, ids_train, headers, y_test, x_test, ids_test, headers_test
	y_train, x_train, ids_train, headers = pickle.load(open('train.pickle', 'rb'))
	y_test, x_test, ids_test, headers_test = pickle.load(open('test.pickle', 'rb'))

def build_k_indices(y, k_fold, seed=1):
	""" Build k indices for k-fold."""
	totalLength = y.shape[0]
	intervalLength = int(totalLength / k_fold) # Length of an internval
	np.random.seed()
	indices = np.random.permutation(totalLength)
	k_indices = [indices[k * intervalLength: (k + 1) * intervalLength] for k in range(k_fold)]
	return np.array(k_indices)

def processing(x, deg, jet_mod=False, jet_num=0, mean=None, std=None):
	""" Process the features. """
	long_tails=[0, 1, 2, 3, 5, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29]
	# Get the valid and invalid values (-999)
	missing_mask = x == -999
	correct_mask = x != -999
	
	# Log transform all the long tails
	for i in long_tails:
		# Only the valid values
		x[correct_mask[:,i],i] = np.log(1 + x[correct_mask[:,i],i])

	# Difference between angles
	angle = [15, 18, 20]
	diff01 = np.abs(x[:,angle[0]] - x[:,angle[1]]).reshape((len(x), 1))
	diff02 = np.abs(x[:,angle[0]] - x[:,angle[2]]).reshape((len(x), 1))
	diff12 = np.abs(x[:,angle[1]] - x[:,angle[2]]).reshape((len(x), 1))
	
	x = np.hstack((x, diff01, diff02, diff12))
			
	# Exclude some invalid variables depending of the jet_num
	if jet_mod:
		features_excluded = [[4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28], [4, 5, 6, 12, 22,25, 26, 27, 28], [], []]
		excepted = np.setdiff1d(np.arange(x.shape[1]), features_excluded[jet_num])
		x = x[:,excepted]

	# Standardize
	x, mean, std = standardize(x, mean, std)

	# Build polynomial features
	x = build_poly(x, deg)
	
	return x, mean, std

def show_x(x):
	for i in range(len(x[0])):
		array = x[:,i]
		plt.hist(array, 250)
		plt.title("Variable %i: %s"%(i, headers[i+2]))
		plt.show()
	return x

def grid_search():
	# Load dataset
	reload_dataset()

	# Parameters
	k_fold = 10

	file = open("trace.txt", "w") 

	# We process a grid search for each "jet"
	for i in range(2,-1,-1):
		print("Jet: " + str(i))
		best = 0
		best_lambda = 0
		best_degree = 0
		best_gamma = 0
		# Select the corresponding rows
		jet_mask_train = x_train[:,22] == i
		jet_mask_test = x_test[:,22] == i
		if i == 2 :
			# 2 and 3 are treated the same way
			jet_mask_train = np.asarray(x_train[:,22]==i) + np.asarray(x_train[:,22]==3) 
			jet_mask_test = np.asarray(x_test[:,22]==i) + np.asarray(x_test[:,22]==3) 
			
		x_jet_train, x_jet_test = x_train[jet_mask_train], x_test[jet_mask_test]
		y_jet_train, y_jet_test = y_train[jet_mask_train], y_test[jet_mask_test]
		
		k_indices = build_k_indices(y_jet_train, k_fold)

		for deg in range(1,20,1):
			print("Degré : " + str(deg))
			for lambda_ in np.logspace(-20,0,20):
				print("Lambda : " + str(lambda_))
				for gamma in np.logspace(-20,0,20):
					print("Gamma : " + str(gamma))

					# Process features
					x_jet_train_processed, mean, std = processing(x_jet_train, deg, True, i)

					# Split the dataset for cross validation
					accuracies = []
					for k in range(k_fold):
						new_accuracy = cross_validation(y_jet_train, x_jet_train_processed, k_indices, k, lambda_, gamma)
						accuracies.append(new_accuracy)
					new_accuracy = np.asarray(accuracies).mean()

					# Is it the best model so far ?
					if new_accuracy>best:
						best=new_accuracy
						print("New best accuracy ! Accuracy : {} Lambda: {} Gamma: {} Degree: {} Jet: {}".format(new_accuracy, lambda_, gamma, deg, i))
						file.write("New best accuracy ! Accuracy : {} Lambda: {} Gamma: {} Degree: {} Jet: {}".format(new_accuracy, lambda_, gamma, deg, i))
						best_lambda = lambda_
						best_gamma = gamma
						best_degree = deg
					else:
						print("Accuracy : " + str(new_accuracy))

	file.close()

if __name__ == '__main__':
	# Load dataset
	reload_dataset()

	k_fold = 10

	ids_ = []
	pred_ = []

	# We train a classifier for each "jet"
	for i in range(2,-1,-1):
		print("Jet: " + str(i))

		# Select the corresponding rows
		jet_mask_train = x_train[:,22] == i
		jet_mask_test = x_test[:,22] == i
		if i == 2 :
			# 2 and 3 are treated the same way
			jet_mask_train = np.asarray(x_train[:,22]==i) + np.asarray(x_train[:,22]==3) 
			jet_mask_test = np.asarray(x_test[:,22]==i) + np.asarray(x_test[:,22]==3) 
			
		x_jet_train, x_jet_test = x_train[jet_mask_train], x_test[jet_mask_test]
		y_jet_train, y_jet_test = y_train[jet_mask_train], y_test[jet_mask_test]
		
		k_indices = build_k_indices(y_jet_train, k_fold)

		# Best values for lambda and deg
		lambdas = [1e-15, 1e-15, 1e-13]
		degs = [11, 12, 14]

		lambda_= lambdas[i]
		deg = degs[i]

		accuracies = []
		numbers = []

		# Process features
		x_jet_train_processed, mean, std = processing(x_jet_train, deg, True, i)
		x_jet_test_processed, _, _ = processing(x_jet_test, deg, True, i, mean, std)

		# Training
		#w,loss =least_squares(y_jet_train, x_jet_train)
		#w, loss = least_squares_SGD(y_jet_train, x_jet_train, np.zeros(x_jet_train.shape[1]), 200, 3e-2)
		w, loss = ridge_regression(y_jet_train, x_jet_train_processed, lambda_)
		#w, loss = reg_logistic_regression_SGD((y_train2 == 1).astype(float), train_processed_poly, 1e-5,
		#	np.zeros(train_processed_poly.shape[1]), 2000, 1e-7)
		#print("Loss = %f"%(loss))

		# Prediction
		y_jet_test_pred = predict_labels(w, x_jet_test_processed)

		pred_.append(y_jet_test_pred)
		ids_.append(ids_test[jet_mask_test])

	with open("submission", 'w') as csvfile:
		for i in range(3):
			ids = ids_[i]
			y_pred = pred_[i]
			fieldnames = ['Id', 'Prediction']
			writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
			if i ==0:
				writer.writeheader()
			for r1, r2 in zip(ids, y_pred):
				writer.writerow({'Id':int(r1),'Prediction':int(r2)})

