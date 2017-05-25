import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold 
import random
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import operator


def featureEngineer(data):
	"""
	create or drop features to make algorithmn work
	parameter: 
	----------
	data: original dataframe that contains both of features and results 
	return:
	---------- 
	data: dataframe that includes new features and drops redundent features
	-------
	"""
	return data 


def dataSplit(data, ratio = 0.7):
	"""
	split data into training and validation set
	parameter: 
	----------
	data: dataframe that contains all observations

	return:
	---------- 
	train: dataframe that contains 70% (by default) of observations as training set
	validation: dataframe that contains 30% of observatiosn as validation set
			 
	"""

	return train, validation

def glmPossion(train_X, train_Y, val_X, val_Y):
	"""
	running the model of generalized linear regression, parameters tuning
	parameters:
	---------- 
	X: ndarray, predictors
	Y: ndarray, reponsors
	return:
	---------- 
	par: a list of parameters that gives the best performance
	"""
	return val_rmlse

def randomForest(train_X, train_Y, val_X, val_Y):
	"""
	running random forest, parameters tuning
	parameters:
	---------- 
	X: ndarray, predictors
	Y: ndarray, reponsors
	return:
	---------- 
	par: a list of parameters that gives the best performance
	"""
	return val_rmlse

def gradientBoost(train_X, train_Y, val_X, val_Y):
	"""
	running gradient boost, parameters tuning
	parameters:
	---------- 
	X: ndarray, predictors
	Y: ndarray, reponsors
	return:
	---------- 
	par: a list of parameters that gives the best performance
	"""
	return val_rmlse

def xgBoost(train_X, train_Y, val_X, val_Y):
	"""
	running xgBoost model, parameters tuning
	parameters:
	---------- 
	X: ndarray, predictors
	Y: ndarray, reponsors
	return:
	---------- 
	par: a list of parameters that gives the best performance
	"""
	return val_rmlse


def cal_rmlse(pred, actual):
    """
    evaluation of model
    parameters:
    -----------
    pred: array or list
          prediction
    actual: array or list
        	actual target value
    return:
    -----------
    rmlse
    """
    rmlse = np.sqrt(np.mean((np.log(np.array(pred) + 1)- np.log(np.array(actual) + 1))**2))
    return rmlse


def main(filename):
	# load data
	data = pd.read_csv("train.csv", header=0)
	# feature engineer
	data = featureEngineer(data)
	# split data to train and validation
	train, validation = dataSplit(data)
	# get predictive and response variables 
	train_X = train.drop(["casual", "registered","count"], axis=1)
	train_Y = train["count"]
	val_X = validation.drop(["casual", "registered","count"], axis=1)
	val_Y = validation["count"]

	# model selection
	models_rmlse = {}
	models_rmlse["glm"] = glmPossion(features, count)
	models_rmlse["Random Forest"] = randomForest(features, count)
	models_rmlse["Gradient Boost"] = gradientBoost(features, count)
	models_rmlse["XgBoost"] = xgBoost(features, count)

	print "The model that gives the best performance on validation data is %s"%(sorted(x.items(), key=operator.itemgetter(1))[-1][0])


if __name__ == '__main__':
	main(sys.argv[0])

