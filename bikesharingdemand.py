import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold 
import random
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import operator


def featureEngineer(data, feature_to_drop):
    """
    create or drop features to make algorithmn work
    parameter: 
    ----------
    data: original dataframe that contains both of features and results 
    feature_to_drip: list of features to drop from dataset
    return:
    ---------- 
    data: dataframe that includes new features and drops redundent features
    -------
    """
    dt = pd.to_datetime(data["datetime"])  # convert string to datetime type
    data["year"] = dt.map(lambda x: x.year)
    data["month"] = dt.map(lambda x: x.month)
    data["day"] = dt.map(lambda x: x.day)
    data["hour"] = dt.map(lambda x: x.hour)
    data["weekday"] = dt.map(lambda x: x.weekday())
    # remove outliers detected from temp vs atemp scatter plot
    train_all = data[np.array(data["temp"] / data["atemp"]) < 2]
    train_all = train_all.drop(feature_to_drop,
        axis=1)  # include all features and target
    return train_all


def dataSplit(data, ratio = 0.7):
    """
    split data into training and validation set
    parameter: 
    ----------
    data: dataframe that contains all observations
    
    return:
    ---------- 
    train_feature: array that contains 70% (by default) of observations. 
    only feature included
    train_target_count: training set target using COUNT
    train_target_casual: training set target using CASUAL
    train_target_registered: training set target using REGISTERE
    valid_feature: array that contains 30% (by default) of observations. 
    only feature included
    valid_target_count: validation set target using COUNT
    valid_target_casual: validation set target using CASUAL
    valid_target_registered: validation set target using REGISTERE
    """
    header1 = list(data.columns)
    print "columns:", header1
    temp = data.values
    train_size = int(temp.shape[0] * ratio)
    train_sample_indices = random.sample(range(temp.shape[0]),
                                         train_size)
    train = temp[train_sample_indices, :]
    validation_sample_indices = [i for i in range(temp.shape[0]) if
                                 i not in train_sample_indices]
    validation = temp[validation_sample_indices, :]

    train_target_count = temp[train_sample_indices, header1.index(
        'count')].astype(int)  # target value for train data
    train_target_casual = temp[train_sample_indices, header1.index(
        'casual')].astype(int)
    train_target_registered = temp[train_sample_indices, header1.index(
        'registered')].astype(int)
    # all features for model input
    train_feature = np.delete(train, [header1.index('count'), header1.index(
        'casual'), header1.index('registered')],axis=1)

    valid_target_count = temp[validation_sample_indices, header1.index(
        'count')].astype(int)  # target value for validation data
    valid_target_casual = temp[validation_sample_indices, header1.index(
        'casual')].astype(int)  # target value for validation data
    valid_target_registered = temp[validation_sample_indices, header1.index(
        'registered')].astype(int)  # target value for validation data
    valid_feature = np.delete(validation, [header1.index('count'),
                                           header1.index('casual'),
                                           header1.index('registered')],axis=1)

    print "train shape: ", np.shape(train)
    print "train features shape: ", np.shape(train_feature)
    print "train target shape: ", np.shape(train_target_count), np.shape(
        train_target_registered), np.shape(train_target_casual)
    print "validation feature shape: ", np.shape(valid_feature)
    print "validation target shape: ", np.shape(valid_target_count), np.shape(
        valid_target_registered), np.shape(valid_target_casual)
    print "data preparation done."
    return train_feature, train_target_casual, train_target_registered, \
           train_target_count, valid_feature, valid_target_casual, \
           valid_target_count, valid_target_registered

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
	kf = KFold(n_splits=10)
	train_X = sm.add_constant(train_X) 

	
	family = [sm.families.Poisson(), sm.families.NegativeBinomial()] # link functions

	neg_rmlse = [] 
	pois_rmlse = []

	for train_index, test_index in kf.split(train_X):
		# possion regression
	    glm_pois = sm.GLM(train_Y[train_index], train_X[train_index, :], family=family[0]).fit()
	    pred = glm_pois.predict(train_X[test_index, :]) # prediction of test set
	    pois_rmlse.append(cal_rmlse(train_Y[test_index], pred))
	    # negative binomial regression
	    glm_neg = sm.GLM(train_Y[train_index], train_X[train_index, :], family=family[1]).fit()
	    pred = glm_neg.predict(train_X[test_index, :]) # prediction of test set
	    neg_rmlse.append(cal_rmlse(train_Y[test_index], pred))
	print "average rmlse of possion regression:", np.mean(pois_rmlse)
	print "average rmlse of negative binomial regression:", np.mean(neg_rmlse)

	zipped = zip(family, [np.mean(pois_rmlse), np.mean(neg_rmlse)]) 


	val_X = sm.add_constant(val_X)
	glm = sm.GLM(val_Y, val_X, family=sorted(zipped, key=operator.itemgetter(1))[0][0]).fit() # using the family that yielded the lowest average rmlse
	pred = glm.predict(val_X)
	val_rmlse = cal_rmlse(val_Y, pred)
	print "rmlse of validation set:", val_rmlse

	# visualize prediction vs actual values
	fig = plt.figure(figsize=(12,10))
	#fig.suptitle('Negative Binomial Regression', fontsize=14, fontweight='bold')
	ax = fig.add_subplot(111)
	# ax1.scatter(pred, val_Y - pred)
	# ax1.set_title("Residuals Vs Fitted")
	# ax1.set_xlabel("Fitted Values")
	# ax1.set_ylabel('Residuals')
	# ax2 = fig.add_subplot(212)
	ax.scatter(pred, val_Y)
	ax.set_title("Negative Binomial Regression")
	ax.set_xlabel("predicted")
	ax.set_ylabel("actual")
	plt.show()
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


def main():
    # load data
    print "loading data..."
    data = pd.read_csv("train.csv", header=0)
    # feature engineer
    print "feature engineering..."
    data = featureEngineer(data, ['datetime', 'weekday', 'temp'])
    # split data to train and validation
    train_feature, train_target_casual, train_target_registered, train_target_count, \
  	valid_feature, valid_target_casual, valid_target_count, valid_target_registered = dataSplit(data)

    # # model selection
    models_rmlse = {}
    models_rmlse["glm"] = glmPossion(train_feature, train_target_count, valid_feature, valid_target_count) 
    # models_rmlse["Random Forest"] = randomForest(train_X, train_Y, val_X, val_Y)
    # models_rmlse["Gradient Boost"] = gradientBoost(train_X, train_Y, val_X, val_Y)
    # models_rmlse["XgBoost"] = xgBoost(train_X, train_Y, val_X, val_Y)

    print "The model that gives the best performance on validation data is %s"%(sorted(models_rmlse.items(), key=operator.itemgetter(1))[0][0])


if __name__ == '__main__':
    main()

