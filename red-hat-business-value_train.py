'''
@author : github.com/nitish11
@date : September 17th, 2016
@description : Base code for https://www.kaggle.com/c/predicting-red-hat-business-value/data
'''

import pandas as pd
import pickle

print('Data Loading started--')
train_data = pickle.load(open('train_data.p','r'))
print('-'*30)
print('Data Loading done')

import numpy as np
import sklearn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

print train_data.head(5)
print train_data.describe()


########### Features from the training dataset ###########

predictors = ["people_id","activity_id","activity_category","char_1_x","char_2_x","char_3_x", "char_4_x","char_5_x","char_6_x","char_7_x","char_8_x","char_9_x", "char_10_x","char_1_y","group_1","char_2_y","char_3_y","char_4_y","char_5_y","char_11","char_12","char_13","char_14" ,"char_15","char_16","char_17","char_18","char_19","char_20","char_21","char_22","char_23", "char_24","char_25","char_26","char_27","char_28","char_29","char_30","char_31","char_32", "char_33","char_34","char_35","char_36","char_37","char_38"]


########### Select Features from the training dataset ###########

print('-'*30)
print('Feature Selection')
# Perform feature selection
selector = SelectKBest(f_classif, k=10)
selector.fit(train_data[predictors], train_data["outcome"])
# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)
print scores
pickle.dump(selector,open('train_select_model.p','wb'))


########### Linear Regression Classifier and Accuracy Calculation ###########

print('-'*30)
print('LinearRegression')
alg_linear_regression = LinearRegression()
scores=cross_validation.cross_val_score(alg_linear_regression,train_data[predictors],train_data["outcome"],cv=3)
print(scores.mean())
pickle.dump(alg_linear_regression,open('linear_regression_model.p','wb'))


########### Logistic Regression Classifier and Accuracy Calculation ###########

print('-'*30)
print('LogisticRegression')
alg_log_regression = LogisticRegression(random_state=1)
scores=cross_validation.cross_val_score(alg_log_regression,train_data[predictors],train_data["outcome"],cv=3)
print(scores.mean())
pickle.dump(alg_log_regression,open('log_regression_model.p','wb'))


########### Random Forest Classifier and Accuracy Calculation ###########

print('-'*30)
print('RandomForestClassifier')
alg_random_classifier = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
scores=cross_validation.cross_val_score(alg_random_classifier,train_data[predictors],train_data["outcome"],cv=3)
print(scores.mean())
pickle.dump(alg_random_classifier,open('random_forest_model.p','wb'))


###########  Gradient Boosting Classifier and Accuracy Calculation ###########

print('-'*30)
print('GradientBoostingClassifier')
alg_gradient_boosting_classifier = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
scores=cross_validation.cross_val_score(alg_gradient_boosting_classifier,train_data[predictors],train_data["outcome"],cv=3)
print(scores.mean())
pickle.dump(alg_gradient_boosting_classifier,open('gradient_boosting_model.p','wb'))


########### Ensemble Classifier and Accuracy Calculation ###########

print('-'*30)
print('Ensemble method')
# The algorithms we want to ensemble.
# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.
algorithms = [[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),predictors ],[LogisticRegression(random_state=1), predictors]]
# Initialize the cross validation folds
kf = KFold(train_data.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = train_data["outcome"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(train_data[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.  
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(train_data[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)
# Compute accuracy by comparing to the training data.
accuracy = sum(predictions[predictions == train_data["outcome"]]) / len(predictions)
print(accuracy)


########### Classical Classifier and Accuracy Calculation ###########

print('-'*30)
print('Classical models')

#prepare train and test data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_data[predictors],train_data["outcome"], test_size=.3)

print('-'*30)
print('Decision tree Based outputs')
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#Decison tree classifier
decison_tree_classifier = DecisionTreeClassifier()
#Training Classifier, Prediction and Accuracy Calculation
decison_tree_classifier.fit(x_train, y_train)
pickle.dump(decison_tree_classifier,open('decision_tree_classifier.p','wb'))
predictions = decison_tree_classifier.predict(x_test) 
print accuracy_score(y_test,predictions)

#Nearest Neighbour Classifier
nearest_neighbour_classifier = KNeighborsClassifier()
#Training Classifier, Prediction and Accuracy Calculation
nearest_neighbour_classifier.fit(x_train, y_train)
pickle.dump(nearest_neighbour_classifier,open('kneighbors_classifier.p','wb'))
predictions = nearest_neighbour_classifier.predict(x_test) 
print accuracy_score(y_test,predictions)


########### SVM Classifier and Accuracy Calculation ###########

print('-'*30)
print('SVM outputs')
from sklearn import svm, metrics
# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)
# We learn the digits on the first half of the digits
classifier.fit(x_train, y_train)
pickle.dump(classifier,open('svm_classifier.p','wb'))

# Now predict the value of the digit on the second half:
expected = y_test
predicted = classifier.predict(x_test)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

###############################################################