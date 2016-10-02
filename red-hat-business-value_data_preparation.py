'''
@author : github.com/nitish11
@date : August 7th, 2016
@description : Base code for https://www.kaggle.com/c/predicting-red-hat-business-value/data
'''

import pandas as pd
import pickle

# function to covert non-numeric data into numeric values
def convert_data(non_numeric,key):
    # Use key as a separator to extract numeric value from the data
    if str(non_numeric).isdigit():
    	numeric_value=non_numeric
    else:
	    try:
	    	numeric_value = non_numeric.split(key)[1]
	    except:
	    	print "----",non_numeric
    return numeric_value

#Getting the people data
people_data = pd.read_csv('people.csv')
# print people_data.head(5)
# print people_data.describe()
people_data = people_data*1
# print people_data.dtypes
pickle.dump(people_data,open('people_data.p','wb'))


#Getting activity train data
activity_train_data = pd.read_csv('act_train.csv')
# print activity_train_data.head(5)
# print activity_train_data.describe()
activity_train_data["activity_id"] = [convert_data(x,'_') for x in activity_train_data["activity_id"]]
activity_train_data["activity_category"] = [convert_data(x,' ') for x in activity_train_data["activity_category"]]
pickle.dump(activity_train_data,open('activity_train_data.p','wb'))


#Getting activity test data
activity_test_data = pd.read_csv('act_test.csv').head(10)
pickle.dump(activity_test_data,open('activity_test_data.p','wb'))
print activity_test_data.head(5)
print activity_test_data.describe()

#Merge activity and people data on people_id as key
train_data = pd.merge(activity_train_data, people_data, how='outer', on=None, left_on=['people_id'], right_on=['people_id'], left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True)
# print train_data.head(5)

#Fill missing data
#train_data = train_data.fillna(0)
predictors = ["people_id","activity_id","activity_category","char_1_x","char_2_x","char_3_x", "char_4_x","char_5_x","char_6_x","char_7_x","char_8_x","char_9_x", "char_10_x","char_1_y","group_1","char_2_y","char_3_y","char_4_y","char_5_y","char_11","char_12","char_13","char_14" ,"char_15","char_16","char_17","char_18","char_19","char_20","char_21","char_22","char_23", "char_24","char_25","char_26","char_27","char_28","char_29","char_30","char_31","char_32", "char_33","char_34","char_35","char_36","char_37","char_38"]

for feature in predictors:
    print "Updating ",feature
    train_data[feature] = train_data[feature].fillna(train_data[feature].median())
# print train_data.head(5)


#Replace False with 0 and True with 1

#Change non_numeric data to numeric data 
train_data["people_id"] = [convert_data(x,'_') for x in train_data["people_id"]]
print train_data.head(5)

print train_data["group_1"]
train_data["group_1"] = [convert_data(x,' ') for x in train_data["group_1"]]
print train_data.head(5)

#Removing type features
for x in range(1,100):
    pattern = "type "+str(x)
    train_data = train_data.where(train_data.values != pattern,x)
print train_data.head(5)

#Removing date columns
train_data = train_data.convert_objects(convert_numeric=True)
print train_data.dtypes

#Removing date columns
train_data.drop("date_x",axis=1,inplace=True)
train_data.drop("date_y",axis=1,inplace=True)
print train_data.head(5)

#Dumping the train data
pickle.dump(train_data,open('train_data.p','wb'))

print('-'*30)
print('Data preparation done')

