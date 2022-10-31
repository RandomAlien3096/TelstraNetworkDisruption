import numpy as np
import pandas as pd
import os
import seaborn as sns

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']= 300
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from catboost import CatBoostClassifier, Pool
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#------------------Loading dato set------------------
datadir = 'data'
def str_to_num(string):
    return int(string.split(" ")[1])    #after the split returns the num on the string

#loading test.csv file
test = pd.read_csv('test.csv', 
                   converters = {'location':str_to_num})

#loading train.csv file
train = pd.read_csv('train.csv',
                   converters = {'location':str_to_num})

#loading event_type.csv file
event_type = pd.read_csv('event_type.csv', 
                         converters = {'event_type':str_to_num})

#loading log_feature.csv file
log_failure = pd.read_csv('log_feature.csv', 
                          converters = {'log_feature':str_to_num})

#loading resource_type.csv file
resource_type = pd.read_csv('resource_type.csv', 
                            converters = {'resource_type':str_to_num})

#loading severity_type.csv file
severity_type = pd.read_csv('severity_type.csv', 
                            index_col = 'id',
                            converters = {'severity_type':str_to_num})

#------------------Printing the shape of each data set------------------
print("The size of the test data set is: {}\n".format(test.shape))
print("The size of the train data set is: {}\n".format(train.shape))
print("The size of the events data set is: {}\n".format(event_type.shape))
print("The size of the log feautures data set is: {}\n".format(log_failure.shape))
print("The size of the resource data set is: {}\n".format(resource_type.shape))
print("The size of the severity type data set is: {}\n".format(severity_type.shape))

print("Headers of train data set")
train.head()

print("Headers of test data set")
test.head()

train_1 = train.merge(severity_type, how = 'left', left_on='id', right_on='id')
train_2 = train_1.merge(resource_type, how = 'left', left_on='id', right_on='id')
train_3 = train_2.merge(log_failure, how = 'left', left_on='id', right_on='id')
train_4 = train_3.merge(event_type, how = 'left', left_on='id', right_on='id')
#checking the head after merging

print("The size of the train dataset is: {} ".format(train_4.shape))

print("Deleting duplicates in data set...")
train_4.drop_duplicates(subset= 'id', keep= 'first', inplace = True)

#checking the shape of training file after dropping duplicate records
print("The size of the train dataset is now: {} ".format(train_4.shape))
train_4.head()

#Plotting the amount of faults in the network
plt.figure(figsize = (8,6))
ax = sns.countplot(train_4['fault_severity'])
for i in ax.patches:
    ax.annotate('{:.1f}'.format(i.get_height()), (i.get_x()+ 0.25, i.get_height() + 0.1))
plt.title("Count of fault_severity")
plt.show()

#Plotting a heat map of the fault severity
plt.figure(figsize = (12,12))
sns.set()
sns.heatmap(train_4.corr(), 
            vmax = 0.8, 
            linewidths= 0.01, 
            square= True, annot= True, 
            cmap= 'viridis', 
            linecolor= 'white')
plt.title("Heatmap of Fault Serveity Data")
plt.show()

#loading test.csv file
test_plot = pd.read_csv('test.csv', 
                        index_col = 'id',
                        converters = {'location':str_to_num})

#loading train.csv file
train_plot = pd.read_csv('train.csv',
                         index_col = 'id',
                         converters = {'location':str_to_num})

df = train_plot.append(test_plot)

fig, ax = plt.subplots(figsize=(15,15))
plt.title("Scatter plot Location v ID")
ax.scatter(df.loc[df.fault_severity.isnull(),'location'],
           df.loc[df.fault_severity.isnull()].index, alpha=0.5, color='k', s=2)

ax.scatter(df.loc[df.fault_severity==0,'location'],
           df.loc[df.fault_severity==0].index, alpha=0.5, color='g')

ax.scatter(df.loc[df.fault_severity==1,'location'],
           df.loc[df.fault_severity==1].index, alpha=0.5, color='y')

ax.scatter(df.loc[df.fault_severity==2,'location'], 
           df.loc[df.fault_severity==2].index, alpha=0.5, color='r')

ax.set_xlim((-20,1150))
ax.set_ylim((0,19000))
ax.set_xlabel('Location')
ax.set_ylabel('ID');

#Splitting X(data in array) and y (index of data used)

X = train_4[['id', 'location', 'severity_type', 'resource_type',
       'log_feature', 'volume', 'event_type']]
y = train_4.fault_severity
 
#divide the training set into train/validation set with 25% set aside for validation. 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
categorical_features_indices = np.where(X_train.dtypes == object)[0]

#using pool to make the training and validation sets
train_dataset = Pool(data=X_train,
                     label=y_train,
                     cat_features=categorical_features_indices)

eval_dataset = Pool(data=X_test,
                    label=y_test,
                    cat_features=categorical_features_indices)

#initialize the catboost classifier
model = CatBoostClassifier(iterations=1000,
                           learning_rate=1,
                           depth=2,
                           loss_function='MultiClass',
                           random_seed=3,
                           bagging_temperature=22,
                           od_type='Iter',
                           metric_period=50,
                           od_wait=100)
#Fit model
model.fit(train_dataset, eval_set= eval_dataset, plot= True)
#predicts the actual label or class over the evaluation data set (gives us the final choice)
preds_class = model.predict(eval_dataset) 

#get predicted probabilities for each class (gives us the probabilities of each choice option that it had)
preds_proba = model.predict_proba(eval_dataset)
print('Probabilities of each Label')
print(preds_proba)

print("The shape of the test data set without merging is: {}".format(test.shape))

test_1 = test.merge(severity_type, how = 'left', left_on='id', right_on='id')
test_2 = test_1.merge(resource_type, how = 'left', left_on='id', right_on='id')
test_3 = test_2.merge(log_failure, how = 'left', left_on='id', right_on='id')
test_4 = test_3.merge(event_type, how = 'left', left_on='id', right_on='id')

#removing the duplicates.
test_4.drop_duplicates(subset= 'id', keep= 'first', inplace = True)
 
#checking for any null values. 
test_4.isnull().sum()

print("The shape of the merged test dataset is: {}".format(test_4.shape))

predict_class = model.predict(test_4)
prediction_class_df = pd.DataFrame(predict_class, 
                                   columns = ['fault_severity'])

prediction_catboost = pd.concat([test[['id','location']], prediction_class_df],axis = 1)
prediction_catboost.to_csv('prediction.csv', index = False, header = True)
prediction_catboost.head(15)

predict_test = model.predict_proba(test_4) #using the trained catboost model to get the probabilities of the choices
print("The shape of the prediction test dataset is now: {}".format(predict_test.shape))

pred_df = pd.DataFrame(predict_test, 
                       columns = ['predict_0','predict_1', 'predict_2'])
print("The shape of the prediction data frame is now: {}".format(pred_df.shape))

submission_cat = pd.concat([test[['id']],pred_df],axis=1)
submission_cat.to_csv('sub_cat_1.csv',index=False,header=True)
submission_cat.head(15)

#loading test.csv file
prediction_plot = pd.read_csv('prediction.csv', 
                        index_col = 'id')

fig, ax = plt.subplots(figsize=(15,15))
plt.title("Scatter plot Location v ID")
ax.scatter(prediction_catboost.loc[prediction_catboost.fault_severity.isnull(),'location'],
           prediction_catboost.loc[prediction_catboost.fault_severity.isnull()].index, alpha=0.5, color='k', s=2)

ax.scatter(prediction_catboost.loc[prediction_catboost.fault_severity==0,'location'],
           prediction_catboost.loc[prediction_catboost.fault_severity==0].index, alpha=0.5, color='g')

ax.scatter(prediction_catboost.loc[prediction_catboost.fault_severity==1,'location'],
           prediction_catboost.loc[prediction_catboost.fault_severity==1].index, alpha=0.5, color='y')

ax.scatter(prediction_catboost.loc[prediction_catboost.fault_severity==2,'location'],
           prediction_catboost.loc[prediction_catboost.fault_severity==2].index, alpha=0.5, color='r')

ax.set_xlim((-20,1150))
ax.set_ylim((0,11500))
ax.set_xlabel('Location')
ax.set_ylabel('ID');

model.get_feature_importance()