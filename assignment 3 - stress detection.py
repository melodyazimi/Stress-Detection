#!/usr/bin/env python
# coding: utf-8

# In[2]:


# pip install flirt


# In[1]:


# pip install xgboost


# In[5]:


pip install numpy numba pandas scikit-learn tensorflow matplotlib xgboost flirt


# In[20]:


import pandas as pd
import os
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import math
import flirt
import numpy as np
from datetime import datetime as dt, timedelta
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer 
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, precision_score, recall_score, f1_score, classification_report, mean_squared_error, mean_absolute_error, make_scorer, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_validate, LeaveOneGroupOut, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[21]:


#all user files - they have the same pattern
#create a folder called zips containing all these files
zips = sorted(glob.glob('S*_E4_Data.zip'))
print(zips)


# In[22]:


###Data Preprocessing 

all_dfs = []
subjects = []

#find all the quest files
for filename in sorted(glob.glob('S*_quest.csv')):
    new_df = pd.read_csv(filename, header = None)
    
    #data comes as one column - split by ; for new columns 
    new_df = new_df[0].str.split(';', expand = True)
    #subject number
    subject = new_df.loc[0][1]
    print(new_df)
    
    #grab the row with the actual columns names
    new_df.columns = new_df.iloc[1]
    
    #remove unneeded columns and reindex the df
    new_df = new_df.drop(new_df.index[[0, 1]])
    new_df = new_df.reset_index(drop = True)
    
    #just want to look at columns base and tsst
    detect_df = new_df[['Base', 'TSST']]
    
    all_dfs.append(detect_df) #contains start and end times for base and TSST 
    subjects.append(subject)


# In[23]:


start_ends = {}
#zip together the subject and its associated df to match the start and end times
for sub, times in zip(subjects, all_dfs):
    #grab the start times for both Base and TSST
    start = times.iloc[0]
    end = times.iloc[1]
    
    #assign the base start time and end time
    base_start, base_end = start.loc['Base'], end.loc['Base']

    #assign the stress start and end time
    stress_start, stress_end = start.loc['TSST'], end.loc['TSST']
    
    #build a dictionary - add the starts and ends to their prospective subject 
    start_ends[sub] = {(base_start, base_end, stress_start, stress_end)}
    


# In[24]:


def get_stress(zips, start_ends):
    task_times_dict = {}
    total_features_list = []

    for zip_file in zips:
        zip_subject = zip_file.split("_")[0]

        #get features using a 60 second window with a 10 second step
        features = flirt.simple.get_features_for_empatica_archive(zip_file, 60, 10, True, True, False)
        
        #data cleaning/preproccessing
        features.index = pd.to_datetime(features.index).tz_localize(None)
        features = features.drop(['eda_phasic_entropy', 'eda_tonic_entropy'], axis=1)
#         feature_cols = feature.columns
#         features[feature.columns] = features.fillna(feature_cols, df[columns_to_fill].mean())
        

        #grab the first time stamp
        first_time_period = features.index[0]
        #dt_time_period = pd.to_datetime(first_time_period, '%Y-%m-%d %H-%M-%S')

        #the actual start time is 60 seconds before the first timestamp
        actual_first_time = first_time_period - timedelta(seconds = 60)


        #For each subject get the rows associated with base and TSST times
        if zip_subject in start_ends:
            task_times_dict[zip_subject] = [] #create empty dictionary value for each subject
            time_values = start_ends[zip_subject]

            for times in time_values:
                for timestamp in times:
                    time = float(timestamp)
                    minutes = int(time)
                    seconds = int(round(time - minutes, 2)*100)
                    total_time = timedelta(minutes = minutes, seconds = seconds)
                    #get exact times for the tasks done (base and TSST) based on the start time for each subject
                    task_time = actual_first_time + total_time

                    task_times_dict[zip_subject].append(task_time)

        #find the rows for base activity (no stress)
        filtered_features_base = ((features.index >= task_times_dict[zip_subject][0]) & 
                                  (features.index <= task_times_dict[zip_subject][1]))

        #find the rows for TSST activity (stress)
        filtered_features_tsst = ((features.index >= task_times_dict[zip_subject][2]) & 
                                  (features.index <= task_times_dict[zip_subject][3]))

        #make a new column and make it 1 or 0 depending on if its a base of stress activity
        features["stress"] = np.nan
        features.loc[filtered_features_base, "stress"] = 0
        features.loc[filtered_features_tsst, "stress"] = 1

        #drop the rows that aren't associated with base and tsst activites
        features_df = features[features["stress"].notnull()]

        features_df["subject"] = zip_subject #add subject number to associated task rows

        #add the dataframe to a final df list
        total_features_list.append(features_df)
        
        

    #concatenate all the dataframes in the list
    total_features_df = pd.concat(total_features_list, ignore_index = True)
    
    return(total_features_df)

    


# In[25]:


#concat all dataframes from every subject together
features_df = get_stress(zips, start_ends)


# In[26]:


##Training Classifiers
X = features_df.drop(columns = ["stress", "subject"])

# Replace positive and negative infinity values with NaNs
X.replace([np.inf, -np.inf], np.nan, inplace=True)
# Impute missing values (NaNs) with the mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

y = features_df["stress"] #labels

#groups will be the subjects
groups = features_df["subject"]


# In[27]:


##Logistic Regression

#scale the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2)

#train the model on log regression
logreg_model = LogisticRegression(max_iter=1000, solver='lbfgs')
logreg_model.fit(X_train, y_train)

#build predictions
log_y_preds = logreg_model.predict(X_test)


#classification report
class_report_log = classification_report(y_test, log_y_preds)
print("Classification Report:")
print(class_report_log)


# In[28]:


##Logistic regression - Leave one subject out
logo = LeaveOneGroupOut()
logo_preds = cross_val_predict(logreg_model, x_scaled, y, cv = logo, groups = groups)

#Classification Report
class_report_logo =classification_report(y, logo_preds)
print('Classification Report:')
print(class_report_logo)

#Roc Auc Score
roc_auc = cross_val_score(logreg_model, x_scaled, y, cv=logo, scoring = 'roc_auc', groups=groups)
print('Average AUROC Score:', (np.mean(roc_auc)))


# In[29]:


##Random Forest

#create train and test splits - not scaled 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#build and train the random forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

#make predictions
rf_preds = rf_model.predict(X_test)

#Build classification report
rf_class_report = classification_report(rf_preds, y_test)
print("Classification Report:")
print(rf_class_report)


# In[30]:


##XGBoost

#build and train the model
xgb_model = xgboost.XGBClassifier(use_label_encoder = False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

#make predictions
xgb_preds = xgb_model.predict(X_test)

#build classification report
class_report_xgb = classification_report(y_test, xgb_preds)
print("Classification Report:")
print(class_report_xgb)


# In[31]:


##Keras Tensorflow - Neural Network

#use scaled data to create train and test split
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2)

#number of features in the data
feature_shape = X_train.shape[1]

#build keras model
model = Sequential() #function to build layers 
#add the fully connected layers with 64, 32 and 1 number of neurons
#use of Dense for classification model
model.add(Dense(64, activation = "relu", input_shape = (feature_shape, )))
model.add(Dense(32, activation = "relu"))
#sigmoid creates a binary classification - one neuron for single output
model.add(Dense(1, activation = "sigmoid"))
model.compile(optimizer="adam", loss = "binary_crossentropy", metrics = ["accuracy"])


#fit the model
model_fit = model.fit(X_train, y_train, epochs = 20, batch_size = 64, validation_split = 0.2, verbose = 0)

#evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose = 0)
print(test_accuracy)

#make predictions
keras_preds = model.predict(X_test)
#make predictions into binary probabilities
bins = (keras_preds > 0.5).astype(int)


keras_class = classification_report(y_test, bins)
print("Classification Report:")
print (keras_class)


# <!-- #features_df['time'] = features_df['datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")
# 
# #assign the first time available 
# first_dt = features_df.iloc[0]['datetime']
# 
# #bs = base_start.replace(".", ":")
# #base_s = "00:" + "0" + bs
# #b_mins = int(base_start.split(".")[0])
# #b_secs = int(base_start.split(".")[1])
# 
# first_time = dt.strftime(first_dt, "%Y-%m-%d %H:%M:%S")
# time_f = dt.strptime(first_time, "%Y-%m-%d %H:%M:%S")
# 
# #Change the base start into a proper timestamp
# #base_stimestamp = pd.Timestamp(base_s).strftime("%Y-%m-%d %H:%M:%S")
# #baseline_s = dt.strptime(base_stimestamp, "%Y-%m-%d %H:%M:%S")
# 
# 
# #change baseline start into a datetime type
# #baseline_start = np.datetime64(baseline_s)
# 
# #Need to fix this ASAP -- convert to seconds 
# start_limit = time_f + timedelta(minutes = int(b_st[0]), seconds = int(b_st[1]))
# 
# #end_limit = time_f + timedelta(minutes = 26, seconds = 32)
# 
# #stress_s_limit = time_f + timedelta(minutes = 39, seconds = 55)
# #stress_e_limit = time_f + timedelta(minutes = 50, seconds = 30)
# 
# print(end_limit)
# print(first_time)
# print(time_f) -->
