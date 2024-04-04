import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import joblib


#this program closely follows the steps delineated in the tutorial:https://www.datacamp.com/tutorial/random-forests-classifier-python
def train_azm():
    #1. Load data
    labels_azm = pd.read_csv('Data\metadata.csv',usecols=["Sample_ID","azm_sr"])
    features_azm = pd.read_csv('Data\zm_sr_gwas_filtered_unitigs.Rtab',sep=' ',index_col=0)


    #2. Remove N/A 
    n_labels_azm=labels_azm.dropna()

    #3. inner join between the two dataframes to only keep complete data
    transposed_features_azm = features_azm.transpose();
    joined = pd.merge(n_labels_azm,transposed_features_azm,how='inner',left_on="Sample_ID",right_index=True)


    #4. un-join to separate into X and Y
    #get list of unitigs titles so we can use the index when predicting => but remove it from the dataset itself ...or can just refer to index... hm, maybe not necessary

  
    processed_labels = joined['azm_sr']
    processed_features = joined.drop(['Sample_ID','azm_sr'],axis=1)
    unitigs = processed_features.columns

    #Splitting the data
    features_train, features_test, labels_train, labels_test = train_test_split(processed_features,processed_labels,test_size=0.67) 
    

    #Training the model
    rf = RandomForestClassifier()
    rf.fit(features_train,labels_train)
    joblib.dump(rf,"./random_forest_azm.joblib")


    #Predict with testing set
    labels_prediction = rf.predict(features_test)
    print(labels_prediction)

    #Metrics
    accuracy = accuracy_score(labels_prediction, labels_test)
    precision = precision_score(labels_test, labels_prediction)
    recall = recall_score(labels_test,labels_prediction)
    print(accuracy)
    print(precision)
    print(recall)
    c_m = confusion_matrix(labels_test,labels_prediction)
    ConfusionMatrixDisplay(confusion_matrix=c_m).plot()
    plt.show()
    
    #save unitigs
    with open(r'.\Data\azm_unitigs.txt','w') as file:
        for unitig in unitigs:
            file.write("%s\n" % unitig)
    file.close()
    
    return unitigs

def train_cip():
     #1. Load data
    labels_azm = pd.read_csv('Data\metadata.csv',usecols=["Sample_ID","cip_sr"])
    features_azm = pd.read_csv('Data\cip_sr_gwas_filtered_unitigs.Rtab',sep=' ',index_col=0)


    #2. Remove N/A 
    n_labels_azm=labels_azm.dropna()

    #3. inner join between the two dataframes to only keep complete data
    transposed_features_azm = features_azm.transpose();
    joined = pd.merge(n_labels_azm,transposed_features_azm,how='inner',left_on="Sample_ID",right_index=True)


    #4. un-join to separate into X and Y
    #get list of unitigs titles so we can use the index when predicting => but remove it from the dataset itself ...or can just refer to index... hm, maybe not necessary

  
    processed_labels = joined['cip_sr']
    processed_features = joined.drop(['Sample_ID','cip_sr'],axis=1)
    unitigs = processed_features.columns

    #Splitting the data
    features_train, features_test, labels_train, labels_test = train_test_split(processed_features,processed_labels,test_size=0.67) 
    

    #Training the model
    rf = RandomForestClassifier()
    rf.fit(features_train,labels_train)

    joblib.dump(rf,"./random_forest_cip.joblib")

    #Predict with testing set
    labels_prediction = rf.predict(features_test)
    print(labels_prediction)

    #Metrics
    accuracy = accuracy_score(labels_prediction, labels_test)
    precision = precision_score(labels_test, labels_prediction)
    recall = recall_score(labels_test,labels_prediction)
    print(accuracy)
    print(precision)
    print(recall)
    c_m = confusion_matrix(labels_test,labels_prediction)
    ConfusionMatrixDisplay(confusion_matrix=c_m).plot()
    plt.show()

    #save unitigs
    with open(r'.\Data\cip_unitigs.txt','w') as file:
        for unitig in unitigs:
            file.write("%s\n" % unitig)
    file.close()

    return unitigs
def train_cfx():
     #1. Load data
    labels_cfx = pd.read_csv('Data\metadata.csv',usecols=["Sample_ID","cfx_sr"])
    features_cfx = pd.read_csv('Data\cfx_sr_gwas_filtered_unitigs.Rtab',sep=' ',index_col=0)


    #2. Remove N/A 
    to_delete =[]
    s = 0 

    #while s < len(labels_cfx): #gives the number of rows --> number of samples WRONG!! REMOVING MESSES UP THE INDICES!!
        #if np.isnan(labels_cfx.loc[s].at["cfx_sr"]) :
          #  to_delete.append(labels_cfx.loc[s].at["Sample_ID"])
         #   labels_cfx.drop(index=s,inplace=True)
        #s = s +1
    n_labels_cfx=labels_cfx.dropna()
 
    #3. inner join between the two dataframes to only keep complete data
    transposed_features_azm = features_cfx.transpose();
    joined = pd.merge(n_labels_cfx,transposed_features_azm,how='inner',left_on="Sample_ID",right_index=True)


    #4. un-join to separate into X and Y
    #get list of unitigs titles so we can use the index when predicting => but remove it from the dataset itself ...or can just refer to index... hm, maybe not necessary

  
    processed_labels = joined['cfx_sr']
    processed_features = joined.drop(['Sample_ID','cfx_sr'],axis=1)
    unitigs = processed_features.columns

    #Splitting the data
    features_train, features_test, labels_train, labels_test = train_test_split(processed_features,processed_labels,test_size=0.67) 
   
    labels_train2=labels_train.tolist()

    
    #Training the model
    rf_cfx = RandomForestClassifier()
    rf_cfx.fit(features_train,labels_train)  
    joblib.dump(rf_cfx,"./random_forest_cfx.joblib")    


    #Predict with testing set
    labels_prediction = rf_cfx.predict(features_test)
    print(labels_prediction)
    

    #Metrics
    accuracy = accuracy_score(labels_prediction, labels_test)
    precision = precision_score(labels_test, labels_prediction)
    recall = recall_score(labels_test,labels_prediction)
    print(accuracy)
    print(precision)
    print(recall)
    c_m = confusion_matrix(labels_test,labels_prediction)
    ConfusionMatrixDisplay(confusion_matrix=c_m).plot()
    plt.show()
    #save unitigs
    with open(r'.\Data\cfx_unitigs.txt','w') as file:
        for unitig in unitigs:
            file.write("%s\n" % unitig)
    file.close()
    return unitigs
   
train_azm()
train_cip()
train_cfx()