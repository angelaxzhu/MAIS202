import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

#Preprocessing
 #1. Load data
labels = pd.read_csv('Data\metadata.csv',usecols=["Sample_ID","azm_sr"])
features = pd.read_csv('Data\zm_sr_gwas_filtered_unitigs.Rtab',sep=' ')
print(features.shape)
print(labels.shape)
 #2. Remove N/A 
to_delete =[]
s = 0 
while s < len(labels): #gives the number of rows --> number of samples
    if np.isnan(labels.loc[s].at["azm_sr"]):
        to_delete.append(labels.loc[s].at["Sample_ID"])
        labels.drop(index=s,inplace=True)
    s = s +1
transposed_features= features.transpose();

  #3. inner join between the two dataframes to only keep complete data
joined = pd.merge(labels,transposed_features,how='inner',left_on="Sample_ID",right_index=True)


  #4. un-join to separate into X and Y
processed_labels = joined['azm_sr']
processed_features = joined.drop(['Sample_ID','azm_sr'],axis=1)

#Splitting the data
features_train, features_test, labels_train, labels_test = train_test_split(processed_features,processed_labels,test_size=0.67) 

#Training the model
rf = RandomForestClassifier()
rf.fit(features_train,labels_train)

#Predict with testing set
labels_prediction = rf.predict(features_test)

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
