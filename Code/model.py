import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
#this program closely follows the steps delineated in the tutorial:https://www.datacamp.com/tutorial/random-forests-classifier-python

#Preprocessing
 #1. Load data
labels_azm = pd.read_csv('Data\metadata.csv',usecols=["Sample_ID","azm_sr"])

labels_azm_cfx_cip= pd.read_csv('Data\metadata.csv',usecols=["Sample_ID","azm_sr","cip_sr","cfx_sr"],)

features_azm = pd.read_csv('Data\zm_sr_gwas_filtered_unitigs.Rtab',sep=' ',index_col=0)
features_cfz =pd.read_csv('Data\cfx_sr_gwas_filtered_unitigs.Rtab',sep=' ',index_col = 0)
features_cip=pd.read_csv('Data\cip_sr_gwas_filtered_unitigs.Rtab', sep=' ', index_col=0)

print(features_azm.shape)
print(labels_azm.shape)
 
  #2. Join the features together in case there are redundant unitigs
features_azm_cfz = pd.merge(features_azm,features_cfz, how = 'outer',on= 'pattern_id')
features_azm_cfz_cip = pd.merge(features_azm_cfz,features_cip,how="outer",on='pattern_id')
transposed_features_azm_cfz_cip = features_azm_cfz_cip.transpose();


  #3. Remove N/A 
to_delete =[]
s = 0 
while s < len(labels_azm): #gives the number of rows --> number of samples
    if np.isnan(labels_azm_cfx_cip.loc[s].at["azm_sr"]) or np.isnan(labels_azm_cfx_cip.loc[s].at["cip_sr"]) or np.isnan(labels_azm_cfx_cip.loc[s].at["cfx_sr"]) :
        to_delete.append(labels_azm_cfx_cip.loc[s].at["Sample_ID"])
        labels_azm_cfx_cip.drop(index=s,inplace=True)
    s = s +1

  #35. inner join between the two dataframes to only keep complete data
joined = pd.merge(labels_azm_cfx_cip,transposed_features_azm_cfz_cip,how='inner',left_on="Sample_ID",right_index=True)

  #4. un-join to separate into X and Y
  #get list of unitigs titles so we can use the index when predicting => but remove it from the dataset itself ...or can just refer to index... hm, maybe not necessary
processed_features_azm_cfx_cip = joined.drop(['Sample_ID','azm_sr','cip_sr','cfx_sr'],axis=1)
unitigs = processed_features_azm_cfx_cip.columns

  #5 Turn separate labels into one vector label 

processed_labels_azm = joined['azm_sr']
processed_labels_cip = joined['cip_sr']
processed_labels_cfx = joined['cfx_sr']
joined_labels = []
row = 0 


while row < processed_labels_azm.size:
    temp= []
    temp.append(processed_labels_azm.iloc[row])
    temp.append(processed_labels_cip.iloc[row])
    temp.append(processed_labels_cfx.iloc[row])
    joined_labels.append(temp)
    row = row + 1
joined_labels_df = pd.DataFrame(joined_labels)

    

#Splitting the data
features_train, features_test, labels_train, labels_test = train_test_split(processed_features_azm_cfx_cip,joined_labels,test_size=0.67) 

#Training the model
rf = RandomForestClassifier()
rf.fit(features_train,labels_train)

#Save weights somewhere 

#Predict with testing set
labels_prediction = rf.predict(features_test)
print(labels_prediction)

#Converting labels into text
def text(labels_prediction):
    txt_labels_prediction = []
    sample_label = 0
    print(len(labels_prediction))
    while sample_label < len(labels_prediction):
        if np.array_equal(labels_prediction[sample_label],[0.,0.,1.]):
            txt_labels_prediction.append("Cefixime Resistant")
        elif np.array_equal(labels_prediction[sample_label],[0.,1.,0.]):
            txt_labels_prediction.append("Ciprofloxacin Resistant")
        elif np.array_equal(labels_prediction[sample_label],[1.,0.,0.]):
            txt_labels_prediction.append("Azithromycin Resistant")
        elif np.array_equal(labels_prediction[sample_label],[1.,1.,0.]):
            txt_labels_prediction.append("Azithromycin and Ciprofloxacin Resistant")
        elif np.array_equal(labels_prediction[sample_label],[0.,1.,1.]):
            txt_labels_prediction.append("Azithromycin and Cefixime Resistant")
        elif np.array_equal(labels_prediction[sample_label],[1.,0.,1.]):
            txt_labels_prediction.append("Azithromycin and Cefixime Resistant")
        elif np.array_equal(labels_prediction[sample_label],[1.,1.,1.]):
            txt_labels_prediction.append("Azithromycin, Ciprofloxacin and Cefixime Resistant")
        else:
            txt_labels_prediction.append("No Resistance")
        sample_label = sample_label + 1
    return txt_labels_prediction;
t_labels_prediction = text(labels_prediction)
print(t_labels_prediction)
t_labels_test= text(labels_test)
print(pd.crosstab(t_labels_test,t_labels_prediction,rownames=['Actual resistance'],colnames=['Predicted resistance']))
            
#Metrics
#accuracy = accuracy_score(labels_prediction, labels_test)
#precision = precision_score(labels_test, labels_prediction) 
#recall = recall_score(labels_test,labels_prediction)
#print(accuracy)
#print(precision)
#print(recall)
#c_m = confusion_matrix(labels_test,labels_prediction)
#ConfusionMatrixDisplay(confusion_matrix=c_m).plot()
#plt.show()
