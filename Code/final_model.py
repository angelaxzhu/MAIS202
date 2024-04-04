import joblib
import numpy as np
import pandas as pd

#note: find is not the best algorithm cuz it can include overlapping sequences... 

def predict_azm(seq):
    unitigs = find_unitigs(seq,"azm")
    arr_unitigs=np.array(unitigs)
    twoarr_unitigs = arr_unitigs.reshape(1,-1)
    loaded_rf = joblib.load("./Model/random_forest_azm.joblib")
    result = loaded_rf.predict(twoarr_unitigs)
    if result == 1:
        result_w = "Resistant"
    elif result == 0:
        result_w = "Not Resistant"
    else:
        print("what") 
    return result_w

def predict_cfx(seq):
    unitigs = find_unitigs(seq,"cfx")
    arr_unitigs=np.array(unitigs)
    twoarr_unitigs = arr_unitigs.reshape(1,-1)
    loaded_rf = joblib.load("./Model/random_forest_cfx.joblib")
    result = loaded_rf.predict(twoarr_unitigs)
    if result == 1:
        result_w = "Resistant"
    elif result == 0:
        result_w = "Not Resistant"
    else:
        print("what") 
    return result_w

def predict_cip(seq):
    unitigs = find_unitigs(seq,"cip")
    arr_unitigs=np.array(unitigs)
    twoarr_unitigs = arr_unitigs.reshape(1,-1)
    loaded_rf = joblib.load("./Model/random_forest_cip.joblib")
    result = loaded_rf.predict(twoarr_unitigs)
    if result == 1:
        result_w = "Resistant"
    elif result == 0:
        result_w = "Not Resistant"
    else:
        print("what") 
    return result_w

def find_unitigs(seq,ab):
    unitigs_list = []
    if ab == "azm":
        unitigs = []
        #csv to list
        unitigs_file = open(r'.\Data\azm_unitigs.txt','r')
        #remove linebreak
        for line in unitigs_file:
            uni = line[:-1]
            unitigs.append(uni)

        for u in unitigs:
            if seq.find(u) == -1: 
                unitigs_list.append(0)
            else:
                unitigs_list.append(1)

    elif ab == "cfx":
        unitigs = []
        #csv to list
        unitigs_file = open(r'.\Data\cfx_unitigs.txt','r')
        #remove linebreak
        for line in unitigs_file:
            uni = line[:-1]
            unitigs.append(uni)

        for u in unitigs:
            if seq.find(u) == -1: 
                unitigs_list.append(0)
            else:
                unitigs_list.append(1)

    elif ab =="cip": 
        unitigs = []
        #csv to list
        unitigs_file = open(r'.\Data\cip_unitigs.txt','r')
        #remove linebreak
        for line in unitigs_file:
            uni = line[:-1]
            unitigs.append(uni)

        for u in unitigs:
            if seq.find(u) == -1: 
                unitigs_list.append(0)
            else:
                unitigs_list.append(1)

    return unitigs_list

def model_test(seq):
    results_azm = predict_azm(seq)
    results_cip = predict_cip(seq)
    results_cfx = predict_cfx(seq)

    results = []
    results.append(results_azm)
    results.append(results_cip)
    results.append(results_cfx)

    return results

#make test sample
def generate_test():
    seq=""
    unitigs_l_cip = pd.read_csv('Data\cip_sr_gwas_filtered_unitigs.Rtab',sep=' ',index_col=0)
    unitigs_l_azm = pd.read_csv('Data\zm_sr_gwas_filtered_unitigs.Rtab',sep=' ',index_col=0)
    unitigs_l_cfx = pd.read_csv('Data\cfx_sr_gwas_filtered_unitigs.Rtab',sep=' ',index_col=0)
    unitigs_list_cip = (unitigs_l_cip['SRR1661209']).tolist()
    unitigs_list_azm = (unitigs_l_azm['SRR1661209']).tolist()
    unitigs_list_cfx = (unitigs_l_cfx['SRR1661209']).tolist()
    unitigs_n_cip = unitigs_l_cip.index.values.tolist()  
    unitigs_n_azm = unitigs_l_azm.index.values.tolist()  
    unitigs_n_cfx = unitigs_l_cfx.index.values.tolist()  
    unitigs_list = unitigs_list_cip + unitigs_list_azm + unitigs_list_cfx
    unitigs_n = unitigs_n_cip + unitigs_n_azm + unitigs_n_cfx
    #print('names',unitigs_n)
    i = 0
    while i < len(unitigs_list):
        if unitigs_list[i]==1:
            seq = seq + unitigs_n[i]
        i = i + 1
    
    s = open(r'.\Data\test_sequence.txt','w')
    s.write(seq)
    s.close()

    return seq

generate_test()