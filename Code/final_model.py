import joblib
import numpy as np
import pandas as pd

#note: find is not the best algorithm cuz it can include overlapping sequences... 

def predict_azm(seq):
    unitigs = find_unitigs(seq,"azm")
    print(unitigs)
    arr_unitigs=np.array(unitigs)
    print("arr",arr_unitigs)
    twoarr_unitigs = arr_unitigs.reshape(1,-1)
    print("reshape",twoarr_unitigs)
    loaded_rf = joblib.load("./random_forest_azm.joblib")
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
    loaded_rf = joblib.load("./random_forest_cfx.joblib")
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
    loaded_rf = joblib.load("./random_forest_cip.joblib")
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

#make test sample for azm 
def generate_test():
    seq=""
    unitigs_l = pd.read_csv('Data\zm_sr_gwas_filtered_unitigs.Rtab',sep=' ',index_col=0)
    unitigs_list = (unitigs_l['SRR1661154']).tolist()
    unitigs_n = unitigs_l.index.values.tolist()  
    i = 0
    while i < len(unitigs_list):
        if unitigs_list[i]==1:
            seq = seq + unitigs_n[i]
        i = i + 1

    return seq

