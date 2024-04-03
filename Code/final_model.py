def predict_azm(unitigs):
    return result
def predict_cfx(unitigs):
    return result

def predict_cip(unitigs):
    return result
def find_unitigs(seq):
    return unitigs_list

def model_test(seq):
    unitigs = find_unitigs(seq)
    results_azm = predict_azm(unitigs)
    results_cip = predict_cip(unitigs)
    results_cfx = predict_cfx(unitigs)

    results = []
    results.append(results_azm)
    results.append(results_cip)
    results.append(results_cfx)

    return results

