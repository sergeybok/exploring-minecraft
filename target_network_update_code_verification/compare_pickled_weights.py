import pickle
import numpy as np

fp1 = open('curiosity_model/target_network_saved_weights.pickle', 'r')
fp2 = open('curiosity_model/target_network_saved_weights_with_scope_method.pickle', 'r')
a = pickle.load(fp1)
b = pickle.load(fp2)
for i in range(len(a)):
    if(np.mean(a[i]-b[i])<0.000002):
        print('match')
    else:
        print('no match')