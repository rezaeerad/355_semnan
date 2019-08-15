#!/usr/bin/env python
from sklearn.externals import joblib
import numpy as np
from numpy import dot
from numpy.linalg import norm

def get_sepsis_score(data, model):

    num_rows = len(data)

    s_m = np.load('all_mean.npy', allow_pickle=True)
    ns_m = np.load('all_mean.npy', allow_pickle=True)
    s_m = np.delete(s_m, [40])
    ns_m = np.delete(ns_m, [40])

    # data = np.delete(data, [34, 35, 36, 37, 38], 1)
    flag = 0
    if np.isnan(data[0, 0]):
        flag = 1

    for i in range(40):
        for j in range(np.size(data, 0)):
            if np.isnan(data[j, i]):
                if j>0:
                    before = data[0:(j),i]
                    before = before[~np.isnan(before)]
                else:
                    before = []

                if np.size(before)>0:
                    data[j, i] = before[-1]
                else:
                    d = np.nan_to_num(data[j, :])
                    cos_s = dot(d, s_m) / (norm(d) * norm(s_m))
                    cos_ns = dot(d, ns_m) / (norm(d) * norm(ns_m))
                    if flag == 1 and j==0:
                        data[j, i] = ns_m[i]
                    else:
                        if cos_s > cos_ns:
                            data[j, i] = s_m[i]
                        else:
                            data[j, i] = ns_m[i]


    M1 = joblib.load('model-saved.pkl')
    # predicted = M1.predict(data)

    ####### End Impute
    if num_rows==1:
        label = 0
        score = 0.4
    else:
        predicted = M1.predict(data)
        if predicted[num_rows - 1] == 0:
            score = 0.4
        else:
            score = 0.6
        label = predicted[num_rows - 1]

    # ###################################

    # return score_final, label_final


    return score, label

def load_sepsis_model():

    return None
