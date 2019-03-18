import numpy as np
import pandas as pd

def variance(data, feature_name):
    return np.var(data[feature_name])

def fisher_score(data, feature_name, class_label='class'):
    feature = data[feature_name]
    m = np.mean(feature)
    
    numerator = 0
    denominator = 0
    for c in set(data[class_label]):
        n_i = len(data[data[class_label]==c])
        m_i = np.mean(data[data[class_label]==c][feature_name])
        v_i = np.var(data[data[class_label]==c][feature_name])
        numerator += n_i * pow(m_i - m, 2)
        denominator += n_i * v_i
        
    return numerator/denominator

def rank_features(data, features, method='variance'):
    rankings = dict.fromkeys(features)
    
    if method == 'variance':
        for f in features:
            rankings[f] = variance(data, f)
    elif method == 'fisher_score':   
        for f in features:
            rankings[f] = fisher_score(data, f)
    else:
        raise Exception('Invalid argument value!')
        
    f_ranking_tuple = [(feature, rankings[feature]) for feature in sorted(rankings, key=rankings.get, reverse=True)]
    
    return f_ranking_tuple
