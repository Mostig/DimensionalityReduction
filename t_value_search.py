from LaplacianScore import LaplacianScore
import numpy as np
import pandas as pd

attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
class_label = ['class']
data = pd.read_table('data/iris.data', sep=',', names=attributes+class_label)

print("-------------Laplacian Score--------------------")
print("-------------t value search---------------------")

results = []
correct_values = []

t_values = np.arange(-100, 100, 1)
for t in t_values:
    print('---test value: ', t, '---')
    l = LaplacianScore(k_nearest_neighbors=5, constant_t=t, denominator_value=0.0001)
    l.fit(data.iloc[:, 0:4])
    results += l.feature_ranking_tuples
    if  l.features_ordered[0:2] == [2, 3] or  l.features_ordered[0:2] == [3, 2]:
        correct_values += t
        print('found!')
        print('\tresult: ', l.features_ordered)
    
        
        