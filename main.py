import numpy as np
import pandas as pd

from filter_methods import *

attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
class_label = ['class']
data = pd.read_table('data/iris.data', sep=',', names=attributes+class_label)

print(rank_features(data, attributes, 'variance'))
print(rank_features(data, attributes, 'fisher_score'))