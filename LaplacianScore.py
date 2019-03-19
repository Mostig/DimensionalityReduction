import numpy as np
import pandas as pd
import math 
import sys

class LaplacianScore:
    
    def __init__(self, k_nearest_neighbors=5, constant_t=1, denominator_value=0.0001):
        self.k_nearest_neighbors = k_nearest_neighbors
        self.constant_t = constant_t
        self.denominator_value = denominator_value
        
    def fit(self, data):
        self.features_num = len(data.columns)
        self.graph = self.neighbor_graph(data)
        self.weight_matrix = self.weight_matrix(data)
        self.diagonal = self.diagonal(len(data))
        self.graph_laplacian = self.graph_laplacian()
        self.feature_ranking_tuples = self.features_rankings(data)
        
        self.features_ordered = [f_r[0] for f_r in self.feature_ranking_tuples]
        self.features_names = [i for i in data.columns]
        self.features_ordered_names = [self.features_names[i] for i in self.features_ordered]
        
        
    def neighbor_graph(self, data):
        """
        data doesn't contain class column
        for each example find k nearest neighbors 
        """
        graph = np.zeros(dtype=np.bool, shape=(len(data), len(data))) # zeros & bool -> all False
    
        for i in range(len(data)):
            distance_from_i = np.zeros(dtype=np.float64, shape=len(data))
            
            for j in range(len(data)):
                if i is not j:
                    for f in range(len(data.columns)):
                        distance_from_i[j] += pow(data.iloc[i, f] - data.iloc[j, f], 2)
                        
                    distance_from_i[j] = math.sqrt(distance_from_i[j])
                else:
                    distance_from_i[j] = sys.float_info.max
                    
            index_value_tuple = [(index_distance[0], index_distance[1]) for index_distance in sorted(enumerate(distance_from_i), key=lambda x: x[1])]
            
            for j in range(self.k_nearest_neighbors):
                graph[i][index_value_tuple[j][0]] = graph[index_value_tuple[j][0]][i] = True
            
        return graph

    def weight_matrix(self, data):
        matrix = np.zeros(shape=(len(data), len(data)))
        
        for i in range(len(data)):
            for j in range(i+1): # to limit calculations
                if self.graph[i][j]:
                    distance = 0
                    for f in range(self.features_num):
                        distance += pow(data.iloc[i, f] - data.iloc[j, f], 2)
                    matrix[i][j] = matrix[j][i] = pow(math.e, -distance/self.constant_t)
                else:
                    matrix[i][j] = 0
        
        return matrix

    def diagonal(self, size):
        matrix = np.zeros(shape=(size, size))
        
        for i in range(size):
            matrix[i][i] = sum(self.weight_matrix[i])
                
        return matrix

    def estimated_feature_matrix(self, data, feature_num):    
        # compute nominator
        # transpose identity matrix * diagonal matrix * identity matrix
        f_r = data.iloc[:, feature_num].values
        numerator = sum(np.matmul(f_r, self.diagonal))
        
        # compute denominator
        # transpose identity matrix * diagonal matrix * identity matrix
        denominator = 0
        for i in range(len(self.diagonal)):
            denominator += self.diagonal[i][i]
            
        if denominator == 0:
            result = numerator / self.denominator_value
        else:
            result = numerator / denominator
        result = result * np.ones(shape=(len(data)))
        
        return f_r - result
    
    def graph_laplacian(self):
        return self.diagonal - self.weight_matrix


    def features_rankings(self, data):
        rankings = dict.fromkeys([i for i in range(self.features_num)])
        
        for f_num in range(self.features_num):
            est_feature_matrix = self.estimated_feature_matrix(data, f_num)
            
            numerator = np.matmul(est_feature_matrix, self.graph_laplacian)
            numerator = np.matmul(numerator, est_feature_matrix)

            denominator = np.matmul(est_feature_matrix, self.diagonal)
            denominator = np.matmul(denominator, est_feature_matrix)
            
            if denominator != 0:
                rankings[f_num] = numerator / denominator
            else:
                rankings[f_num] = numerator / self.denominator_value
                
        return [(feature, rankings[feature]) for feature in sorted(rankings, key=rankings.get, reverse=True)]
        