import numpy as np
import torch


def euc_dist_sq(data1, data2):
    '''
    inputs:
        data1 - numpy array of data points (n1, d)
        data2 - numpy array of data points (n2, d)
    '''
    n1, d1 = data1.shape
    n2, d2 = data2.shape
    assert d1 == d2, f"the embedding dimension of data1, data2 are different {d1} != {d2}."
    d = d1
    c = np.reshape(data1,[n1,1,d]) - np.reshape(data2,[1,n2,d])
    dist_sq = np.sum(np.square(c),axis=2)
    return dist_sq

def hamming_dist_sq(data1, data2):
    '''
    inputs:
        data1 - numpy array of data points (n1, d)
        data2 - numpy array of data points (n2, d)
    '''
    n1, d1 = data1.shape
    n2, d2 = data2.shape
    assert d1 == d2, f"the embedding dimension of data1, data2 are different {d1} != {d2}."
    d = d1
    c = (np.reshape(data1,[n1,1,d]) != np.reshape(data2,[1,n2,d])) * 1.0
    dist_sq = np.square(np.sum(c,axis=2))
    return dist_sq

def kmeans_pp(data, k, dist='euclidean', init_ind=None):
    '''
    initialized the centroids for K-means++
    inputs:
        data - numpy array of data points having shape (n, d)
        k - number of clusters (k <= n)
        dist - the name of metric
        init_ind - int (if None, random init index)
    '''
    ## initialize the centroids list and add
    ## a randomly selected data point to the list
    centroids = []
    selected_indices = []
    
    if init_ind is None:
        init_ind = np.random.randint(data.shape[0])
    centroids.append(data[init_ind, :])
    selected_indices.append(init_ind)
    
    if dist == 'euclidean':
        d_sq_func = euc_dist_sq
    elif dist == 'hamming':
        d_sq_func = hamming_dist_sq

    ## compute remaining centroids
    for _ in range(k - 1):
        all_indices = list(range(data.shape[0]))
        unselected_indices = list(set(all_indices) - set(selected_indices))

        d_sq_to_centroid = d_sq_func(data[unselected_indices], data[selected_indices])
        min_d_sq_to_centroid = np.min(d_sq_to_centroid, axis=1)
        if np.sum(min_d_sq_to_centroid)==0:
            break
        #prob = min_d_sq_to_centroid / np.sum(min_d_sq_to_centroid)
        #next_centroid_ind = all_indices[np.random.choice(unselected_indices, p=prob)]
        next_centroid_ind = unselected_indices[ 
                    np.argmax(min_d_sq_to_centroid)
                ]

        selected_indices.append(next_centroid_ind)

        centroids.append(data[next_centroid_ind, :])
    return np.array(centroids), selected_indices




