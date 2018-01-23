import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

# pandas Dataset
ratings_cols = ['user id','movie id','rating','timestamp']
movies_cols = ['movie id','movie title','release date', 'video release date','IMDb URL','unknown','Action',
               'Adventure','Animation','Childrens','Comedy','Crime', 'Documentary','Drama','Fantasy',
               'Film - Noir','Horror', 'Musical','Mystery','Romance','Sci -Fi','Thriller','War','Western']
users_cols = ['user id','age','gender','occupation','zip code']

users = pd.read_csv('ml-100k/u.user', sep='|',
                      names=users_cols, encoding='latin-1')
movies = pd.read_csv('ml-100k/u.item', sep='|',
                        names=movies_cols, encoding='latin-1')
ratings = pd.read_csv('ml-100k/u.data', sep='\t',
                        names=ratings_cols, encoding='latin-1')

#data matrix
fill_value = 0
rat_df = ratings.pivot(index='user id', columns='movie id', values='rating').fillna(fill_value)
#conversion to numpy array
x = rat_df.as_matrix()

#approximate x using NMF
nmf = NMF(n_components=25)
x_appr = nmf.fit_transform(x)

#evaluate sparsity for different number of prototypes
def eval_num_prototypes(x, n_components=[400, 600]):
    size = x.shape[0] * x.shape[1]
    densities = []
    print('number of users {}'.format(x.shape[0]))
    print('original density:', round(np.count_nonzero(x) / size, 4))
    for n in n_components:
        nmf = NMF(n_components=n)
        x_appr = nmf.fit_transform(x)
        sparsity = round(np.count_nonzero(x_appr) / size, 4)
        densities.append(sparsity)
        print('density for {} prototypes: {}'.format(n, sparsity))
    return n_components, densities

n, s = eval_num_prototypes(x)

#TODO: try initializing with non-zero fill_value and clip after approximation