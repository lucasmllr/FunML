import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from copy import deepcopy

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


def fill(x, method='all'):
    '''initializes zero entries in x in different ways.'''

    min_x = np.min(x[np.where(x!=0)])
    filled = deepcopy(x)

    if method == 'individual':
        rat_mean = np.mean(x, axis=0)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if filled[i, j] == 0:
                    filled[i, j] = rat_mean[j]
        return filled, rat_mean

    elif method == 'all':
        total_mean = np.mean(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if filled[i, j] == 0:
                    filled[i, j] = total_mean
        return filled, total_mean

    elif method == 'all_ratings':
        mean_rating = np.mean(x[x != 0])
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if filled[i, j] == 0:
                    filled[i, j] = mean_rating
        return filled, mean_rating

    else:
        raise ValueError('no valid method for filling the blanks. choose among: all, individual, all_ratings')


def approximate(x, n_components=25, clip_value=None):
    '''approximates x using NMF with n_components, and clips values smaller than clip_value'''

    nmf = NMF(n_components)
    W = nmf.fit_transform(x)
    x_appr = nmf.inverse_transform(W)

    if clip_value:
        x_appr[x_appr < clip_value] = 0

    return x_appr


def eval_density(x):
    'returning fraction of non-zero entries in x'

    size = x.shape[0] * x.shape[1]
    density = round(np.count_nonzero(x) / size, 4)

    return density


def recommend(to_user, x, x_appr):
    '''function to find recommendations for a given user'''

    #movie ids start at 1
    user_row = to_user - 1
    seen = np.argwhere(x[user_row] != 0)
    appr = np.argwhere(x_appr[user_row] != 0)
    recom = appr[appr not in seen].squeeze()
    #movie ids start at 1
    ids = recom + 1

    rec_movies = movies.iloc[recom]
    #print(rec_movies.columns)
    rec_titles = rec_movies['movie title']
    #print(rec_titles)
    return rec_titles


def check_approximation(x):
    'plotting non-zero entries as a funcntion of column number, i.e. movie id.'

    x_ax = np.linspace(0, x.shape[1] - 1, x.shape[1])
    non_zeros = np.count_nonzero(x, axis=0)
    plt.plot(x_ax, non_zeros, label='non-zeros')
    plt.title('non zero entris for every movie')
    plt.show()

    return


if __name__ == "__main__":

    # data matrix
    fill_value = 0
    rat_df = ratings.pivot(index='user id', columns='movie id', values='rating').fillna(fill_value)
    # conversion to numpy array
    x_or = rat_df.as_matrix()

    x, mean_ratings = fill(x_or, method='individual')
    x_appr = approximate(x, n_components=20, clip_value=2)
    #check_approximation(x_appr)

    print('density of original ratings:', eval_density(x_or))
    print('density of approximation after clipping:', eval_density(x_appr))

    for user in [12]:
        rec = recommend(to_user=user, x=x_or, x_appr=x_appr)
        print('recommendations are:')
        print(rec)

