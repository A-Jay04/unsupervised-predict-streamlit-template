"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pickle
# data science imports
import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# utils import
from fuzzywuzzy import fuzz

ratings = pd.read_csv('resources/data/ratings.csv')
ratings.drop('timestamp', axis=1, inplace=True)
movies = pd.read_csv('resources/data/movies.csv')

def fuzzy_matching(mapper, fav_movie, verbose=True):
    """
    return the closest match via fuzzy ratio.
    match will always be found due to adding selection
    to the training data.
    
    Note: code has been edited from online sources. See notebook
    for details.
    
    Parameters
    ----------    
    mapper: dict, map movie title name to index of the movie in data

    fav_movie: str, name of user input movie
    
    verbose: bool, print log if True

    Return
    ------
    index of the closest match
    """
    match_tuple = []
    # get match
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]

    return match_tuple[0][1]

def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):
    """
    return top n similar movie recommendations based on user's input movie

    Note: code has been edited from online sources. 
    See notebook for details.

    Parameters
    ----------
    model_knn: sklearn model, knn model (trained)

    data: movie-user matrix

    mapper: dict, map movie title name to index of the movie in data

    fav_movie: str, name of user input movie

    n_recommendations: int, top n recommendations

    Return
    ------
    list of top n similar movie recommendations
    """

    # get input movie index
    idx = fuzzy_matching(mapper, fav_movie, verbose=True)
    # inference
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
    # get list of raw idx of recommendations
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # save recommendations
    rec = []
    for i, (idx, dist) in enumerate(raw_recommends):
        rec.append(reverse_mapper[idx])
    return rec

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """

    # load the ratings file. loading each time ensures that each selection
    # is seen as a new user.
    ratings = pd.read_csv('resources/data/ratings.csv')
    ratings.drop('timestamp', axis=1, inplace=True)
    
    # get rating frequency
    df_movies_cnt = pd.DataFrame(ratings.groupby('movieId').size(), columns=['count'])

    # filter data via movies
    # change the amount of ratings a movie needs to stay in the system
    popularity_thres = 24
    popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))
    df_ratings_drop_movies = ratings[ratings.movieId.isin(popular_movies)]

    # get number of ratings given by every user
    df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('userId').size(), columns=['count'])

    # filter ratings via users
    # change the number to remove users who rate too little
    ratings_thres = 20
    active_users = list(set(df_users_cnt.query('count >= @ratings_thres').index))
    df_ratings_drop_users = df_ratings_drop_movies[df_ratings_drop_movies.userId.isin(active_users)]

    #reload csv of movies
    movies = pd.read_csv('resources/data/movies.csv')
    movies = movies.set_index('movieId')

    # add the selection of movies to the training set
    # this is a fix for the cold start issue and the user 
    # who made the selection has now been added to the system
    df_ratings_drop_users = df_ratings_drop_users.append({'userId':672, 'movieId': movies[movies.title == movie_list[0]].index[0], 'rating': 5 } , ignore_index=True)
    df_ratings_drop_users = df_ratings_drop_users.append({'userId':672, 'movieId': movies[movies.title == movie_list[1]].index[0], 'rating': 5 } , ignore_index=True)
    df_ratings_drop_users = df_ratings_drop_users.append({'userId':672, 'movieId': movies[movies.title == movie_list[2]].index[0], 'rating': 5 } , ignore_index=True)
    df_ratings_drop_users = df_ratings_drop_users.drop_duplicates() 

    # pivot and create movie-user matrix
    movie_user_mat = df_ratings_drop_users.pivot(index='movieId', columns='userId', values='rating').fillna(0)

    # create mapper from movie title to index
    movie_to_idx = {
        movie: i for i, movie in 
        enumerate(list(movies.loc[movie_user_mat.index].title)) #check index hasnt been made already
    }
    
    # transform matrix to scipy sparse matrix
    movie_user_mat_sparse = csr_matrix(movie_user_mat.values)
    
    #initiate the KNN model and fit the data
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
    model_knn.fit(movie_user_mat_sparse)

    final_list = []

    #make individual recommendations for each selected movie

    rec1 =   make_recommendation(
            model_knn=model_knn,
            data=movie_user_mat_sparse,
            fav_movie=movie_list[0],
            mapper=movie_to_idx,
            n_recommendations=15)

    rec2 =   make_recommendation(
            model_knn=model_knn,
            data=movie_user_mat_sparse,
            fav_movie=movie_list[1],
            mapper=movie_to_idx,
            n_recommendations=15)

    rec3 =   make_recommendation(
            model_knn=model_knn,
            data=movie_user_mat_sparse,
            fav_movie=movie_list[2],
            mapper=movie_to_idx,
            n_recommendations=15)

    # append all recommendations to one list 
    for i in rec1:
        final_list.append(i)
    for i in rec2:
        final_list.append(i)
    for i in rec3:
        final_list.append(i)

    final_f = []
    
    # ensure the recommendations provided do not include the
    # selected movies or duplicated items
    for i in final_list:
        if i not in movie_list:
            final_f.append(i)

    final_f = list(dict.fromkeys(final_f))
    
    return final_f[0:10]
