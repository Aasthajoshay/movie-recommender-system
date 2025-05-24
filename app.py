import pickle
import streamlit as st
import requests
import pandas as pd



def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    recommended_movies=[]
    for i in movies_list:
        # fetch the movie poster
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies



movies_dict= pickle.load(open('movie_dict.pkl','rb'))
movies=pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl','rb'))
st.header('MovieFlick: Movie Recommender System')


movie_list = movies['title'].values
selected_movie_name=st.selectbox(
    'Type or select a movie from the dropdown',movie_list)

if st.button('Show Recommendation'):
   recommendations=recommend(selected_movie_name)
   for i in recommendations:
       st.write(i)


