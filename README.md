[![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?logo=visual-studio-code&logoColor=white)](#)
[![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)](#)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](#)
[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](#)
[![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)](#)
![Awesome](https://img.shields.io/badge/Awesome-ffd700?logo=awesome&logoColor=black)


<a href="#">
  <img width="100%" src="https://editor.analyticsvidhya.com/uploads/76889recommender-system-for-movie-recommendation.jpg" alt="Cedit">
</a>

# Build a recommendation system from scratch: Movilens data

## Overview
A major challenge in recommendation systems is the occurrence of fallacious recommendations that are completely irrelevant to the user’s preferences, which can negatively impact user experience and satisfaction. This study employs one of the most effective approaches for generating relevant recommendations: \textit{Collaborative Filtering}, combined with the Alternating Least Squares (ALS) algorithm for Matrix Factorization to make recommendations of movies for new user.

## Data
This study is based on MovieLens datasets the smallest (ml-latest-small) contains $100836$ ratings across $9742$ movies and 
the large(ml-25m) contains $25000095$ ratings across 62423 movies (Available at http://grouplens.org/datasets). Users were selected at random for inclusion. All selected users had rated at least 20 movies. In order to test the model, we further split these datasets into training and test sets where each user appear in both datasets but with different number of ratings for the same user on different movies.

## Tasks Performed

1. **Implemente a personnalized data structure:**

   The movies ratings and user are written in array and dictionnaries format to allow for quick search and model fitting optimization.
   This is really important to dela with 25 millions observations.

3. **Power law and ratings distributions**

4. **First Model: Biases only**
   
  Here we work on the assumptions that only the users and items biases explain the ratings distribution.

6. **Second model: Biases + embeddings**
   
   We add the users and items embeddings to the biases.

8. **PLot movies embeddings**
  
9. **Deployment: Streamlit**
    
   - Streamlit app : https://movielens-recommender-system.streamlit.app/






