# PMF_ratings
Implementation of a recommender system using probabilistic matrix factorization

# Introduction

Matching customers to products is an important practical problem that many companies would be interested in being able to do well. Products information provided by users can be used to develop recommendation systems to help recommend to customers new products that they may like.

Ratings prediction is a common application of such models and is explored in this project. Starting with a database of users/objects ratings, a matrix factorization model is used to predict customers ratings about products they had not rated yet depending on:
- how they rated other products,
- how other customers had rated any of the products.


A rating dataset(1) from the MovieLens website is used for this project.
It contains 100,000 ratings applied to 9,000 movies by 700 users.
Ratings scores are given between 0.5 and 5.

(1) https://grouplens.org/datasets/movielens/
