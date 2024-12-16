import numpy as np
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
#if i precompute Vaj - va
#then apply mask - i can do mat mult 

users_matidx_to_data_map = {} #might need a 2 way map (2 maps)
users_data_to_matidx_map = {} 
movies_matidx_to_data_map = {} #might need a 2 way map (map idx in mat to number given in the dataset)
movies_data_to_matidx_map = {}

user_matidx = 0
movie_matidx = 0

vote_matrix = np.zeros((28978, 1821)) #n users x m movies
weight_matrix = np.zeros((28978, 28978)) #n users x n users


with open('D:\\school\\7-1\\CS-6375-ML\\proj3\\netflix\\TrainingRatings.txt', 'r') as file:
    for line in file:  #MovieID,UserID,Rating 
        rating = line.split(",")
        rating[0], rating[1], rating[2] = int(rating[0]), int(rating[1]), float(rating[2])
        
        if rating[1] not in users_data_to_matidx_map:
            users_data_to_matidx_map[rating[1]] = user_matidx
            users_matidx_to_data_map[user_matidx] = rating[1]
            user_matidx += 1
        
        if rating[0] not in movies_data_to_matidx_map:
            movies_data_to_matidx_map[rating[0]] = movie_matidx
            movies_matidx_to_data_map[movie_matidx] = rating[0]
            movie_matidx += 1

        vote_matrix[users_data_to_matidx_map[rating[1]]][movies_data_to_matidx_map[rating[0]]] = rating[2]

print("Blank matrices made")

user_mean_list = []
for x in tqdm(range(user_matidx), desc="Calculating mean votes"):
    user_vote_mean = np.sum(vote_matrix[x]) / np.count_nonzero(vote_matrix[x])
    user_mean_list.append(user_vote_mean)
user_mean_list = np.array(user_mean_list)
np.savetxt('user_means.txt', user_mean_list, delimiter=',', fmt='%f')
user_mean_list_T = np.atleast_2d(user_mean_list).T

#create difference matrix
#subtract from vote matrix the user mean list (column vector)
non_zero_mask = vote_matrix != 0
vote_diff_matrix = np.where(non_zero_mask, vote_matrix - user_mean_list_T, 0)



for a in tqdm(range(user_matidx), desc="Calculating weights"):
    for i in range(a, user_matidx):
        #einsum to compute the numerator and left/right components
        common_vote_mask = (vote_matrix[a] != 0) & (vote_matrix[i] != 0)

        
        a_diff = np.where(common_vote_mask, vote_diff_matrix[a], 0)
        i_diff = np.where(common_vote_mask, vote_diff_matrix[i], 0)
        
        top = np.einsum('j,j->', a_diff, i_diff)
        left = np.einsum('j,j->', a_diff, a_diff)
        right = np.einsum('j,j->', i_diff, i_diff)

        weight_matrix[a, i] = weight_matrix[i, a] = top / math.sqrt(left * right) if left > 0 and right > 0 else 0

np.savetxt('weight_matrix.txt', weight_matrix, delimiter=',', fmt='%f')

def get_prediction(a, j):
    if vote_matrix[a, j] != 0:
        print("User a has already rated film j!")
        return
    
    pred = user_mean_list[a]
    
    #all users who have rated movie `j`
    user_indices = vote_matrix[:, j].nonzero()[0]
    
    vote_diffs = vote_matrix[user_indices, j].flatten() - np.array([user_mean_list[i] for i in user_indices])
    
    weights = weight_matrix[a, user_indices].flatten()
    kappa = np.einsum('i->', np.abs(weights))

    add = np.einsum('i,i->', weights, vote_diffs)
    if kappa == 0:
        return pred 
    kappa = 1/ kappa
    pred += kappa * add
    return pred

se = 0.0
ae = 0.0
counter = 0
with open('D:\\school\\7-1\\CS-6375-ML\\proj3\\netflix\\TestingRatings.txt', 'r') as file:
    lines = file.readlines()
for line in tqdm(lines, desc="Making predictions", unit="rating"):
    rating = line.split(",")
    rating[0], rating[1], rating[2] = int(rating[0]), int(rating[1]), float(rating[2])
    predicted_rating = get_prediction(users_data_to_matidx_map[rating[1]], movies_data_to_matidx_map[rating[0]])
    se += (rating[2] - predicted_rating) ** 2
    ae += abs(rating[2] - predicted_rating)
    counter += 1

mae = float(ae) / counter
mse = float(se) / counter
rmse = math.sqrt(mse)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

