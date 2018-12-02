# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 11:02:21 2018
@author: Chithra Menon
"""

#import relevant libraries
import os.path
import time
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise import accuracy
from surprise.model_selection import KFold
from collections import defaultdict

#print start time of the model
print('Start time:: '+ time.asctime( time.localtime(time.time())))

#define precision and recall functions
def precision_recall_at_k(predictions, k=10, threshold=4):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls 


#function which returns the top n recommendations   
def recommend_movies(p1, userId, movies_df,ratings_df, num_recommendations):
    
    #select rows specific to userId from predicted file
    user_data = p1[p1.userId == userId]
    
    #merge movies_df file with user_data file for userId
    user_full= user_data.astype(dict(movieId=int)).merge(movies_df.astype(dict(movieId=int)), 'left')
    
    #select movies not already rated by user 
    subtracted = movies_df[~movies_df['movieId'].isin(user_full['movieId'])]
    
    #select movies already rated by user
    rated = ratings_df[ratings_df.userId == (userId)]
    already_rated = (rated.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').sort_values(['rating'], ascending=False))
  
    #for loop to update 'estimate' column
#    for i, row in subtracted.iterrows():
#        estimate1 = (algo.predict(userId,row.movieId)).est
#        subtracted.loc[i,'estimate']  = estimate1
        
    for row in subtracted.itertuples(index=True, name='Pandas'):
        estimate1 = (algo.predict(userId,row.movieId)).est
        subtracted.loc[row.Index,'estimate'] = estimate1
            
    #sort the file based on estimated rating and select top 10 ratings
    sub =(subtracted.sort_values(by='estimate',ascending=False).iloc[:num_recommendations, :-1])
    
    return sub, already_rated

# Read 1Million dataset 
dir_path = os.path.abspath(os.path.dirname(__file__)) 
input_path = os.path.join(dir_path,'data')
ratings_list = [i.strip().split("::") for i in open(os.path.join(input_path, "ratings.dat")).readlines()]
users_list = [i.strip().split("::") for i in open(os.path.join(input_path,"users.dat")).readlines()]
movies_list = [i.strip().split("::") for i in open(os.path.join(input_path,"movies.dat")).readlines()]    
#ratings_list = [i.strip().split("::") for i in open(r'C:\Users\Chithra Menon\Downloads\ratings.dat').readlines()]
#users_list = [i.strip().split("::") for i in open(r'C:\Users\Chithra Menon\Downloads\users.dat').readlines()]
#movies_list = [i.strip().split("::") for i in open(r'C:\Users\Chithra Menon\Downloads\movies.dat').readlines()]


#Convert datasets into arrays
ratings = np.array(ratings_list)
users = np.array(users_list)
movies = np.array(movies_list)

#Convert arrays into dataframes
ratings_df = pd.DataFrame(ratings_list, columns = ['userId', 'movieId', 'rating', 'timestamp'], dtype = 'int')
movies_df = pd.DataFrame(movies, columns = ['movieId', 'title', 'genres'])
movies_df['movieId'] = movies_df['movieId'].apply(pd.to_numeric)
#movies_df['estimate']=""

## Read 100K dataset
#ratings= pd.read_csv(r'C:\Users\Chithra Menon\Downloads\ratings20M.csv')
#ratings_df = pd.DataFrame(ratings, columns = ['userId', 'movieId', 'rating', 'timestamp'], dtype = int)
#
#movies_df= pd.read_csv(r'C:\Users\Chithra Menon\Downloads\movies.csv')
#movies_df['movieId'] = movies_df['movieId'].apply(pd.to_numeric)
#movies_df['estimate']=""

#drop unnecessary columns and ready it for the surprise package
ratings_df.drop(columns=['timestamp'])
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

raw_ratings = data.raw_ratings

# Split dataset into independent testset and trainset
# A = 30% of the data, B = 70% of the data
threshold = int(.3 * len(raw_ratings))
A_raw_ratings = raw_ratings[:threshold]
B_raw_ratings = raw_ratings[threshold:]


#Assign 70% of total dataset to be the trainset
data.raw_ratings = B_raw_ratings  # data is now the set B

trainset = data.build_full_trainset()
 

#initialise variables to calculate mean rmse, precision and recall after 5x2 CV
sum_rmse = 0
sum_precision = 0
sum_recall=0

#define the algorithm for the model
algo=SVD(n_factors=200, n_epochs=50, lr_all=0.007, reg_all = 0.01,
         init_std_dev=0.1)  

#Iterate 5 times over the 2 Folds of training dataset
for iterations in range(5):
    
    print("Iteration %2d:" % (iterations+1))
    
    # Split the dataset in 2 folds with KFold
    
    kf = KFold(n_splits=2, shuffle = True)
    
    for fold, (trainset, testset) in enumerate(kf.split(data)):
        algo.fit(trainset)
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions, verbose= False)
        precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=3.5)
        print("Fold %2d rmse, precision, recall: %0.6f %0.6f %0.6f" % 
              (fold+1, rmse, 
              (sum(prec for prec in precisions.values()) / len(precisions)),
              (sum(rec for rec in recalls.values()) / len(recalls))))
        sum_rmse = sum_rmse + rmse
        sum_precision = sum_precision + (sum(prec for prec in precisions.values()) / len(precisions))
        sum_recall = sum_recall + (sum(rec for rec in recalls.values()) / len(recalls))


#calculate mean rmse, precision and recall after 5x2 CV
mean_rmse = sum_rmse/10
mean_precision = sum_precision/10
mean_recall = sum_recall/10

print("Mean training rmse, precision, recall: %0.6f %0.6f %0.6f" % (mean_rmse, mean_precision, mean_recall))

# retrain on the whole train set 
trainset = data.build_full_trainset()                                       
algo.fit(trainset) 

# Compute unbiased accuracy on independent testset
testset = data.construct_testset(A_raw_ratings)  # testset is now the set A
predictions = algo.test(testset)
print('Independent test data accuracy,', end=' ')
accuracy.rmse(predictions)


#convert testet predictions file into dataframe
predict=np.array(predictions)
pred=pd.DataFrame(predict)
pred.columns=['userId','movieId','rating','estimate','details']
p1=pred.drop(columns=['rating','details'])

#display actual and estimated side by side for selected user
p2= pred.drop(columns=['details'])
user_datap2 = p2[p2.userId == 2]
user_fullp2= user_datap2.astype(dict(movieId=int)).merge(movies_df.astype(dict(movieId=int)), 'left')
user_fullp2.to_csv(r'C:\Users\Chithra Menon\Downloads\user_fullp2.csv',sep=',') 

#Add an estimate column to the movies_df file to get estimated ratings
movies_df['estimate']=""

#function call for new predictions for user
recommended, already_rated = recommend_movies(p1, 2, movies_df, ratings_df, 10)           
      
#save output prediction files to path    
out_dir =   os.path.join(dir_path,'predictions') 
recommended.to_csv(os.path.join(out_dir,'recommended.csv'),sep=',') 
already_rated.to_csv(os.path.join(out_dir,'already_rated.csv'),sep=',')

#print the end time for model 
print('End time::'+time.asctime( time.localtime(time.time())))
