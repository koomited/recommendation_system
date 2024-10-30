import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from scipy.stats import norm
from collections import defaultdict
import random
from tqdm import tqdm 

np.random.seed(42)

params = {'pdf.fonttype': 3, 'axes.labelsize': 18, 'xtick.labelsize':18
, 'ytick.labelsize':18, 'legend.fontsize':18, "font.size":18}
plt.rcParams.update(params)


class AlternatingLeastSquare:
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        # Mappings for user and movie indexes
        self.map_user_to_idx = {}
        self.map_idx_to_user = []
        self.data_by_user_id = defaultdict(list)

      

        self.map_movie_to_idx = {}
        self.map_idx_to_movie = []
        self.data_by_movie_id = defaultdict(list)

        
       
        

    def data_indexing(self):
        with open(self.data_dir, "r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                user_id = int(row["userId"])
                movie_id = int(row["movieId"])
                rating = float(row["rating"])

                # Handle user indexing
                if user_id not in self.map_user_to_idx:
                    self.map_user_to_idx[user_id] = len(self.map_idx_to_user)
                    self.map_idx_to_user.append(user_id)
                user_idx = self.map_user_to_idx[user_id]

                # Handle movie indexing
                if movie_id not in self.map_movie_to_idx:
                    self.map_movie_to_idx[movie_id] = len(self.map_idx_to_movie)
                    self.map_idx_to_movie.append(movie_id)
                movie_idx = self.map_movie_to_idx[movie_id]

                # Append data for user and movie
                self.data_by_user_id[user_idx].append((movie_id, rating))
                self.data_by_movie_id[movie_idx].append((user_id, rating))
        self.data_by_movie_id = [data for data in self.data_by_movie_id.values()]
        self.data_by_user_id = [data for data in self.data_by_user_id.values()]


    def get_data_by_user_id(self, user_id):
        position = self.map_idx_to_user.index(user_id)
        user_data = self.data_by_user_id[position]
        return user_data

    def get_data_by_movie_id(self, movie_id):
        position = self.map_idx_to_movie.index(movie_id)
        movie_data = self.data_by_movie_id[position]
        return movie_data

    def average_rating_per_movie(self):
        average_rating_per_movie = []
        for movie in self.data_by_movie_id:
            rating_sum = 0
            for rating in movie:
                rating_sum+=rating[1]
            average_rating_per_movie.append(rating_sum/len(movie))
        return average_rating_per_movie

    
    def plot_power_law(self, fig_name):
        number_of_movies_per_user = [len(user_movies) for user_movies in self.data_by_user_id]
        number_of_users_per_movie = [len(movie_users) for movie_users in self.data_by_movie_id]
        minimum_number_rating = min(number_of_movies_per_user)

        users_ratings_number_frequency = {num_ratings:  number_of_movies_per_user.count(num_ratings)  for num_ratings in number_of_movies_per_user }
        movies_ratings_number_frequency = {num_ratings: number_of_users_per_movie.count(num_ratings)  for num_ratings in number_of_users_per_movie }

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(users_ratings_number_frequency.keys(),  users_ratings_number_frequency.values(),
                  marker = ".", ls = "none", color = "m", label="Users")
        ax.loglog(movies_ratings_number_frequency.keys(), movies_ratings_number_frequency.values() ,
                  marker = ".", ls = "none", color = "blue", label="Movies")
        ax.axvline(x = minimum_number_rating, color = 'r', ls="--")
        ax.text(10**1*2.3, 10**3*1.8, "Minimun ratings number", color ="r")#fontsize = 12
        ax.set_xlabel("Degree")
        ax.set_ylabel("Frequencies")
        # ax.set_title("Ratings: Log log plot")
        ax.legend(bbox_to_anchor=(1.06, .6), loc="center left", frameon=False)
        plt.savefig(f"plots/{fig_name}.pdf", format="pdf", bbox_inches="tight", dpi = 1000)
        
        plt.show()


    def plot_average_rating_hist(self, fig_name):
   
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(self.average_rating_per_movie(),stat="probability", bins=10, kde=True, kde_kws={"bw_adjust":3}, color="m")
        # sns.distplot(self.average_rating_per_movie(), kde=False, fit=norm, color="m")
        ax.set_xlabel("Average ratings")
        # ax.set_ylabel("Frequencies")
        # ax.set_title("Average Rating per Movie")
        plt.savefig(f"plots/{fig_name}.pdf", format="pdf", bbox_inches="tight", dpi=1000)

        plt.show()

    def line_plot(self, data_train, data_test, xaxis, yaxis, fig_name):

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1,len(data_train)+1), data_train,  color='blue', lw=1, label='Training')
        ax.plot(range(1,len(data_test)+1), data_test, color='red', lw=1, label='Testing')
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)
        ax.legend(bbox_to_anchor=(1.06, .6), loc="center left", frameon=False)

        plt.savefig(f"plots/{fig_name}.pdf", format="pdf", bbox_inches="tight", dpi=1000)

        plt.show()

    def plot_training_loss_only(self, losses_train, xaxis, yaxis, fig_name):

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1,len(losses_train)+1), losses_train,  color='blue', lw=1, label='Training')
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)
        # ax.legend(loc="upper right", frameon=False)
   
        # ax.set_title(title, fontsize=18, pad=20)

        plt.savefig(f"plots/{fig_name}.pdf", format="pdf", bbox_inches="tight", dpi=1000)

        plt.show()

    def plot_power_law(self, fig_name):
        number_of_movies_per_user = [len(user_movies) for user_movies in self.data_by_user_id]
        number_of_users_per_movie = [len(movie_users) for movie_users in self.data_by_movie_id]
        minimum_number_rating = min(number_of_movies_per_user)

        users_ratings_number_frequency = {num_ratings:  number_of_movies_per_user.count(num_ratings)  for num_ratings in number_of_movies_per_user }
        movies_ratings_number_frequency = {num_ratings: number_of_users_per_movie.count(num_ratings)  for num_ratings in number_of_users_per_movie }

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(users_ratings_number_frequency.keys(),  users_ratings_number_frequency.values(),
                  marker = ".", ls = "none", color = "m", label="Users")
        ax.loglog(movies_ratings_number_frequency.keys(), movies_ratings_number_frequency.values() ,
                  marker = ".", ls = "none", color = "blue", label="Movies")
        ax.axvline(x = minimum_number_rating, color = 'r', ls="--")
        ax.text(10**1*2.3, 10**3*1.8, "Minimun ratings number", color ="r")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Frequencies")
        ax.legend(bbox_to_anchor=(1.06, .6), loc="center left", frameon=False)
        plt.savefig(f"plots/{fig_name}.pdf", format="pdf", bbox_inches="tight", dpi = 1000)
        
        plt.show()


    def plot_average_rating_hist(self, fig_name):
   
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(self.average_rating_per_movie(),stat="probability", bins=10, kde=True, kde_kws={"bw_adjust":3}, color="m")
        ax.set_xlabel("Average ratings")
        plt.savefig(f"plots/{fig_name}.pdf", format="pdf", bbox_inches="tight", dpi=1000)

        plt.show()

    def line_plot(self, data_train, data_test, xaxis, yaxis, fig_name):

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1,len(data_train)+1), data_train,  color='blue', lw=1, label='Training')
        ax.plot(range(1,len(data_test)+1), data_test, color='red', lw=1, label='Testing')
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)
        ax.legend(bbox_to_anchor=(1.06, .6), loc="center left", frameon=False)

        plt.savefig(f"plots/{fig_name}.pdf", format="pdf", bbox_inches="tight", dpi=1000)

        plt.show()

    def plot_training_loss_only(self, losses_train, xaxis, yaxis, fig_name):

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1,len(losses_train)+1), losses_train,  color='blue', lw=1, label='Training')
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)


        plt.savefig(f"plots/{fig_name}.pdf", format="pdf", bbox_inches="tight", dpi=1000)

        plt.show()

    
    def alternating_least_square(self, data_user, data_movie, lambd=0.05, gamma=0.05, epochs = 1000):
        # number of users
        M = len(data_user)
        # number of items
        N = len(data_movie)
        user_biases = np.zeros(M)
        item_biases = np.zeros(N)
 
        losses = []
        rmses = []
        losses_test=[]
        rmses_test = []

        with tqdm(range(epochs), unit='epoch') as tepochs:
            tepochs.set_description('Training')
            for i in tepochs:
                # users biases computation
                for m in range(M):
                    bias = 0
                    item_counter = 0
                    for n, r in data_user[m]:
                        bias += lambd*(r - item_biases[self.map_movie_to_idx[n]]) 
                        item_counter += 1
                    bias = bias/(lambd * item_counter + gamma) 
                    user_biases[m] = bias

                # items biases computation
                for n in range(N):
                    bias = 0
                    user_counter = 0
                    for m, r in data_movie[n]:
                        bias += lambd*(r - user_biases[self.map_user_to_idx[m]]) 
                        user_counter += 1
                    bias = bias/(lambd*user_counter + gamma) 
                    item_biases[n] = bias

                loss, rmse = self.loss_rmse_function(data_user, user_biases, item_biases, lambd = lambd, gamma = gamma)
                loss_test, rmse_test = self.loss_rmse_function(self.data_by_user_test, user_biases, item_biases, lambd = lambd, gamma = gamma)
                losses.append(loss)
                rmses.append(rmse)
                losses_test.append(loss_test)
                rmses_test.append(rmse_test)
                tepochs.set_postfix(test_rmse=rmse_test)
                
        return user_biases, item_biases, losses, rmses, losses_test, rmses_test

        
    def  loss_rmse_function(self, data, user_biases, item_biases, lambd = 0.5, gamma = 0.5):
        M = len(data)
        loss= gamma*np.sum(user_biases**2)/2 + gamma*np.sum(item_biases**2)/2
        rmse_list=[]
        for m in range(M):
            ratings_loss = []
            
            for n, r in data[m]:
                ratings_loss.append((r-user_biases[m] -item_biases[self.map_movie_to_idx[n]])**2)# 
                rmse_list.append((r-user_biases[m] -item_biases[self.map_movie_to_idx[n]])**2)# 
            loss+= lambd*sum(ratings_loss)/2 
        rmse = np.sqrt(np.mean(rmse_list))
        return loss, rmse
  
    def train_test_split(self):
        self.map_user_to_idx = {}
        self.map_idx_to_user = []
        self.data_by_user_train = []
        self.data_by_user_test = []

        self.map_movie_to_idx = {}
        self.map_idx_to_movie = []
        self.data_by_movie_train = []
        self.data_by_movie_test = []

        with open(self.data_dir, "r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                user_id = int(row["userId"])
                movie_id = int(row["movieId"])
                rating = float(row["rating"])

                # Assign randomly to train or test
                flip_coin = random.random()
                is_train = flip_coin < 0.5

                # Handle user mapping
                if user_id not in self.map_user_to_idx:
                    self.map_idx_to_user.append(user_id)
                    self.map_user_to_idx[user_id] = len(self.map_idx_to_user) - 1
                    self.data_by_user_train.append([])
                    self.data_by_user_test.append([])

                user_idx = self.map_user_to_idx[user_id]

                # Handle movie mapping
                if movie_id not in self.map_movie_to_idx:
                    self.map_idx_to_movie.append(movie_id)
                    self.map_movie_to_idx[movie_id] = len(self.map_idx_to_movie) - 1
                    self.data_by_movie_train.append([])
                    self.data_by_movie_test.append([])

                movie_idx = self.map_movie_to_idx[movie_id]

                # Add to train or test set for both user and movie
                if is_train:
                    self.data_by_user_train[user_idx].append((movie_id, rating))
                    self.data_by_movie_train[movie_idx].append((user_id, rating))
                else:
                    self.data_by_user_test[user_idx].append((movie_id, rating))
                    self.data_by_movie_test[movie_idx].append((user_id, rating))

                








