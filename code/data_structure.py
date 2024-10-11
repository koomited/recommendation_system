import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from scipy.stats import norm
class dataIndexing:
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.map_user_to_idx = {}
        self.map_idx_to_user = []
        self.data_by_user_id = []

        self.map_movie_to_idx = {}
        self.map_idx_to_movie = []
        self.data_by_movie_id = []

    def get_data(self):
        with open(self.data_dir, "r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                user_id = int(row["userId"])
                movie_id = int(row["movieId"])
                rating = float(row["rating"])
                if user_id not in self.map_idx_to_user:
                    self.map_idx_to_user.append(user_id)
                    self.map_user_to_idx[user_id] = self.map_idx_to_user.index(user_id)
                    self.data_by_user_id.append([(movie_id, rating)])
                else:
                    self.data_by_user_id[self.map_idx_to_user.index(user_id)].append((movie_id, rating))

                if movie_id not in self.map_idx_to_movie:
                    self.map_idx_to_movie .append(movie_id)
                    self.map_movie_to_idx[movie_id] = self.map_idx_to_movie.index(movie_id)
                    self.data_by_movie_id.append([(user_id, rating)])
                else:
                    self.data_by_movie_id[self.map_idx_to_movie.index(movie_id)].append((user_id, rating))

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

    def plot_power_law(self):
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
        ax.set_title("Ratings: Log log plot")
        ax.legend(bbox_to_anchor=(1.06, .6), loc="center left",
                       title_fontsize=15, frameon=False)
        plt.show()


    def plot_average_rating_hist(self):
   
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(self.average_rating_per_movie(),stat="probability", bins=10, kde=True, kde_kws={"bw_adjust":3}, color="m")
        # sns.distplot(self.average_rating_per_movie(), kde=False, fit=norm, color="m")
        ax.set_xlabel("Average ratings")
        # ax.set_ylabel("Frequencies")
        ax.set_title("Average Rating per Movie")
        plt.show()

    def line_plot(self, data, xaxis, yaxis, title):

        # _, _, losses = self.alternating_least_square_biases(lambd=0.5, gamma=0.5, iterations = 10)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1,len(data)+1), data)
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)
        ax.set_title(title)

        plt.show()
    
    def alternating_least_square_biases(self, lambd=0.5, gamma=0.5, iterations = 1000):
        # number of users
        M = len(self.data_by_user_id)
        # number of items
        N = len(self.data_by_movie_id)
        user_biases = np.zeros(M)
        item_biases = np.zeros(N)

        losses = []
        rmses = []
        for i in range(iterations):
            # users biases computation
            for m in range(M):
                bias = 0
                item_counter = 0
                for n, r in self.data_by_user_id[m]:
                    bias += lambd*(r - item_biases[self.map_idx_to_movie.index(n)])
                    item_counter += 1
                bias = bias/(lambd * item_counter + gamma) 
                user_biases[m] = bias
            # items biases computation
            for n in range(N):
                bias = 0
                user_counter = 0
                for m, r in self.data_by_movie_id[n]:
                    bias += lambd*(r - user_biases[self.map_idx_to_user.index(m)])
                    user_counter += 1
                bias = bias/(lambd*user_counter + gamma) 
                item_biases[n] = bias

            loss, rmse = self.loss_function(user_biases, item_biases, lambd = lambd, gamma = gamma)
            if not i%10:
                print(f"Iteration{i}: loss = {loss}; RMSE = {rmse}")
            losses.append(loss)
            rmses.append(rmse)
        return user_biases, item_biases, losses, rmses

        
    def  loss_function(self, user_biases, item_biases, lambd = 0.5, gamma = 0.5):
        M = len(self.data_by_user_id)
        loss= -gamma/2*np.sum(user_biases**2)
        rmse_list=[]
        for m in range(M):
            user_loss=[]
            for n, r in self.data_by_user_id[m]:
                user_loss.append((r-user_biases[m]-item_biases[self.map_idx_to_movie.index(n)])**2)
                rmse_list.append((r-user_biases[m]-item_biases[self.map_idx_to_movie.index(n)])**2)
            loss+= -sum(user_loss)/2*lambd
        rmse = np.mean(rmse_list)
        return loss, rmse



                








