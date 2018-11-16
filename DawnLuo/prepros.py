import random
from random import shuffle
import numpy as np

def index_to_movieId():
    index_movieId = {}
    count = 0
    with open('./ml-1m/movies.dat', encoding='latin-1') as f:
        for line in f:
            line = line.strip().split('::')
            index_movieId.update({count: line[0]})
            count = count + 1
    return index_movieId

def movieId_to_dic():
    count = 0
    movieId_dic = {}
    # movieId :: name :: genres
    with open('./ml-1m/movies.dat', encoding='latin-1') as f:
        for line in f:
            line = line.strip().split('::')
            movieId_dic.update({line[0]: count})
            count = count +1
    return movieId_dic

def movieName_to_dic():
    count = 0
    movieName_dic = {}
    # movieId :: name :: genres
    with open('./ml-1m/movies.dat', encoding='latin-1') as f:
        for line in f:
            line = line.strip().split('::')
            movieName_dic.update({line[1]:count})
            count = count+1
    return movieName_dic

def movieId_genre_dic():
    movieId_genre = {}
    # movieId :: name :: genres
    with open('./ml-1m/movies.dat', encoding='latin-1') as f:
        for line in f:
            line = line.strip().split('::')
            movieId_genre.update({line[0]: line[2].split('|')})
    return movieId_genre

def get_train_test_set(test_size=0.2):
    user_movieId_pos = {}
    user_movieId_neg = {}
    user_movieId_pos_test = {}

    movieId_Dic = movieId_to_dic()
    count = 0
    # userId :: movieId :: rating :: timestamp
    with open('./ml-1m/ratings.dat', encoding='latin-1') as f:
        for line in f:
            line = line.strip().split('::')
            movieId_rating = set()
            movieId_rating = (movieId_Dic[line[1]],line[2])
            if (float(line[2]) < 3):
                user_movieId_neg.setdefault(int(line[0]), []).append(movieId_rating)
            elif (float(line[2]) > 3):
                if(random.random() <= test_size):
                    user_movieId_pos_test.setdefault(int(line[0]),[]).append(movieId_rating)
                else:
                    user_movieId_pos.setdefault(int(line[0]), []).append(movieId_rating)
    
    # delete user with few movie ratings
    filter_threshold = 4
    count = 0
    for key, value in user_movieId_pos.copy().items():
        if(len(value) < filter_threshold):
            count = count + 1
            del(user_movieId_pos[key])
            if key in user_movieId_pos_test:
                del(user_movieId_pos_test[key])
        else:
            shuffle(user_movieId_pos[key])
    print("delete user with few movie ratings: ", count)

    # delete user in test set but not in training set
    for key, value in user_movieId_pos_test.copy().items():
        if key not in user_movieId_pos:
            del(user_movieId_pos_test[key])
    return user_movieId_pos, user_movieId_neg, user_movieId_pos_test

def user_gender_age_occup_dic():
    user_gender = {} # M: 0, F: 1
    user_age = {}
    user_occup = {}
    # userId :: gender :: age :: occupation :: zip-code
    with open('./ml-1m/users.dat', encoding='latin-1') as f:
        for line in f:
            line = line.strip().split("::")
            if(line[1] == 'M'):
                user_gender.update({int(line[0]):0})
            else:
                user_gender.update({int(line[0]):1})
            
            user_age.update({int(line[0]): int(line[2])/56})
            user_occup.update({int(line[0]): int(line[3])})
    return user_gender, user_age, user_occup

def user_genre_dic():
    user_genre = {}
    user_movieId_pos, user_movieId_neg, user_movieId_pos_test = get_train_test_set()
    movieId_genre = movieId_genre_dic()
    index_movieId = index_to_movieId()
    for key, value in user_movieId_pos.items():
        genre_count = np.zeros(18)
        for index in value:
            id = index_movieId[index[0]]
            genres = movieId_genre[id]
            for genre in genres:
                if(genre == "Action"):
                    genre_count[0] += 1
                elif(genre == "Adventure"):
                    genre_count[1] += 1
                elif(genre == "Animation"):
                    genre_count[2] += 1
                elif(genre == "Children's"):
                    genre_count[3] += 1
                elif(genre == "Comedy"):
                    genre_count[4] += 1
                elif(genre == "Crime"):
                    genre_count[5] += 1
                elif(genre == "Documentary"):
                    genre_count[6] += 1
                elif(genre == "Drama"):
                    genre_count[7] += 1
                elif(genre == "Fantasy"):
                    genre_count[8] += 1
                elif(genre == "Film-Noir"):
                    genre_count[9] += 1
                elif(genre == "Horror"):
                    genre_count[10] += 1
                elif(genre == "Musical"):
                    genre_count[11] += 1
                elif(genre == "Mystery"):
                    genre_count[12] += 1
                elif(genre == "Romance"):
                    genre_count[13] += 1
                elif(genre == "Sci-Fi"):
                    genre_count[14] += 1
                elif(genre == "Thriller"):
                    genre_count[15] += 1
                elif(genre == "War"):
                    genre_count[16] += 1
                elif(genre == "Western"):
                    genre_count[17] += 1
        genre_count = np.divide(genre_count, np.sum(genre_count))
        user_genre.update({key: genre_count})
    return user_genre, user_movieId_pos, user_movieId_neg, user_movieId_pos_test