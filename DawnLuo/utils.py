import numpy as np
from prepros import *
from param import *

user_gender, user_age , user_occupation = user_gender_age_occup_dic()
user_genre, user_movieId_pos, user_movieId_neg, user_movieId_pos_test = user_genre_dic()
movieName_dic = movieName_to_dic()
#user_movieId_pos, user_movieId_neg, user_movieId_pos_test = get_train_test_set()

def read_data(pos, batch_size, data_lst, neg_lst):  # data_lst = u_mid_pos: {use:(mid,rate)}
    batch = {}
    i = pos
    ## SHUFFLE USER
    for key, value in data_lst.copy().items():
        keys.add(key)
        batch.update({key: value})
        del [data_lst[key]]
        pos += 1
        if (pos >= i + batch_size):
            break

    x = np.zeros((batch_size, max_window_size))
    y = np.zeros((batch_size, n_classes), dtype=float)
    ##feature: age and gender
    feature = np.zeros((batch_size,feature_size))
    ##occupation:
    occupation = np.zeros((batch_size, 1))
    ##genre:
    genre = np.zeros((batch_size, genre_size))


    word_num = np.zeros((batch_size))

    line_no = 0

    for key, value in batch.items():
        col_no_x = 0
        col_no_y = 0

        gender = np.zeros(1)
        gender[0] = user_gender[key]

        age = np.zeros(1)
        age[0] = user_age[key]

        occupation[line_no][:] = user_occupation[key]

        genre[line_no][:] = user_genre[key]

        tmp = np.concatenate([gender, age])
        feature[line_no][:] = tmp

        odd = 0
        for i in value:
            # update y: one hot encoding for y has five labels
            if(len(value) < y_size + 4):
                index = int(i[0])
                if(odd % 2 == 0):
                    x[line_no][col_no_x] = index
                    col_no_x += 1
                    x_label.setdefault(key, set()).add(index)
                    odd += 1
                else:
                    y[line_no][index] = 1
                    col_no_y += 1
                    # store in y_label:
                    y_label.setdefault(key, set()).add(index)
                    odd += 1


            else:
                if (col_no_y < y_size):
                    index = int(i[0])
                    y[line_no][index] = 1
                    col_no_y += 1
                    # store in y_label:
                    y_label.setdefault(key,set()).add(index)

                # update x
                else:
                    index = int(i[0])
                    # y[line_no][index] = 1
                    x[line_no][col_no_x] = index
                    col_no_x += 1
                    # store x label
                    x_label.setdefault(key, set()).add(index)

                if col_no_x >= max_window_size:
                    break

		# add negative samples:  set one hot encoding for negative sample = -1
        if key in neg_lst:
            count = 0
            for i in neg_lst[key]:
                index = int(i[0])
                y[line_no][index] = -0.5
                if(count > y_size*3):
                    break
                neg_label.setdefault(key, set()).add(index)
                count = count + 1

        #print("x",x[line_no])
        #print("col_no_x", col_no_x)
        #print("y", y[line_no])
        word_num[line_no] = col_no_x
        line_no += 1

    return x, y, word_num.reshape(batch_size, 1), feature, occupation, genre

#read_data(0, batch_size, user_movieId_pos, user_movieId_neg)
def read_data_test(pos, batch_size, data_lst, neg_lst):  # data_lst = u_mid_pos: {use:(mid,rate)}
    batch = {}
    i = pos
    ## SHUFFLE USER
    for key, value in data_lst.copy().items():
        batch.update({key: value})
        del [data_lst[key]]
        pos += 1
        if (pos >= i + batch_size):
            break

    x = np.zeros((batch_size, max_window_size))
    y = np.zeros((batch_size, n_classes), dtype=float)
    y_train = np.zeros((batch_size, n_classes), dtype=float)
    y_count_total = []

    # feature: age and gender
    feature = np.zeros((batch_size, feature_size))
    # occupation:
    occupation = np.zeros((batch_size, 1))
    # genre:
    genre = np.zeros((batch_size, genre_size))

    word_num = np.zeros((batch_size))

    line_no = 0

    for key, value in batch.items():
        col_no_x = 0

        gender = np.zeros(1)
        gender[0] = user_gender[key]

        age = np.zeros(1)
        age[0] = user_age[key]

        occupation[line_no][:] = user_occupation[key]

        genre[line_no][:] = user_genre[key]
        tmp = np.concatenate([gender, age])
        feature[line_no][:] = tmp

        # update y: one hot encoding for y has labels
        for i in value:
            index = int(i[0])
            y[line_no][index] = 1


        # update x: retrive original x_label used in training
        for index in x_label[key]:
            x[line_no][col_no_x] = index
            col_no_x += 1

        y_count = 0
        # update y used in training:
        for index in y_label[key]:
            index = int(i[0])
            y_train[line_no][index] = 1
            y_count += 1
        y_count_total.append(y_count)

        # add negative samples:  set one hot encoding for negative sample = -1
        count_y = 0
        if key in neg_lst:
            for i in neg_lst[key]:
                index = int(i[0])
                if i in neg_label[key]:
                    y_train[line_no][index] = -0.5
                else:
                    y[line_no][index] = -0.5
                    count_y += 1
                    if (count_y > y_size * 3):
                        break
        word_num[line_no] = col_no_x
        line_no += 1

    return x, y, word_num.reshape(batch_size, 1), y_train, feature, occupation, genre, y_count_total
