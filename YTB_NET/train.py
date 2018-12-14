import os
import numpy as np 
import tensorflow as tf 
from sklearn.metrics import roc_auc_score
from param import *
from utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


weights = {
	'h1': tf.Variable(tf.random_normal([input_size, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def mlp(x, weights, biases):
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	dlayer_1 = tf.nn.dropout(layer_1, 0.5)
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	return layer_2

embedding = {
    'input':tf.Variable(tf.random_uniform([n_classes, emb_size], -1.0, 1.0))
    # 'output':tf.Variable(tf.random_uniform([len(label_dict)+1, emb_size], -1.0, 1.0))
}
embedding_occ = {
    'input':tf.Variable(tf.random_uniform([21, occupation_emb_size], -1.0, 1.0))
    # 'output':tf.Variable(tf.random_uniform([len(label_dict)+1, emb_size], -1.0, 1.0))
}

# initialize batch parameter
word_num = tf.placeholder(tf.float32, shape=[None, 1])
x_batch = tf.placeholder(tf.int32, shape=[None, max_window_size])   ###max_window_size
y_batch = tf.placeholder(tf.float32, shape=[None, n_classes]) ###one-hot
feature_batch = tf.placeholder(tf.float32, shape=[None, feature_size])
occupation_batch =  tf.placeholder(tf.int32, shape=[None, 1])
occupation_embedding = tf.squeeze(tf.nn.embedding_lookup(embedding_occ['input'], occupation_batch))
genre_batch = tf.placeholder(tf.float32, shape=[None, genre_size])

input_embedding = tf.nn.embedding_lookup(embedding['input'], x_batch)
project_embedding = tf.div(tf.reduce_sum(input_embedding, 1),word_num)
project_embedding = tf.concat([project_embedding, feature_batch],1)
project_embedding = tf.concat([project_embedding, occupation_embedding],1)
project_embedding = tf.concat([project_embedding, genre_batch],1)
check_op = tf.add_check_numerics_ops()

# Construct model
pred = mlp(project_embedding, weights, biases)
pred = tf.nn.l2_normalize(pred,0)
embedding['input'] = tf.nn.l2_normalize(embedding['input'],0)
score = tf.matmul(pred, tf.transpose(embedding['input']))
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = score, labels = y_batch)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

out_layer = tf.nn.sigmoid(score)


def test():
    remove_key = set(user_movieId_pos_test.keys()) - keys
    for i in remove_key:
        del (user_movieId_pos_test[i])

    test_lst = user_movieId_pos_test
    batch_size = len(test_lst)
    total_batch = int(len(test_lst) / batch_size)

    # top k accuracy:
    k = 10
    rec_count = 0
    hit = 0
    test_count = 0
    avg_cost = 0

    for i in range(total_batch):
        copy = user_movieId_pos_test.copy()
        x, y, word_number, y_train, feature, occupation, genre, y_count_total = read_data_test(i * batch_size, batch_size, copy, user_movieId_neg)
        out_score = out_layer.eval({x_batch: x, word_num: word_number,
                                    feature_batch: feature, occupation_batch: occupation,
                                    genre_batch: genre})

        # cost
        c = cost.eval({x_batch: x, word_num: word_number, y_batch: y,
                       feature_batch: feature, occupation_batch: occupation,
                       genre_batch: genre})
        print("validation cost", c)

        # get roc
        # calculate recall and precision
        y_true = []
        y_pred = []
        for row_x, row_out, row_y, row_y_train, y_number in zip(x, out_score, y, y_train, y_count_total):
            # set the training labels' prob as 0
            tmmmmmp = 0
            for col in row_x:
                row_out[int(col)] = 0

            train_label = np.where(row_y_train == 1)[0]
            for col in train_label:
                row_out[int(col)] = 0

            # get roc
            pos_label = np.where(row_y == 1)[0]
            for col in pos_label:
                y_true.append(1)
                y_pred.append(row_out[int(col)])
                # print("pos_label score", row_out[int(col)])


            neg_label = np.where(row_y == -0.5)[0]
            for col in neg_label:
                y_true.append(0)
                y_pred.append(row_out[int(col)])
                # print("neg_label score", row_out[int(col)])

            # get top k index
            top_k = np.argsort(row_out)[::-1][:k]
            # print("predict", top_k)
            # print("real_y",  np.where(row_y == 1))
            # print("real_x", row_x)
            # print("index",top_k)
            for index in top_k:
                if(row_y[index] == 1):
                    hit += 1
                    tmmmmmp +=1
            rec_count += k
            test_count += y_number
            #print("tmmmmmp:", tmmmmmp)
    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    #print("hit: ", hit, "rec_count:", rec_count)


    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auc = roc_auc_score(y_true, y_pred)

    print("auc", auc)
    print('precision=%.4f\trecall=%.4f\n' %
            (precision, recall))


# run
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    #start_time = time.time()
    total_batch = int(len(user_movieId_pos) / batch_size)
    print("total_batch of training data: ", total_batch)
    for epoch in range(training_epochs):
        avg_cost = 0.
        copy = user_movieId_pos.copy()

        for i in range(total_batch):
            x, y, word_number, feature, occupation, genre = read_data(i * batch_size, batch_size, copy, user_movieId_neg)
            #print(x)
            #print(word_number)
            #print(y)
            _, c, a = sess.run([optimizer, cost, check_op],
                               feed_dict=({x_batch: x, word_num: word_number, y_batch: y,
                                           feature_batch: feature, occupation_batch: occupation,
                                           genre_batch: genre}))

            # print("loss", l)
            avg_cost += c / total_batch



        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
              "{:.9f}".format(avg_cost))

        test()
