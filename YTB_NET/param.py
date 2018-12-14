from prepros import movieName_to_dic
import tensorflow as tf

batch_size = 50
emb_size = 40
max_window_size = 100
occupation_emb_size = 3
feature_size = 2
genre_size = 18
input_size = emb_size + feature_size + occupation_emb_size + genre_size
movieName_dic = movieName_to_dic()
n_classes = len(movieName_dic)
print("Class number: ", n_classes)

training_epochs = 3000
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.01
display_step = 1

y_size = 50

# Network Parameters
n_hidden_1 = 50 # 1st layer number of features
n_hidden_2 = 40 # 2nd layer number of features

# gloab variable
keys = set()
y_label = {}
x_label = {}
neg_label = {}