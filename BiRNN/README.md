For the recommendation N-top Movie problem, RNNs can taking the evolution of users taste into account.  BiRNN consists of forward and backward RNN structure (GRU cell).  In the forward
RNN, the input sequence is sorted in ascending order by timestamps (chronological order) for each user.  The backward RNN takes the input sequence in the reverse order.  To compute the final pre-
diction.  The output of both GRU is fed into the last (softmax) output layer.  We recommend the N-movies that have the highest values in the output layer.
