# Data Preparation

u.data is copy of rating.dat

To split it 80%/20%, run:

./mku.sh

## More information:

u1.base and u1.test   -- The data sets u1.base and u1.test through u5.base and u5.test are 80%/20% splits of the u data into training and test data. Each of u1, ..., u5 have disjoint test sets; this if for 5 fold cross validation (where you repeat your experiment with each training and test set and average the results). These data sets can be generated from u.data by mku.sh.

ua.base and ua.test   -- The data sets ua.base, ua.test, ub.base, and ub.test split the u data into a training set and a test set with exactly 10 ratings per user in the test set.  The sets ua.test and ub.test are disjoint.  These data sets can be generated from u.data by mku.sh.     

allbut.pl  -- The script that generates training and test sets where all but n of a users ratings are in the training data.

mku.sh     -- A shell script to generate all the u data sets from u.data.
