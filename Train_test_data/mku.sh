#!/bin/bash

trap `rm -f tmp.$$; exit 1` 1 2 15

for i in 1 2 3 4 5
do
	head -`expr $i \* 200000` u.data | tail -200000 > tmp.$$
	sort -t"	" -k 1,1n -k 2,2n tmp.$$ > u$i.test
	head -`expr \( $i - 1 \) \* 200000` u.data > tmp.$$
	tail -`expr \( 5 - $i \) \* 200000` u.data >> tmp.$$
	sort -t"	" -k 1,1n -k 2,2n tmp.$$ > u$i.base
done

/home/tatyana/CMPT726ML/Movie_rec_sys/ml-latest/allbut.pl ua 1 10 1000000 u.data
sort -t"	" -k 1,1n -k 2,2n ua.base > tmp.$$
mv tmp.$$ ua.base
sort -t"	" -k 1,1n -k 2,2n ua.test > tmp.$$
mv tmp.$$ ua.test

/home/tatyana/CMPT726ML/Movie_rec_sys/ml-latest/allbut.pl ub 11 20 1000000 u.data
sort -t"	" -k 1,1n -k 2,2n ub.base > tmp.$$
mv tmp.$$ ub.base
sort -t"	" -k 1,1n -k 2,2n ub.test > tmp.$$
mv tmp.$$ ub.test

