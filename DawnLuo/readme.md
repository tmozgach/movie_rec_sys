## Youtube Recommendation Model

> Description

Treat each movie as a class and change recommendation to classification problem. Model will give out probability for each movie and choose top-K out. Then see how many of the top-K hit the test label.

> Feature

userId, movieId, genres

> Run

```shell
> python3 train.py
```

Tested with tensorflow 1.7.0 (No try on the new version 1.12.0)

> **Problem**

It seems training dose not converge, so it fails. -_-||

> May due to

1. Information embedding is not good, just change Id and genres to number vectors. I think it is not good, _YouTube paper_ did not give embedding detials. We could refer to this [repo](https://github.com/chengstone/movie_recommender) (we see this Friday) to see how to embed info.

2. Maybe the data set is too small to be suitable for this model. We know Google have good computer and large number of data. But our hardware, dataset and time are limited.