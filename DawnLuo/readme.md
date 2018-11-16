## Youtube Recommendation Model

> Description

Treat each movie as a class and change recommendation to a classification problem. Model will give out probability for each movie and choose top-K. See how many of the top-K hit the test label.

> Feature

userId, movieId, genres

> Run

```shell
> python3 train.py
```

Tested with tensorflow 1.7.0

> Problem

Training dose not converge, so it fails.

> May due to

1. Information embedding is not good, just change Id to number and genres to number vectors. I think it is not good, youtube paper did not give embedding detials. We could refer to this [repo](https://github.com/chengstone/movie_recommender) (we see this Friday) to see how to embed info.

2. Maybe the data set is too small and not suitable for this model. We know Google have google computer and large number of data. But out hardware, dataset and time is limited.