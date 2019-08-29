"""
Neighborhood-based Collaborative filtering. Include: Item-based CF and User-based CF
Two Neighborhood-Based CF modes: spark, local
    spark:
        A straightforward implementation of Neighborhood-Based CF algorithm with Spark.
        Good for the situation that the data are extremely large.
        This implementation needs to shuffle about (n_bucket_block * n_bucket_block * n_item_block) keys and duplicate
        similarity matrix and rating matrix  (number of duplicates grows with n_bucket_block and n_item_block). So
        there is a trade-off between time and space while adjusting n_bucket_block and n_item_block.
    local:
        Local implementation in local machine.
        Good for the situation that similarity matrix and rating history can be cached in the memory.
        Faster but also more strict in data size limitation.
"""
from pyspark import SparkContext, RDD
from datming.similar_items import JaccardSimilarity

from numbers import Number
from collections import defaultdict
import heapq

__all__ = [
    "UserBasedCF", "ItemBasedCF", "NeighborhoodBasedCF"
]


class NeighborhoodBasedCF(object):
    def __init__(self, k=float("inf"), lsh_params=None,
                 n_bucket_block=1, n_cross_block=1, n_item_block=1,
                 bucket_block_size=None, cross_block_size=None, item_block_size=None, **params):
        self._k = k if isinstance(k, int) else float("inf")

        self._n_bucket_block = n_bucket_block
        self._n_item_block = n_item_block
        self._n_cross_block = n_cross_block
        self._bucket_block_size = bucket_block_size
        self._item_block_size = item_block_size
        self._cross_block_size = cross_block_size
        self._n_buckets = 0
        self._n_items = 0

        self._lsh_params = lsh_params if lsh_params is not None else dict()
        self._num_partitions_of_train = self._num_partitions_of_test = 0
        self._data = DataContainer()

    def fit(self, train):
        """
        :param train: RDD<bucket, item, rating>
        :return:
        """
        train, self._num_partitions_of_train = self._check_data(train=train)
        self._init_parameters(train)
        similarity = self._calculate_similarity(train, self._lsh_params).cache()
        train = self._blocking_matrix(rating=train)
        similarity = self._blocking_matrix(similarity=similarity)
        print(sorted(similarity.collect()))
        self._data.add(train=train, similarity=similarity)
        return self

    def predict(self, test):
        test, self._num_partitions_of_test = self._check_data(test=test)
        test = self._blocking_matrix(rating=test)
        train, test, similarity = self._group_by_blocks(
            self._data.train, test, self._data.similarity,
            self._n_bucket_block, self._n_cross_block, self._n_item_block
        )

        prediction = self._make_prediction(
            train=train, test=test, similarity=similarity, k=self._k
        )

        return prediction

    def fit_predict(self, train, test):
        return self.fit(train).predict(test)

    @staticmethod
    def _hash(hashable):
        if isinstance(hashable, Number):
            return int(hashable)
        else:
            return hash(hashable)

    @staticmethod
    def _check_data(train=None, test=None):
        # Data-type check
        if isinstance(train, RDD):
            is_legal_train = train.map(
                lambda u: len(u) >= 3 and u[0] is not None and u[1] is not None and isinstance(u[2], Number)
            ).reduce(lambda u1, u2: u1 and u2)
            if not is_legal_train:
                raise ValueError("Parameter train should be an RDD<(user, item, rating)>")
            num_partitions_of_train = train.getNumPartitions()
            return train, num_partitions_of_train

        if isinstance(test, RDD):
            is_legal_test = test.map(
                lambda u: len(u) >= 2 and u[0] is not None and u[1] is not None
            ).reduce(lambda u1, u2: u1 and u2)
            if not is_legal_test:
                raise ValueError("Parameter train should be an RDD<(user, item, rating)>")
            num_partitions_of_test = test.getNumPartitions()
            return test, num_partitions_of_test

        raise ValueError("RDD train/test need to be input.")

    @staticmethod
    def _calculate_similarity(train, lsh_params):
        """
        Calculate Jaccard Similarity from train-RDD.
        :param train:
        :return: RDD<int, int, float>
            RDD<bucket, bucket, similarity>
        """
        train = train.map(lambda u: (u[0], u[1]))\
            .groupByKey().map(lambda u: (u[0], list(u[1]))).cache()
        similarity_among_buckets = JaccardSimilarity(**lsh_params)._lsh_predict(train)
        return similarity_among_buckets

    def _init_parameters(self, train):
        """
        :param train:
            RDD<bucket, item, rating>
        :return: model

        _n_buckets/_n_items:
            The number of distinct buckets/items in the train RDD.
        _bucket_block_size/_cross_block_size/_item_block_size:
            The size of blocks when dividing buckets/cross buckets/items into blocks.
        _n_bucket_block/_n_cross_block/_n_item_block:
            The number of blocks when dividing buckets/cross buckets/items into blocks.
        """
        self._n_buckets = train.map(lambda u: u[0]).distinct().count()
        if self._n_buckets <= self._k:
            self._k = float("inf")

        if self._bucket_block_size is None:
            # Interpret bucket_block_size from n_bucket_block
            self._bucket_block_size = self._n_buckets // self._n_bucket_block + 1
        else:
            self._n_bucket_block = self._n_buckets // self._bucket_block_size + 1

        if self._cross_block_size is None:
            self._cross_block_size = self._n_buckets // self._n_cross_block + 1
        else:
            self._n_cross_block = self._n_buckets // self._cross_block_size + 1

        self._n_items = train.map(lambda u: u[1]).distinct().count()
        if self._item_block_size is None:
            self._item_block_size = self._n_items // self._n_item_block + 1
        else:
            self._n_item_block = self._n_item // self._item_block_size + 1
        return self

    def _blocking_matrix(self, rating=None, similarity=None):
        """
        :param train:
            RDD<bucket, item, rating>
        :param similarity:
            RDD<bucket, bucket, similarity>
        :return:
            RDD<(bucket_block, item_block), (bucket, item, rating)> or
            RDD<(bucket_block, bucket_block), (bucket, bucket, similarity)>
        """
        _hash = self._hash
        n_buckets, bucket_block_size = self._n_buckets, self._bucket_block_size
        n_items, item_block_size = self._n_items, self._item_block_size
        cross_block_size = self._cross_block_size
        if rating is not None:
            rating = rating.map(
                lambda u: ((_hash(u[0]) % n_buckets // cross_block_size, _hash(u[1]) % n_items // item_block_size), u)
            ).cache()
            rating.count()
            return rating

        if similarity is not None:
            similarity = similarity.flatMap(lambda u: [(u[0], u[1], u[2]), (u[1], u[0], u[2])]).map(
                lambda u: (
                    (_hash(u[0]) % n_buckets // bucket_block_size, _hash(u[1]) % n_buckets // cross_block_size), u
                )
            ).cache()
            similarity.count()
            return similarity

    @staticmethod
    def _group_by_blocks(train, test, similarity,
                         n_bucket_block, n_cross_block, n_item_block):
        """
        :param train:
            RDD<(cross_block, item_block), (cross_bucket, item, rating)>
        :param test:
            RDD<(bucket_block, item_block), (bucket, item)>
        :param similarity:
            RDD<(bucket_block, cross_block), (bucket, cross_bucket, similarity)>
        :param n_bucket_block:
        :return:
        """
        """
        train -> RDD<(b, bucket_block, item_block), ((bucket, item, rating), 0)>, b=1, ..., n_bucket_block
        test -> RDD<(bucket_block, b, item_block), ((bucket, item), 1)>, b=1, ..., n_cross_block
        similarity-> RDD<(bucket_block, bucket_block, i), ((bucket, bucket, similarity), 2)>, i=1, ..., n_item_block
        """
        train = train.flatMap(
            lambda u: (
                ((b, u[0][0], u[0][1]), (u[1], 0))
                for b in range(n_bucket_block)
            )
        ).cache()
        test = test.flatMap(
            lambda u: (
                ((u[0][0], c, u[0][1]), (u[1], 1))
                for c in range(n_cross_block)
            )
        ).cache()
        similarity = similarity.flatMap(
            lambda u:  (
                ((u[0][0], u[0][1], i), (u[1], 2))
                for i in range(n_item_block)
            )
        ).cache()
        return train, test, similarity

    @classmethod
    def _make_prediction(cls, train, test, similarity, k):
        """
        :param train:
            RDD<(b, cross_block, item_block), ((cross_bucket, item, rating), 0)>, b=1, ..., n_bucket_block
        :param test:
            RDD<(bucket_block, b, item_block), ((bucket, item), 1)>, b=1, ..., n_bucket_block
        :param similarity:
            RDD<(bucket_block, cross_block, i), ((bucket, cross_bucket, similarity), 2)>, i=1, ..., n_item_block
        :return:
        """

        prediction = train.union(test).union(similarity)\
            .groupByKey().flatMap(lambda u: cls._calculate_per_block(u, k)).cache()

        if k == float("inf"):
            prediction = prediction.reduceByKey(
                lambda u1, u2: (u1[0] + u2[0], u1[1] + u2[1])
            ).map(
                lambda u: (u[0][0], u[0][1], u[1][1] / u[1][0])
            )
        else:
            # k is a finite number
            prediction = prediction.groupByKey().flatMap(lambda u: cls._sum_up_blocks(u, k)).cache()

        return prediction

    @staticmethod
    def _calculate_per_block(key_values, k):
        key, values = key_values
        ratings = dict()
        similarities = defaultdict(list)
        to_be_predicted = list()
        for value in values:
            if value[1] == 0:
                # train
                bucket, item, rating = value[0]
                ratings[(bucket, item)] = rating
            elif value[1] == 1:
                # test
                to_be_predicted.append(value[0])
            elif value[1] == 2:
                # similarity
                bucket1, bucket2, sim = value[0]
                similarities[bucket1].append((bucket2, sim))

        for bucket, item in to_be_predicted:

            heap = []
            for bucket2, sim in similarities[bucket]:
                if (bucket2, item) in ratings:
                    if k == float("inf"):
                        yield ((bucket, item), (sim, sim * ratings[(bucket2, item)]))
                    else:
                        heap.append(((key[0], key[2]), ((bucket, item), (sim, sim * ratings[(bucket2, item)]))))
            if k != float("inf"):
                yield ((key[0], key[2]), ((bucket, item), (10**-10, -10**-10)))
                yield from heapq.nlargest(k, heap, key=lambda u: u[1][1][0])
            else:
                yield ((bucket, item), (10 ** -10, -10 ** -10))

    @staticmethod
    def _sum_up_blocks(key_values, k):
        key, values = key_values
        ratings = defaultdict(list)
        for (bucket, item), (sim, rating) in values:
            ratings[(bucket, item)].append((sim, rating))
        for bucket, item in ratings:
            n_largest = heapq.nlargest(k, ratings[(bucket, item)], lambda u: u[0])
            predicted_rating = sum(r for _, r in n_largest) / sum(sim for sim, _ in n_largest)
            yield (bucket, item, predicted_rating)


class DataContainer(object):
    def __init__(self, **kwargs):
        self.train = self.test = self.similarity = None
        self.__dict__.update(kwargs)

    def add(self, **kwargs):
        self.__dict__.update(kwargs)


class ItemBasedCF(NeighborhoodBasedCF):
    def __init__(self, k=float("inf"), mode="spark-light", lsh_params=None,
                 n_user_block=1, n_item_block=1, user_block_size=None, item_block_size=None):
        super().__init__(k=k, mode=mode, lsh_params=lsh_params,
                         n_bucket_block=n_item_block, n_item_block=n_user_block,
                         bucket_block_size=item_block_size, item_block_size=user_block_size)

    @staticmethod
    def _check_data(train=None, test=None):
        """
        :param train:
            RDD<(user, item, rating)>
        :param test:
            RDD<(user, item)>
        :return:
        """
        super()._check_data(train=train, test=test)
        if train is not None:
            return train.map(lambda u: (u[1], u[0], u[2]))

        if test is not None:
            return test.map(lambda u: (u[1], u[0]))


class UserBasedCF(NeighborhoodBasedCF):
    def __init__(self, k=float("inf"), mode="spark-light", lsh_params=None,
                 n_user_block=1, n_item_block=1, n_cross_block=1,
                 cross_block_size=None, user_block_size=None, item_block_size=None, **params):
        super().__init__(k=k, mode=mode, lsh_params=lsh_params,
                         n_bucket_block=n_user_block, n_item_block=n_item_block, n_cross_block=n_cross_block,
                         bucket_block_size=user_block_size, item_block_size=item_block_size,
                         cross_block_size=cross_block_size, **params)


if __name__ == '__main__':
    import numpy as np

    train = SparkContext.getOrCreate().parallelize(np.random.randint(0, 100, (5000, 3))).cache()
    test = SparkContext.getOrCreate().parallelize(np.random.randint(0, 100, (1000, 2))).cache()

    np.random.seed(0)
    print(
        sorted(UserBasedCF(k=2, n_user_block=4, n_cross_block=4, n_item_block=4,
                           threshold=0.01, n_bands=100, signature_length=200, random_seed=0)
               .fit_predict(train, test).collect(), key=lambda u: u[2], reverse=True)
    )

    np.random.seed(0)
    print(
        sorted(UserBasedCF(k=2, n_user_block=10, n_cross_block=10, n_item_block=10,
                           threshold=0.01, n_bands=100, signature_length=200, random_seed=0)
               .fit_predict(train, test).collect(), key=lambda u: (u[2], u[0], u[1]), reverse=True)
    )
