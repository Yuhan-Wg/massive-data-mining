"""
Neighborhood-based Collaborative filtering. Include: Item-based CF and User-based CF
A straightforward implementation of Neighborhood-Based CF algorithm with Spark.
This implementation needs to shuffle about (n_bucket_block * n_bucket_block * n_item_block) keys and duplicate
    similarity matrix and rating matrix  (number of duplicates grows with n_bucket_block and n_item_block). So
    there is a trade-off between time and space while adjusting n_bucket_block and n_item_block.

>> sc = SparkContext.getOrCreate()
>> train = sc.parallelize([(1, 2, 5.), (1, 3, 3.), (2, 3, 4.)])
>> test = sc.parallelize([(2, 2)])
>> result = UserBasedCF().fit_predict(train, test).collect()

"""
from pyspark import SparkContext, RDD
from datming.similarity import JaccardSimilarity
from datming.utils import hash2int

from numbers import Number
from types import GeneratorType
from typing import Hashable, Iterator
from collections import defaultdict
import heapq

__all__ = [
    "UserBasedCF", "ItemBasedCF", "NeighborhoodBasedCF"
]


class NeighborhoodBasedCF(object):
    """
    Base class of Item-based/User-based Collaborative Filtering.
    """
    def __init__(self, k: Number=float("inf"), maximum_num_partitions=None,
                 n_bucket_block: int=1, n_cross_block: int=1, n_item_block: int=1,
                 bucket_block_size: int=None, cross_block_size: int=None, item_block_size: int=None,
                 seed: int=None, **params):
        self._k = k if isinstance(k, int) else float("inf")
        self._maximum_num_partitions = maximum_num_partitions \
            if isinstance(maximum_num_partitions, int) and maximum_num_partitions > 0 else float("inf")

        self._n_bucket_block = n_bucket_block
        self._n_item_block = n_item_block
        self._n_cross_block = n_cross_block
        self._bucket_block_size = bucket_block_size
        self._item_block_size = item_block_size
        self._cross_block_size = cross_block_size
        self._n_buckets = 0
        self._n_items = 0

        self._seed = seed if isinstance(seed, int) else np.random.randint(0, 2**32-1)
        self._lsh_params = params
        self._lsh_params["seed"] = seed
        self._num_partitions_of_train = self._num_partitions_of_test = 0
        self._data = DataContainer()

    def fit(self, train: RDD):
        """
        :param train: RDD<(Hashable, Hashable, float)>
            = RDD<(bucket, item, rating)>
        :return: self
        """
        # check the data type and initialize parameters
        train = self._check_data(train=train)
        self.__init_parameters(train)

        similarity = self.__calculate_similarity(train,
                                                 self._lsh_params, self._maximum_num_partitions).cache()
        # similarity = RDD<(Hashable, Hashable, float)>

        train = self.__blocking_matrix(train=train)
        # train = RDD<((int, int), (Hashable, Hashable, float))>

        similarity = self.__blocking_matrix(similarity=similarity)
        # similarity = RDD<((int, int), (Hashable, Hashable, float))>

        self._data.add(train=train, similarity=similarity)
        return self

    def predict(self, test: RDD) -> RDD:
        """
        :param test: RDD<(Hashable, Hashable)>
            = RDD<(bucket, item)>
        :return: RDD<(Hashable, Hashable, float)>
            = RDD<(bucket, item, predicted_rating)>
        """
        test = self._check_data(test=test)
        test = self.__blocking_matrix(test=test)
        # test = RDD<((int, int), (Hashable, Hashable))>

        train, test, similarity = self.__group_by_blocks(
            self._data.train, test, self._data.similarity,
            self._n_bucket_block, self._n_cross_block, self._n_item_block
        )
        # train, test, similarity = RDD<(bucket_block, cross_block, item_block), (value_i, i)>, i=0, 1, 2
        # train, value_0 = (bucket, item, rating)
        # test, value_1 = (bucket, item)
        # similarity, value_2 = (bucket, item, similarity)

        prediction = self.__make_prediction(
            train=train, test=test, similarity=similarity, k=self._k,
            maximum_num_partitions=self._maximum_num_partitions
        )
        prediction.count()
        return prediction

    def fit_predict(self, train, test):
        return self.fit(train).predict(test)

    @staticmethod
    def _check_data(train: RDD=None, test: RDD=None) -> (RDD, int):
        # Data-type check
        if isinstance(train, RDD):
            is_legal_train = train.map(
                lambda u: len(u) >= 3 and u[0] is not None and u[1] is not None and isinstance(u[2], Number)
            ).reduce(lambda u1, u2: u1 and u2)
            if not is_legal_train:
                raise ValueError("Parameter train should be an RDD<(user, item, rating)>")
            num_partitions_of_train = train.getNumPartitions()
            return train

        if isinstance(test, RDD):
            is_legal_test = test.map(
                lambda u: len(u) >= 2 and u[0] is not None and u[1] is not None
            ).reduce(lambda u1, u2: u1 and u2)
            if not is_legal_test:
                raise ValueError("Parameter train should be an RDD<(user, item, rating)>")
            num_partitions_of_test = test.getNumPartitions()
            return test

        raise ValueError("RDD train/test need to be input.")

    @staticmethod
    def __calculate_similarity(train: RDD, lsh_params: dict, maximum_num_partitions: int) -> RDD:
        """
        Calculate Jaccard Similarity from train-RDD.
        :param train: RDD<(Hashable, Hashable, float)>
        :return: RDD<Hashable, Hashable, float>
            = RDD<bucket, bucket, similarity>
        """
        train = train.map(lambda u: (u[0], u[1]))\
            .groupByKey().map(lambda u: (u[0], list(u[1]))).cache()
        similarity_among_buckets = JaccardSimilarity(**lsh_params).predict(train).cache()
        if similarity_among_buckets.getNumPartitions() > maximum_num_partitions:
            similarity_among_buckets = similarity_among_buckets.coalesce(maximum_num_partitions).cache()
        return similarity_among_buckets

    def __init_parameters(self, train: RDD):
        """
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

        # For the bucket dimension.
        if self._bucket_block_size is None:
            # Interpret bucket_block_size from n_bucket_block
            self._bucket_block_size = self._n_buckets // self._n_bucket_block + 1
        else:
            self._n_bucket_block = self._n_buckets // self._bucket_block_size + 1

        # For the cross dimension.
        if self._cross_block_size is None:
            self._cross_block_size = self._n_buckets // self._n_cross_block + 1
        else:
            self._n_cross_block = self._n_buckets // self._cross_block_size + 1

        # For the item dimension
        self._n_items = train.map(lambda u: u[1]).distinct().count()
        if self._item_block_size is None:
            self._item_block_size = self._n_items // self._n_item_block + 1
        else:
            self._n_item_block = self._n_item // self._item_block_size + 1
        return self

    def __blocking_matrix(self, train: RDD=None, test: RDD=None, similarity=None) -> RDD:
        """
        Divide matrix into blocks for the purpose of reduce key number.
        :param train: RDD<(Hashable, Hashable, float)>
            = RDD<bucket, item, rating>
        :param test: RDD<(Hashable, Hashable)>
            = RDD<bucket, item>
        :param similarity: RDD<(Hashable, Hashable, float)>
            RDD<bucket, bucket, similarity>
        :return: RDD<(int, int)(Hashable, Hashable, float)>
            = RDD<(bucket_block, item_block), (bucket, item, rating)> or
              RDD<(bucket_block, bucket_block), (bucket, bucket, similarity)>
        """
        seed = self._seed
        n_bucket_block = self._n_bucket_block
        n_item_block = self._n_item_block
        n_cross_block = self._n_cross_block

        if train is not None:
            train = train.map(
                lambda u: (
                    (hash2int(u[0], max_value=n_cross_block, seed=seed),
                     hash2int(u[1], max_value=n_item_block, seed=seed)), u)
            ).cache()
            train.count()
            return train

        if test is not None:
            test = test.map(
                lambda u: (
                    (hash2int(u[0], max_value=n_bucket_block, seed=seed),
                     hash2int(u[1], max_value=n_item_block, seed=seed)), u)
            ).cache()
            test.count()
            return test

        if similarity is not None:
            similarity = similarity.flatMap(lambda u: [(u[0], u[1], u[2]), (u[1], u[0], u[2])]).map(
                lambda u: (
                    (hash2int(u[0], max_value=n_bucket_block, seed=seed),
                     hash2int(u[1], max_value=n_cross_block, seed=seed)), u)
            ).cache()
            similarity.count()
            return similarity

    @staticmethod
    def __group_by_blocks(train: RDD, test: RDD, similarity: RDD,
                          n_bucket_block: int, n_cross_block: int, n_item_block: int) -> (RDD, RDD, RDD):
        """
        :param train:
            RDD<(cross_block, item_block), (cross_bucket, item, rating)>
        :param test:
            RDD<(bucket_block, item_block), (bucket, item)>
        :param similarity:
            RDD<(bucket_block, cross_block), (bucket, cross_bucket, similarity)>
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
        )
        test = test.flatMap(
            lambda u: (
                ((u[0][0], c, u[0][1]), (u[1], 1))
                for c in range(n_cross_block)
            )
        )
        similarity = similarity.flatMap(
            lambda u:  (
                ((u[0][0], u[0][1], i), (u[1], 2))
                for i in range(n_item_block)
            )
        )
        return train, test, similarity

    @classmethod
    def __make_prediction(cls, train, test, similarity, k, maximum_num_partitions) -> RDD:
        """
        :param train:
            RDD<(b, cross_block, item_block), ((cross_bucket, item, rating), 0)>, b=1, ..., n_bucket_block
        :param test:
            RDD<(bucket_block, b, item_block), ((bucket, item), 1)>, b=1, ..., n_bucket_block
        :param similarity:
            RDD<(bucket_block, cross_block, i), ((bucket, cross_bucket, similarity), 2)>, i=1, ..., n_item_block
        :return: RDD<(Hashable, Hashable, float)>
            RDD<(bucket, item, predicted_rating)>
        """
        prediction = train.union(test).union(similarity).coalesce(maximum_num_partitions)
        prediction = (
            prediction.groupByKey()
                      .flatMap(lambda u: cls.__calculate_per_block(u, k))
                      .coalesce(maximum_num_partitions)
                      .groupByKey()
                      .flatMap(lambda u: cls.__sum_up_blocks(u, k))
                      .coalesce(maximum_num_partitions).cache()
        )
        return prediction

    @staticmethod
    def __calculate_per_block(key_values: (tuple, Iterator), k: int) -> GeneratorType:
        """
        Find k nearest neighbors in each block for each to-be-predicted bucket-item pair.
        """
        (bucket_block, cross_block, item_block), values = key_values
        # values = list(values)
        """
        Read matrix elements from iterator. 
        The elements have three sources: train-RDD, test-RDD, similarity-RDD.
        """
        ratings = dict()
        similarities = defaultdict(list)
        to_be_predicted = list()
        for value in values:
            if value[1] == 0:
                # train
                cross, item, rating = value[0]
                ratings[(cross, item)] = rating
            elif value[1] == 1:
                # test
                bucket, item = value[0]
                to_be_predicted.append((bucket, item))
            elif value[1] == 2:
                # similarity
                bucket, cross, sim = value[0]
                similarities[bucket].append((cross, sim))

        """
        Find k nearest neighbors for each bucket-item pair in test.
        """
        for bucket, item in to_be_predicted:
            if k != float("inf"):
                heap = []
            else:
                summary = [0, 0]

            for cross, sim in similarities[bucket]:
                # Similar bucket
                if (cross, item) in ratings:
                    # If this similar bucket has the rated the item
                    if k != float("inf"):
                        heap.append(
                            ((bucket_block, item_block),
                             ((bucket, item), (sim, sim * ratings[(cross, item)])))
                        )
                    else:
                        summary[0] += sim
                        summary[1] += sim * ratings[(cross, item)]
            if k != float("inf"):
                if len(heap) == 0:
                    # prevent none value
                    yield ((bucket_block, item_block), ((bucket, item), (0, 0)))
                else:
                    yield from heapq.nlargest(k, heap, key=lambda u: (u[1][1][0]))
            else:
                yield ((bucket_block, item_block), ((bucket, item), tuple(summary)))

    @staticmethod
    def __sum_up_blocks(key_values: ((Hashable, Hashable), (float, float)), k: int) -> GeneratorType:
        """
        Sum up predicted ratings from all blocks.
        :param key_values:
            = RDD<(bucket, item), (similarity, rating)>
        :param k:
        :return: Generator<(bucket, item, predicted_rating)>
        """
        key, values = key_values
        ratings = defaultdict(list)
        for (bucket, item), (sim, rating) in values:
            ratings[(bucket, item)].append((sim, rating))
        for bucket, item in ratings:
            if k == float("inf"):
                n_largest = ratings[(bucket, item)]
            else:
                n_largest = heapq.nlargest(k, ratings[(bucket, item)])

            sum_sim = sum(sim for sim, _ in n_largest)
            if sum_sim != 0:
                predicted_rating = sum(r for _, r in n_largest) / sum_sim
            else:
                predicted_rating = None
            yield (bucket, item, predicted_rating)

    @staticmethod
    def evaluate(truth: RDD, prediction: RDD) -> float:
        """
        Calculate RMSE between truth and predictions.
        :param truth: RDD<Hashable, Hashable, float> = RDD<(bucket, item, rating)>
        :param prediction: RDD<Hashable, Hashable, float> = RDD<(bucket, item, rating)>
        :return: float = RMSE
        """
        truth = truth.map(lambda u: ((u[0], u[1]), u[2]))
        prediction = prediction.map(lambda u: ((u[0], u[1]), u[2]))
        return truth.join(prediction).map(lambda u: (u[1][0] - u[1][1]) ** 2).mean() ** 0.5


class DataContainer(object):
    def __init__(self, **kwargs):
        self.train = self.test = self.similarity = None
        self.__dict__.update(kwargs)

    def add(self, **kwargs):
        self.__dict__.update(kwargs)


class ItemBasedCF(NeighborhoodBasedCF):
    """
    Item-based CF sub-classing from NeighborhoodBasedCF
    """
    def __init__(self, n_user_block=1, n_item_block=1,
                 user_block_size=None, item_block_size=None, **kwargs):
        super().__init__(n_bucket_block=n_item_block, n_item_block=n_user_block,
                         bucket_block_size=item_block_size, item_block_size=user_block_size, **kwargs)

    @staticmethod
    def _check_data(train=None, test=None):
        """
        :param train:
            RDD<(user, item, rating)>
        :param test:
            RDD<(user, item)>
        :return:
        """
        NeighborhoodBasedCF._check_data(train=train, test=test)
        if train is not None:
            return train.map(lambda u: (u[1], u[0], u[2]))

        if test is not None:
            return test.map(lambda u: (u[1], u[0]))


class UserBasedCF(NeighborhoodBasedCF):
    """
    User-based CF sub-classing from NeighborhoodBasedCF
    """
    def __init__(self, n_user_block=1, n_item_block=1,
                 user_block_size=None, item_block_size=None, **kwargs):
        super().__init__(n_bucket_block=n_user_block, n_item_block=n_item_block,
                         bucket_block_size=user_block_size, item_block_size=item_block_size, **kwargs)
