"""
Savasere,Omiecinski, and Navathe (SON) Algorithm:
    Pass1:
    -> Repeatedly read small subsets of buckets into memory.
    -> An itemset is a candidate if it is found frequent in any one of the buckets
    Pass2:
    -> Count all candidate itemsets.

This algorithm is implemented with pyspark.
"""
from datming.freq_itemset.apriori import APriori
from pyspark import SparkContext
from collections import defaultdict


class SON(object):
    def __init__(self, support,num_partitions, **kwargs):
        self._support = support
        self._num_partitions = num_partitions

    def count(self, spark_context, file):
        spark_rdd_file = spark_context.textFile(file)\
            .repartition(self._num_partitions)\
            .map(self._interpret_line_of_input)

        candidates = spark_rdd_file.mapPartitions(self._find_candidates)\
            .distinct().collect()

        set_candidates = set(candidates)
        del candidates

        counts = spark_rdd_file.flatMap(
            lambda bucket: self._count_candidates_in_bucket(bucket, set_candidates)
        ).reduceByKey(lambda x, y: x + y)\
            .filter(lambda candidateCount: candidateCount[1] >= self._support)

        dict_count = defaultdict(dict)
        for key,val in counts.collect():
            dict_count[len(key)][key] = val

        return dict_count

    @staticmethod
    def _interpret_line_of_input(line):
        return list(int(item) for item in line.strip().split(","))

    def _find_candidates(self, iterator):
        iterable = list(iterator)
        sub_support = self._support//self._num_partitions
        apriori = APriori(support=sub_support)
        candidates = list()
        for c in apriori.predict(iterable=iterable):
            candidates.extend(c.keys())
        return candidates

    @staticmethod
    def _count_candidates_in_bucket(bucket, candidates):
        set_bucket = set(bucket)
        for c in candidates:
            if c.issubset(set_bucket):
                yield (c, 1)


if __name__ == '__main__':
    son = SON(support=2,num_partitions=2)
    sc = SparkContext()
    for s in son.count(spark_context=sc, file = ["1,2,3","2,3","1,4","2,4","1,2,3,4"]).items():
        print(s)