"""
FP-Growth
"""
from collections import defaultdict
from pyspark import SparkContext


class FPGrowth(object):
    def __init__(self, support):
        self._support = support
        self._root = None

    def count(self, iterable):
        frequent_items = self._generate_frequent_item(
            self._decorate_iterable(iterable)
        )
        self._root, frequent_items_next_node = self._construct_fptree(
            self._decorate_iterable(iterable), frequent_items
        )
        dict_count = defaultdict(dict)
        for key, val in self._count_frequent_set(frequent_items_next_node, frequent_items):
            dict_count[len(key)][key] = val
        return dict_count

    @staticmethod
    def _decorate_iterable(iterable):
        for bucket in iterable:
            yield (bucket, 1)

    def _generate_frequent_item(self, iterable):
        frequent_items = defaultdict(int)
        for bucket, count in iterable:
            for item in set(bucket):
                frequent_items[item] += count
        return {
            key: val for key, val in frequent_items.items() if val >= self._support
        }

    @staticmethod
    def _construct_fptree(iterable, frequent_items):
        root = FPGrowth.FPTreeNode(name="root", count=-1)
        frequent_items_next_node = dict()
        for bucket, count in iterable:
            freq_bucket = sorted(
                [
                    (item, frequent_items[item])
                    for item in bucket
                    if item in frequent_items
                ],
                key=lambda x: (x[1], x[0]), reverse=True
            )

            prev_node = root
            for item, _ in freq_bucket:
                if item in prev_node.children:
                    child = prev_node.children.get(item)
                    child.count += count
                else:
                    child = FPGrowth.FPTreeNode(
                        name=item,
                        count=count,
                        parent=prev_node
                    )
                    prev_node.children[item] = child
                    if item not in frequent_items_next_node:
                        frequent_items_next_node[item] = [child, child]
                    else:
                        frequent_items_next_node[item][1].next_node = child
                        frequent_items_next_node[item][1] = child

                prev_node = child
        return root, {
            item: next_node[0] for item, next_node in frequent_items_next_node.items()
        }

    def _count_frequent_set(self, frequent_items_next_node, frequent_items):
        for item in frequent_items_next_node:
            next_node = frequent_items_next_node[item]
            new_buckets = list()
            while next_node:
                count = next_node.count
                bucket = []
                temp = next_node.parent
                while temp.count > 0:
                    bucket.append(temp.name)
                    temp = temp.parent
                if bucket:
                    new_buckets.append((bucket, count))
                next_node = next_node.next_node

            new_frequent_items = self._generate_frequent_item(
                new_buckets
            )
            if new_frequent_items:
                _, new_frequent_items_next_node = self._construct_fptree(
                    new_buckets, new_frequent_items
                )
                yield from (
                    (frozenset({item}) | freq_set, freq_count)
                    for freq_set, freq_count in self._count_frequent_set(new_frequent_items_next_node,
                                                                         new_frequent_items)
                )
            yield (frozenset({item}), frequent_items[item])

    class FPTreeNode(object):
        def __init__(self, name, count: int=0,
                     next_node=None,
                     parent=None):
            self.name = name
            self.count = count
            self.next_node = next_node
            self.parent = parent
            self.children = dict()

        def __repr__(self):
            return "<{0}: count={1}, nextNode={2}, parent={3}, children=({4})>".format(
                self.name, self.count,
                self.next_node.name if self.next_node else "None",
                self.parent.name if self.parent else "None",
                ",".join(str(node.name) for node in self.children.values())
                )


class FPGrowthSparkModel(object):
    def __init__(self, support):
        self._support = support

    def iter_count(self, iterable, from_file=True):
        sc = SparkContext.getOrCreate()
        if from_file:
            fptree = self._rdd_from_file(iterable)
        else:
            fptree = sc.parallelize(iterable).map(
                lambda x: ((frozenset({}), tuple(x)), 1)
            ).cache()

        while True:
            freq_items, fptree = self._loop_count(fptree)
            if freq_items:
                yield freq_items
            else:
                break

    def _loop_count(self, rdd_file):
        """
        :param rdd_file:
        :return:
        """
        # Filter out all frequent items. (On specific condition)
        frequent_items = rdd_file.flatMap(self._pop_item)\
            .reduceByKey(lambda x, y: x+y)\
            .filter(lambda x: x[1] >= self._support).cache()

        # Pop out all frequent sets in this loop
        collected_freq_items = dict(frequent_items.map(
            lambda x: (x[0][0] | frozenset({x[0][1]}), x[1])
        ).collect())

        # Maintain the frequent items.
        # rdd: (condition, count_of_frequent_items)
        freq_table = frequent_items\
            .map(lambda x: (x[0][0], (x[0][1], x[1])))\
            .groupByKey().map(lambda x: (x[0], list(x[1])))

        fptree = rdd_file\
            .map(lambda x: (x[0][0], (x[0][1], x[1])))\
            .join(freq_table)\
            .flatMap(self._generate_path)\
            .reduceByKey(lambda x, y: x+y)
        return collected_freq_items, fptree

    @staticmethod
    def _construct_ftree(bucket):
        exist_set, next_path, count = bucket
        for item in next_path:
            yield ((exist_set, item), count)

    @staticmethod
    def _generate_path(key_values):
        exist_set = key_values[0]
        (next_path, count), list_of_freq = key_values[1]
        dict_of_freq = dict(list_of_freq)
        new_path = sorted([
            item for item in next_path if item in dict_of_freq
        ], key=lambda item: (dict_of_freq[item], item), reverse=True)
        if len(new_path) > 1:
            for idx in range(1, len(new_path)):
                yield (
                    (exist_set | frozenset({new_path[idx]}), tuple(new_path[:idx])),
                    count
                )

    @staticmethod
    def _pop_item(bucket):
        (exist_set, next_path), count = bucket
        for item in next_path:
            yield ((exist_set, item), count)

    def _rdd_from_file(self, file):
        sc = SparkContext.getOrCreate()
        rdd_file = sc.textFile(file).map(self._interpret_line).cache()
        return rdd_file

    def _interpret_line(self, line):
        return ((frozenset({}),
                tuple(self.line_interpreter(line))),
                1)

    @staticmethod
    def line_interpreter(line):
        return (int(item) for item in line.strip().split(","))


if __name__ == '__main__':
    fp_growth = FPGrowth(2)
    for k, v in fp_growth.count([[1, 2, 3], [2, 3], [1, 4], [2, 4], [1, 2, 3, 4]]).items():
        print(k, v)

    fp_growth_spark = FPGrowthSparkModel(2)
    for k in fp_growth_spark.iter_count([[1, 2, 3], [2, 3], [1, 4], [2, 4], [1, 2, 3, 4]], from_file=False):
        print(k)
