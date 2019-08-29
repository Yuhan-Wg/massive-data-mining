"""
Some join operations in Spark.
"""
from pyspark import RDD


def join_multiple_keys(left: RDD, right: RDD, n: int) -> RDD:
    """
    Join RDDs with multiple keys.
        ((key1, key2, ...), value_left) x (key_i, value_right_i) ->
        ((key1, key2, ...), (value_left, value_right_1, value_right_2, ...))
    :param left: RDD<tuple<int>, value>
    :param right: RDD<int, value>
    :param n: int, the length of the key in left-RDD
    :return: joint RDD.
    """
    left = left.map(
        lambda u: (-1, (u[0], (u[1],)))
    )  # (_, (tuple<key>, tuple<value>))
    right = right.map(
        lambda u: (u[0], (u[1],))
    ).cache()  # (_, tuple<value>)
    for key_order in range(n):
        left = left.map(
            lambda u: (u[1][0][key_order], u[1])  # (_, (tuple<key>, tuple<value>))
        ).join(
            right  # (_, ((tuple<key>, tuple<value>), tuple<value>))
        ).map(
            lambda u: (-1, (u[1][0][0], u[1][0][1] + u[1][1]))
        )  # (_, (tuple<key>, tuple<value>))

    left = left.map(
        lambda u: u[1]
    )  # (tuple<key>, tuple<value>)
    return left


