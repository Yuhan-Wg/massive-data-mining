from pyspark import SparkContext
import numpy as np
import unittest


class TestJoinMultipleKeysMethod(unittest.TestCase):
    def test_random_data(self):
        from datming.utils import join_multiple_keys
        left = SparkContext.getOrCreate().parallelize(
            [
                (tuple(np.random.randint(0, 5, 3)), np.random.randint(0, 10))
                for _ in range(100)
            ]
        )
        right = SparkContext.getOrCreate().parallelize(
            [
                (np.random.randint(0, 5), np.random.randint(0, 10))
                for _ in range(20)
            ]
        )
        joint = join_multiple_keys(left, right, n=3).collect()
        print(joint)
        self.assertTrue(len(joint[0][0]) == 3 and len(joint[0][1]) == 4)

if __name__ == '__main__':
    unittest.main()
