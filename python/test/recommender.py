import unittest


class TestRecommenderSystem(unittest.TestCase):
    def test_consistency_neighborhood_based_with_random_data(self):
        from datming.recommender import ItemBasedCF, UserBasedCF
        from pyspark import SparkContext
        import numpy as np

        train_data = list(set([
            tuple(np.random.randint(0, 100, 2)) + (np.random.randint(0, 5),)
            for _ in range(2000)
        ]))

        test_data = list(set([
            tuple(np.random.randint(0, 100, 2))
            for _ in range(200)
        ]))
        train_data = SparkContext.getOrCreate().parallelize(train_data).cache()
        test_data = SparkContext.getOrCreate().parallelize(test_data).cache()

        result1 = (
            UserBasedCF(k=float("inf"), n_user_block=4, n_cross_block=4, n_item_block=4,
                        maximum_num_partitions=10, threshold=0.01, n_bands=100, signature_length=200, seed=0)
            .fit_predict(train_data, test_data)
        )

        result2 = (
            UserBasedCF(k=float("inf"), n_user_block=10, n_cross_block=10, n_item_block=10,
                        maximum_num_partitions=10, threshold=0.01, n_bands=100, signature_length=200, seed=0)
            .fit_predict(train_data, test_data)
        )

        self.assertLessEqual(
            UserBasedCF.evaluate(result1, result2), 10**-4
        )

        result3 = (
            ItemBasedCF(k=2, n_user_block=4, n_cross_block=4, n_item_block=4, maximum_num_partitions=10,
                        threshold=0.01, n_bands=100, signature_length=200, seed=0)
            .fit_predict(train_data, test_data)
        )

        result4 = (
            ItemBasedCF(k=2, n_user_block=10, n_cross_block=10, n_item_block=10, maximum_num_partitions=10,
                        threshold=0.01, n_bands=100, signature_length=200, seed=0)
            .fit_predict(train_data, test_data)
        )

        self.assertLessEqual(
            ItemBasedCF.evaluate(result3, result4), 10**-4
        )


if __name__ == '__main__':
    unittest.main()
