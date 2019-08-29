import unittest


class TestJaccardSimilarityMethod(unittest.TestCase):
    def test_random_data(self):
        from datming.similarity import JaccardSimilarity, jaccard_similarity
        from pyspark import SparkContext
        import numpy as np
        test_data = [
                        (_, set(np.random.randint(0, 500, np.random.randint(5, 20))))
                        for _ in range(1000)
        ]

        _threshold = 0.1
        result = JaccardSimilarity(threshold=_threshold, n_bands=200, signature_length=400).predict(
            SparkContext.getOrCreate().parallelize(test_data)
        ).collect()
        SparkContext.getOrCreate().stop()

        truth = set()
        for i, arr1 in test_data[:-1]:
            for j, arr2 in test_data[i + 1:]:
                if jaccard_similarity(arr1, arr2) >= _threshold:
                    truth.add((i, j))
        lsh_result = set([
            (i, j) for i, j, _ in result
        ])

        print("number of LSH-selected pairs: ", len(lsh_result))
        print("number of true pairs: ", len(truth))
        TP_rate = len(lsh_result & truth) / (len(truth) + 10**-10)
        FN_rate = len(truth - lsh_result) / (len(truth) + 10**-10)
        print("TP rate=", TP_rate)
        print("FN rate=", FN_rate)
        del lsh_result

        test_data = dict(test_data)
        sum_square_error = 0
        count = 0
        for itemA, itemB, lsh_similarity in result:
            truth = len(set(test_data[itemA]) & set(test_data[itemB])) / \
                    len(set(test_data[itemA]) | set(test_data[itemB]))
            sum_square_error += (lsh_similarity - truth) ** 2
            count += 1
        print("RMSE=", np.sqrt(sum_square_error / count))
        self.assertLessEqual(FN_rate, 0.5)
        self.assertLessEqual(0.5, TP_rate)

    def test_simple_data(self):
        from datming.similarity import JaccardSimilarity
        from pyspark import SparkContext
        data = SparkContext.getOrCreate().parallelize([(1, (1, 2, 3)), (2, (1, 2, 4)), (3, (1, 3))])
        print(JaccardSimilarity(threshold=0.5, n_bands=10, signature_length=20, seed=0).predict(data).collect())
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
