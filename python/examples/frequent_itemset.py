"""
Example of A-Priori Algorithm.
"""
from datming.freq_itemset import *
from pyspark import SparkContext
import os
import json
import time
import sys


class DataPrepare(object):
    def __init__(self, path, k=70):
        self.threshold = k
        self.output_path = os.path.join(path, "yelp_processed/buckets.txt")
        if not os.path.exists(self.output_path):
            self.fetch_data_path(path)

        self.iterable = self.read_data()

    def fetch_data_path(self, path):
        file_path = os.path.join(path, "yelp_dataset/review.json")
        business = os.path.join(path, "yelp_dataset/business.json")
        output_path = os.path.join(path, "yelp_processed/buckets.txt")

        business_condition = self.business_condition(business)
        itemsets = self.itemsets(file_path, business_condition)
        self.write_data(output_path, itemsets)
        return output_path

    @staticmethod
    def business_condition(path):
        business_set = set()
        with open(path, "r") as  file:
            for line in file:
                _dict = json.loads(line.strip())
                if _dict["state"] == "NV":
                    business_set.add(_dict["business_id"])
        return business_set

    def itemsets(self, path, business_condition):
        index_dict = dict()
        itemsets = dict()
        count = 0
        with open(path, 'r') as file:
            for line in file:
                _dict = json.loads(line.strip())
                user_id = _dict["user_id"]
                business_id = _dict["business_id"]
                if business_id not in business_condition:
                    continue
                if business_id not in index_dict:
                    index_dict[business_id] = count
                    count += 1
                if user_id not in itemsets:
                    itemsets[user_id] = list()
                itemsets[user_id].append(index_dict[business_id])
        return [
            l for l in itemsets.values() if len(l) >= self.threshold
        ]

    @staticmethod
    def write_data(path, itemsets):
        with open(path, "w") as file:
            _iterator = itemsets.__iter__()
            bucket = next(_iterator)
            file.write(",".join([str(i) for i in bucket]))
            for bucket in _iterator:
                file.write("\n")
                file.write(",".join([str(i) for i in bucket]))

    def read_data(self):
        with open(self.output_path, "r") as file:
            iterable = [
                [int(item) for item in line.strip().split(",")]
                for line in file
            ]
        return iterable

    def __iter__(self):
        yield from self.iterable


class Configuration(object):
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.start_time = time.time()

    def __call__(self, func):
        def run(*args, **kwargs):
            start = time.time()
            func(self.start_time, self.input_path, self.output_path, *args, **kwargs)
            end = time.time()
            print("Duration:", (end - start))
        return run


def model(start_time, input_path, output_path, Model, **kwargs):
    file = DataPrepare(input_path)
    frequent_itemsets = Model(**kwargs).count(file)

    with open(os.path.join(output_path, "%s.txt" % Model.__name__), 'w') as file:
        for idx, itemsets in frequent_itemsets.items():
            print("Pass{0} finished, duration = {1}".format(idx + 1, time.time() - start_time))
            file.write(", ".join(["(%s)" % (",".join([str(i) for i in _tuple])) for _tuple in itemsets]))
            file.write("\n")



def spark_model(start_time, input_path, output_path, SparkModel, **kwargs):
    sc = SparkContext(appName="SON")
    frequent_itemsets = SparkModel(**kwargs).count(
        spark_context=sc,
        file=os.path.join(input_path, "yelp_processed/buckets.txt"))

    with open(os.path.join(output_path, "%s.txt" % SparkModel.__name__), 'w') as file:
        for idx,itemsets in frequent_itemsets.items():
            print("Pass{0} finished, duration = {1}".format(idx, time.time()-start_time))
            file.write(", ".join(["(%s)" % (",".join([str(i) for i in _tuple])) for _tuple in itemsets ]))
            file.write("\n")


if __name__ == "__main__":
    case = str(sys.argv[1]).strip().lower()
    conf = Configuration(
        input_path="../../input",
        output_path="../../output"
    )
    model_parameters = {
        "apriori": {
            "Model": APriori,
            "support": 70
        },
        "pcy": {
            "Model": PCY,
            "support": 70,
            "num_buckets": 1000,
        },
        "sampling": {
            "Model": RandomSampling,
            "support": 70,
            "sampling_rate": 0.1,
            "random_seed": 233
        },
        "toivonen": {
            "Model": Toivonen,
            "support": 100,
            "sampling_rate": 0.3,
            "adjust_rate": 0.8,
            "random_seed": 233
        },
        "eclat": {
            "Model": Eclat,
            "support": 70
        }
    }
    conf(model)(**model_parameters[case])

    if case.strip().lower() == "son":
        conf(spark_model)(SparkModel=SON, support=200, num_partitions=8)
