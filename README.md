### Massive Data Mining Algorithms

This repo is to implement data mining algorithms on massive data that can not be cached in memory. The algorithms are implemented mainly with Apache Spark. This project is not finished.

 Part of codes are from the assignments in INF 553 class (Spring19) at USC. The details of algorithms can be found in http://www.mmds.org and "The Mining of Massive Dataset" book.

### List of Algorithms

| Algorithm | Category| Python
| ---- | ---- |---- |
A-Priori | Frequent Item-set | [Local](python/datming/freq_itemset/apriori.py)
FP-Growth | Frequent Item-set | [Distributed](python/datming/freq_itemset/fpgrowth.py)
Eclat | Frequent Item-set | [Local](python/datming/freq_itemset/eclat.py)
PCY | Frequent Item-set | [Local](python/datming/freq_itemset/pcy.py)
SON | Frequent Item-set | [Distributed](python/datming/freq_itemset/son.py)
Toivonen | Frequent Item-set | [Local](python/datming/freq_itemset/toivonen.py)
LSH of Jaccard Similarity | Similar Items | [Distributed](python/datming/similar_items/jaccard_similarity.py)
LSH of Euclidean Similarity | Similar Items | [Distributed](python/datming/similar_items/lsh_euclidean.py)
LSH of Cosine Similarity | Similar Items | [Distributed](python/datming/similar_items/lsh_cosine.py)
Girvan-Newman Algorithm | Graph Community Searching | [Distributed](python/datming/graph/community/girvan_newman.py)
Spectral Clustering | Graph Community Searching | [Distributed](python/datming/graph/community/spectral_clustering.pu)
Big-CLAM | Graph Community Searching | [Distributed](python/datming/graph/community/big_clam.py)
PageRank | Link Analysis | [Distributed](python/datming/graph/link_analysis/page_rank.py)
Trust-Rank | Link Analysis | [Distributed](python/datming/graph/link_analysis/trust_rank.py)
K-Means | Clustering | [Distributed](python/datming/clustering/k_means.py)
BFR | Clustering | [Local](python/datming/clustering/bfr.py)
Neighborhood-based CF | Recommender System | [Distributed](python/datming/recommender/neighbohood_based_cf.py)

### Process
In the present, I'm still debugging, testing the codes. More algorithm implementations are on the track.
