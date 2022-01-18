Text Clustering in flair
----------

In this package text clustering is implemented. This module has the following
clustering algorithms implemented:
- k-Means
- BIRCH
- Expectation Maximization

Each of the implemented algorithm needs to have an instanced DocumentEmbedding. This embedding will 
transform each text/document to a vector. With these vectors the clustering algorithm can be performed.

---------------------------

k-Means
------
k-Means is a classical and well known clustering algorithm. k-Means is a partitioning-based Clustering algorithm. 
The user defines with the parameter *k* how many clusters the given data has. 
So the choice of *k* is very important. 
More about k-Means can be read on the official [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).


```
from flair.models import ClusteringModel
from flair.datasets import TREC_6
from flair.embeddings import SentenceTransformerDocumentEmbeddings
from sklearn.cluster import KMeans

embeddings = SentenceTransformerDocumentEmbeddings()

# store all embeddings in memory which is required to perform clustering
corpus = TREC_6(memory_mode='full').downsample(0.05)

model = KMeans(n_clusters=6)

clustering_model = ClusteringModel(
    model=model,
    corpus=corpus,
    label_type="question_class",
    embeddings=embeddings
)

clustering_model.fit()
```

BIRCH
---------
BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) is a hierarchical clustering algorithm. 
BIRCH is specialized to handle large amounts of data. BIRCH scans the data a single time and builds an internal data 
structure. This data structure contains the data but in a compressed way.
More about BIRCH can be read on the official [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html).



    from sklearn.cluster import Birch
    from flair.datasets import TREC_6
    from flair.embeddings import SentenceTransformerDocumentEmbeddings
    from flair.models import ClusteringModel

    embeddings = SentenceTransformerDocumentEmbeddings()
    
    # store all embeddings in memory which is required to perform clustering
    corpus = TREC_6(memory_mode='full').downsample(0.05)

    model = Birch(n_clusters=6)

    clustering_model = ClusteringModel(
        model=model,
        corpus=corpus,
        label_type="question_class",
        embeddings=embeddings
    )

    clustering_model.fit()



Expectation Maximization
--------------------------
Expectation Maximization (EM) is a different class of clustering algorithms called soft clustering algorithms. 
Here each point isn't directly assigned to a cluster by a hard decision. 
Each data point has a probability to which cluster the data point belongs. The Expectation Maximization (EM) 
algorithm is a soft clustering algorithm.
More about EM can be read on the official [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).



    from sklearn.mixture import GaussianMixture
    from flair.datasets import TREC_6
    from flair.embeddings import SentenceTransformerDocumentEmbeddings
    from flair.models import ClusteringModel
    
    embeddings = SentenceTransformerDocumentEmbeddings()

    # store all embeddings in memory which is required to perform clustering
    corpus = TREC_6(memory_mode='full').downsample(0.05)

    model = GaussianMixture(n_components=6)

    clustering_model = ClusteringModel(
        model=model,
        corpus=corpus,
        label_type="question_class",
        embeddings=embeddings
    )

    clustering_model.fit()


---------------------------

Evaluation
---------
The result of the clustering can be evaluated. For this we will use the
[NMI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html).
(Normalized Mutual Info) score.

    clustering_model.fit()
    clustering_model.evaluate()


The result of the evaluation can be seen below with the SentenceTransformerDocumentEmbeddings:


| Clustering Algorithm     |    Dataset    | NMI |
|--------------------------|:-------------:|----:|
| k Means                  | StackOverflow |   / |
| BIRCH                    | StackOverflow |   / |
| Expectation Maximization | 20News group  |   / |
