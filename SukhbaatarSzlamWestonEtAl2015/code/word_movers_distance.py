from collections import Counter
import numpy as np
#from sklearn.metrics import euclidean_distances
from pyemd import emd as pyemd

def distance_matrix(pts):
    """Returns matrix of pairwise Euclidean distances. Vectorized numpy version."""
    return np.sum((pts[None,:] - pts[:, None])**2, -1)**0.5


def word_movers_distance(a, b, embeddings):
    """Word Mover's Distance.
    
    A measure of text similarity: earth mover's distance in embedding metric space.
    Computes the distance between texts `a` and `b`.
    
    Parameters
    ----------
    a: iterable
        One of the two documents to compute similarity between.
        Represented as a sequence of indexes into the `embeddings`.
    b: iterable
        The other document to compute similarity between.
    embeddings: array-like, shape: (vocab_size, n_features)
        Word representations. The embedding of the first word in `a`
        should be `embeddings[a[0]]`.
    
    Returns
    -------
    distance: double,
        The distance between `a` and `b`.
    
    References
    ----------
    Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, Kilian Q. Weinberger,
    From Word Embeddings To Document Distances. ICML 2015
    http://matthewkusner.com/publications/WMD.pdf
    
    Notes
    -----
    This implementation is OK for one-off cases.  If looping over multiple documents,
    precomputing can speed things up. (Code coming soon.)
    
    
    """
    counter_a = Counter(a)
    counter_b = Counter(b)
    uniq_words = list(set(counter_a.keys()).union(counter_b.keys()))

    bow_a = np.array([counter_a[w] for w in uniq_words], dtype=np.double)
    bow_b = np.array([counter_b[w] for w in uniq_words], dtype=np.double)

    bow_a /= bow_a.sum()
    bow_b /= bow_b.sum()

    print(embeddings[uniq_words].shape)
    print(embeddings[uniq_words])
    D = distance_matrix(embeddings[uniq_words])
    print(uniq_words)
    print(D)
    print(bow_a)
    print(bow_b)
    return pyemd(bow_a, bow_b, D)


if __name__ == '__main__':
    from time import time
    vocab_size = 50
    n_words = 5
    
    # make up random word embeddings
    embeddings = np.random.randn(vocab_size, 100)
    embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    
    # make up two random texts
    a = np.random.random_integers(vocab_size, size=n_words) - 1 
    b = np.random.random_integers(vocab_size, size=n_words) - 1
    
    t0 = time()
    print(a)
    print(b)
    print(embeddings.shape)
    dist = word_movers_distance(a, b, embeddings)
    t = time() - t0
    print("Distance = {:.4f}, computed in {:.2}s.".format(dist, t))
