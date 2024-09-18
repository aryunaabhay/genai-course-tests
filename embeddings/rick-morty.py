import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Union
import csv

MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

def read_quotes() -> list[str]:
    quotes = []
    with open('rick_and_morty_quotes.txt', 'r') as file:
        for line in file:
            quotes.append(line.strip())
    return quotes

def generate_embeddings(input_data: Union[str, list[str]]) -> np.ndarray:
   
    return model.encode(input_data)

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Euclidean distance between `v1` and `v2`.
    """
    dist = v1 - v2
    return np.linalg.norm(dist, axis=len(dist.shape)-1)


def find_nearest_neighbors(query: np.ndarray,
                           vectors: np.ndarray,
                           k: int = 1) -> np.ndarray:
    """
    Find k-nearest neighbors of a query vector.

    Parameters
    ----------
    query : np.ndarray
        Query vector.
    vectors : np.ndarray
        Vectors to search.
    k : int, optional
        Number of nearest neighbors to return, by default 1.

    Returns
    -------
    np.ndarray
        The indices of the `k` nearest neighbors of `query` in `vectors`.
    """
    distances = euclidean_distance(query, vectors)
    indices = np.argsort(distances)[:k]
    return indices


rick_and_morty_quotes = read_quotes()
embeddings = generate_embeddings(rick_and_morty_quotes)


'''for sentence, embedding in zip(rick_and_morty_quotes[:3], embeddings[:3]):
    dimensions = embedding.shape
    print(dimensions)
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
'''

query_text = "Are you the cause of your parents' misery?"
query_embedding = model.encode(query_text)
indices = find_nearest_neighbors(query_embedding, embeddings, k=3)

for i in indices:
    print(rick_and_morty_quotes[i])

