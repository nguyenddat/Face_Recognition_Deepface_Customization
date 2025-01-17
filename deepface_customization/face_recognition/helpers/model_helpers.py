import os
from typing import *

import numpy as np
import pickle

def l2_normalize(
    x: Union[np.ndarray, list],
    axis: Union[int, None] = None,
    epsilon: float = 1e-10
) -> np.ndarray:
    x = np.asarray(x)
    norm = np.linalg.norm(x, axis = axis, keepdims = True)
    return x / (norm + epsilon)

def find_cosine_distance(
    alpha_embedding: Union[np.ndarray, list], 
    beta_embedding: Union[np.ndarray, list]
) -> Union[np.float64, np.ndarray]:
    alpha_embedding = np.asarray(alpha_embedding)
    beta_embedding = np.asarray(beta_embedding)

    if alpha_embedding.ndim == 1 and beta_embedding.ndim == 1:
        dot_product = np.dot(alpha_embedding, beta_embedding)
        alpha_norm = np.linalg.norm(alpha_embedding)
        beta_norm = np.linalg.norm(beta_embedding)
        distances = 1 - dot_product / (alpha_norm * beta_norm)
    elif alpha_embedding.ndim == 2 and beta_embedding.ndim == 2:
        alpha_norm = l2_normalize(alpha_embedding, axis = 1)
        beta_norm = l2_normalize(beta_embedding, axis = 1)
        cosine_similarities = np.dot(alpha_norm, beta_norm.T)
        distances =  1 - cosine_similarities
    else:
        raise ValueError("Embeddings must be 1D or 2D")
    
    return distances
    
def find_euclidean_distance(
    alpha_embedding: Union[np.ndarray, list], 
    beta_embedding: Union[np.ndarray, list]
) -> Union[np.float64, np.ndarray]:
    alpha_embedding = np.asarray(alpha_embedding)
    beta_embedding = np.asarray(beta_embedding)

    if alpha_embedding.ndim == 1 and beta_embedding.ndim == 1:
        distances = np.linalg.norm(alpha_embedding - beta_embedding)
    elif alpha_embedding.ndim == 2 and beta_embedding.ndim == 2:
        diff = (
            alpha_embedding[None, :, :] - beta_embedding[:, None, :]
        )  
        distances = np.linalg.norm(diff, axis=2)
    else:
        raise ValueError("Embeddings must be 1D or 2D, but received")
    return distances

def find_distance(
    alpha_embedding: Union[np.ndarray, list],
    beta_embedding: Union[np.ndarray, list],
    distance_metric: str
) -> Union[np.float64, np.ndarray]:
    alpha_embedding = np.asarray(alpha_embedding) 
    beta_embedding = np.asarray(beta_embedding) 
    
    if (alpha_embedding.ndim != beta_embedding.ndim) or (alpha_embedding.ndim not in (1, 2)):
        raise ValueError("Both embeddings must be 1D or 2D")
    
    if distance_metric == "cosine":
        distance = find_cosine_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean":
        distance = find_euclidean_distance(alpha_embedding, beta_embedding)
    else:
        raise ValueError("Invalid distance metric passed")
    return np.round(distance, 6)

def find_threshold(
    model_name: str,
    distance_metric: str
) -> float:
    base_threshold = {"cosine": 0.40, "euclidean": 0.55}

    thresholds = {
        "VGG-Face": {
            "cosine": 0.68,
            "euclidean": 1.17
        }
    }

    threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)
    return threshold
