from typing import Dict, List, Annotated
import numpy as np
from math import log2
from random import random
from math import ceil

# in this file we will implement a better version of VecDB
# we have to use better indexing method to speed up the retrival process
# we will use FAISS to build index

class vector:
    def __init__(self, id, vect) -> None:
        self.id = id
        self.vect = vect
        self.centroid = None # the centroid that this vector belongs to


class VecDBIF:
    def __init__(self, file_path = "saved_db.csv", new_db = True) -> None:
        
        # hyperparameters
        self.num_vectors_per_cluster = 10000
        self.centroids = []
        # we will store the clusters in files
        # each file will have a centroid id
        # and the vectors that belong to that centroid

        # file path to save the db        
        self.file_path = file_path
        if new_db:
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass
    
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]): # anonoated is a type hint means that the list has 70 elements of type float
        # create a list to store all the vectors
        db_vectors = []
        with open(self.file_path, "a+") as fout:
            for row in rows:
                id, embed = row["id"], row["embed"]
                row_str = f"{id}," + ",".join([str(e) for e in embed])
                fout.write(f"{row_str}\n")
                v = vector(id, embed)
                db_vectors.append(v)
        # build index
        self._build_index(db_vectors)

    # TODO: change this function to retreive from the indexed Inverted file index db
    def retrive(self, query: Annotated[List[float], 70], top_k = 5):
        return [int(random() * 1000) for _ in range(top_k)]
        pass
    
    

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self, db_vectors):
        # fist pick random centroids
        num_centroids = ceil(len(db_vectors) / self.num_vectors_per_cluster)
        print("num_centroids", num_centroids)
        centroids = []
        clusters = {}
        for _ in range(num_centroids):
            # we have to make sure that the centroid is not already picked
            centroid_i = int(random() * len(db_vectors))
            while db_vectors[centroid_i] in centroids:
                centroid_i = int(random() * len(db_vectors))
            centroids.append(db_vectors[centroid_i])

        # now we have the centroids, then we have to assign each vector to the closest centroid and then recalculate the centroids again and again until the centroids are not changing
        # to do that we will keep track of the old centroids
        old_centroids = []
        while centroids != old_centroids:
            old_centroids = centroids
            clusters = {}
            for vec in db_vectors:
                best_centroid = None
                best_score = -1
                for centroid in centroids:
                    score = self._cal_score(vec.vect, centroid.vect)
                    if score > best_score:
                        best_score = score
                        best_centroid = centroid
                clusters[best_centroid] = clusters.get(best_centroid, []) + [vec]
            # now update the centroids by taking the mean of the vectors in the cluster
            for centroid in clusters:
                mean = np.mean([vec.vect for vec in clusters[centroid]], axis=0)
                centroid.vect = mean

        self.centroids = centroids

        # now we have the clusters
        # we will save the clusters in files
        # each file will have a centroid id
        # and the vectors that belong to that centroid

        # create a file for each centroid
        print("Start storing index")

        for centroid in clusters:
            with open(f"cluster_{centroid.id}.csv", "w") as fout:
                for vec in clusters[centroid]:
                    row_str = f"{vec.id}," + ",".join([str(e) for e in vec.vect])
                    fout.write(f"{row_str}\n")

        print("Done building index")
