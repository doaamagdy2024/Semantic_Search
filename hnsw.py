from typing import Dict, List, Annotated
import numpy as np
from math import log2
from random import random

# in this file we will implement a better version of VecDB
# we have to use better indexing method to speed up the retrival process
# we will use FAISS to build index

class node:
    def __init__(self, index, id, vect: Annotated[List[float], 70]) -> None:
        self.index = index
        self.id = id
        self.vect =  vect
        self.neighbors = []


class VecDBWorst:
    def __init__(self, file_path = "saved_db.csv", new_db = True) -> None:
        self.file_path = file_path
        if new_db:
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass
    
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]): # anonoated is a type hint means that the list has 70 elements of type float
        # create a list to store all the vectors
        query_vectors = []
        with open(self.file_path, "a+") as fout:
            for row in rows:
                id, embed = row["id"], row["embed"]
                row_str = f"{id}," + ",".join([str(e) for e in embed])
                fout.write(f"{row_str}\n")
                query_vectors.append(embed)
        # build index                
        self._build_index()

    def retrive(self, query: Annotated[List[float], 70], top_k = 5):
        scores = []
        with open(self.file_path, "r") as fin:
            for row in fin.readlines():
                row_splits = row.split(",")
                id = int(row_splits[0])
                embed = [float(e) for e in row_splits[1:]]
                score = self._cal_score(query, embed)
                scores.append((score, id))
        # here we assume that if two rows have the same score, return the lowest ID
        scores = sorted(scores, reverse=True)[:top_k]
        return [s[1] for s in scores]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self, query_vector: list[node]):
        # we will use hnsw to build index
        # first we need to load all the vectors from the file
        # and during the loading process, we will build the index
        # the implementation of hnsw is as follows:
        # 1. build a graph with each node representing a vector
        # the structure is a list of layers and each layer is a list of nodes and each node is a list of k connected neighbors
        hnsw_structure: List[List[node]] = []
        # 2. for each layer, we will build a graph with each node representing a vector

        # hyperparameters
        m = 5  # the number of neighbors to be connected
        ef = 200
        m0 = 2*m
        level_mult = 1 / log2(m)

        for v in query_vector:
            # generate a random number l that is smaller than or equal the number of layers
            level = int(-log2(random()) * level_mult) + 1

            # case 1
            # the level is empty, we will the required number of layers
            layers = max(len(hnsw_structure), level)

            if level > len(hnsw_structure):
                add_layers = 0
                for i in range(level - len(hnsw_structure)):
                    hnsw_structure.append([])
                    # then insert this vector into the new level
                    hnsw_structure[layer].append(v)
                    add_layers += 1
                layers -= add_layers

            if len(hnsw_structure) == 0:
                continue

            entry_node: node = hnsw_structure[layers][0]
            nearest_neighbor_score = self._cal_score(v.vect, entry_node.vect)
            for layer in range(layers, 0):
                if layer == 0:
                    M = m0
                else:
                    M = m
                # check on ef nodes in this layer
                # find the nearest neighbor of v in this layer
                # and connect v and its nearest neighbor

                # check if this vector should be inserted in this layer
                if layer <= level:
                    neighbors_to_connect = []
                    neighbors_to_connect.append(entry_node)
                    iterations = max(ef, len(hnsw_structure[layer]))
                    for i in range(iterations):
                        distances = []
                        for node in entry_node.neighbors:
                            # calculate the distance between v and each node in this layer
                            # and find the nearest one
                            distances.append(self._cal_score(v.vect, node.vect))
                        # find the nearest neighbor
                        min_score = min(distances)
                        if min_score <= nearest_neighbor_score:
                            nearest_neighbor_score = min_score
                            entry_node = hnsw_structure[layer][distances.index(min_score)] 
                            neighbors_to_connect.append(entry_node)
                        else:
                            break
                    # connect v and its nearest m neighbors
                    k = min(M, len(neighbors_to_connect))
                    hnsw_structure[layer].append(v)
                    hnsw_structure[layer][v.index].append(neighbors_to_connect[:k])

                else:
                    iterations = max(ef, len(hnsw_structure[layer]))
                    for i in range(iterations):
                        distances = []
                        for node in entry_node.neighbors:
                            # calculate the distance between v and each node in this layer
                            # and find the nearest one
                            distances.append(self._cal_score(v.vect, node.vect))
                        # find the nearest neighbor
                        min_score = min(distances)
                        if min_score <= nearest_neighbor_score:
                            nearest_neighbor_score = min_score
                            entry_node = hnsw_structure[layer][distances.index(min_score)] 
                            neighbors_to_connect.append(entry_node)
                        else:
                            break


                


