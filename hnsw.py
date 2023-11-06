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
        


class VecDBhnsw:
    def __init__(self, file_path = "saved_db.csv", new_db = True) -> None:
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
                n = node(0, id, embed)
                db_vectors.append(n)
        # build index
        self._build_index(db_vectors)

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

    def _build_index(self, db_vectors):
        # we will use hnsw to build index
        # first we need to load all the vectors from the file
        # and during the loading process, we will build the index
        # the implementation of hnsw is as follows:
        # 1. build a graph with each node representing a vector
        # the structure is a list of layers and each layer is a list of nodes and each node is a list of k connected neighbors
        hnsw_structure: List[List[node]] = []
        # let's make the structure list of lists of dicts where the key is the node id and the value is the node
        #hnsw_structure: List[List[Dict[int, node]]] = []
        # 2. for each layer, we will build a graph with each node representing a vector

        # hyperparameters
        m = 5  # the number of neighbors to be connected
        ef = 200
        m0 = 2*m
        level_mult = 1 / log2(m)
        it = -1

        for v in db_vectors:

                        

            # print("inserting vector", v.index)
            # print("the vector is ", v)
            # generate a random number l that is smaller than or equal the number of layers
            level = int(-log2(random()) * level_mult) + 1
            # print("the level", level)

            # case 1
            # the level is empty, we will the required number of layers
            layers = max(len(hnsw_structure), level)

            if level > len(hnsw_structure):
                add_layers = 0
                layer_index = len(hnsw_structure)
                for i in range(level - len(hnsw_structure) + 1):
                    hnsw_structure.append([])
                    # then insert this vector into the new level
                    # print("appending the vector", v)
                    # print("the layer", i)
                    v.index = len(hnsw_structure[layer_index])
                    # print("the index during inserting ", v.index)
                    # print("the hnsw layer during inserting", hnsw_structure[layer_index])
                    
                    hnsw_structure[layer_index].append(v)
                    layer_index += 1
                    add_layers += 1
                layers -= add_layers
                # print("after appending layers", hnsw_structure)
                if layers < 0:
                    continue

            if len(hnsw_structure) == 0:
                continue
            
            # print("the layers", layers)
            # print("the the hnsw before inserting the vector ", hnsw_structure)
            entry_node: node = hnsw_structure[layers-1][0]
            nearest_neighbor_score = self._cal_score(v.vect, entry_node.vect)
            for layer in range(layers-1, -1, -1):
                it += 1
                # let's make sure that all the neighbors of each node in each layer has index less than or equal to the layer length
                for l in hnsw_structure:
                    for node in l:
                        for neighbor in node.neighbors:
                            if neighbor.index > len(l):
                                print("error in it ", it)
                                print("the node index", node.index)
                                print("the layer length", len(l))
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
                    

                    iterations = min(ef, len(hnsw_structure[layer]))
                    sorted_distances = []
                    for i in range(iterations):
                        distances = dict()
                        for node in entry_node.neighbors:
                            # calculate the distance between v and each node in this layer
                            # and find the nearest one
                            distance = self._cal_score(v.vect, node.vect)
                            node_index = node.index
                            distances[node_index] = distance
                            #distances.append(self._cal_score(v.vect, node.vect))
                        # find the nearest neighbor
                        if len(distances) == 0:
                            break
                        # sort the distances accroding to the value not the key
                        sorted_distances = sorted(distances.items(), key=lambda x: x[1]) # will distances be sorted? actually it will not be sorted
                        min_score = sorted_distances[0][1]
                        if min_score <= nearest_neighbor_score:
                            nearest_neighbor_score = min_score
                            entry_node = hnsw_structure[layer][sorted_distances[0][0]]
                            neighbors_to_connect.append(entry_node)
                        else:
                            break
                    # connect v and its nearest m neighbors
                    k = min(M, len(neighbors_to_connect))
                    # print("number of k neighbors", k)
                    # print("the neighbors to connect", neighbors_to_connect[:k])
                    v.neighbors = neighbors_to_connect[:k]
                    v.index = len(hnsw_structure[layer])
                    # print("the index during inserting ", v.index)
                    # print("the hnsw layer during inserting", hnsw_structure[layer])

                    hnsw_structure[layer].append(v)
                    # print("after appending the vector", hnsw_structure)
                    # now insert this node in the neighbors of its nearest neighbors
                    for neighbor in sorted_distances[:k]:
                        if v.index >= len(hnsw_structure[layer]):
                            print("error in inserting the vector", v.index)
                        hnsw_structure[layer][neighbor[0]].neighbors.append(v)

                    #hnsw_structure[layer][v.index].append(neighbors_to_connect[:k])
                # if not, we will find the nearest neighbor of v in this layer to be the entry node of the next layer
                else:
                    sorted_distances = []
                    iterations = min(ef, len(hnsw_structure[layer]))
                    for i in range(iterations):
                        distances = {}
                        for node in entry_node.neighbors:
                            # calculate the distance between v and each node in this layer
                            # and find the nearest one
                            distance = self._cal_score(v.vect, node.vect)
                            node_index = node.index
                            distances[node_index] = distance                        # find the nearest neighbor
                        if len(distances) == 0:
                            break
                        sorted_distances = sorted(distances.items(), key=lambda x: x[1]) # will distances be sorted? actually it will not be sorted
                        min_score = sorted_distances[0][1]
                        # print("the sorted distances", sorted_distances)
                        if min_score <= nearest_neighbor_score:
                            nearest_neighbor_score = min_score
                            # print("I am here ", hnsw_structure[layer])
                            entry_node = hnsw_structure[layer][sorted_distances[0][0]]
                        else:
                            break


        # print("finish building index")
        # print(hnsw_structure)


                


