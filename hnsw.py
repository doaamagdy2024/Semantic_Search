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
        self.hnsw_structure: List[Dict[int, node]]= []
        # hyperparameters
        self.m = 5  # the number of neighbors to be connected
        self.ef = 200 # the number of neighbors to be explored - ef stands for exploration factor
        self.ef_search = 100 # the number of neighbors to be explored during search
        self.m0 = 2*self.m # the number of neighbors to be connected in the first layer
        self.level_mult = 1 / log2(self.m) # the number of layers to be added to the graph
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

    # TODO: change this function to retreive from the indexed hnsw db
    def retrive(self, query: Annotated[List[float], 70], top_k = 5):
        for layer in range(len(self.hnsw_structure)):
            print("the layer number is: ", layer)
            print("the layer nodes are: ", self.hnsw_structure[layer].keys())
        print("start retriving")

        scores = []
        # now we will search for the top k most similar vectors to the query vector from the db hnsw
        entry_node: node = self.hnsw_structure[0].get(list(self.hnsw_structure[0].keys())[0])
        min_score = self._cal_score(query, entry_node.vect)
        # scores.append((min_score, entry_node.id))
        for layer in range(len(self.hnsw_structure)):
        # for layer in self.hnsw_structure:
            # now search in all the neighbors of the entry node and find the min
            for neighbor in entry_node.neighbors:
                score = self._cal_score(query, neighbor.vect)
                if score < min_score:
                    min_score  = score
                    entry_node = neighbor

            
            if layer == len(self.hnsw_structure) - 1:
                break

            entry_node = self.hnsw_structure[layer+1].get(entry_node.id)

        # now we have the entry node of the last layer which is the nearest neighbor (greadily) of the query vector
        # we will search in the neighbors of this node to find the top k most similar vectors to the query vector
        # we will use the same greedy search
        print("the nearest neighbor is: ", entry_node.id)
        print("the nearest neighbor score is: ", min_score)
        
        # we will find the min neighbor score in the neighbors of the entry node
        min_score = float("inf")
        visited_nodes = set()
        visited_nodes.add(entry_node.id)
        while len(scores) < top_k:
            print("the new entry node is: ", entry_node.id)

            for neighbor in entry_node.neighbors:
                score = self._cal_score(query, neighbor.vect)
                if score < min_score:
                    min_score  = score
                    entry_node = neighbor
                scores.append((score, entry_node.id))
            scores = sorted(scores, key=lambda x: x[0], reverse=True)

            # now take the nearest neighbor to the query vector from the neighbors of the entry node
            # to be the entry node of the next iteration so that we explore the neighbors of this node
            new_node_to_visit = False
            for node in scores:
                if node[1] not in visited_nodes:
                    entry_node = self.hnsw_structure[len(self.hnsw_structure) - 1].get(node[1])
                    visited_nodes.add(node[1])
                    new_node_to_visit = True
                    break
            if new_node_to_visit == False:
                break
                    

        print(scores)
        l = [s[1] for s in scores]
        print(l)
        return [s[1] for s in scores]
    

    def retrive2(self, query: Annotated[List[float], 70], top_k = 5):
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
        self.hnsw_structure = []
        # 2. for each layer, we will build a graph with each node representing a vector
        #print("the total number of vectors is: ", len(db_vectors))
        for v in db_vectors:
            # generate a random number l that is smaller than or equal the number of layers
            level = int(-log2(random()) * self.level_mult) + 1

            # case 1
            # the level is empty, we will the required number of layers
            layers = max(len(self.hnsw_structure), level)

            if level > len(self.hnsw_structure):
                add_layers = 0
                layer_index = len(self.hnsw_structure)
                for i in range(level - len(self.hnsw_structure)):
                    #self.hnsw.structure.insert(0, {})
                    self.hnsw_structure.append({})
                    # then insert this vector into the new level
                    self.hnsw_structure[layer_index][v.id] = v
                    layer_index += 1
                    add_layers += 1
                layers -= add_layers
                if layers < 0:
                    continue

            if len(self.hnsw_structure) == 0:
                continue
            
            entry_node: node = self.hnsw_structure[layers-1].get(list(self.hnsw_structure[layers-1].keys())[0])
            if entry_node is None:
                print("error_0")
            nearest_neighbor_score = self._cal_score(v.vect, entry_node.vect)

            for layer in range(layers-1, -1, -1):
                if layer == 0:
                    M = self.m0
                else:
                    M = self.m

                # check if this vector should be inserted in this layer
                if layer < level:
                    neighbors_to_connect = []
                    neighbors_to_connect.append(entry_node)             

                    iterations = min(self.ef, len(self.hnsw_structure[layer]))
                    sorted_distances = []
                    for i in range(iterations):
                        distances = dict()
                        for node in entry_node.neighbors:
                            # calculate the distance between v and each node in this layer
                            # and find the nearest one
                            distance = self._cal_score(v.vect, node.vect)
                            distances[node.id] = distance
                        # find the nearest neighbor
                        if len(distances) == 0:
                            self.hnsw_structure[layer][v.id] = v
                            if layer > 0:
                                entry_node = self.hnsw_structure[layer-1].get(entry_node.id)
                                neighbors_to_connect.append(entry_node)
                                break
                            break

                        # sort the distances accroding to the value not the key
                        sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=True) # will distances be sorted? actually it will not be sorted
                        min_score = sorted_distances[0][1]
                        if min_score <= nearest_neighbor_score:
                            nearest_neighbor_score = min_score
                            if layer > 0:
                                entry_node = self.hnsw_structure[layer-1].get(sorted_distances[0][0])
                                neighbors_to_connect.append(entry_node)
                        else:
                            break
                    if len(sorted_distances) == 0:
                        continue
                    # connect v and its nearest m neighbors
                    k = min(M, len(neighbors_to_connect))
                    v.neighbors = neighbors_to_connect[:k]
                    self.hnsw_structure[layer][v.id] = v
                    # now insert this node in the neighbors of its nearest neighbors
                    for neighbor in sorted_distances[:k]:
                        if neighbor:
                            self.hnsw_structure[layer].get(neighbor[0]).neighbors.append(v)

                # if not, we will find the nearest neighbor of v in this layer to be the entry node of the next layer
                else:
                    sorted_distances = []
                    iterations = min(self.ef, len(self.hnsw_structure[layer]))
                    for i in range(iterations):
                        distances = {}
                        for node in entry_node.neighbors:
                            # calculate the distance between v and each node in this layer
                            # and find the nearest one
                            distance = self._cal_score(v.vect, node.vect)
                            distances[node.id] = distance                        # find the nearest neighbor
                        if len(distances) == 0:
                            entry_node = self.hnsw_structure[layer-1].get(entry_node.id)
                            break
                        sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=True) # will distances be sorted? actually it will not be sorted
                        min_score = sorted_distances[0][1]
                        if min_score <= nearest_neighbor_score:
                            nearest_neighbor_score = min_score
                            if layer > 0:
                                entry_node = self.hnsw_structure[layer-1].get(sorted_distances[0][0])
                        else:
                            break
        self.hnsw_structure.reverse()

        #print("the size of the vectors in the lowest layer is: ", len(self.hnsw_structure[0]))
        
        #print(self.hnsw_structure)
        print("finished building index")