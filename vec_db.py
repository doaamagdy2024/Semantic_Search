from typing import Dict, List, Annotated
import numpy as np
from math import log2
from random import random
from math import ceil
from sklearn.cluster import KMeans
import heapq
import pickle
import os
# in this file we will implement a better version of VecDB
# we have to use better indexing method to speed up the retrival process
# we will use FAISS to build index


import shutil
import os

def copy_files(src_folder, dest_folder):
    # Make sure the source folder exists
    if not os.path.exists(src_folder):
        print(f"Source folder '{src_folder}' does not exist.")
        return

    # Make sure the destination folder exists, create it if necessary
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Get a list of all files in the source folder
    files = os.listdir(src_folder)

    # Copy each file from the source folder to the destination folder
    for file in files:
        src_path = os.path.join(src_folder, file)
        dest_path = os.path.join(dest_folder, file)
        shutil.copy(src_path, dest_path)
        #print(f"File '{file}' copied to '{dest_folder}'.")

class vector:
    def __init__(self, id, vect) -> None:
        self.id = id
        self.vect = vect
        self.centroid = None # the centroid that this vector belongs to


class VecDB:
    def __init__(self, file_path = "10K", new_db = True) -> None:
        
        # hyperparameters
        self.num_vectors_per_cluster = 200
        self.centroids = []
        self.kmeans = None
        # we will store the clusters in files
        # each file will have a centroid id
        # and the vectors that belong to that centroid

        # file path to save the db        
        self.file_path = file_path
        if new_db:
            # delete files in the folder if exist
            if os.path.exists(self.file_path):
                for file in os.listdir(self.file_path):
                    os.remove(f"{self.file_path}/{file}")
    
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]], src = "10K", dest = "10K", new_db = True): # anonoated is a type hint means that the list has 70 elements of type float
        # create a list to store all the vectors
        db_vectors = []
        with open(f"{self.file_path}/old_db.csv", "w") as fout:
            for row in rows:
                id, embed = row["id"], row["embed"]
                row_str = f"{id}," + ",".join([str(e) for e in embed])
                fout.write(f"{row_str}\n")
                v = vector(id, embed)
                db_vectors.append(v)
        # build index
        self._build_index(db_vectors, src, dest, new_db)

    # TODO: change this function to retreive from the indexed Inverted file index db
    def retrive(self, query: Annotated[List[float], 70], top_k = 5):
        # now we need to find the n closest centroids to the query
        # we will use the kmeans model to find the closest centroids
        # gen n centroids where n is a hyperparameter
        n = 5 # number of nearest centroids to get
        ###########################################################################
        # load the kmeans model from the pickle file
        with open(f"{self.file_path}/old_kmeans.pickle", "rb") as fin:
            self.kmeans = pickle.load(fin)
        ###########################################################################
        
        nearest_centroids = sorted(self.kmeans.cluster_centers_, key=lambda centroid: self._cal_score(query, centroid), reverse=True)[:n]
        # # now we need to get the label of each centroid
        nearest_centroids = [self.kmeans.predict([centroid])[0] for centroid in nearest_centroids]
        print("nearest_centroids_kmeans", nearest_centroids)

        # now we need to search in the files of the nearest centroids
        # we will get the top k vectors from each file
        # then we will sort them by their score
        # then we will return the top k vectors
        # we will use a heap to do that cause it is faster
        heap = [] 
        heapq.heapify(heap)
        for centroid in nearest_centroids:
            # open the file of the centroid
            f = open(f"{self.file_path}/cluster_{centroid}.csv", "r")
            # read the file line by line
            while True:
                line = f.readline()
                if not line:
                    break
                # split the line to get the id and the vector
                # the first number will be the id
                # the rest of the numbers will be the vector
                id = int(line.split(",")[0])
                vect = [float(e) for e in line.split(",")[1:]]
                # calculate the score of the vector
                score = self._cal_score(query, vect)
                # add it to the heap
                heapq.heappush(heap, (-score, id, vect))
            f.close()
        # now we have the top k vectors in the heap
        # we will pop them from the heap and return them
        ids_scores = []
        for _ in range(top_k):
            score, id, vect = heapq.heappop(heap)
            ids_scores.append(id)
        #################################################
        # now we have the top k ids
        # write them to a csv file
        with open("test/old_ids.csv", "w") as fout:
            for id in ids_scores:
                fout.write(f"{id}\n")
        #################################################
        return ids_scores


    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
    
    
    def build_part_of_db(self, db_vectors, src, dest):
        # first copy the files of the clusters from the old db
        copy_files(src, dest)
        # now we have the centorids and the clusters
        # we need to assign each vector to the closest centroid
        # we will use the kmeans model to find the closest centroid of each vector
        # then insert this vector to the file of this centroid

        for vec in db_vectors:
            centroid = self.kmeans.predict([vec.vect])[0]
            # now we have the centroid
            # we need to insert this vector to the file of this centroid
            with open(f"{dest}/cluster_{centroid}.csv", "a") as fout:
                row_str = f"{vec.id}," + ",".join([str(e) for e in vec.vect])
                fout.write(f"{row_str}\n")

        print("Done building part of db")


    def _build_index(self, db_vectors, src, dest, new_db = True):
        # now let's create the centroids on part of the vectors only to speed up the process
        if new_db == False:
            self.build_part_of_db(db_vectors, src, dest)
            return

        # number of vectors to use to create the centroids
        n_vectors_train = ceil(len(db_vectors) * 0.5)
        
        num_centroids = ceil(n_vectors_train / self.num_vectors_per_cluster)

        print("n_vectors_train", n_vectors_train)
        print("num_centroids", num_centroids)

        self.kmeans = KMeans(n_clusters=num_centroids, random_state=0).fit([vec.vect for vec in db_vectors[:n_vectors_train]]) 
        
        #self.kmeans = KMeans(n_clusters=num_centroids, random_state=0).fit([vec.vect for vec in db_vectors])
        
        # now Kmeans has the centroids
        # now we need to assign each vector to the closest centroid
        self.centroids = self.kmeans.cluster_centers_
        clusters = {}
        
        print("self.centroids", self.centroids)
        # we can find the closest centroid by fit function
        for vec in db_vectors:
            centroid = self.kmeans.predict([vec.vect])[0]
            clusters[centroid] = clusters.get(centroid, []) + [vec]

        # now store each cluster in a file
        # create a file for each centroid
        print("Start storing index")
        for centroid in clusters:
            with open(f"{self.file_path}/cluster_{centroid}.csv", "w") as fout:
                for vec in clusters[centroid]:
                    row_str = f"{vec.id}," + ",".join([str(e) for e in vec.vect])
                    fout.write(f"{row_str}\n")

        ##################################################################################
        # store the centroids in a csv file
        with open("test/old_centroids.csv", "w") as fout:
            for centroid in self.centroids:
                row_str = ",".join([str(e) for e in centroid])
                fout.write(f"{row_str}\n")

        ##################################################################################
        # save the kmeans model to a pickle file
        with open(f"{self.file_path}/old_kmeans.pickle", "wb") as fout:
            pickle.dump(self.kmeans, fout)
            
        print("Done building index")
