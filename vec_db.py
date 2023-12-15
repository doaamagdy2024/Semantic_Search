from typing import Dict, List, Annotated
import numpy as np
from math import log2
from random import random
from math import ceil
from sklearn.cluster import KMeans
import heapq
import os
import pickle
import scipy.cluster.vq as VQ
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

# in this file we will implement a better version of VecDB
# we have to use better indexing method to speed up the retrival process
# we will use inverted file index

class vector:
    def __init__(self, id, vect) -> None:
        self.id = id
        self.vect = vect
        self.centroid = None # the centroid that this vector belongs to


class VecDB:
    def __init__(self, files_path = "saved_db.csv",  new_db = True) -> None:
        # hyperparameters
        self.num_vectors_per_cluster = 50
        self.centroids = []
        # we will store the clusters in files
        # each file will have a centroid id
        # and the vectors that belong to that centroid

        # file path to save the db        
        self.files_path = files_path
        if new_db:
            # delete files in the folder if exist
            if os.path.exists(self.files_path):
                for file in os.listdir(self.files_path):
                    os.remove(f"{self.files_path}/{file}")
    # add parameter to check if not the first time to insert records
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]], first_insert = True): # anonoated is a type hint means that the list has 70 elements of type float
        # create a list to store all the vectors
        db_vectors = []
        with open(f"{self.files_path}/saved_db.csv", "a+") as fout:
            # the previous line will open the file in append mode
            for row in rows:
                id, embed = row["id"], row["embed"]
                row_str = f"{id}," + ",".join([str(e) for e in embed])
                fout.write(f"{row_str}\n")
                v = vector(id, embed)
                db_vectors.append(v)
                
        # build index
        self._build_index(db_vectors, build_on_part = not first_insert)

    # TODO: change this function to retreive from the indexed Inverted file index db
    def retrive(self, query: Annotated[List[float], 70], top_k = 5):
        #######################################################
        print("query", query)
        #######################################################
        # first open the file
        f = open("centroids.pkl", "rb")

        # loop over all the lines in the file to read vector by vector
        self.centroids = []
        while True:
            try:
                v = pickle.load(f)
                self.centroids.append(v)
            except:
                break
        f.close()

        # remove the IDs from the centroids
        centroids_with_no_id = [centroid[1:] for centroid in self.centroids]

        # print("centroids_with_no_id", centroids_with_no_id)
    
        #print("centroids with no id are", centroids_with_no_id)
        # print("query is", query)
        nearest_centroid, distance = VQ.vq(query, centroids_with_no_id)
        print("Before: nearest_centroid", nearest_centroid)
        print("distance", distance)

        #nearest_centroid = self.centroids[nearest_centroid[0]][0]
        print("After: nearest_centroid", nearest_centroid)

        
        # print("BEFORE: nearest_centroids", nearest_centroid)

      
        # now we need to search in the files of the nearest centroids
        # we will get the top k vectors from each file
        # then we will sort them by their score
        # then we will return the top k vectors
        # we will use a heap to do that cause it is faster
        heap = [] 
        heapq.heapify(heap)
        for centroid in nearest_centroid:
            # open the file of the centroid
            f = open(f"cluster_{centroid}.pkl", "rb")

            # load vector by vector from the file
            while True:
                try:
                    v = pickle.load(f)
                    id = v[0]
                    vect = v[1:]
                    score = self._cal_score(query, vect)
                    heapq.heappush(heap, (-score, id, vect))
                except:
                    break
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
        with open("test/new_ids.csv", "w") as fout:
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

    # add a parameter to build the index on the whole db or on part of it
    def _build_index(self, db_vectors, build_on_part = False):
        # first check the build_on_part parameter
        if not build_on_part:
            # build the index on the whole db
            self._build_index_on_whole_db(db_vectors)
        else:
            # build the index on part of the db
            self._build_index_on_part_of_db(db_vectors)


    def _build_index_on_whole_db(self, db_vectors):
        # now let's create the centroids on part of the vectors only to speed up the process
        # number of vectors to use to create the centroids
        n_vectors_train = ceil(len(db_vectors) * 0.5)
        
        num_centroids = ceil(n_vectors_train / self.num_vectors_per_cluster)

        print("n_vectors_train", n_vectors_train)
        print("num_centroids", num_centroids)

        kmeans = KMeans(n_clusters=num_centroids, random_state=0).fit([vec.vect for vec in db_vectors[:n_vectors_train]]) 
        
        
        # now Kmeans has the centroids
        # now we need to assign each vector to the closest centroid
        self.centroids = kmeans.cluster_centers_  # those are the centroid's vectors
        clusters = {}
        # we can find the closest centroid by fit function
        for vec in db_vectors:
            centroid = kmeans.predict([vec.vect])[0]
            clusters[centroid] = clusters.get(centroid, []) + [vec]

        # now store each cluster in a file
        # create a file for each centroid if not exist
        print("Start storing index")
        i = 0
        for centroid in clusters:
            # we will create new file in the folder self.files_path
            with open(f"cluster_{centroid}.pkl", "wb") as fout:
                for vec in clusters[centroid]:
                    i += 1
                    v = [vec.id] + vec.vect
                    pickle.dump(v, fout)
        
        ###############################################################
        v_list = []
        # we need to store the centroids in a file
        with open(f"centroids.pkl", "wb") as fout:
            # write the centroid id, and its vector
            for centroid in self.centroids:
                label = kmeans.predict([centroid])[0]
                v = [label] + list(centroid)
                ##################
                v_list.append(v)
                ##################
                # fout.write(f"{row_str}\n")
                pickle.dump(v, fout)
        
        ############################################################################################################
        # # we need to store the centroids in a readable file
        with open(f"test/new_centroids.csv", "w") as fout:
            # write the centroid id, and its vector
            for centroid in v_list:
                row_str = f"{centroid[0]}," + ",".join([str(e) for e in centroid[1:]])
                fout.write(f"{row_str}\n")
        ############################################################################################################
        
        # # we need to store the kmeans model to use it later
        # # we will store it in a file
        # # first create the file
        # with open(f"{self.files_path}/kmeans.npy", "wb") as fout:
        #     np.save(fout, kmeans.cluster_centers_)
        print("Done building index")


    def _build_index_on_part_of_db(self, db_vectors):
        # # read the kmeans model from the file
        # kmeans = KMeans(n_clusters=len(np.load(f"{self.files_path}/kmeans.npy")), random_state=0)
        # kmeans.cluster_centers_ = np.load(f"{self.files_path}/kmeans.npy")
        
        clusters = {}
        self.centroids = np.load(f"{self.files_path}/centroids.pkl")
        # we can find the closest centroid by vq function of scipy
        centroids_with_no_id = [centroid[1:] for centroid in self.centroids]
        clusters_out_of_prediction, _ = VQ.vq([vec.vect for vec in db_vectors], centroids_with_no_id)
        # add the vectors to the clusters
        for i, vec in enumerate(db_vectors):
            centroid_label = self.centroids[clusters_out_of_prediction[i]][0]
            clusters[centroid_label] = clusters.get(centroid_label, []) + [vec]

        # for vec in db_vectors:
        #     centroid = VQ.vq([vec.vect], centroids_with_no_id)[0][0]
        #     centroid_id = self.centroids[centroid][0]
        #     clusters[centroid] = clusters.get(centroid_id, []) + [vec]
        
        
        # for vec in db_vectors:
        #     centroid = kmeans.predict([vec.vect])[0]
        #     clusters[centroid] = clusters.get(centroid, []) + [vec]

        # now store each cluster in a file
        # create a file for each centroid if not exist
        print("Start storing index")
        for centroid in clusters:
            # we will append to the file
            with open(f"{self.files_path}/cluster_{centroid}.pkl", "a+") as fout:
                for vec in clusters[centroid]:
                    row_str = f"{vec.id}," + ",".join([str(e) for e in vec.vect])
                    fout.write(f"{row_str}\n")
        print("Done building index")


# old

    def _build_index_old(self, db_vectors):
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
            with open(f"cluster_{centroid.id}.pkl", "w") as fout:
                for vec in clusters[centroid]:
                    row_str = f"{vec.id}," + ",".join([str(e) for e in vec.vect])
                    fout.write(f"{row_str}\n")

        print("Done building index")



