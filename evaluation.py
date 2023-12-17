import numpy as np
from worst_case_implementation import VecDBWorst
from hnsw import VecDBhnsw
import time
from dataclasses import dataclass
from typing import List
from vec_db import VecDB
# from inverted_file_index import VecDBIF
from math import ceil
AVG_OVERX_ROWS = 10

QUERY_SEED_NUMBER = 10
DB_SEED_NUMBER = 50



@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

def run_queries(db, np_rows, top_k, num_runs):
    results = []
    for _ in range(num_runs):
        rng = np.random.default_rng(QUERY_SEED_NUMBER)
        query = rng.random((1, 70), dtype=np.float32)
        
        tic = time.time()
        db_ids = db.retrive(query, top_k)
        toc = time.time()
        run_time = toc - tic
        
        tic = time.time()
        actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(np_rows, axis=1) * np.linalg.norm(query)), axis= 1).squeeze().tolist()[::-1]
        toc = time.time()
        print("actual_ids = ", actual_ids[:10])
        # calculate the recall
        # recall = len(set(db_ids).intersection(set(actual_ids))) / len(set(actual_ids))
        # print("recall = ", recall)

        np_run_time = toc - tic
        
        results.append(Result(run_time, top_k, db_ids, actual_ids))
    return results

def eval(results: List[Result]):
    # scores are negative. So getting 0 is the best score.
    scores = []
    run_time = []
    count_true = 0
    for res in results:
        run_time.append(res.run_time)
        # case for retireving number not equal to top_k, socre will be the lowest
        if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
            scores.append( -1 * len(res.actual_ids) * res.top_k)
            continue
        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 3:
                    score -= ind
                else:
                    count_true += 1
            except:
                score -= len(res.actual_ids)
        scores.append(score)
    print("count_true = ", count_true)

    return sum(scores) / len(scores), sum(run_time) / len(run_time)


if __name__ == "__main__":
    #num_records = 10**7*2
    rng = np.random.default_rng(DB_SEED_NUMBER)
    #records_np = rng.random((num_records, 70), dtype=np.float32)

    # new_db = True

    # # # # create an obj from class db
    # db = VecDB(new_db=new_db, file_path="10M")

    # len_db = 0
    # for x in range(10):
    #     records_np = rng.random((10**6, 70), dtype=np.float32)
    #     recrods_dict = [{"id": i + len_db, "embed": list(row)} for i, row in enumerate(records_np)]
    #     len_db += len(records_np)
    #     del records_np
    #     db.insert_records(recrods_dict, first_batch=x == 0, new_db=new_db, src= "0", dest="10M")
    #     new_db = False


    db = VecDB(new_db=False, file_path="10M")
    # now run the queries
    records_np = rng.random((10*10**6, 70), dtype=np.float32)
    res = run_queries(db, records_np, 5, 1)

    print("restul for 10M records")
    print(eval(res))





    # first insert the first 100K records ----------------------------------------------------------------
    # print("inserting 100K records---------------------------------------------------------------")
    # records_dict = [{"id": i, "embed": list(row)} for i, row in enumerate(records_np[:100000])]
    # db.insert_records(records_dict, True, src = "100K", dest = "100K", new_db= new_db)
    # new_db = False

    # # now run the queries
    # res = run_queries(db, records_np[:100000], 5, 1)

    # print("restul for 100K records")
    # print(eval(res))

    # now insert up to 1M records ----------------------------------------------------------------------
    # so we need to insert 900K records
    #print("inserting 1M records---------------------------------------------------------------")
    # records_dict = [{"id": i, "embed": list(row)} for i, row in enumerate(records_np[0:1000000])]
    # db.insert_records(records_dict, True, new_db=True, src= "100K", dest="1M")

    # # now run the queries  
    # res = run_queries(db, records_np[0:1000000], 5, 1)

    # print("restul for 1M records")
    # print(eval(res))

    # # # now insert up to 5M records ----------------------------------------------------------------------
    # # # so we need to insert 4M records
    # # # insert them million by million
    # print("inserting 5M records---------------------------------------------------------------")
    # for j in range(1, 5):
    #     records_dict = [{"id": i + 1000000 + 1000000 * j, "embed": list(row)} for i, row in enumerate(records_np[1000000 * j:1000000 * (j + 1)])]
    #     db.insert_records(records_dict, first_batch=False, new_db=False, src= "1M", dest="5M")
        
    # # now run the queries
    # res = run_queries(db, records_np[0:5*10**6], 5, 1)

    # print(f"restul for 5M records")
    # print(eval(res))

    # # now insert up to 10M records ----------------------------------------------------------------------
    # # so we need to insert 5M records
    # # insert them million by million
    # print("inserting 10M records---------------------------------------------------------------")
    # for j in range(5):
    #     records_dict = [{"id": i + 5000000 + 1000000 * j, "embed": list(row)} for i, row in enumerate(records_np[1000000 * j:1000000 * (j + 1)])]
    #     db.insert_records(records_dict, j == 0, new_db=False, src= "5M", dest="10M")
        

    # # now run the queries
    # res = run_queries(db, records_np[0:10*10**6], 5, 1)

    # print(f"restul for 10M records")
    # print(eval(res))

    # # now insert up to 15M records ----------------------------------------------------------------------
    # # so we need to insert 5M records
    # # isnert them million by million
    # print("inserting 15M records---------------------------------------------------------------")
    # for j in range(5): 
    #     records_dict = [{"id": i + 10000000 + 1000000 * j, "embed": list(row)} for i, row in enumerate(records_np[1000000 * j:1000000 * (j + 1)])]
    #     db.insert_records(records_dict, j == 0, new_db=False, src= "10M", dest="15M")
        
    # # now run the queries
    # res = run_queries(db, records_np[0:15*10**6], 5, 1)

    # print(f"restul for 15M records")
    # print(eval(res))

    # # now insert up to 20M records ----------------------------------------------------------------------
    # # so we need to insert 5M records
    # # insert them million by million
    # print("inserting 20M records---------------------------------------------------------------")
    # for j in range(5):
    #     records_dict = [{"id": i + 15000000 + 1000000 * j, "embed": list(row)} for i, row in enumerate(records_np[1000000 * j:1000000 * (j + 1)])]
    #     db.insert_records(records_dict, j == 0, new_db=False, src= "15M", dest="20M")
        
    # # now run the queries
    # res = run_queries(db, records_np[0:20*10**6], 5, 1)

    # print(f"restul for 20M records")
    # print(eval(res))

    






    
    

































    # if we want to calculate the recall
    # we need to get the actual ids
    # db_worest = VecDBWorst()
    # db_worest.insert_records(records_dict)
    # res_best = run_queries(db_worest, records_np, 5, 1)
    # print(eval(res_best))

    # recall is the true positive / (true positive + false negative)
    # recall = len(set(res[0].db_ids).intersection(set(res_best[0].db_ids))) / len(set(res_best[0].db_ids))
    # print("recall = ", recall)

    
    # records_np = np.concatenate([records_np, np.random.random((90000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # print(eval(res))

    # records_np = np.concatenate([records_np, np.random.random((900000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    # records_np = np.concatenate([records_np, np.random.random((4000000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    # records_np = np.concatenate([records_np, np.random.random((5000000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    # records_np = np.concatenate([records_np, np.random.random((5000000, 70))])
    # records_dict = [{"id": i +  _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    