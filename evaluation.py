import numpy as np
from worst_case_implementation import VecDBWorst
from hnsw import VecDBhnsw
import time
from dataclasses import dataclass
from typing import List
from vec_db import VecDB
from inverted_file_index import VecDBIF
from math import ceil
AVG_OVERX_ROWS = 10

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

def run_queries(db, np_rows, top_k, num_runs):
    results = []
    for _ in range(num_runs):
        query = np.random.random((1,70))
        
        tic = time.time()
        db_ids = db.retrive(query, top_k)
        toc = time.time()
        run_time = toc - tic
        
        tic = time.time()
        actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(np_rows, axis=1) * np.linalg.norm(query)), axis= 1).squeeze().tolist()[::-1]
        toc = time.time()
        np_run_time = toc - tic
        
        results.append(Result(run_time, top_k, db_ids, actual_ids))
    return results

def eval(results: List[Result]):
    # scores are negative. So getting 0 is the best score.
    scores = []
    run_time = []
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
            except:
                score -= len(res.actual_ids)
        scores.append(score)

    return sum(scores) / len(scores), sum(run_time) / len(run_time)


if __name__ == "__main__":
    num_records = 10000
    new_db = True
    # create the db
    db = VecDB(new_db=new_db, file_path="10K")
    worst_db = VecDBWorst(new_db=new_db)
    ivf_db = VecDBIF(new_db=new_db)
    # generate random records with ceil(num_records / 1M) vectors each time
    num_of_iterations = ceil(num_records / 1000000)
    if num_of_iterations == 0:
        num_of_iterations = 1
    # insert the records in the db but take care as the number of records may be less than 1M
    # check the number of records if less than 1M then insert the number of records only
    # we will use a fixed seed to generate random records
    rnd = np.random.RandomState(50)
    
    records_np = rnd.random((num_records, 70))
    for i in range(num_of_iterations):
        num_records_to_insert = num_records - (i * 1000000)
        records_to_insert = records_np[i * 1000000: i * 1000000 + num_records_to_insert - 1]
        records_dict = [{"id": i + (i * 1000000), "embed": list(row)} for i, row in enumerate(records_np)]
        db.insert_records(records_dict, first_insert = i == 0)
        
    # insert the records in the worst db
    records_dict = [{"id": i, "embed": list(row)} for i, row in enumerate(records_np)]
    worst_db.insert_records(records_dict)
    ivf_db.insert_records(records_dict)

    # for i in range(num_of_iterations):
    #     num_records_to_insert = num_records - (i * 1000000)
    #     records_np = rnd.random((num_records_to_insert, 70))
    #     records_dict = [{"id": i + (i * 1000000), "embed": list(row)} for i, row in enumerate(records_np)]
    #     db.insert_records(records_dict, first_insert = i == 0)
    res = run_queries(db, records_np, 5, 1)
    res_worst = run_queries(worst_db, records_np, 5, 1)
    res_ivf = run_queries(ivf_db, records_np, 5, 1)
    eval_res = eval(res)
    eval_res_worst = eval(res_worst)
    eval_res_ivf = eval(res_ivf)
    
    # calculate the recall
    recall = len(set(res[0].db_ids).intersection(set(res_worst[0].db_ids))) / len(set(res_worst[0].db_ids))
    
    print("recall = ", recall)
    print("db = ", eval_res)
    print("worst = ", eval_res_worst)
    print("ivf = ", eval_res_ivf)


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

    