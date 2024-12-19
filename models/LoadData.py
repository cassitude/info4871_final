from numpy import *
import random


def load_rating_data(file_path="data/ml-32m/ratings.csv", delim=','):
    prefer = []
    for line in open(file_path, 'r'):
        (userid, movieid, rating) = line.split(delim)
        if userid == "User-ID":
            continue
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)
        prefer.append([uid, mid, rat])
    data = array(prefer)
    return data
