
import requests
import timeit
import itertools
from multiprocessing.dummy import Pool as ThreadPool
from random import shuffle
import pandas as pda
import pickle

from mongohelper import MongoHelper
mongo = MongoHelper('imdb')
from table import Table

# Functionality within this package requires that the __main__ module be importable by the children.
# This is covered in Programming guidelines however it is worth pointing out here. This means that some examples,
# such as the multiprocessing.Pool examples will not work in the interactive interpreter

# Bound methods are not picklable thus define here
def fetch_document(tup):
    id, title, year = tup
    api_url = 'http://www.omdbapi.com/?t=%s&y=%d&r=json&plot=short&type=movie&tomatoes=true' % (unicode(title).encode("utf-8"), year)
    try:
        json_file = requests.get(api_url).json()
        if 'Error' in json_file:
            return None
        else:
            return {'id': id, 'info': [json_file]}
    except:
        return None

def pending_documents():
    titles = {k: v for k, vs in mongo.collect('titles_map').iteritems() for v in vs}
    years = {
        id: year
        for id, years in mongo.collect('release_years').iteritems()
        for year in years
        if id in titles
    }
    stored = None
    if 'omdb_files' in mongo.db.collection_names():
        stored = mongo.collect('omdb_files')
    pending = [
        (id, titles[id], year)
        for id, year in years.iteritems()
        if not stored or id not in stored
    ]
    shuffle(pending)
    return pending

def fetch_documents(pending, nbunch, ntimes, nworkers):
    for i in range(ntimes):
        start = timeit.default_timer()
        fetched = []
        pool = ThreadPool(nworkers)
        try:
            for x in pool.imap_unordered(fetch_document, pending[i*nbunch:(i+1)*nbunch]):
                if x:
                    fetched.append(x)
            if fetched:
                mongo.db['omdb_files'].insert_many(fetched)
                stop = timeit.default_timer()
                print len(fetched), stop-start
        finally:
            pool.close()
            pool.join()

pending = pending_documents()
print len(pending)
fetch_documents(pending, 1000, 100, 32)
