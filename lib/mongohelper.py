import pymongo
import timeit

from table import Table


class MongoHelper:

    # Establish connection to the MongoDB instance
    def __init__(self, dbname):
        try:
            self.client = pymongo.MongoClient()
            print self.client
            self.db = self.client[dbname]
            print self.db
        except pymongo.errors.ConnectionFailure, e:
            print e

    def close(self):
        self.client.close()

    # Create an index on ids to search for documents a lot faster
    def create_index(self, coll):
        start = timeit.default_timer()
        self.db[coll].create_index([('id', pymongo.ASCENDING)], unique=True)
        stop = timeit.default_timer()
        self.tdata.append(('Create index', coll, '', stop - start))

    def drop_index(self, coll):
        self.db[coll].drop_index('id')

    # Convert each column of df to list of documents in a format allowed by MongoDB
    def to_dict(self, df):
        start = timeit.default_timer()
        d_raw = df.to_dict(orient='list')
        ids = d_raw['id']
        val_lists = {}
        for coll, val_list in d_raw.iteritems():
            if coll != 'id':
                val_lists[coll] = [{'id': ids[i], 'info': val}
                                   for i, val in enumerate(val_list)]
                stop = timeit.default_timer()
                self.tdata.append(('Convert dataframe', coll,
                                   len(val_list), stop - start))
        return val_lists

    # Push columns in a dataframe to MongoDB as new collections or append to existing ones
    def store(self, *dfs, **params):
        print '# MongoDB'
        for df in dfs:
            t = Table()
            self.tdata = []
            start = timeit.default_timer()
            val_lists = self.to_dict(df)
            for coll, val_list in val_lists.iteritems():
                if params.get('append'):
                    self.drop_index(coll)
                else:
                    self.drop(coll)
                self.db[coll].insert_many(val_list)
                stop = timeit.default_timer()
                self.tdata.append(
                    ('Store documents', coll, len(val_list), stop - start))
                self.create_index(coll)
            t.from_tuples(self.tdata, columns=[
                          'Operation', 'Collection', 'Count', 'Time (sec)'])
            if not params.get('silent'):
                t.display()
                self.stats(colls=val_lists.keys())

    # Search collection for ids and retrieve matched documents
    def collect(self, coll, filter=None, applymap=None):
        if self.exists(coll):
            id_infos = {d['id']: d['info'] for d in self.db[
                coll].find(filter=filter, projection={'_id': 0})}
            if applymap:
                if filter:
                    mapper = {d['id']: d['info'] for d in self.db[applymap].find(
                        filter={'id': {'$in': [v for vs in id_infos.values() for v in vs]}},
                        projection={'_id': 0})}
                else:
                    mapper = {d['id']: d['info'] for d in self.db[applymap].find(projection={'_id': 0})}
                for id, info in id_infos.iteritems():
                    id_infos[id] = [i for item in info for i in mapper[item]]
            return id_infos
        else:
            print 'Collection "%s" not found' % coll
            return None

    # Drop collection using collection name
    def drop(self, coll):
        if self.exists(coll):
            self.db[coll].drop()

    # Collection exists?
    def exists(self, coll):
        return coll in self.db.collection_names()

    # Print some basic statistics on collections
    def stats(self, colls=None, total=False):
        if not colls:
            colls = self.db.collection_names(include_system_collections=False)
        if colls:
            # Keys of dict are in arbitrary order
            stats_count, stats_size, stats_storage = [], [], []
            for name in colls:
                collstats = self.db.command('collstats', name)
                stats_count.append(collstats['count'])
                stats_size.append(collstats['size'])
                stats_storage.append(collstats['storageSize'])
            t = Table()
            t.from_dict({
                'collections': colls,
                'count': stats_count,
                'size (MB)': map(lambda x: x / (1024**2), stats_size),
                'storage (MB)': map(lambda x: x / (1024**2), stats_storage)
            })
            t.sort_values(by='count', ascending=False)
            t.display()
            print
            if total:
                t = Table()
                t.from_dict({
                    '': ['total'],
                    'count': [sum(stats_count)],
                    'size (MB)': [sum(stats_size) / (1024**2)],
                    'storage (MB)': [sum(stats_storage) / (1024**2)]
                })
                t.display()
        else:
            print 'No collections'
