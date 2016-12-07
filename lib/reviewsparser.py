from os import listdir
from os.path import isfile, join
import re
import codecs
from collections import defaultdict
import pandas as pda
import timeit

from table import Table

class ReviewsParser:
    def __init__(self, reviews_dir, urls_path):
        self.reviews_dir = reviews_dir
        self.urls_path = urls_path

    def review_ids(self):
        with codecs.open(self.urls_path, 'r', 'utf-8') as f:
            start = timeit.default_timer()
            review_ids = {
                i: re.search(r'tt\d+', line).group()
                for i, line in enumerate(f.readlines())
            }
            stop = timeit.default_timer()
            self.tdata.append(('Parse ids', len(review_ids), stop-start))
            return review_ids
        return None

    def review_paths(self):
        start = timeit.default_timer()
        review_paths = [
            join(self.reviews_dir, path)
            for path in listdir(self.reviews_dir)
            if isfile(join(self.reviews_dir, path)) and re.search(r'.txt', path)
        ]
        stop = timeit.default_timer()
        self.tdata.append(('Parse paths', len(review_paths), stop-start))
        return review_paths

    def join(self, review_ids, review_paths):
        start = timeit.default_timer()
        id_reviews = defaultdict(list)
        for path in review_paths:
            with codecs.open(path, 'r', 'utf-8') as f:
                i = int(re.search(r'(\d+)_\d+.txt', path).group(1))
                id_reviews[review_ids[i]].extend(f.readlines())
        stop = timeit.default_timer()
        self.tdata.append(('Join', len(id_reviews), stop-start))
        return id_reviews

    def to_df(self, id_reviews):
        start = timeit.default_timer()
        df = pda.DataFrame.from_dict({'id': id_reviews.keys(), 'user_reviews': id_reviews.values()})
        stop = timeit.default_timer()
        self.tdata.append(('Convert to dataframe', len(df.index), stop-start))
        return df

    def parse(self):
        t = Table()
        self.tdata = []
        review_ids = self.review_ids()
        review_paths = self.review_paths()
        reviews_df = self.to_df(self.join(review_ids, review_paths))
        print '# ReviewsParser'
        t.from_tuples(self.tdata, columns=['Operation', 'Count', 'Time (sec)'])
        t.display()
        return reviews_df
