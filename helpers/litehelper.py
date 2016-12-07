import pandas as pda
import re
import timeit
import sqlite3
import numpy as np

from table import Table

class LiteHelper:

    def __init__(self, path):
        self.con = sqlite3.connect(path)
        pda.options.mode.chained_assignment = None

    # Pipeline for data preprocessing: executes methods one by one and returns a dataframe
    # col_extract: regular expression keyed by column on which it must be applied
    # col_map: map function keyed by column
    # col_reduce: reduce function keyed by column
    # log: Show how pipeline performs
    def pipeline(self,
                 stmt,
                 col_filter=None,
                 col_drop=None,
                 col_extract=None,
                 col_map=None,
                 col_reduce=None,
                 col_encode=None):
        print '# SQLite'
        t = Table()
        self.tdata = []
        df = self.read_sql(stmt, self.con)
        df = self.drop_duplicates(df)
        if col_filter:
            dfs = self.filter(df, col_filter, col_drop)
        else:
            dfs = (df,)
        t.from_tuples(self.tdata, columns=['Operation', 'Column', 'Count', 'Time (sec)'])
        t.display()
        dfs_export = ()
        for df in dfs:
            t = Table()
            self.tdata = []
            if col_extract:
                fd = self.filter_dict(col_extract, self.column_list(df))
                if fd:
                    df = self.extract(df, fd)
            if col_map:
                fd = self.filter_dict(col_map, self.column_list(df))
                if fd:
                    df = self.map(df, fd)
            df = self.group(df)
            if col_reduce:
                fd = self.filter_dict(col_reduce, self.column_list(df))
                if fd:
                    df = self.reduce(df, fd)
            if col_encode:
                fd = self.filter_dict(col_encode, self.column_list(df))
                if fd:
                    df, df_map = self.encode(df, fd)
                    dfs_export += (df_map,)
            t.from_tuples(self.tdata, columns=['Operation', 'Column', 'Count', 'Time (sec)'])
            t.display()
            dfs_export += (df,)
        return dfs_export

    # Execute a query on SQLite and transform result into a dataframe
    def read_sql(self, stmt, con):
        start = timeit.default_timer()
        df = pda.read_sql(stmt, con)
        stop = timeit.default_timer()
        self.tdata.append(('Read SQL', self.column_list(df), len(df.index), (stop-start)))
        df = df.dropna()
        return df

    # Filters dataframe by values in a column and returns dataframes (multiple possible)
    def filter(self, df, col_filter, col_drop):
        dfs = ()
        for (filter_col, filter_val), col_rename in col_filter.iteritems():
            start = timeit.default_timer()
            new_df = df[df[filter_col] == filter_val]
            new_df.rename(columns=col_rename, inplace=True)
            if col_drop:
                new_df.drop(col_drop, axis=1, inplace=True)
            stop = timeit.default_timer()
            self.tdata.append((
                'Filter',
                'df[{}=={}] {}'.format(
                    filter_col,
                    filter_val,
                    ', '.join([old+'->'+new for old, new in col_rename.iteritems()])),
                len(new_df.index),
                (stop-start)))
            new_df = new_df.dropna()
            dfs += (new_df,)
        return dfs

    # Returns list of columns
    def column_list(self, df):
        return df.columns.values.tolist()

    # If filering took place, we must work with multiple dataframes then
    # In this case we must filter out those parameters (keys in dict) which do not correspond to the particular dict
    def filter_dict(self, d, key_lst):
        return {k: v for k, v in d.iteritems() if k in key_lst}

    # Drop duplicates
    def drop_duplicates(self, df):
        start = timeit.default_timer()
        df = df.drop_duplicates()
        stop = timeit.default_timer()
        self.tdata.append(('Drop duplicates', self.column_list(df), len(df.index), (stop-start)))
        return df

    # Parse strings using regular expressions provided in col_extract
    def extract(self, df, col_extract):
        for col, reg in col_extract.iteritems():
            n = re.compile(reg).groups
            search_res = re.search(r'<(.*?)>', reg)
            groups = [
                search_res.group(i)
                for i in range(1, n+1)
            ]
            for group in groups:
                start = timeit.default_timer()
                df[group] = df[col].str.extract(reg, expand=True).ix[:, 0]
                stop = timeit.default_timer()
                self.tdata.append(('Parse', col+'->'+group, len(df[group].index), (stop-start)))
            if col not in groups:
                df = df.drop(col, 1)
        df = df.dropna()
        return df

    # Apply map on columns using functions provided in col_map
    def map(self, df, col_map):
        for col, f in col_map.iteritems():
            start = timeit.default_timer()
            df[col] = df[col].map(f)
            stop = timeit.default_timer()
            self.tdata.append(('Map', col, len(df[col].index), (stop-start)))
        df = df.dropna()
        return df

    # Group rows by id (first column) and put them into a list
    def group(self, df):
        start = timeit.default_timer()
        df = df.groupby('id').aggregate(lambda x: list(x)).reset_index()
        stop = timeit.default_timer()
        self.tdata.append(('Group', self.column_list(df), len(df.index), (stop-start)))
        return df

    # Reduce lists (number of rows remains the same)
    def reduce(self, df, col_reduce):
        for col, f in col_reduce.iteritems():
            start = timeit.default_timer()
            df[col] = df[col].map(f)
            stop = timeit.default_timer()
            self.tdata.append(('Reduce', col, len(df[col].index), (stop-start)))
        df = df.dropna()
        return df

    # Encode column by building a map automatically (only if number of values is low)
    def encode(self, df, col_encode):
        dfs = (df,)
        for col, new_col in col_encode.iteritems():
            start = timeit.default_timer()
            mapper = {i: key for i, key in enumerate(set([i for item in df[col] for i in item]))}
            f = {key: i for i, key in mapper.iteritems()}
            df[col] = df[col].apply(lambda x: map(lambda y: f[y], x))
            # MongoDB helper takes dataframes as input, so translate map dict into df
            # Do not forget to put each element to a list for the MongoHelper
            dfs += (pda.DataFrame.from_dict({'id': mapper.keys(), new_col: map(lambda x: [x], mapper.values())}),)
            stop = timeit.default_timer()
            self.tdata.append(('Encode', col, '{} ({}), {} ({})'.format(len(df[col].index), col, len(mapper), new_col), (stop-start)))
        return dfs
