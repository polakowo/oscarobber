import pandas as pda

class Table:
    def __init__(self):
        # Store dataframe as class variable to access it from everywhere
        self.df = None

    def from_dict(self, d):
        # Takes rows keyed by columns
        self.df = pda.DataFrame.from_dict(d)

    def from_tuples(self, d, columns):
        # Takes list of rows represented by tuples and list of columns
        self.df = pda.DataFrame(d, columns=columns)

    def sort_values(self, by=None, ascending=True):
        self.df.sort_values(by=by, ascending=ascending, inplace=True)

    def display(self):
        print
        print self.df
        print
