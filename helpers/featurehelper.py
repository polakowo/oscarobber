from __future__ import division
import sklearn.preprocessing as pp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import timeit
from collections import defaultdict, Counter
import scipy as sp
import math
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from os import path
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.misc import imread
import matplotlib.colors as colors
import matplotlib.cm as cmx
import nltk, re
from nltk.corpus import stopwords
nltk.data.path.append("/Users/olegpolakow/Documents/nltk_data")

from table import Table

class FeatureHelper:
    def __init__(self):
        self.stopwords_set = set(sorted(stopwords.words()))
        pass

    def tokenize(self, text, tags=None):
        tokens = [
                w.lower()
                for w in nltk.regexp_tokenize(text, r'\w+')
                if w.lower() not in self.stopwords_set and len(w) > 2
            ]
        if tags:
            tagged = nltk.pos_tag(tokens, tagset='universal')
            return [token for token, tag in tagged if tag in tags]
        return tokens

    def vectorize_documents(self, documents_by_id):
        tab = Table()
        start = timeit.default_timer()
        documents_by_id = {k: set(map(lambda x: x.lower(), vs)) for k, vs in documents_by_id.iteritems()}
        doc_index = {k: i for i, k in enumerate(sorted(documents_by_id.keys()))}
        token_docs_freq = Counter([token.lower() for docs in documents_by_id.values() for token in docs])
        feature_index = {k: i for i, k in enumerate(sorted(token_docs_freq.keys()))}
        X = sp.sparse.dok_matrix((len(doc_index), len(feature_index)), dtype=np.float64)
        N = len(documents_by_id)
        for doc, tokens in documents_by_id.iteritems():
            token_freq = Counter(tokens)
            for token in tokens:
                tf = token_freq[token] / len(token_freq)
                idf = np.log10(N / token_docs_freq[token])
                X[doc_index[doc],feature_index[token]] = tf * idf
        X = X.tocsr()
        stop = timeit.default_timer()
        tab.from_tuples([(len(doc_index), len(feature_index), stop-start)],
                        columns=['Documents', 'Features', 'Time (sec)'])
        tab.display()
        return X, doc_index, feature_index

    def sparse_cosim(self, X):
        tab = Table()
        start = timeit.default_timer()
        X_cosim = cosine_similarity(X)
        stop = timeit.default_timer()
        tab.from_tuples([(X_cosim.shape, stop-start)], columns=['Shape', 'Time (sec)'])
        tab.display()
        return X_cosim

    def most_similar_to(self, X_cosim, doc_index, document):
        tab = Table()
        start = timeit.default_timer()
        doc_index_rev = {v: k for k, v in doc_index.iteritems()}
        doc_row = X_cosim[doc_index[document]]
        row_sorted_idx = np.argsort(doc_row)
        similar_documents = list(reversed([
                (doc_index_rev[i], doc_row[i])
                for i in row_sorted_idx[row_sorted_idx!=doc_index[document]]
                if doc_row[i]
            ]))
        stop = timeit.default_timer()
        tab.from_tuples([(len(similar_documents), stop-start)], columns=['Similar documents', 'Time (sec)'])
        tab.display()
        return similar_documents

    def top_n_most_similar_pairs(self, X_cosim, doc_index, n):
        pass

    def document_vector(self, X, doc_index, feature_index, document):
        feature_index_rev = {v: k for k, v in feature_index.iteritems()}
        doc_tfidf = X[doc_index[document]]
        return sorted([
            (feature_index_rev[idx], doc_tfidf[0,idx])
            for idx in doc_tfidf.nonzero()[1]
        ], key=lambda x: x[1], reverse=True)

    def transform_vector_wordcloud(self, document_vector):
        max_tfidf = max(zip(*document_vector)[1])
        document_freq = [(feature, int(math.ceil(tfidf*100/max_tfidf))) for feature, tfidf in document_vector]
        return document_freq

    def color_func(self, word, font_size, position, orientation, random_state=None, **kwargs):
        oldmin, oldmax, newmin, newmax = 20, 150, 100, 220
        rgb = int((((font_size - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin)
        if rgb > newmax:
            rgb = newmax
        return "rgb(%d, %d, %d)" % (random.randint(50, rgb), random.randint(50, rgb), random.randint(50, rgb))

    def wordclouds(self, title_document_freq):
        plt.close('all')
        plcount = int(math.ceil(len(title_document_freq)/2))
        fig, pl = plt.subplots(plcount, 2, sharey=True, figsize=(12, plcount*4), facecolor='#252525')
        if plcount>1:
            pl_matrix = np.array(pl)
        else:
            pl_matrix = np.array([list(pl)])

        for i, (title, document_freq) in enumerate(title_document_freq.iteritems()):
            ax = pl_matrix[int(math.floor(i/2)),int(i%2)]
            wc = WordCloud(
                font_path='/Users/olegpolakow/Library/Fonts/KenyanCoffee.ttf',
                background_color='#252525',
                max_words=15,
                margin=10,
                width=800,
                height=600
            ).fit_words(document_freq) # Fit_words produces more correct results compared to generate_from_text
            default_colors = wc.to_array()
            ax.imshow(wc.recolor(color_func=self.color_func, random_state=3), aspect='auto')
            ax.set_title(title, color='white', fontsize=16, family='Helvetica Neue')
        for i in range(pl_matrix.shape[0]):
            for j in range(pl_matrix.shape[1]):
                pl_matrix[i][j].axis('off')
        fig.tight_layout(pad=0.01, w_pad=1, h_pad=2.0)
        plt.show()

    def pairs_cosim(self, X_cosim, doc_index, doc_pairs):
        tab = Table()
        start = timeit.default_timer()
        pair_cosine = {}
        for doc1, doc2 in set(doc_pairs):
            if doc1 in doc_index.keys() and doc2 in doc_index.keys():
                pair_cosine[(doc1, doc2)] = pair_cosine[(doc2, doc1)] = X_cosim[doc_index[doc1]][doc_index[doc2]]
        stop = timeit.default_timer()
        tab.from_tuples([(int(len(pair_cosine)/2), stop-start)], columns=['Pairs', 'Time (sec)'])
        tab.display()
        return pair_cosine

    def bin_integers(self, integers):
        xy = Counter(integers)
        return xy.keys(), xy.values()

    def bin_floats(self, floats, bins):
            y, bins = np.histogram(floats, bins=bins)
            x = (bins[:-1] + bins[1:]) / 2
            return x, y

    def map_colors(self, y, cmap):
            cNorm = colors.Normalize(vmin=min(y), vmax=max(y))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
            return [scalarMap.to_rgba(v) for v in y]

    def plot_dist(self, values, cmap, desc, cumulative=None):
        plt.close('all')
        fig, pl = plt.subplots()
        val = map(lambda x: round(x, 3), values)
        x, y = self.bin_floats(val, len(set(val)))
        if cumulative:
            y = np.cumsum(y)/sum(y)
        pl.scatter(x, y, marker='o', s=20, lw=0, c=y, cmap=cmap, zorder=3)
        offset = .1
        pl.set_xlim(self.rescale(x, offset))
        pl.set_ylim(self.rescale(y, 3 / 2 * offset))
        pl.set_ylabel('Count')
        pl.set_xlabel(desc)
        pl.set_title(desc + ' distribution')
        pl.grid(zorder=0)
        fig.set_size_inches(12, 4)
        fig.tight_layout(pad=0.01, w_pad=1, h_pad=2.0)
        plt.show()

    def plot_tokens_freq(self, legend_token_counter, cmap, n):
        plt.close('all')
        fig = plt.figure()
        clrs = self.map_colors(range(0, len(legend_token_counter)), cmap)
        glob_counter = defaultdict(list)
        for token_counter in legend_token_counter.values():
            for token, c in token_counter.items():
                glob_counter[token].append(c)
        for token in glob_counter:
            glob_counter[token] = sum(glob_counter[token])
        top_tokens = zip(*sorted(glob_counter.items(), key=lambda x: x[1], reverse=True)[:n])[0]
        xx = x = range(len(top_tokens))
        yy = []
        markers = []
        for i, (legend, token_counter) in enumerate(legend_token_counter.items()):
            y = [token_counter[token] if token in token_counter else 0 for token in top_tokens]
            yy.extend(y)
            marker, = plt.plot(x, y, lw=1, c=clrs[i], zorder=3)
            markers.append(marker)
            plt.fill_between(x, y, 0, facecolor=clrs[i], alpha=0.3)
        plt.legend(
            markers,
            legend_token_counter.keys(),
            fontsize='medium'
        )
        plt.xticks(x, list(top_tokens), rotation='vertical')
        offset = .1
        plt.xlim(self.rescale(xx, offset))
        plt.ylim(self.rescale(yy, 3 / 2 * offset))
        plt.ylabel('Frequency')
        plt.title('Tokens by count')
        plt.grid(zorder=0)
        fig.set_size_inches(12, 5)
        fig.tight_layout(pad=0.01, w_pad=1, h_pad=2.0)
        plt.show()

    def rescale(self, axis_values, offset):
        minx, maxx = min(axis_values), max(axis_values)
        return [minx - offset * (maxx - minx), maxx + offset * (maxx - minx)]

    def plot_pairs_cosim(self, pair_cosimx, pair_cosimy, cmap, root=None):
        plt.close('all')
        fig, pl = plt.subplots()
        common_pairs = list(set.intersection(set(pair_cosimx), set(pair_cosimy)))
        x, y = [], []
        for e in common_pairs:
            x.append(pair_cosimx[e])
            y.append(pair_cosimy[e])
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        f = lambda x: m * x + b
        d = [abs(y[j] - f(xv)) for j, xv in enumerate(x)]
        pl.scatter(x, y, marker='o', s=20, lw=0, c=d, cmap=cmap, zorder=3)
        xr = [min(x), max(x)]
        yr = [f(min(x)), f(max(x))]
        rmarker, = pl.plot(xr, yr, lw=1, c='k', zorder=4)
        pl.legend(
            [rmarker],
            ['$r^2:%.2f, S:%.2f, y=%.2fx+%.2f$'%(r_value**2, std_err, m, b)]
        )
        offset = .1
        pl.set_xlim(self.rescale(x, offset))
        pl.set_ylim(self.rescale(y, 3 / 2 * offset))
        pl.set_xlabel('Similarity on keywords')
        pl.set_ylabel('Similarity on plots')
        pl.set_title('Similarity on plots vs keywords')
        pl.grid(zorder=0)
        if root:
            root_pairs = [e for e in common_pairs if root in e]
            if root_pairs:
                for root_pair in root_pairs:
                    pl.plot(
                        pair_cosimx[root_pair], pair_cosimy[root_pair],
                        lw=0, c='magenta',
                        marker='o', ms=10,
                        zorder=5
                    )
        fig.set_size_inches(12, 4)
        fig.tight_layout(pad=0.01, w_pad=1, h_pad=2.0)
        plt.show()

    def plot_attr_vs_attr(self, t_xattrs, t_yattr, cmap, root=None):
        plt.close('all')
        plcount = int(math.ceil(len(t_xattrs)/2))
        fig, pl = plt.subplots(plcount, 2)
        if plcount>1:
            pl_matrix = np.array(pl)
        else:
            pl_matrix = np.array([list(pl)])
        for i, (xt, xattr) in enumerate(t_xattrs):
            ax = pl_matrix[int(math.floor(i/2)),int(i%2)]
            yt, yattr = t_yattr
            intsct = set.intersection(set(xattr.keys()), set(yattr.keys()))
            x, y = [], []
            for j in intsct:
                x.append(xattr[j])
                y.append(yattr[j])
            m, b, r_value, p_value, std_err = stats.linregress(x, y)
            f = lambda x: m * x + b
            d = [abs(y[j] - f(xv)) for j, xv in enumerate(x)]
            ax.scatter(x, y, marker='o', s=20, lw=0, c=d, cmap=cmap, zorder=3)
            xr = [min(x), max(x)]
            yr = [f(min(x)), f(max(x))]
            rmarker, = ax.plot(xr, yr, lw=1, c='k', zorder=4)
            ax.legend(
                [rmarker],
                ['$r^2:%.2f, S:%.2f, y=%.2fx+%.2f$' %
                    (r_value**2, std_err, m, b)],
                fontsize='medium'
            )
            offset = .1
            ax.set_xlim(self.rescale(x, offset))
            ax.set_ylim(self.rescale(y, 3 / 2 * offset))
            ax.set_xlabel(xt)
            ax.set_ylabel(yt)
            ax.set_title(xt + ' vs. ' + yt)
            ax.grid(zorder=0)
            if root and root in intsct:
                ax.plot(
                    xattr[root], yattr[root],
                    lw=0, c='magenta',
                    marker='o', ms=10,
                    zorder=5
                )
        if len(t_xattrs)%2>0:
            pl_matrix[int(math.floor((len(t_xattrs)-1)/2))][1].axis('off')
        fig.set_size_inches(12, 3 * plcount)
        fig.tight_layout(pad=0.01, w_pad=1, h_pad=2.0)
        plt.show()
