from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import random
from collections import Counter, defaultdict
import math
import six
import operator
import timeit
import warnings
import community
from scipy import stats

from table import Table


class NetworkHelper:

    def __init__(self):
        warnings.filterwarnings('ignore')
        pass

    # Generate graph

    def BipGraph(self, nbottom, ntop, edges, dir=False):
        start = timeit.default_timer()
        if dir:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        G.add_edges_from(edges)
        G.add_nodes_from(nbottom, bipartite=0)
        G.add_nodes_from(ntop, bipartite=1)
        stop = timeit.default_timer()
        if dir:
            self.graph_info(G.to_undirected(), stop-start)
            self.degree_info(G)
        else:
            self.graph_info(G, stop-start)
        return G

    # Graph information

    def p(self, G):
        n = G.number_of_nodes()
        l = G.number_of_edges()
        return 2 * l / (n * (n - 1))

    def graph_info(self, G, time=None):
        deg = nx.degree(G).values()
        tab = Table()
        tdata = [(
            len(G),
            len(G.edges()),
            nx.number_connected_components(G),
            len(max(list(nx.connected_component_subgraphs(G)), key=len)),
            nx.average_clustering(G),
            self.p(G),
            min(deg),
            max(deg),
            np.median(deg),
            np.mean(deg)
        )]
        columns = ['N', 'L', 'Components', 'Larg. component',
                   'C', 'p', 'k_min', 'k_max', 'k_median', 'k_mean']
        if time:
            tdata[0] += (time,)
            columns.append('Time (sec)')
        tab.from_tuples(tdata, columns=columns)
        tab.display()

    def degree_info(self, G):
        deg = nx.degree(G).values()
        indeg = G.in_degree().values()
        outdeg = G.out_degree().values()
        tab = Table()
        tdata = [('Degree', min(deg), max(deg), np.median(deg), np.mean(deg)),
                 ('In-degree', min(indeg), max(indeg), np.median(indeg), np.mean(indeg)),
                 ('Out-degree', min(outdeg), max(outdeg), np.median(outdeg), np.mean(outdeg))]
        columns = ['', 'k_min', 'k_max', 'k_median', 'k_mean']
        tab.from_tuples(tdata, columns=columns)
        tab.display()

    # Connected components

    def depth(self, G, root=None):
        if not nx.is_connected(G):
            gc = max(nx.connected_component_subgraphs(G), key=len)
        else:
            gc = G
        tab = Table()
        tdata = []
        start = start_main = timeit.default_timer()
        if not root:
            root = random.choice(gc.nodes())
        nodes = defaultdict(set)
        lvl_nodes = set()
        next_lvl_nodes = set([root])
        lvl = 0
        nodes[lvl].add(root)
        for n1, n2 in nx.bfs_edges(gc, root):
            if n1 not in lvl_nodes:
                stop = timeit.default_timer()
                tdata.append((lvl, len(next_lvl_nodes), len(set([n for s in nodes.values() for n in s]))/len(G), stop-start))
                start = timeit.default_timer()
                lvl += 1
                lvl_nodes = next_lvl_nodes.copy()
                next_lvl_nodes = set()
            next_lvl_nodes.add(n2)
            nodes[lvl].add(n2)
        stop = timeit.default_timer()
        tdata.append((lvl, len(next_lvl_nodes), len(set([n for s in nodes.values() for n in s]))/len(G), stop-start))
        tab.from_tuples(tdata, columns=['Level', 'Nodes', 'Visited', 'Time (sec)'])
        tab.display()
        return nodes

    def network_area(self, G, depth=None, root=None):
        start = timeit.default_timer()
        if not nx.is_connected(G):
            gc = max(nx.connected_component_subgraphs(G), key=len)
        else:
            gc = G
        if depth:
            if not root:
                root = random.choice(gc)
            nodes = set([root])
            lvl_nodes = set()
            next_lvl_nodes = set([root])
            lvl = 0
            for n1, n2 in nx.bfs_edges(gc, root):
                if n1 not in lvl_nodes:
                    if lvl >= depth:
                        break
                    else:
                        lvl += 1
                        lvl_nodes = next_lvl_nodes.copy()
                        next_lvl_nodes = set()
                next_lvl_nodes.add(n2)
                nodes.add(n1)
                nodes.add(n2)
            area = nx.Graph(gc.subgraph(nodes))
        else:
            area = gc
        stop = timeit.default_timer()
        self.graph_info(area, time=(stop - start))
        return area

    # Bipartite

    def bipartite_sets(self, G):
        start = timeit.default_timer()
        bottom_nodes = set(n for n, d in G.nodes(
            data=True) if d['bipartite'] == 0)
        top_nodes = set(G) - bottom_nodes
        stop = timeit.default_timer()
        tab = Table()
        tab.from_tuples([(len(top_nodes), len(bottom_nodes), (stop - start))],
                        columns=['Top nodes', 'Bottom nodes', 'Time (sec)'])
        tab.display()
        return bottom_nodes, top_nodes

    def bipartite_projection(self, G, nodes):
        start = timeit.default_timer()
        bp = bipartite.projected_graph(G, nodes)
        stop = timeit.default_timer()
        self.graph_info(bp, time=(stop - start))
        return bp

    # Friendship paradox

    def friendship_paradox(self, G, nmax):
        start = timeit.default_timer()
        deg_tup = [
            (G.degree(n), np.mean(G.degree(nbunch=G.neighbors(n)).values()))
            for n in random.sample(G, nmax)
        ]
        smaller = sum(map(lambda x: x[0] < x[1], deg_tup))
        same = sum(map(lambda x: x[0] == x[1], deg_tup))
        bigger = sum(map(lambda x: x[0] > x[1], deg_tup))
        stop = timeit.default_timer()
        tab = Table()
        tab.from_tuples([(nmax, smaller, same, bigger, (stop - start))], columns=[
            'Nodes', 'Lower', 'Same', 'Higher', 'Time (sec)'])
        tab.display()

    # Centralities

    def calc_centr(self, G, depth=None, root=None):
        if depth:
            g = self.network_area(G, depth=depth, root=root)
        else:
            g = G
        cdict = {
            'Degree': lambda g: nx.degree_centrality(g),
            'Closeness': lambda g: nx.closeness_centrality(g),
            'Betweenness': lambda g: nx.betweenness_centrality(g),
            'Eigenvector': lambda g: nx.eigenvector_centrality_numpy(g)
        }
        tab = Table()
        tdata = []
        centr = {}
        for ty, c in cdict.iteritems():
            start = timeit.default_timer()
            centr[ty] = c(g)
            stop = timeit.default_timer()
            tdata.append((ty, (stop - start)))
        tab.from_tuples(tdata, columns=['Centrality', 'Time (sec)'])
        tab.display()
        return centr

    def plot_centr_dist(self, G, centr, cmap, root=None):
        plt.close('all')
        fig, ax = plt.subplots(2, 2)
        for i, (t, cent) in enumerate(centr.items()):
            val = map(lambda x: round(x, 3), cent.values())
            x, y = self.bin_floats(val, len(set(val)))
            ij = bin(i)[2:].zfill(2)
            pl = ax[int(ij[0])][int(ij[1])]
            pl.scatter(x, y, marker='o', s=20, lw=0, c=y, cmap=cmap, zorder=3)
            offset = .1
            pl.set_xlim(self.rescale(x, offset))
            pl.set_ylim(self.rescale(y, 3 / 2 * offset))
            pl.set_ylabel('Count')
            pl.set_xlabel('Centrality')
            pl.set_title(t + ' centrality distribution')
            pl.grid(zorder=0)
            if root:
                pl.axvline(x=centr[t][root], color='magenta', lw=2, zorder=5)
        fig.set_size_inches(12, 6)
        fig.tight_layout(pad=0.01, w_pad=1, h_pad=2.0)
        plt.show()

    def draw_graph_centr(self, G, cmap, depth=None, root=None, nclasses=None, savefig=None):
        plt.close('all')
        fig = plt.figure(figsize=(12, 10), facecolor='#252525')

        if depth:
            if not root:
                root = random.choice(G.nodes())
            comp = self.network_area(G, depth=depth, root=root)
        else:
            comp = G.subgraph(init_nodes)
        nodes = comp.nodes()
        edges = comp.edges()

        centr = self.calc_centr(comp)

        pos = nx.spring_layout(comp)
        xpos = [x for x, _ in pos.values()]
        ypos = [y for _, y in pos.values()]
        xlen = max(xpos)-min(xpos)
        ylen = max(ypos)-min(ypos)
        offset = .3

        poss = {
            centr.keys()[0]: {k: (x, y) for k, (x, y) in pos.iteritems()},
            centr.keys()[1]: {k: (x+xlen+offset, y+ylen+offset) for k, (x, y) in pos.iteritems()},
            centr.keys()[2]: {k: (x+xlen+offset, y) for k, (x, y) in pos.iteritems()},
            centr.keys()[3]: {k: (x, y+ylen+offset) for k, (x, y) in pos.iteritems()}
        }

        node_degree = nx.degree(comp)
        for title, cent in centr.iteritems():
            nodes, cents = zip(*sorted(cent.items(), key=lambda x: x[1]))
            degrees = [node_degree[n] for n in nodes]
            node_size = map(lambda x: 20 + 50 * x / max(degrees), degrees)
            nx.draw_networkx_nodes(
                comp,
                poss[title],
                nodelist=nodes,
                node_size=node_size,
                node_color=cents,
                node_shape='s',
                linewidths=.5,
                with_labels=False,
                cmap=cmap
            )

            edges = comp.edges()
            degree_mins = map(lambda x: min([node_degree[x[0]], node_degree[x[1]]]), edges)
            edgewidths = map(lambda x: .5 + 3 * x / max(degree_mins), degree_mins)
            nx.draw_networkx_edges(
                comp,
                poss[title],
                edgelist=edges,
                width=edgewidths,
                edge_color=degree_mins,
                alpha=.3,
                edge_cmap=self.grayify_cmap(plt.cm.cool),
                edge_vmin=-max(degree_mins),
                edge_vmax=max(degree_mins)
            )
            plt.text(
                np.mean([x for x, _ in poss[title].values()])-.1,
                max([y for _, y in poss[title].values()])+.1,
                title, color='white'
            )

        cut = .1
        lim_f = lambda minx, maxx: [minx - cut * (maxx - minx), maxx + cut * (maxx - minx)]
        minx = min([x for pos in poss.values() for x, y in pos.values()])
        maxx = max([x for pos in poss.values() for x, y in pos.values()])
        miny = min([y for pos in poss.values() for x, y in pos.values()])
        maxy = max([y for pos in poss.values() for x, y in pos.values()])
        plt.xlim(lim_f(minx, maxx))
        plt.ylim(lim_f(miny, maxy))
        if root:
            if depth:
                plt.title('Graph [$root=%d$, $level=%d$, $N=%d$, $L=%d$]' %
                          (root, depth, len(comp), len(comp.edges())), color='white')
            else:
                plt.title('Graph [$root=%d$, $N=%d$, $L=%d$]' %
                          (root, len(comp), len(comp.edges())), color='white')
        else:
            plt.title('Graph [$N=%d$, $L=%d$]' %
                      (len(comp), len(comp.edges())), color='white')
        plt.axis('off')
        if savefig:
            plt.savefig(savefig, transparent=True, dpi=500, bbox_inches='tight')
        plt.show()

    # Attributes

    def attach_attr_from_mongo(self, mongo, G, *attrs):
        tab = Table()
        tdata = []
        for attr in attrs:
            start = timeit.default_timer()
            node_attr = mongo.collect(attr, filter={'id': {'$in': G.nodes()}})
            nodes_by_degree = [int(v) for v, _ in Counter(
                [e for v in node_attr.values() for e in v]).most_common()]
            reduce_func = lambda x: int(
                nodes_by_degree[min(map(nodes_by_degree.index, x))])
            attr_dict = {k: reduce_func(v) for k, v in node_attr.iteritems()}
            nx.set_node_attributes(G, attr, attr_dict)
            stop = timeit.default_timer()
            setsorted = sorted(set(attr_dict.values()))
            tdata.append((attr, len(setsorted), min(setsorted),
                          max(setsorted), (stop - start)))
        tab.from_tuples(
            tdata,
            columns=['Attribute', 'Unique', 'Min', 'Max', 'Time (sec)']
        )
        tab.sort_values(by=['Unique'], ascending=False)
        tab.display()

    # Assortativity

    def classify(self, node_attr, nclasses):
        setsorted = set(node_attr.values())
        xmin, xmax = min(setsorted), max(setsorted)
        c = lambda x: int((((x - xmin) * (nclasses - 1)) / (xmax - xmin)) + 1)
        return {
            n: c(v)
            for n, v in node_attr.iteritems()
        }

    # As attributes only numeric values are allowed (also for attribute_assortativity_coefficient) to save up the space
    # Those values which are >1000 perform very slow -> scale values ->
    # performance >100x faster
    def assortativity(self, G, attr_assorttype):
        tab = Table()
        tdata = []
        assorttypes = {
            'attribute': lambda x, y: nx.attribute_assortativity_coefficient(x, y),
            'numeric': lambda x, y: nx.numeric_assortativity_coefficient(x, y)
        }
        rescale_func = {
            'attribute': lambda x, xsetsorted: x,
            'numeric': lambda x, setsorted: self.classify(x, setsorted, 100) if max(setsorted) > 1000 else x
        }
        for attr, assorttype in attr_assorttype.iteritems():
            start = timeit.default_timer()
            node_attr = nx.get_node_attributes(G, attr)
            setsorted = set(node_attr.values())
            if len(setsorted) > 1:
                if max(setsorted) > 1000:
                    node_attr = self.classify(node_attr, 100)
                G_assort = nx.Graph(G.subgraph(node_attr.keys()))
                nx.set_node_attributes(G_assort, attr, node_attr)
                coef = assorttypes[assorttype](G_assort, attr)
            else:
                coef = np.nan
            stop = timeit.default_timer()
            tdata.append((assorttype, attr, coef, (stop - start)))
        start = timeit.default_timer()
        coef = nx.degree_pearson_correlation_coefficient(G)
        deg = set(nx.degree(G).values())
        stop = timeit.default_timer()
        tdata.append(('degree', '', coef, (stop - start)))
        tab.from_tuples(
            tdata,
            columns=['Type', 'Attribute', 'Coef.', 'Time (sec)']
        )
        tab.sort_values(by=['Coef.'], ascending=False)
        tab.display()

    def plot_attr_dist(self, G, attrs, cmap, root=None):
        plt.close('all')
        fig, pl = plt.subplots(len(attrs), 2)
        if len(attrs)>1:
            pl_matrix = np.array(pl)
        else:
            pl_matrix = np.array([list(pl)])
        for i, attr in enumerate(attrs):
            node_attr = nx.get_node_attributes(G, attr)
            x, y = self.bin_integers(node_attr.values())
            pl_matrix[i][0].scatter(x, y, marker='o', s=20, lw=0, c=y, cmap=cmap, zorder=3)
            offset = .1
            pl_matrix[i][0].set_xlim(self.rescale(x, offset))
            pl_matrix[i][0].set_ylim(self.rescale(y, 3 / 2 * offset))
            pl_matrix[i][0].set_xlabel(attr)
            pl_matrix[i][0].set_ylabel('Count')
            pl_matrix[i][0].set_title(attr + ' distribution')
            pl_matrix[i][0].grid(zorder=0)
            if root and root in node_attr:
                pl_matrix[i][0].axvline(x=node_attr[root], color='magenta', lw=2)
            deg = nx.degree(G)
            x_dd = defaultdict(list)
            for n, v in node_attr.iteritems():
                x_dd[v].append(deg[n])
            x, y = [], []
            for n, vs in x_dd.iteritems():
                for v in vs:
                    x.append(n)
                    y.append(v)
            m, b, r_value, p_value, std_err = stats.linregress(x, y)
            f = lambda x: m * x + b
            d = [abs(y[j] - f(xv)) for j, xv in enumerate(x)]
            pl_matrix[i][1].scatter(x, y, marker='o', s=20, lw=0, c=d, cmap=cmap, zorder=3)
            xr = [min(x), max(x)]
            yr = [f(min(x)), f(max(x))]
            rmarker, = pl_matrix[i][1].plot(xr, yr, lw=1, c='k', zorder=4)
            pl_matrix[i][1].legend(
                [rmarker],
                ['$r^2:%.2f, S:%.2f, y=%.2fx+%.2f$' %
                    (r_value**2, std_err, m, b)],
                fontsize='medium'
            )
            pl_matrix[i][1].set_xlim(self.rescale(x, offset))
            pl_matrix[i][1].set_ylim(self.rescale(y, 3 / 2 * offset))
            pl_matrix[i][1].set_xlabel(attr)
            pl_matrix[i][1].set_ylabel('avg(Degree)')
            pl_matrix[i][1].set_title(attr + ' vs. average degree')
            pl_matrix[i][1].grid(zorder=0)
            if root and root in node_attr and root in deg:
                pl_matrix[i][1].plot(
                    node_attr[root], deg[root],
                    lw=0, c='magenta',
                    marker='o', ms=10,
                    zorder=5
                )
        fig.set_size_inches(12, 3 * len(attrs))
        fig.tight_layout(pad=0.01, w_pad=1, h_pad=2.0)
        plt.show()

    def plot_attr_vs_attr(self, G, yattr_xattr, cmap, root=None):
        plt.close('all')
        plcount = int(math.ceil(len(yattr_xattr)/2))
        fig, pl = plt.subplots(plcount, 2)
        if plcount>1:
            pl_matrix = np.array(pl)
        else:
            pl_matrix = np.array([list(pl)])
        for i, (y_attr, x_attr) in enumerate(yattr_xattr):
            ax = pl_matrix[int(math.floor(i/2)),int(i%2)]
            node_x_attr = nx.get_node_attributes(G, x_attr)
            node_y_attr = nx.get_node_attributes(G, y_attr)
            node_intsct = set.intersection(set(node_x_attr.keys()), set(node_y_attr.keys()))
            x, y = [], []
            for n in node_intsct:
                x.append(node_x_attr[n])
                y.append(node_y_attr[n])
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
            ax.set_xlabel(x_attr)
            ax.set_ylabel(y_attr)
            ax.set_title(x_attr + ' vs. ' + y_attr)
            ax.grid(zorder=0)
            if root and root in node_intsct:
                ax.plot(
                    node_x_attr[root], node_y_attr[root],
                    lw=0, c='magenta',
                    marker='o', ms=10,
                    zorder=5
                )
        if len(yattr_xattr)%2>0:
            pl_matrix[int(math.floor((len(yattr_xattr)-1)/2))][1].axis('off')
        fig.set_size_inches(12, 3 * plcount)
        fig.tight_layout(pad=0.01, w_pad=1, h_pad=2.0)
        plt.show()

    # Communities

    def modularity(self, G, attrs, classes=None):
        tab = Table()
        tdata = []
        for attr in attrs:
            start = timeit.default_timer()
            node_attr = nx.get_node_attributes(G, attr)
            setsorted = set(node_attr.values())
            if len(setsorted) > 1:
                if classes and attr in classes:
                    node_attr = self.classify(node_attr, classes[attr])
                part_G = nx.Graph(G.subgraph(node_attr.keys()))
                M = community.modularity(node_attr, part_G)
            else:
                M = np.nan
            stop = timeit.default_timer()
            tdata.append((attr, len(set(node_attr.values())), M, (stop - start)))
        tab.from_tuples(
            tdata, columns=['Attribute', 'Communities', 'M', 'Time (sec)'])
        tab.sort_values(by='M', ascending=False)
        tab.display()

    def find_communities(self, G):
        tab = Table()
        start = timeit.default_timer()
        communities = community.best_partition(G)
        M = community.modularity(communities, G)
        stop = timeit.default_timer()
        tab.from_tuples([(len(set(communities.values())), M,
                          (stop - start))], columns=['Unique', 'M', 'Time (sec)'])
        tab.sort_values(by='M', ascending=False)
        tab.display()
        return communities

    def attach_communities(self, G, communities):
        nx.set_node_attributes(G, 'communities', communities)

    # Draw distribution

    def bin_integers(self, integers):
        xy = Counter(integers)
        return xy.keys(), xy.values()

    def bin_floats(self, floats, bins):
        y, bins = np.histogram(floats, bins=bins)
        x = (bins[:-1] + bins[1:]) / 2
        return x, y

    def rescale(self, axis_values, offset):
        minx, maxx = min(axis_values), max(axis_values)
        return [minx - offset * (maxx - minx), maxx + offset * (maxx - minx)]

    def rescale_loglog(self, axis_values, offset):
        log = lambda x: math.log(x, 10) if x > 0 else - \
            math.log(abs(x), 10) if x < 0 else 0
        minx, maxx = min(axis_values), max(axis_values)
        return [10**(log(minx) - offset * (log(maxx) - log(minx))),
                10**(log(maxx) + offset * (log(maxx) - log(minx)))]

    def map_colors(self, y, cmap):
        cNorm = colors.Normalize(vmin=min(y), vmax=max(y))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        return [scalarMap.to_rgba(v) for v in y]

    def plot_degree_dist(self, G, degrees, cmap, root=None):
        plt.close('all')
        fig, (ax_lin, ax_loglog) = plt.subplots(1, 2)
        x, y = self.bin_integers(filter(None, degrees))
        ax_lin.scatter(x, y, marker='o', s=20, lw=0, c=y, cmap=cmap, zorder=3)
        offset = .1
        ax_lin.set_xlim(self.rescale(x, offset))
        ax_lin.set_ylim(self.rescale(y, 3 / 2 * offset))
        ax_lin.set_xlabel('Degree')
        ax_lin.set_ylabel('Count')
        ax_lin.set_title('Degree distribution linear')
        ax_lin.grid(zorder=0)
        ax_loglog.scatter(x, y, marker='o', s=20, lw=0, c=y, cmap=cmap, zorder=3)
        ax_loglog.set_yscale('log')
        ax_loglog.set_xscale('log')
        ax_loglog.set_xlim(self.rescale_loglog(x, offset))
        ax_loglog.set_ylim(self.rescale_loglog(y, 3 / 2 * offset))
        ax_loglog.set_xlabel('Degree')
        ax_loglog.set_ylabel('Count')
        ax_loglog.set_title('Degree distribution loglog')
        ax_loglog.grid(zorder=0)
        if root:
            ax_lin.axvline(x=G.degree(root), color='magenta', lw=2)
            ax_loglog.axvline(x=G.degree(root), color='magenta', lw=2)
        fig.set_size_inches(12, 3)
        fig.tight_layout(pad=0.01, w_pad=1, h_pad=2.0)
        plt.show()

    # Draw graph

    def grayify_cmap(self, cmap):
        cmap = plt.cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))
        RGB_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
        colors[:, :3] = luminance[:, np.newaxis]
        return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)

    def draw_graph_attr(self, G, attr, vcmap, depth=None, root=None, nclasses=None, nlabels=None,
                        mongo=None, labelsmap=None, savefig=None):
        plt.close('all')
        fig = plt.figure(figsize=(8, 6), facecolor='#252525')

        node_attr = nx.get_node_attributes(G, attr)
        setsorted = set(node_attr.values())

        if len(setsorted) > 1:
            if nclasses:
                node_attr = {n: v for n, v in self.classify(
                    node_attr, nclasses).iteritems()}
        if isinstance(vcmap, dict):
            node_attr = {n: v for n, v in node_attr.iteritems() if v in vcmap}
        else:
            node_attr = {n: v for n, v in node_attr.iteritems()}

        init_nodes = set(n for e in G.edges(node_attr.keys()) for n in e)
        if depth:
            if not root:
                root = random.choice(list(init_nodes))
            comp = self.network_area(G.subgraph(
                init_nodes), depth=depth, root=root)
        else:
            comp = G.subgraph(init_nodes)
        nodes = comp.nodes()
        edges = comp.edges()

        attr_nodes = defaultdict(list)
        for n in nodes:
            if n in node_attr:
                attr_nodes[node_attr[n]].append(n)
            else:
                attr_nodes[-1].append(n)

        print 'Attribute values:', ', '.join(map(str, attr_nodes.keys()))

        pos = nx.spring_layout(comp)
        node_degree = nx.degree(comp)
        for a, ns in sorted(attr_nodes.items(), key=lambda x: x[0]):
            degrees = [node_degree[n] for n in ns]
            node_size = map(lambda x: 20 + 50 * x / max(node_degree.values()), degrees)
            node_color = degrees if isinstance(vcmap, dict) else [a]*len(ns)
            if a==-1:
                cmap = self.grayify_cmap(plt.cm.cool)
            else:
                cmap = vcmap[a] if isinstance(vcmap, dict) else vcmap
            vmax = max(map(abs, degrees)) if isinstance(vcmap, dict) else max(map(abs, attr_nodes.keys()))
            vmin = min(degrees)-vmax if isinstance(vcmap, dict) else min(attr_nodes.keys())-vmax
            nx.draw_networkx_nodes(
                comp,
                pos,
                nodelist=ns,
                node_size=node_size,
                node_color=node_color,
                node_shape='s',
                linewidths=.5,
                with_labels=False,
                cmap=cmap,
                vmin=vmin,  # Second half of color map
                vmax=vmax
            )

        attr_edges = defaultdict(list)
        edge_degree = {}
        for n1, n2 in edges:
            if n1 in node_attr and n2 in node_attr and node_attr[n1] == node_attr[n2]:
                attr_edges[node_attr[n1]].append((n1, n2))
            else:
                attr_edges[-1].append((n1, n2))
            edge_degree[(n1, n2)] = min([node_degree[n1], node_degree[n2]])
        for a, es in sorted(attr_edges.items(), key=lambda x: x[0]):
            degrees = [edge_degree[e] for e in es]
            edge_width = map(lambda x: .5 + 3 * x / max(edge_degree.values()), degrees)
            edge_color = degrees if isinstance(vcmap, dict) else [a]*len(es)
            if a==-1:
                cmap = self.grayify_cmap(plt.cm.cool)
            else:
                cmap = vcmap[a] if isinstance(vcmap, dict) else vcmap

            edge_vmax = max(map(abs, degrees)) if isinstance(vcmap, dict) else max(map(abs, attr_nodes.keys()))
            edge_vmin = min(degrees)-edge_vmax if isinstance(vcmap, dict) else min(attr_nodes.keys())-edge_vmax
            nx.draw_networkx_edges(
                comp,
                pos,
                edgelist=es,
                width=edge_width,
                edge_color=edge_color,
                alpha=.3,
                edge_cmap=cmap,
                edge_vmin=edge_vmin,
                edge_vmax=edge_vmax
            )
        if nlabels:
            offset = .05
            for p in pos:  # raise text positions
                pos[p][1] += offset
            for a, nl in nlabels.iteritems():
                attr_node_degree = {n: node_degree[n] for n in attr_nodes[a]}
                labels = {n: n for n, _ in sorted(
                    attr_node_degree.items(), key=operator.itemgetter(1), reverse=True)[:nl]}
                if mongo and labelsmap:
                    if isinstance(labelsmap, dict):
                        labels = {
                            k: u'{}'.format(mongo.collect(mapc, filter={'id': abs(v)})[abs(v)][0])
                            for k, v in labels.iteritems()
                            for mapc, f in labelsmap.iteritems() if f(v)
                        }
                    else:
                        labels = {
                            k: u'{}'.format(mongo.collect(labelsmap, filter={'id': abs(v)})[abs(v)][0])
                            for k, v in labels.iteritems()
                        }
                else:
                    labels = {k: str(v) for k, v in labels.iteritems()}
                nx.draw_networkx_labels(
                    comp,
                    pos,
                    labels,
                    font_size=10,
                    font_color='white'
                )
        else:
            offset = 0
        if root:
            if depth:
                plt.title('Graph [$root=%d$, $level=%d$, $N=%d$, $L=%d$]' %
                          (root, depth, len(comp), len(comp.edges())), color='white')
            else:
                plt.title('Graph [$root=%d$, $N=%d$, $L=%d$]' %
                          (root, len(comp), len(comp.edges())), color='white')
        else:
            plt.title('Graph [$N=%d$, $L=%d$]' %
                      (len(comp), len(comp.edges())), color='white')
        plt.axis('off')
        cut = .1
        lim_f = lambda minx, maxx: [minx - cut * (maxx - minx), maxx + cut * (maxx - minx)]
        posxx = [xx for xx, _ in pos.values()]
        posyy = [yy for _, yy in pos.values()]
        plt.xlim(lim_f(min(posxx), max(posxx)))
        plt.ylim(lim_f(min(posyy) - offset, max(posyy)))
        if savefig:
            plt.savefig(savefig, transparent=True, dpi=500, bbox_inches='tight')
        plt.show()

    def write_graph(self, G, path):
        nx.write_gpickle(G, path)

    def read_graph(self, path):
        G = nx.read_gpickle(path)
        self.graph_info(G)
        return G
