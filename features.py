import heapq
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from girg_core import neighbour_pairs

# Node features for the seed-level models. They come in two groups, so the
# models can be compared by what they are allowed to see:
# SEED: what a seed knows about itself (weight, degree, label).
# LOCAL: hand-crafted descriptions of its surroundings, from one-hop
# quantities (clustering, triangles, same-label neighbours) up to
# the distance to the nearest label boundary and the size of its
# same-label cluster, which reach further.
# Positions are never used: a real network does not come with them.

SEED = ['log_w', 'log_deg', 'label_0', 'label_1', 'label_2']
LOCAL = ['same_nbr_frac', 'mean_edge_bias', 'log_triangles', 'clustering',
         'mean_nbr_log_deg', 'log_nbr_weight', 'log_two_hop', 'core',
         'boundary_dist', 'log_label_cluster']


def boundary_distance(indptr, indices, labels):
    # Graph distance from every vertex to the nearest vertex that has
    # a neighbour with a different label (0 means the vertex itself is 
    # on a boundary), by a breadth-first search from all boundary vertices 
    # at once. Vertices with no boundary in their component get -1.
    n = labels.size
    src, nbr = neighbour_pairs(indptr, indices, np.arange(n))
    on_boundary = np.zeros(n, dtype=bool)
    on_boundary[src[labels[src] != labels[nbr]]] = True
    dist = np.full(n, -1)
    frontier = np.where(on_boundary)[0]
    dist[frontier] = 0
    level = 0
    while frontier.size:
        level += 1
        _, nb = neighbour_pairs(indptr, indices, frontier)
        nb = np.unique(nb[dist[nb] < 0])
        dist[nb] = level
        frontier = nb
    return dist


def core_numbers(indptr, indices):
    # k-core number of every vertex, by repeatedly peeling off the vertex of
    # smallest remaining degree. We keep the candidates in a heap and skip
    # stale entries, which is quick enough for n up to a few 10^5.
    n = indptr.size - 1
    deg = np.diff(indptr).astype(np.int64)
    core = np.zeros(n, dtype=np.int64)
    removed = np.zeros(n, dtype=bool)
    heap = [(int(deg[v]), int(v)) for v in range(n)]
    heapq.heapify(heap)
    k = 0
    while heap:
        dv, v = heapq.heappop(heap)
        if removed[v] or dv != deg[v]:
            continue
        k = max(k, dv)
        core[v] = k
        removed[v] = True
        for x in indices[indptr[v]:indptr[v + 1]]:
            if not removed[x]:
                deg[x] -= 1
                heapq.heappush(heap, (int(deg[x]), int(x)))
    return core


def node_features(indptr, indices, rows, cols, w, labels, m, b_plus, b_minus):
    # Returns a dict of feature arrays, one entry per name in SEED + LOCAL.
    n = w.size
    deg = np.diff(indptr)
    f = {'log_w': np.log(w), 'log_deg': np.log1p(deg)}
    for k in range(m):
        f[f'label_{k}'] = (labels == k).astype(float)
    A = csr_matrix((np.ones(indices.size), indices, indptr), shape=(n, n))
    src, nbr = neighbour_pairs(indptr, indices, np.arange(n))
    same = labels[src] == labels[nbr]
    safe_deg = np.maximum(deg, 1)
    f['same_nbr_frac'] = np.bincount(src, weights=same, minlength=n) / safe_deg
    B = np.where(same, b_plus, b_minus)
    f['mean_edge_bias'] = np.bincount(src, weights=B, minlength=n) / safe_deg
    # Triangles through each vertex: (A^2 * A) summed over a row, halved, and
    # the number of vertices within two hops. On dense graphs (Facebook) A^2
    # does not fit in memory at once, so we form it a block of rows at a time.
    tri = np.zeros(n)
    two_hop = np.zeros(n)
    block = max(1, int(2e7 // max(1, (deg.astype(float)**2).mean())))
    for a in range(0, n, block):
        Ab = A[a:a + block]
        A2b = Ab @ A
        tri[a:a + block] = np.asarray(A2b.multiply(Ab).sum(axis=1)).ravel() / 2
        two_hop[a:a + block] = np.diff(((Ab + A2b) > 0).astype(np.int8).tocsr().indptr)
    f['log_triangles'] = np.log1p(tri)
    pairs = deg * (deg - 1) / 2
    f['clustering'] = np.where(pairs > 0, tri / np.maximum(pairs, 1), 0.0)
    f['mean_nbr_log_deg'] = np.bincount(src, weights=np.log1p(deg[nbr]), minlength=n) / safe_deg
    f['log_nbr_weight'] = np.log1p(np.bincount(src, weights=w[nbr], minlength=n))
    f['log_two_hop'] = np.log1p(two_hop)
    f['core'] = core_numbers(indptr, indices).astype(float)
    dist = boundary_distance(indptr, indices, labels)
    f['boundary_dist'] = np.where(dist < 0, 10, np.minimum(dist, 10)).astype(float)
    keep = labels[rows] == labels[cols]
    S = coo_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(n, n))
    _, comp = connected_components(S, directed=False)
    f['log_label_cluster'] = np.log(np.bincount(comp)[comp])
    return f