import numpy as np
from scipy.spatial import cKDTree

# Mostly the same code as in the first paper. I added the function 
# build_GIRG_full to also return the positions and edge list; build_GIRG 
# works as before. Both experiments use this code. It follows Definitions 
# 2.1 and 2.7: Pareto weights with scale 1, uniform positions on the torus,
# L-infinity distance, and the threshold r_uv = (w_u * w_v / W_n)**(1/d).
# Social labels are drawn uniformly from m labels, with an independent
# transmission coin for each ordered pair.


def sample_weights(n, beta, w_min=1.0):
    # Inverse-CDF sampling for the Pareto law P(W >= w) = w^-(beta - 1).
    # u is uniform on (0, 1), so (1 - u) is too. Using it instead of u
    # avoids a division-by-zero edge case if u ever landed exactly on 0.
    u = np.random.uniform(0, 1, n)
    return w_min * (1 - u)**(-1 / (beta - 1))


def build_GIRG(n, d, beta, weights=None, w_min=1.0):
    # If a weight sequence is passed in, we keep it and only resample the
    # positions. Experiment 2 needs this, since Theorem 3.3 is a statement
    # conditional on the weights and averages over the positions.
    indptr, indices, weights, W_n, _, _, _ = build_GIRG_full(n, d, beta, weights, w_min)
    return indptr, indices, weights, W_n


def build_GIRG_full(n, d, beta, weights=None, w_min=1.0):
    # Same graph as build_GIRG, with the same random draws in the same order,
    # so a fixed seed gives exactly the graphs of the first paper. It also
    # returns the positions, which the spatial labels need, and the edge list
    # (rows[e], cols[e]) with rows < cols, which the threshold sweep needs.
    if weights is None:
        weights = sample_weights(n, beta, w_min)
    positions = np.random.uniform(0, 1, (n, d))
    W_n = weights.sum()
    # Checking all n^2 pairs is too slow for n = 10^5, so we group the
    # vertices into weight classes [2^k, 2^(k+1)) and, for each pair of
    # classes, only look at pairs closer than the largest radius the two
    # classes can produce. Exact radii are then checked pair by pair below,
    # so the grouping only speeds things up and does not change the graph.
    cls = np.floor(np.log2(weights / w_min)).astype(int)
    groups = [np.where(cls == k)[0] for k in np.unique(cls)]
    trees = [cKDTree(positions[g], boxsize=1.0) for g in groups]
    w_top = [weights[g].max() for g in groups]
    rows, cols = [], []
    for i in range(len(groups)):
        for j in range(i, len(groups)):
            # No two points on the torus are further apart than 1/2 in the
            # L infinity norm, so a search radius above 1/2 is never needed.
            r_search = min((w_top[i] * w_top[j] / W_n)**(1 / d), 0.5)
            pairs = trees[i].sparse_distance_matrix(
                trees[j], r_search, p=np.inf, output_type='ndarray')
            u = groups[i][pairs['i']]
            v = groups[j][pairs['j']]
            r = (weights[u] * weights[v] / W_n)**(1 / d)
            keep = pairs['v'] <= r
            if i == j:
                # Inside one class each pair shows up twice, plus u with
                # itself at distance 0, so we keep only u < v.
                keep &= u < v
            rows.append(u[keep])
            cols.append(v[keep])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    # Store the undirected graph in compressed (CSR) form: the neighbours
    # of u are indices[indptr[u]:indptr[u + 1]]. This is what lets the
    # cascade below handle a whole generation at once.
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    order = np.argsort(src, kind='stable')
    indices = dst[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return indptr, indices, weights, W_n, positions, rows, cols


def sample_labels(n, m):
    # Each vertex gets one of m labels, independently and uniformly.
    return np.random.randint(0, m, n)


def expected_bias(m, b_plus, b_minus):
    # B = b^+/m + (1 - 1/m) b^-, the expected bias factor from Section 2.3.
    return b_plus / m + (1 - 1 / m) * b_minus


def R0_limit(d, beta, tau, m, b_plus, b_minus):
    # Limiting R0 = 2^d tau B E[W^2]/E[W] from Lemma 3.1. For Pareto weights
    # with scale 1, E[W] = (beta-1)/(beta-2) and E[W^2] = (beta-1)/(beta-3).
    ratio = (beta - 2) / (beta - 3)
    return 2**d * tau * expected_bias(m, b_plus, b_minus) * ratio


def R0_finite(weights, W_n, d, tau, m, b_plus, b_minus):
    # R0^(n) = 2^d tau B sum_v w_v^2 / W_n from Section 3.2. For a fixed
    # weight sequence, Theorem 3.3 gives E[Z_s | w] <= (W_n/n) (R0^(n))^s.
    return 2**d * tau * expected_bias(m, b_plus, b_minus) * (weights**2).sum() / W_n


def max_radius(weights, d):
    # Largest connection radius max_{u != v} r_uv, attained by the two largest weights.
    top2 = np.sort(weights)[-2:]
    return (top2[0] * top2[1] / weights.sum())**(1 / d)


def neighbour_pairs(indptr, indices, frontier):
    # Returns every (infector, neighbour) pair for the vertices in frontier.
    starts = indptr[frontier]
    counts = indptr[frontier + 1] - starts
    total = counts.sum()
    src = np.repeat(frontier, counts)
    offset = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    nbr = indices[np.repeat(starts, counts) + offset]
    return src, nbr


def run_cascade(indptr, indices, weights, labels, tau, b_plus, b_minus,
                seed=None, max_size=None):
    # Independent cascade of Definition 2.7. Returns, for each generation s,
    # the number of newly infected vertices |I_s| and their weight Z_s.
    # If max_size is given, we stop once that many vertices are infected.
    # Experiment 1 uses this to avoid simulating a large cascade to the end
    # when we only need to know that it became large.
    n = len(weights)
    if seed is None:
        seed = np.random.randint(0, n)
    infected = np.zeros(n, dtype=bool)
    infected[seed] = True
    frontier = np.array([seed])
    counts, Zs = [1], [weights[seed]]
    total = 1
    while frontier.size:
        src, nbr = neighbour_pairs(indptr, indices, frontier)
        # Attempts are only made on vertices that are not yet infected.
        fresh = ~infected[nbr]
        src, nbr = src[fresh], nbr[fresh]
        # Each attempt uses its own coin xi_uv with success probability
        # tau * B_uv. Two infectors trying the same vertex in the same
        # generation use two different coins, as in the model.
        q = tau * np.where(labels[src] == labels[nbr], b_plus, b_minus)
        hit = np.random.random(nbr.size) < q
        frontier = np.unique(nbr[hit])
        if frontier.size == 0:
            break
        infected[frontier] = True
        counts.append(frontier.size)
        Zs.append(weights[frontier].sum())
        total += frontier.size
        if max_size is not None and total >= max_size:
            break
    return np.array(counts), np.array(Zs)


def fresh_offspring(indptr, indices, labels, tau, b_plus, b_minus, n_samples):
    # Offspring of an edge-reached vertex in a fully susceptible population.
    # We pick a uniform (vertex, neighbour) incidence, which is the same as a
    # uniform edge with a random orientation, so the reached endpoint v is
    # sampled proportionally to its degree. We then count v's successful
    # transmissions to all its neighbours except the one it was reached from.
    n = len(indptr) - 1
    deg = np.diff(indptr)
    reached = np.repeat(np.arange(n), deg)
    parent = indices
    k = np.random.randint(0, len(indices), n_samples)
    out = np.empty(n_samples)
    for i, e in enumerate(k):
        v, p = reached[e], parent[e]
        nb = indices[indptr[v]:indptr[v + 1]]
        nb = nb[nb != p]
        q = tau * np.where(labels[nb] == labels[v], b_plus, b_minus)
        out[i] = (np.random.random(nb.size) < q).sum()
    return out