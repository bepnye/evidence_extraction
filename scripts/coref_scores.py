'''
CoNLL-2011/2012 scores for coreference detection.


## References

- **Scoring Coreference Partitions of Predicted Mentions: A Reference Implementation.** Sameer
Pradhan, Xiaoqiang Luo, Marta Recasens, Eduard Hovy, Vincent Ng and Michael Strube. *Proceedings
of the 52nd Annual Meeting of the Association for Computational Linguistics*, Baltimore, MD,
June 2014. ([pdf](http://aclweb.org/anthology/P/P14/P14-2006.pdf))
- **BLANC: Implementing the Rand Index for Coreference Evaluation.** Marta Recasens and Eduard
Hovy In: *Natural Language Engineering* 17 (4). Cambridge University Press, 2011.
([pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.300.9229&rep=rep1&type=pdf))
- **An Extension of BLANC to System Mentions.** Xiaoqiang Luo, Sameer Pradhan, Marta Recasens and
Eduard Hovy. *Proceedings of the 52nd Annual Meeting of the Association for Computational
Linguistics*, Baltimore, MD, June 2014. ([pdf](http://aclweb.org/anthology/P/P14/P14-2005.pdf))
The reference implementation : <https://github.com/conll/reference-coreference-scorers>
'''
import math

import itertools as it
import typing as ty
import numpy as np

from scipy.optimize import linear_sum_assignment

def hmean(r: float, p: float) -> float:
    if p+r > 0.0:
        return 2*p*r/(p+r)
    return 0.0
    

# OPTIMIZE: This could be made faster by dealing separately with singletons
# OPTIMIZE: This could be made faster (in average) by sorting `cluster_lst` by decreasing length
def links_from_clusters(
    clusters: ty.Iterable[ty.Set],
) -> ty.Tuple[
    ty.Sequence[ty.Tuple[ty.Hashable, ty.Hashable]],
    ty.Sequence[ty.Tuple[ty.Hashable, ty.Hashable]],
]:
    r'''
    Return a `(coreference_links, non-coreference_links)` tuple corresponding to a clustering.
    '''
    clusters_lst = list(clusters)
    elements = sorted(set.union(*clusters_lst))
    C = []
    N = []
    for i, j in it.combinations(elements, 2):
        if i == j:
            continue
        elif any(c for c in clusters_lst if i in c and j in c):
            C.append((i, j))
        else:
            N.append((i, j))
    return C, N


def trace(cluster: ty.Set, partition: ty.Iterable[ty.Set]) -> ty.Iterable[ty.Set]:
    r'''
    Return the partition of `#cluster` induced by `#partition`, that is
    ```math
    \{C∩A|A∈P\} ∪ \{\{x\}|x∈C∖∪P\}
    ```
    Where `$C$` is `#cluster` and `$P$` is `#partition`.

    This assume that the elements of `#partition` are indeed pairwise disjoint.
    '''
    remaining = set(cluster)
    for a in partition:
        common = remaining.intersection(a)
        if common:
            remaining.difference_update(common)
            yield common
    for x in sorted(remaining):
        yield set((x,))


def muc(
    key: ty.Sequence[ty.Set], response: ty.Sequence[ty.Set]
) -> ty.Tuple[float, float, float]:
    r'''
    Compute the MUC `$(R, P, F₁)$` scores for a `#response` clustering given a `#key` clustering,
    that is
    ```math
    R &= \frac{∑_{k∈K}(\#k-\#p(k, R))}{∑_{k∈K}(\#k-1)}\\
    P &= \frac{∑_{r∈R}(\#r-\#p(r, K))}{∑_{r∈R}(\#r-1)}\\
    F &= 2*\frac{PR}{P+R}
    ```
    with `$p(x, E)=\{x∩A|A∈E\}$`.

    In the edge case where all clusters in either `#key` or `#response` are singletons, `$P$`, `$R$`
    and `$F$` are defined to be `$0$`, following the reference implementation (since singleton
    clusters where not considered in Vilain et al. (1995).

    Note: This implementation is significantly different from the reference one (despite
    implementing the formulae from Pradahan et al. (2014) in that the reference use the ordering of
    mentions in documents to consistently assign a non-problematic spanning tree (viz. a chain) to
    each cluster, thus avoiding the issues that led Vilain et al. (1995) to define MUC by the
    formulae above.
    '''
    # Edge case
    if all(len(k) == 1 for k in key) or all(len(r) == 1 for r in response):
        return 0.0, 0.0, 0.0
    R = sum(len(k) - sum(1 for _ in trace(k, response)) for k in key) / sum(
        len(k) - 1 for k in key
    )
    P = sum(len(r) - sum(1 for _ in trace(r, key)) for r in response) / sum(
        len(r) - 1 for r in response
    )
    F = hmean(R, P)
    return R, P, F


def b_cubed(
    key: ty.Sequence[ty.Set], response: ty.Sequence[ty.Set]
) -> ty.Tuple[float, float, float]:
    r'''
    Compute the B³ `$(R, P, F₁)$` scores for a `#response` clustering given a `#key` clustering,
    that is
    ```math
    R &= \frac{∑_{k∈K}∑_{r∈R}\frac{(\#k∩r)²}{\#k}}{∑_{k∈K}\#k}\\
    P &= \frac{∑_{r∈R}∑_{k∈K}\frac{(\#r∩k)²}{\#r}}{∑_{r∈R}\#r}\\
    F &= 2*\frac{PR}{P+R}
    ```
    '''
    R = math.fsum(
        len(k.intersection(r)) ** 2 / len(k) for k in key for r in response
    ) / sum(len(k) for k in key)
    P = math.fsum(
        len(r.intersection(k)) ** 2 / len(r) for r in response for k in key
    ) / sum(len(r) for r in response)
    F = hmean(R, P)
    return R, P, F


def ceaf(
    key: ty.Sequence[ty.Set],
    response: ty.Sequence[ty.Set],
    score: ty.Callable[[ty.Set, ty.Set], float],
) -> ty.Tuple[float, float, float]:
    r'''
    Compute the CEAF `$(R, P, F₁)$` scores for a `#response` clustering given a `#key` clustering
    using the `#score` alignment score function, that is
    ```math
    R &= \frac{∑_{k∈K}C(k, A(k))}{∑_{k∈K}C(k, k)}\\
    P &= \frac{∑_{r∈R}C(r, A⁻¹(r))}{∑_{r∈R}C(r, r)}\\
    F &= 2*\frac{PR}{P+R}
    ```
    Where `$C$` is `#score` and `$A$` is a one-to-one mapping from key clusters to response
    clusters that maximizes `$∑_{k∈K}C(k, A(k))$`.
    '''
    cost_matrix = np.array([[-score(k, r) for r in response] for k in key])
    # TODO: See https://github.com/allenai/allennlp/issues/2946 for ideas on speeding
    # the next line up
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_score = -cost_matrix[row_ind, col_ind].sum()
    R = total_score / math.fsum(score(k, k) for k in key)
    P = total_score / math.fsum(score(r, r) for r in response)
    F = hmean(R, P)
    return R, P, F


def ceaf_m(
    key: ty.Sequence[ty.Set], response: ty.Sequence[ty.Set]
) -> ty.Tuple[float, float, float]:
    r'''
    Compute the CEAFₘ `$(R, P, F₁)$` scores for a `#response` clustering given a `#key` clustering,
    that is the CEAF score for the `$Φ_3$` score function
    ```math
    Φ_3: (k, r) ⟼ \#k∩r
    ```
    '''

    def Φ_3(k, r):
        return len(k.intersection(r))

    return ceaf(key, response, Φ_3)


def ceaf_e(
    key: ty.Sequence[ty.Set], response: ty.Sequence[ty.Set]
) -> ty.Tuple[float, float, float]:
    r'''
    Compute the CEAFₑ `$(R, P, F₁)$` scores for a `#response` clustering given a `#key`
    clustering, that is the CEAF score for the `$Φ₄$` score function (aka the Sørensen–Dice
    coefficient).
    ```math
    Φ₄: (k, r) ⟼ \frac{2×\#k∩r}{\#k+\#r}
    ```
    Note: this use the original (Luo, 2005) definition as opposed to Pradhan et al. (2014)'s one
    which inlines the denominators.
    '''

    def Φ_4(k, r):
        return 2 * len(k.intersection(r)) / (len(k) + len(r))

    return ceaf(key, response, Φ_4)


# COMBAK: Check the numeric stability
def blanc(
    key: ty.Sequence[ty.Set], response: ty.Sequence[ty.Set]
) -> ty.Tuple[float, float, float]:
    r'''
    Return the BLANC `$(R, P, F)$` scores for a `#response` clustering given a `#key` clustering.

    ## Notes

      - Mention identifiers have to be comparable
      - To ensure the compliance with the reference implementation, the edge cases results are
        those from Recasens and Hovy (2011) rather than from the more recent Luo et al. (2014) when
        those two disagree. This has an effect for the N-6 testcase, where according to Luo et al.
        (2014), BLANC should be `$\frac{0+F_n}{2}$` since `$C_k=∅$` and `$C_r≠∅$`, but according to
        Recasens and Hovy (2011), BLANC should be `$F_n$`.
    '''
    C_score, N_score = detailed_blanc(key, response)
    if C_score is None:
        assert N_score is not None
        return N_score
    if N_score is None:
        assert C_score is not None
        return C_score
    return np.mean((C_score, N_score), axis=0).tolist()


def detailed_blanc(
    key: ty.Sequence[ty.Set], response: ty.Sequence[ty.Set]
) -> ty.Tuple[
    ty.Union[ty.Tuple[float, float, float], None],
    ty.Union[ty.Tuple[float, float, float], None],
]:
    '''Return BLANC `$(R, P, F)$` scores for coreference and non-coreference respectively.'''

    # Edge case : a single mention in both `key` and `response` clusters
    # in that case, `C_k`, `C_r`, `N_k` and `N_r` are all empty, so we need a separate examination
    # of the mentions to know if we are very good or very bad.
    if len(key) == len(response) == 1 and len(key[0]) == len(response[0]) == 1:
        if key[0] == response[0]:
            return ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
        else:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    C_k, N_k = map(set, links_from_clusters(key))
    C_r, N_r = map(set, links_from_clusters(response))

    TP_c = C_k.intersection(C_r)
    TP_n = N_k.intersection(N_r)

    if not C_k or not C_r:
        R_c, P_c, F_c = (1.0, 1.0, 1.0) if C_k == C_r else (0.0, 0.0, 0.0)
    else:
        R_c, P_c = len(TP_c) / len(C_k), len(TP_c) / len(C_r)
        F_c = 2 * len(TP_c) / (len(C_k) + len(C_r))

    if not N_k or not N_r:
        R_n, P_n, F_n = (1.0, 1.0, 1.0) if N_k == N_r else (0.0, 0.0, 0.0)
    else:
        R_n, P_n = len(TP_n) / len(N_k), len(TP_n) / len(N_r)
        F_n = 2 * len(TP_n) / (len(N_k) + len(N_r))

    # Edge cases
    if not C_k:
        return (None, (R_n, P_n, F_n))
    if not N_k:
        return ((R_c, P_c, F_c), None)

    return ((R_c, P_c, F_c), (R_n, P_n, F_n))


def conll2012(key: ty.Sequence[ty.Set], response: ty.Sequence[ty.Set]) -> float:
    r'''
    Return the CoNLL-2012 scores for a `#response` clustering given a `#key` clustering, that is,
    the average of the MUC, B³ and CEAFₑ scores.
    '''
    return np.mean([metric(key, response)[2] for metric in (muc, b_cubed, ceaf_e)])
