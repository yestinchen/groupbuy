'''
PTS: pareto tree search.
Depth-first enumeration of the capped candidate tree with an ideal-point bound
and raw-cost prefix dominance. Exhausted subtrees retire. `exhausted` records
structural completion of the fixed capped tree; exact Pareto completeness
assumes exact arithmetic. The implementation uses floats, a 1e-9 feasibility
tolerance, and nine-decimal rounding of returned objective vectors.
'''

from __future__ import annotations

import threading
import time
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from gquery.alg.common import Item
from gquery.alg.candidates import (
    PState, _SuffixBounds, _non_dominated_final, reduce_candidates,
)


@dataclass
class PTSResult:
    frontier: List[PState]
    n_evals: int
    n_before: int
    n_after: int
    iterations: int
    solve_time_s: float
    reduction_time_s: float
    '''
    exhausted = root retired: every branch under the candidate cap was evaluated
    or pruned. This is a structural certificate, subject to floating comparisons.
    '''
    exhausted: bool = False
    # n_evals = n_action_evals + n_rollout_evals, archive scans counted separately
    n_action_evals: int = 0
    n_rollout_evals: int = 0
    # ideal-point + prefix-dominance entries scanned
    n_bound_evals: int = 0
    # entries examined when inserting into P
    n_frontier_comparisons: int = 0
    n_pruned_bound: int = 0
    n_pruned_prefix: int = 0
    n_frontier_gains: int = 0
    n_created_nodes: int = 0
    n_retired_nodes: int = 0
    # Iterations that drain a selected node without creating a child.
    n_drained_nodes: int = 0


@dataclass
class _Node:
    depth: int
    choice: List[int]
    quality: float
    cost: float
    rating: float
    parent: Optional["_Node"] = None
    children: List["_Node"] = field(default_factory=list)
    untried: Optional[Deque[int]] = None
    dead: bool = False
    evaluated: bool = False


def _propagate_exhaustion(node: _Node, n: int) -> int:
    """Retire evaluated terminals or fully processed internal nodes, then parents.

    None means actions have not been initialized, so it cannot certify closure.
    Returns the number of newly retired nodes.
    """
    retired = 0
    cur = node
    while cur is not None:
        if not cur.dead:
            ready = (cur.evaluated if cur.depth == n else
                     cur.untried is not None and not cur.untried
                     and all(ch.dead for ch in cur.children))
            if not ready:
                break
            cur.dead = True
            retired += 1
        cur = cur.parent
    return retired


def _select_depth_first(root, n, actions):
    """Finish the first live child's subtree before creating its next sibling."""
    node = root
    while node.depth < n and not node.dead:
        if node.untried is None:
            node.untried = deque(actions(node))
        live = next((ch for ch in node.children if not ch.dead), None)
        if live is None:
            return node
        node = live
    return node


class _PrefixArchive:
    """Weak raw-cost dominance with references to already linked tree nodes."""

    def __init__(self, n):
        self.nodes = [[] for _ in range(n + 1)]
        self.costs = [[] for _ in range(n + 1)]
        self.comparisons = 0

    def dominated(self, depth, q, c, r):
        entries, costs = self.nodes[depth], self.costs[depth]
        for node in entries[:bisect_right(costs, c)]:
            self.comparisons += 1
            if node.quality >= q and node.rating >= r:
                return True
        return False

    def insert_created(self, node):
        # A query never mutates the archive. Only an already linked child enters.
        assert node.parent is not None
        assert node.parent.children[-1] is node
        entries, costs = self.nodes[node.depth], self.costs[node.depth]
        lo = bisect_left(costs, node.cost)
        kept = entries[:lo]
        for old in entries[lo:]:
            self.comparisons += 1
            if not (node.quality >= old.quality and node.rating >= old.rating):
                kept.append(old)
        costs = [old.cost for old in kept]
        idx = bisect_right(costs, node.cost)
        kept.insert(idx, node)
        costs.insert(idx, node.cost)
        self.nodes[node.depth], self.costs[node.depth] = kept, costs


def solve_pts(
    slot_candidates: Sequence[Sequence[Tuple[Item, float]]],
    budget: float,
    alpha: float,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
    expansion_budget: int = 500,
    candidate_limit: int = 50,
    use_reduction: bool = True,
    # ideal-point branch and bound against the frontier
    use_bound: bool = True,
    # extension-safe raw-cost dominance across the tree
    prefix_dominance: bool = True,
    timeout_flag: Optional[threading.Event] = None,
    trace: Optional[List[Tuple[float, 'PState']]] = None,
    # Optional validation trace: (created nodes, drained nodes, retired nodes).
    event_trace: Optional[List[Tuple[int, int, int]]] = None,
) -> PTSResult:
    if expansion_budget < 1:
        raise ValueError("expansion_budget must be >= 1")
    if candidate_limit < 0:
        raise ValueError("candidate_limit must be >= 0 (0 means uncapped)")
    if not slot_candidates or any(len(c) == 0 for c in slot_candidates):
        # no assignment exists. also n=0 would divide by zero in F'
        result = PTSResult([], 0, sum(len(c) for c in slot_candidates), 0, 0,
                           0.0, 0.0, exhausted=True)
        result.output_time_s = result.search_time_s = 0.0
        result.counts_after = result.counts_searched = [len(c) for c in slot_candidates]
        result.completed = True
        result.completion_scope = "fixed_candidate_tree"
        result.termination_reason = "completed"
        return result

    n = len(slot_candidates)
    max_budget = budget * (1.0 + alpha)

    t_red = time.perf_counter()
    n_before = sum(len(c) for c in slot_candidates)
    if use_reduction:
        red_c, kept = reduce_candidates(slot_candidates)
    else:
        red_c = [list(c) for c in slot_candidates]
        kept = [list(range(len(c))) for c in slot_candidates]
    n_after = sum(len(c) for c in red_c)
    reduction_s = time.perf_counter() - t_red

    # F' = l1*q/n - l2*max(0, c-B)/(alpha*B) + l3*r/(5n)
    def pof(q: float, cost: float, r: float) -> float:
        pen = (lambda2 * max(0.0, cost - budget) / (alpha * budget)) \
            if (alpha > 0 and budget > 0) else 0.0
        return lambda1 * q / n - pen + lambda3 * r / (5.0 * n)

    t0 = time.perf_counter()
    evals = 0
    frontier_evals = 0
    n_gains = 0
    rollout_evals = 0
    bound_evals = 0
    n_pruned_bound = 0
    n_pruned_prefix = 0
    root = _Node(0, [], 0.0, 0.0, 0.0)
    n_created = 1
    n_retired = 0
    n_drained = 0
    frontier: List[PState] = []

    # objective vectors of `frontier`, kept in parallel and sorted by overage
    f_q: List[float] = []
    f_ov: List[float] = []
    f_r: List[float] = []
    '''
    vectors already offered, accepted or not. the greedy completion is
    deterministic so the same bundle recurs hundreds of times, and exact ties
    dominate nothing - without this the archive scan goes quadratic in repeats.
    rejecting a seen vector is sound: accepted ones are represented, rejected
    ones were dominated by an incumbent that is only ever replaced by a dominator.
    '''
    seen: set = set()

    def add_to_frontier(s: PState) -> bool:
        '''returns True if s entered as a new non-dominated point.'''
        nonlocal frontier_evals
        ov_s = s.cost - budget
        if ov_s < 0.0:
            ov_s = 0.0
        key = (s.quality, ov_s, s.rating)
        if key in seen:
            return False
        seen.add(key)
        q_s, r_s = s.quality, s.rating
        hi = bisect_right(f_ov, ov_s)
        for i in range(hi):
            if f_q[i] >= q_s and f_r[i] >= r_s:
                frontier_evals += i + 1
                return False
        lo = bisect_left(f_ov, ov_s)
        frontier_evals += hi + (len(f_ov) - lo)
        drop = None
        for i in range(lo, len(f_ov)):
            if q_s >= f_q[i] and r_s >= f_r[i]:
                if drop is None:
                    drop = [i]
                else:
                    drop.append(i)
        if drop:
            ds = set(drop)
            keep = [i for i in range(len(f_ov)) if i not in ds]
            frontier[:] = [frontier[i] for i in keep]
            f_q[:] = [f_q[i] for i in keep]
            f_ov[:] = [f_ov[i] for i in keep]
            f_r[:] = [f_r[i] for i in keep]
            hi = bisect_right(f_ov, ov_s)
        frontier.insert(hi, s)
        f_q.insert(hi, q_s)
        f_ov.insert(hi, ov_s)
        f_r.insert(hi, r_s)
        if trace is not None:
            published = PState([kept[d][j] for d, j in enumerate(s.choice)],
                               s.quality, s.cost, s.rating)
            trace.append((time.perf_counter() - t_red, published))
        return True

    '''
    careful: the cap must rank by quality+rating only, independent of the prefix.
    if the top-k set depended on the parent's cost, two same-depth prefixes would
    keep different suffix actions - prefix dominance could then retire a node whose
    dominator cannot reproduce its completions, and `exhausted` would be wrong.
    '''
    capped: List[List[int]] = []
    for d in range(n):
        order = sorted(range(len(red_c[d])),
                       key=lambda j: -(lambda1 * red_c[d][j][1] / n
                                       + lambda3 * red_c[d][j][0].rating / (5.0 * n)))
        capped.append(order[:candidate_limit] if candidate_limit > 0 else order)

    def actions(node: _Node) -> List[int]:
        nonlocal evals
        acts = []
        '''
        test against the cheapest completion of the rest, not just the child's
        own price. affordable-now children can starve every later requirement.
        '''
        rest = min_c[node.depth + 1]
        cands = red_c[node.depth]
        for j in capped[node.depth]:
            evals += 1
            if node.cost + cands[j][0].price + rest > max_budget + 1e-9:
                continue
            acts.append(j)
        # ordering within the retained set may depend on the prefix - it only
        # decides which child gets expanded first, not which children exist
        acts.sort(key=lambda j: -pof(node.quality + cands[j][1],
                                     node.cost + cands[j][0].price,
                                     node.rating + cands[j][0].rating))
        return acts

    '''
    greedy suffixes memoized by (depth, cost). accumulated q and r drop out of
    the per-slot argmax and feasibility tests cost alone, so the suffix depends
    on these two only.
    '''
    suffix_memo: Dict[Tuple[int, float], Optional[Tuple[Tuple[int, ...], float, float, float]]] = {}

    def _greedy_suffix(depth: int, cost: float
                       ) -> Optional[Tuple[Tuple[int, ...], float, float, float]]:
        nonlocal rollout_evals
        ch: List[int] = []
        c = cost
        dq = dr = 0.0
        for d in range(depth, n):
            best_j, best_key = -1, None
            rollout_evals += len(capped[d])
            for j in capped[d]:
                item, qq = red_c[d][j]
                if c + item.price > max_budget + 1e-9:
                    continue
                key = pof(dq + qq, c + item.price, dr + item.rating)
                if best_key is None or key > best_key:
                    best_key, best_j = key, j
            if best_j < 0:
                return None
            it, qq = red_c[d][best_j]
            ch.append(best_j); dq += qq; c += it.price; dr += it.rating
        return (tuple(ch), dq, c - cost, dr)

    '''
    suffix bounds over the capped candidates, the only ones the search may use.
    bounds over the full reduced sets would admit prefixes the capped tree cannot
    complete, and that breaks the `exhausted` certificate.
    '''
    cap_c = [[red_c[d][j] for j in capped[d]] for d in range(n)]
    _sb = _SuffixBounds(cap_c)
    min_c, max_q, max_r = _sb.min_c, _sb.max_q, _sb.max_r

    def _feasible(depth: int, cost: float) -> bool:
        return cost + min_c[depth] <= max_budget + 1e-9

    '''
    ideal(s) = (Q + maxQ[d], max(0, C + minC[d] - B), R + maxR[d]).
    optimistic in every coordinate at once - if a complete solution weakly
    dominates it, no completion of s can contribute a new point.
    '''
    def _bound_pruned(depth: int, q: float, c: float, r: float) -> bool:
        nonlocal bound_evals
        if depth >= n or not f_ov:
            return False
        iq = q + max_q[depth]
        iov = c + min_c[depth] - budget
        if iov < 0.0:
            iov = 0.0
        ir = r + max_r[depth]
        # frontier is sorted by overage, only points at or under iov can dominate
        hi = bisect_right(f_ov, iov)
        for i in range(hi):
            if f_q[i] >= iq and f_r[i] >= ir:
                bound_evals += i + 1
                return True
        bound_evals += hi
        return False

    '''
    prefix dominance per depth on raw cost (q' >= q, c' <= c, r' >= r).
    key: only raw cost is extension-safe, a cheaper prefix leaves more budget for
    the suffix. also merges transpositions since an exact tie is weakly dominated.
    '''
    prefix_archive = _PrefixArchive(n)

    def _next_unpruned_extension(node):
        nonlocal n_pruned_bound, n_pruned_prefix, n_created
        while node.untried:
            cand = node.untried.popleft()
            item, q = red_c[node.depth][cand]
            cq, cc, cr = node.quality + q, node.cost + item.price, node.rating + item.rating
            cd = node.depth + 1
            if use_bound and _bound_pruned(cd, cq, cc, cr):
                n_pruned_bound += 1
                continue
            if prefix_dominance and prefix_archive.dominated(cd, cq, cc, cr):
                n_pruned_prefix += 1
                continue
            child = _Node(cd, node.choice + [cand], cq, cc, cr, parent=node)
            node.children.append(child)
            n_created += 1
            if prefix_dominance:
                prefix_archive.insert_created(child)
            return child
        return None

    def _cheapest_suffix(depth: int, cost: float
                         ) -> Optional[Tuple[Tuple[int, ...], float, float, float]]:
        '''min-cost completion. succeeds exactly when _feasible holds.'''
        nonlocal rollout_evals
        ch: List[int] = []
        c = cost
        dq = dr = 0.0
        for d in range(depth, n):
            best_j, best_p = -1, None
            rollout_evals += len(capped[d])
            for j in capped[d]:
                item, _qq = red_c[d][j]
                if best_p is None or item.price < best_p:
                    best_p, best_j = item.price, j
            if best_j < 0:
                return None
            it, qq = red_c[d][best_j]
            ch.append(best_j); dq += qq; c += it.price; dr += it.rating
        if c > max_budget + 1e-9:
            return None
        return (tuple(ch), dq, c - cost, dr)

    def rollout(node: _Node) -> Optional[PState]:
        key = (node.depth, node.cost)
        if key in suffix_memo:
            suf = suffix_memo[key]
        else:
            suf = _greedy_suffix(node.depth, node.cost)
            suffix_memo[key] = suf
        if suf is None:
            return None
        return complete_from_suffix(node, suf[0])

    def complete_from_suffix(node, choices):
        # Rebuild in requirement order, exactly as explicit tree extensions do.
        q, c, r = node.quality, node.cost, node.rating
        for d, j in enumerate(choices, node.depth):
            item, qq = red_c[d][j]
            q += qq
            c += item.price
            r += item.rating
        if c > max_budget + 1e-9:
            return None
        return PState(node.choice + list(choices), q, c, r)

    it_done = 0
    for iteration in range(1, expansion_budget + 1):
        if root.dead:
            break
        if timeout_flag is not None and timeout_flag.is_set():
            break
        it_done = iteration
        node = _select_depth_first(root, n, actions)
        leaf = node
        if node.depth < n:
            leaf = _next_unpruned_extension(node)
            if leaf is None:
                # Even with live children, draining this list is a one-time event.
                n_drained += 1
                n_retired += _propagate_exhaustion(node, n)
                if event_trace is not None:
                    event_trace.append((n_created, n_drained, n_retired))
                continue

        if leaf.depth == n:
            sol = PState(list(leaf.choice), leaf.quality, leaf.cost, leaf.rating)
            # terminal, nothing left to discover here. retire it
            leaf.evaluated = True
        else:
            sol = rollout(leaf)
            if sol is None and _feasible(leaf.depth, leaf.cost):
                '''
                greedy can starve itself even below budget. a completion still
                exists here, so take the cheapest one instead of killing the subtree.
                '''
                suf = _cheapest_suffix(leaf.depth, leaf.cost)
                if suf is not None:
                    sol = complete_from_suffix(leaf, suf[0])
        if sol is None:
            # really unreachable - no completion fits under (1 + alpha) B
            leaf.dead = True
            n_retired += 1
        elif add_to_frontier(sol):
            n_gains += 1

        n_retired += _propagate_exhaustion(leaf, n)
        if event_trace is not None:
            event_trace.append((n_created, n_drained, n_retired))

    t_output = time.perf_counter()
    front = _non_dominated_final(frontier, budget)
    for s in front:
        s.choice = [kept[i][s.choice[i]] for i in range(len(s.choice))]
    result = PTSResult(front, evals + rollout_evals, n_before, n_after, it_done,
                       time.perf_counter() - t0, reduction_s,
                       exhausted=root.dead,
                       n_action_evals=evals, n_rollout_evals=rollout_evals,
                       n_bound_evals=bound_evals + prefix_archive.comparisons,
                       n_frontier_comparisons=frontier_evals,
                       n_pruned_bound=n_pruned_bound,
                       n_pruned_prefix=n_pruned_prefix,
                       n_frontier_gains=n_gains,
                       n_created_nodes=n_created, n_retired_nodes=n_retired,
                       n_drained_nodes=n_drained)
    result.output_time_s = time.perf_counter() - t_output
    result.search_time_s = t_output - (t_red + reduction_s)
    result.counts_after = [len(c) for c in red_c]
    result.counts_searched = [len(c) for c in capped]
    result.completed = result.exhausted
    result.completion_scope = "fixed_candidate_tree"
    result.termination_reason = ("completed" if result.completed else "cancelled" if
        timeout_flag is not None and timeout_flag.is_set() else "iteration_limit")
    return result
