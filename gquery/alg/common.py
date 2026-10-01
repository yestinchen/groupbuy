'''
shared data structures, the weighted objective
F = l1*Q/n - l2*max(C-B,0)/(alpha*B) + l3*R/(5*n), and dominance helpers.
'''

import threading
import pandas as pd
import math
import numpy as np
from dataclasses import dataclass
import heapq
from typing import Any, List, Tuple, Optional, Dict
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import psutil
    import os
    PSUTIL_AVAILABLE = True
    RESOURCE_AVAILABLE = False
except ImportError:
    PSUTIL_AVAILABLE = False
    try:
        import resource
        RESOURCE_AVAILABLE = True
    except ImportError:
        RESOURCE_AVAILABLE = False

TOP_K_IMPROVEMENT_CAP = 50
TOP_K_REPORT_K_VALUES = [1, 5, 10, 20, 50]

'''
prefilter margin for the fp16 retrieval path. with fp32 accumulation the only
error is input rounding: for unit vectors |dot_fp16 - dot_fp32| <= ~1e-3
(fp16 has ~2^-11 relative rounding per entry; observed max 3.9e-4 on this
catalog). 0.01 is 10x the bound, survivors get an exact fp32 rescore anyway.
'''
FP16_MARGIN = 0.01


def build_top_50_improvements_and_time_to_top_k(
    improvement_heap: List[Tuple[float, float]],
) -> Tuple[List[Tuple[float, float]], Dict[int, Optional[float]]]:
    '''
    time_to_top_k[k] = min(t_1..t_k): earliest time we already had something as
    good as the final k-th best. monotone in k. None if fewer than k improvements.
    '''
    if not improvement_heap:
        return [], {k: None for k in TOP_K_REPORT_K_VALUES}
    sorted_by_value = sorted(improvement_heap, key=lambda x: -x[0])
    top_50 = [(t_s, value) for value, t_s in sorted_by_value[:TOP_K_IMPROVEMENT_CAP]]
    time_to_top_k: Dict[int, Optional[float]] = {}
    for k in TOP_K_REPORT_K_VALUES:
        if k <= len(sorted_by_value):
            times_of_top_k = [sorted_by_value[i][1] for i in range(k)]
            time_to_top_k[k] = min(times_of_top_k)
        else:
            time_to_top_k[k] = None
    return top_50, time_to_top_k


def compute_cost_ratio(current_cost: float, hard_budget: float, alpha: float) -> float:
    '''
    cost term of the objective: max(C - B, 0) / (alpha * B), in [0,1] under the
    relaxed cap. returns 0 for alpha <= 0 or B <= 0. keep all cost-term math in
    this one place.
    '''
    if hard_budget <= 0 or alpha <= 0:
        return 0.0
    overage = max(0.0, current_cost - hard_budget)
    return overage / (alpha * hard_budget)


def pareto_scalar_from_state(
    quality: float,
    current_cost: float,
    remaining_budget: float,
    rating: float,
    alpha: float,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    num_requirements: int,
) -> float:
    # F = l1*Q/n - l2*dC/(alpha*B) + l3*R/(5n)
    if num_requirements <= 0:
        return 0.0
    quality_norm = quality / num_requirements
    rating_norm = rating / (5.0 * num_requirements)
    total_budget = remaining_budget + current_cost
    hard_budget = total_budget / (1.0 + alpha) if (1.0 + alpha) > 0 else total_budget
    cost_ratio = compute_cost_ratio(current_cost, hard_budget, alpha)
    return lambda1 * quality_norm - lambda2 * cost_ratio + lambda3 * rating_norm


def compute_overage(current_cost: float, remaining_budget: float, alpha: float) -> float:
    expanded = current_cost + remaining_budget
    hard_budget = expanded / (1.0 + alpha) if alpha > -1.0 else expanded
    return max(0.0, current_cost - hard_budget)


def is_dominated(s1: Any, s2: Any) -> bool:
    # dominance on cost_overage - note states under budget all tie at 0 on the cost axis
    o1, o2 = s1.cost_overage, s2.cost_overage
    better_in_any = o2 < o1 or s2.quality > s1.quality or s2.rating > s1.rating
    not_worse_in_any = o2 <= o1 and s2.quality >= s1.quality and s2.rating >= s1.rating
    return better_in_any and not_worse_in_any


def is_epsilon_dominated(s1: Any, s2: Any, epsilon: float) -> bool:
    # cost condition: s2.cost_overage <= (1-epsilon) * s1.cost_overage
    if epsilon <= 0:
        return is_dominated(s1, s2)
    o1, o2 = s1.cost_overage, s2.cost_overage
    quality_condition = s2.quality >= (1 - epsilon) * s1.quality
    cost_condition = o2 <= (1 - epsilon) * o1
    rating_condition = s2.rating >= (1 - epsilon) * s1.rating
    all_conditions = quality_condition and cost_condition and rating_condition
    quality_strict = s2.quality > (1 - epsilon) * s1.quality
    cost_strict = o2 < (1 - epsilon) * o1
    rating_strict = s2.rating > (1 - epsilon) * s1.rating
    at_least_one_strict = quality_strict or cost_strict or rating_strict
    return all_conditions and at_least_one_strict


def filter_pareto_non_dominated(
    states: List[Any],
    timeout_flag: Optional[threading.Event] = None,
) -> List[Any]:
    if not states:
        return []
    non_dominated = []
    for i, s1 in enumerate(states):
        if timeout_flag is not None and (i & 0xFF) == 0 and timeout_flag.is_set():
            raise TimeoutError("Timeout exceeded during non-dominated filtering")
        dominated = False
        for j, s2 in enumerate(states):
            if timeout_flag is not None and (j & 0x3FF) == 0 and timeout_flag.is_set():
                raise TimeoutError("Timeout exceeded during non-dominated filtering")
            if i != j and is_dominated(s1, s2):
                dominated = True
                break
        if not dominated:
            non_dominated.append(s1)
    return non_dominated


def filter_pareto_epsilon_non_dominated(
    states: List[Any],
    epsilon: float,
    timeout_flag: Optional[threading.Event] = None,
) -> List[Any]:
    if not states:
        return []
    if epsilon <= 0:
        return filter_pareto_non_dominated(states, timeout_flag=timeout_flag)
    non_dominated = []
    for i, s1 in enumerate(states):
        if timeout_flag is not None and (i & 0xFF) == 0 and timeout_flag.is_set():
            raise TimeoutError("Timeout exceeded during epsilon non-dominated filtering")
        dominated = False
        for j, s2 in enumerate(states):
            if timeout_flag is not None and (j & 0x3FF) == 0 and timeout_flag.is_set():
                raise TimeoutError("Timeout exceeded during epsilon non-dominated filtering")
            if i != j and is_epsilon_dominated(s1, s2, epsilon):
                dominated = True
                break
        if not dominated:
            non_dominated.append(s1)
    return non_dominated


def compute_weighted_objective_components(state: 'BaseState', query: 'GroupQuery', config) -> Dict[str, float]:
    # F = l1*Q/n - l2*max(C-B,0)/(alpha*B) + l3*R/(5*n)
    expanded_budget = query.budget * (1.0 + config.alpha)
    new_cost = expanded_budget - state.remaining_budget
    cost_ratio = compute_cost_ratio(new_cost, query.budget, config.alpha)

    num_requirements = len(query.keywords) if query.keywords is not None else (len(query.query_desc_embeddings) if query.query_desc_embeddings is not None else 1)
    rating_ratio = state.rating / (5 * num_requirements) if num_requirements > 0 else 0.0
    quality_norm = state.quality / num_requirements
    function_value = config.lambda1 * quality_norm - config.lambda2 * cost_ratio + config.lambda3 * rating_ratio

    return {
        "quality": float(state.quality),
        "quality_norm": float(quality_norm),
        "cost_ratio": float(cost_ratio),
        "rating_ratio": float(rating_ratio),
        "function_value": float(function_value),
    }


@dataclass
class GroupQuery:
    id: str
    # one keyword list per requirement
    keywords: Optional[List[List[str]]] = None
    budget: float = 0.0
    # one embedding per requirement
    query_desc_embeddings: Optional[List[List[float]]] = None
    query_desc: Optional[List[str]] = None


@dataclass
class Item:
    id: str
    keywords: List[str]
    price: float
    rating: float = 0.0
    title_embedding: Optional[List[float]] = None
    title: Optional[str] = None


@dataclass
class BaseState:
    remaining_budget: float
    quality: float
    rating: float
    function_value: float


class ItemEmbeddingCache:
    '''keep all item embeddings on GPU, batch the similarity computation.'''

    def __init__(self, items: List[Item], device: Optional[str] = None,
                 fp16_prefilter: bool = True):
        self.items = items
        self.device = device

        if self.device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # gpus here are shared with LLM serving - take the one with
                # the most free memory instead of blindly cuda:0
                free = [torch.cuda.mem_get_info(i)[0]
                        for i in range(torch.cuda.device_count())]
                self.device = f"cuda:{int(np.argmax(free))}"
            else:
                self.device = 'cpu'

        self.fp16_prefilter = False
        # embedding index -> item index
        self.item_indices = []
        # item index -> embedding index
        self.embedding_indices = {}
        embeddings_list = []

        for item_idx, item in enumerate(items):
            if item.title_embedding is not None:
                embedding_idx = len(embeddings_list)
                self.item_indices.append(item_idx)
                self.embedding_indices[item_idx] = embedding_idx
                embeddings_list.append(item.title_embedding)

        if embeddings_list:
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            # prices aligned with embedding rows. float64 so the price mask
            # matches an exact python-float comparison
            prices = np.array([items[i].price for i in self.item_indices],
                              dtype=np.float64)
            self.prices_np = prices
            if TORCH_AVAILABLE:
                self.embeddings_tensor = torch.from_numpy(embeddings_array).to(self.device)
                # normalize once, cosine becomes a dot product
                self.embeddings_tensor = torch.nn.functional.normalize(self.embeddings_tensor, p=2, dim=1)
                self.prices_tensor = torch.from_numpy(prices).to(self.device)
                '''fp16 copy for the prefilter matmul. force fp32 accumulation
                so the FP16_MARGIN bound above actually holds.'''
                self.fp16_prefilter = fp16_prefilter and str(self.device).startswith('cuda')
                if self.fp16_prefilter:
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
                    self.embeddings_fp16 = self.embeddings_tensor.half()
            else:
                self.embeddings_tensor = None
                arr = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                self.embeddings_array = arr
        else:
            self.embeddings_tensor = None
            self.embeddings_array = None
            self.prices_np = None
            self.item_indices = []
            self.embedding_indices = {}

        self.has_embeddings = len(embeddings_list) > 0
    
    def retrieve_candidates_gpu_batch(self, requirement_embeddings,
                                      beta_val: float, max_cost: float) -> List[List[Tuple[Item, float]]]:
        '''threshold retrieval for all requirements of one query at once:
        every item with cos >= beta and price <= max_cost. one matmul over the
        whole catalog and one device sync - no per-item python loop, and the
        masks are applied on the cpu from a single transfer.'''
        if not self.has_embeddings or requirement_embeddings is None:
            return []
        n = len(requirement_embeddings)
        if n == 0:
            return []

        req = np.asarray(requirement_embeddings, dtype=np.float32)
        if TORCH_AVAILABLE:
            req_t = torch.from_numpy(req).to(self.device)
            req_t = torch.nn.functional.normalize(req_t, p=2, dim=1)
            price_ok = (self.prices_tensor <= max_cost).unsqueeze(0)
            out = [[] for _ in range(n)]
            if self.fp16_prefilter:
                '''fp16 matmul with a slack threshold cannot lose a true
                candidate (see FP16_MARGIN), then the few survivors get the
                exact fp32 dot product and the real beta cut. same result as
                the pure fp32 scan at half the matmul cost / memory traffic.'''
                sims16 = torch.mm(req_t.half(), self.embeddings_fp16.t())
                mask = (sims16 >= beta_val - FP16_MARGIN) & price_ok
                nz = torch.nonzero(mask, as_tuple=False)
                if nz.numel() == 0:
                    return out
                r_idx = nz[:, 0]
                c_idx = nz[:, 1]
                vals_t = (req_t[r_idx] * self.embeddings_tensor[c_idx]).sum(dim=1)
                keep = vals_t >= beta_val
                nz = nz[keep]
                vals_t = vals_t[keep]
                if nz.numel() == 0:
                    return out
                rows_cols = nz.cpu().numpy()
                vals = vals_t.cpu().numpy()
            else:
                sims = torch.mm(req_t, self.embeddings_tensor.t())
                # mask on the gpu, transfer only the survivors
                mask = (sims >= beta_val) & price_ok
                nz = torch.nonzero(mask, as_tuple=False)
                if nz.numel() == 0:
                    return out
                rows_cols = nz.cpu().numpy()
                vals = sims[mask].cpu().numpy()
            # nonzero is row-major, so per-requirement order stays ascending
            for (r, c), s in zip(rows_cols, vals):
                out[r].append((self.items[self.item_indices[c]], float(s)))
            return out

        req = req / np.linalg.norm(req, axis=1, keepdims=True)
        sims = req @ self.embeddings_array.T
        price_ok = self.prices_np <= max_cost
        out = []
        for row in sims:
            idx = np.nonzero((row >= beta_val) & price_ok)[0]
            out.append([(self.items[self.item_indices[i]], float(row[i]))
                        for i in idx])
        return out

    def retrieve_candidates_gpu(self, requirement_embedding: List[float],
                                beta_val: float, max_cost: float) -> List[Tuple[Item, float]]:
        if not self.has_embeddings or requirement_embedding is None:
            return []
        return self.retrieve_candidates_gpu_batch(
            [requirement_embedding], beta_val, max_cost)[0]


def query_from_row(row: pd.Series) -> GroupQuery:
    query_desc_embeddings = None
    if 'query_desc_embedding' in row:
        query_desc_embeddings = row['query_desc_embedding']

    query_desc = None
    if 'query_desc' in row:
        query_desc = row['query_desc']

    keywords = None
    if 'keywords' in row:
        keywords = row['keywords']
    elif query_desc_embeddings is not None:
        keywords = [[] for _ in query_desc_embeddings]
    elif query_desc is not None:
        keywords = [[] for _ in query_desc]

    return GroupQuery(
        id=str(row['id']),
        keywords=keywords,
        budget=float(row['budget']),
        query_desc_embeddings=query_desc_embeddings,
        query_desc=query_desc
    )


def item_from_row(row: pd.Series) -> Item:
    keywords = row['title_feats'] if 'title_feats' in row else []
    rating = float(row.get('average_rating', 0.0))
    title_embedding = None
    if 'title_embedding' in row:
        title_embedding = row['title_embedding']

    title = None
    if 'title' in row:
        title = str(row['title'])

    return Item(
        id=str(row['id']),
        keywords=keywords,
        price=float(row['price']),
        rating=rating,
        title_embedding=title_embedding,
        title=title
    )


def calculate_quality_score(item: Item, requirement: List[str], beta: float) -> Tuple[bool, float]:
    if len(requirement) == 0:
        return True, 1.0

    matching_keywords = len(set(item.keywords).intersection(set(requirement)))
    if matching_keywords == 0:
        return False, 0.0
    min_required = math.ceil(beta * len(requirement))
    matches = matching_keywords >= min_required
    quality_score = matching_keywords / len(requirement)
    
    return matches, quality_score


def calculate_quality_score_embedding(item: Item, requirement_embedding: List[float], beta: float, similarity_threshold: float = 0.7) -> Tuple[bool, float]:
    if item.title_embedding is None or requirement_embedding is None:
        return False, 0.0

    item_emb = np.array(item.title_embedding).reshape(1, -1)
    req_emb = np.array(requirement_embedding).reshape(1, -1)

    similarity = cosine_similarity(item_emb, req_emb)[0][0]

    matches = similarity >= beta
    quality_score = similarity

    return matches, quality_score


def create_embedding_cache(items: List[Item], device: Optional[str] = None,
                           fp16_prefilter: bool = True) -> Optional[ItemEmbeddingCache]:
    has_any_embeddings = any(item.title_embedding is not None for item in items)
    if not has_any_embeddings:
        return None

    return ItemEmbeddingCache(items, device=device, fp16_prefilter=fp16_prefilter)


def retrieve_candidates_with_quality(items: List[Item], requirement: List[str], 
                                     beta_val: float, max_cost: float,
                                     use_embeddings: bool = False,
                                     requirement_embedding: Optional[List[float]] = None,
                                     embedding_cache: Optional['ItemEmbeddingCache'] = None) -> List[Tuple[Item, float]]:
    if use_embeddings:
        if embedding_cache is not None:
            return embedding_cache.retrieve_candidates_gpu(requirement_embedding, beta_val, max_cost)

        candidates = [item for item in items if item.price <= max_cost]
        if not candidates:
            return []

        if requirement_embedding is None:
            return []

        valid_candidates = []
        item_embeddings = []
        for item in candidates:
            if item.title_embedding is not None:
                valid_candidates.append(item)
                item_embeddings.append(item.title_embedding)

        if not valid_candidates:
            return []

        req_emb = np.array(requirement_embedding).reshape(1, -1)
        item_emb_matrix = np.array(item_embeddings)

        similarities = cosine_similarity(req_emb, item_emb_matrix)[0]

        valid_items = [
            (item, float(quality))
            for item, quality in zip(valid_candidates, similarities)
            if quality >= beta_val
        ]

        return valid_items
    else:
        candidates = [item for item in items if item.price <= max_cost]
        if not candidates:
            return []

        valid_items = []
        for item in candidates:
            matches, quality = calculate_quality_score(item, requirement, beta_val)
            # skip items that do not match the requirement
            if matches:
                valid_items.append((item, quality))

        return valid_items


def update_function_value(state: BaseState, query: GroupQuery, config) -> None:
    if config.objective == 'quality':
        state.function_value = state.quality
    elif config.objective == 'weighted':
        state.function_value = compute_weighted_objective_components(state, query, config)["function_value"]
    else:
        raise ValueError(f"Invalid objective: {config.objective}")


def get_current_memory_usage_gb() -> float:
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024
    elif RESOURCE_AVAILABLE:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KB on linux
        return usage.ru_maxrss / 1024 / 1024
    else:
        return 0.0


def get_current_memory_usage_mb() -> float:
    return get_current_memory_usage_gb() * 1024


def check_runtime_memory_threshold(threshold_gb: Optional[float], initial_memory_gb: float) -> bool:
    if threshold_gb is None or threshold_gb <= 0:
        return False

    current_memory_gb = get_current_memory_usage_gb()
    runtime_memory_gb = current_memory_gb - initial_memory_gb
    return runtime_memory_gb > threshold_gb


def check_memory_threshold(threshold_mb: Optional[float]) -> bool:
    if threshold_mb is None or threshold_mb <= 0:
        return False

    current_memory = get_current_memory_usage_mb()
    return current_memory > threshold_mb


# save results
def save_as_paramed_file(pd_frame, exp_name, method_name, dataset_dir, num_queries, params):
    base_dir = Path('data_results')
    target_folder = base_dir / exp_name / method_name / f'{dataset_dir}_{num_queries}'
    sorted_params = sorted(params.items())
    params_str = "_".join([f"{k}_{str(v)}" for k, v in sorted_params])
    target_folder.mkdir(parents=True, exist_ok=True)
    target_file = target_folder / f"results_{params_str}.csv"
    pd_frame.to_csv(target_file, index=False)
