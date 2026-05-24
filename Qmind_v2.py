"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            Q M I N D   v2.0                                  ║
║          Proto-AGI Symbolic-Probabilistic Cognitive Architecture             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  A self-evolving, quantum-inspired reasoning substrate combining:            ║
║  · Hybrid memory (working / episodic / semantic / long-term / cold)          ║
║  · Multi-layer inference (deductive / inductive / abductive / causal /       ║
║    analogical / counterfactual / temporal / meta-cognitive)                  ║
║  · True quantum-inspired amplitude mechanics with interference               ║
║  · Contextual semantic collapse & latent meaning fields                      ║
║  · Goal-driven cognition with dynamic strategy selection                     ║
║  · Emergent concept synthesis                                                ║
║  · Contradiction management & uncertainty propagation                        ║
║  · Autonomous curiosity engine                                               ║
║  · Recursive thought simulation                                              ║
║  · Neural-symbolic hybridization via semantic embeddings                     ║
║  · Self-reflective meta-cognition & reasoning trace diagnostics              ║
║                                                                              ║
║  Fully local/offline · Deterministic by default · Modular · Explainable      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  License: MIT                                                                ║
║  Dependencies: networkx numpy scipy scikit-learn qutip                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import re
import time
import uuid
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from qutip import Qobj
from sklearn.metrics.pairwise import cosine_similarity

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    # Persistence
    "graph_file":               "qmind_graph.graphml",
    "episodic_file":            "qmind_episodic.json",
    "unknown_log":              "qmind_unknown.txt",

    # Memory capacities
    "working_memory_capacity":  20,
    "episodic_memory_capacity": 500,
    "cold_cache_capacity":      2000,

    # Reasoning defaults
    "default_depth":            4,
    "complexity_budget":        400,
    "min_amplitude":            0.003,
    "noise_scale":              0.004,

    # Learning
    "base_learning_rate":       0.08,
    "reinforcement_factor":     0.12,

    # Decay
    "decay_factor":             0.92,
    "decay_threshold_s":        600,

    # Emergence
    "emergence_min_shared":     2,      # min shared successors to trigger synthesis
    "emergence_min_conf":       0.55,

    # Curiosity
    "curiosity_gap_threshold":  0.35,   # confidence below this triggers curiosity
    "curiosity_contradiction":  0.4,    # contradiction pressure threshold

    # Embedding dimensionality (lightweight random projection)
    "embed_dim":                64,

    # Semantic collapse
    "collapse_similarity_th":   0.72,
}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ENUMERATIONS
# ══════════════════════════════════════════════════════════════════════════════

class CognitiveGoal(Enum):
    EXPLORE       = auto()   # favour novelty, wide search
    PRECISION     = auto()   # favour high-confidence short paths
    CREATIVE      = auto()   # favour analogical/emergent reasoning
    SAFETY        = auto()   # prune unsafe conclusions
    DISCOVERY     = auto()   # seek unknown regions
    RESOLVE       = auto()   # focus on contradictions
    ABSTRACT      = auto()   # synthesise new concepts

class InferenceMode(Enum):
    DEDUCTIVE     = auto()
    INDUCTIVE     = auto()
    ABDUCTIVE     = auto()
    ANALOGICAL    = auto()
    CAUSAL        = auto()
    TEMPORAL      = auto()
    COUNTERFACTUAL= auto()
    META          = auto()
    TRANSITIVE    = auto()
    HYBRID        = auto()

class MemoryTier(Enum):
    WORKING   = auto()
    EPISODIC  = auto()
    SEMANTIC  = auto()
    LONGTERM  = auto()
    COLD      = auto()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class MemoryTrace:
    """A single episodic memory record."""
    uid:            str
    concept:        str
    query:          str
    conclusions:    List[Tuple[str, float]]
    confidence:     float
    timestamp:      float
    domain:         str
    goal:           str
    success:        bool
    salience:       float = 1.0
    replay_count:   int   = 0

    def decay_salience(self, factor: float = 0.98) -> None:
        self.salience = max(0.01, self.salience * factor)


@dataclasses.dataclass
class ReasoningTrace:
    """Full diagnostic output of one reasoning session."""
    query:              str
    concept:            str
    goal:               CognitiveGoal
    inference_modes:    List[str]
    paths_explored:     int
    paths_used:         int
    inferred_links:     int
    conclusions:        List[Tuple[str, str, float]]   # (concept, safe, prob)
    competing_hyps:     List[Tuple[str, float]]
    contradictions:     List[Tuple[str, str, float]]
    uncertainty:        float
    reasoning_depth:    int
    meta_confidence:    float                          # confidence-about-confidence
    causal_chains:      List[List[str]]
    emergent_concepts:  List[str]
    curiosity_signals:  List[str]
    elapsed_s:          float
    status:             str

    def summary(self) -> str:
        lines = [
            f"  Query          : {self.query}",
            f"  Concept        : {self.concept}",
            f"  Goal           : {self.goal.name}",
            f"  Status         : {self.status}",
            f"  Modes          : {', '.join(self.inference_modes)}",
            f"  Paths (exp/use): {self.paths_explored}/{self.paths_used}",
            f"  Inferred links : {self.inferred_links}",
            f"  Uncertainty    : {self.uncertainty:.3f}",
            f"  Meta-confidence: {self.meta_confidence:.3f}",
            f"  Elapsed        : {self.elapsed_s:.3f}s",
        ]
        if self.conclusions:
            lines.append("  Top conclusions:")
            for c, s, p in self.conclusions[:4]:
                lines.append(f"    [{p:.3f}] {c}" + (" [SAFE-FILTERED]" if s != c else ""))
        if self.contradictions:
            lines.append("  Contradictions:")
            for a, b, p in self.contradictions[:3]:
                lines.append(f"    {a} ↔ {b}  pressure={p:.2f}")
        if self.curiosity_signals:
            lines.append("  Curiosity signals:")
            for sig in self.curiosity_signals[:3]:
                lines.append(f"    ? {sig}")
        if self.emergent_concepts:
            lines.append("  Emergent concepts synthesised:")
            for ec in self.emergent_concepts:
                lines.append(f"    ★ {ec}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HYBRID MEMORY ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class WorkingMemory:
    """
    Bounded LRU active concept store (the 'cognitive spotlight').

    Concepts are scored by: recency × access_frequency × salience.
    The lowest-scoring concept is evicted when capacity is reached.
    """

    def __init__(self, capacity: int = CFG["working_memory_capacity"]) -> None:
        self.capacity  = capacity
        self._store:   Dict[str, Dict] = {}          # concept → metadata
        self._cold:    collections.OrderedDict = collections.OrderedDict()

    def activate(self, concept: str, salience: float = 1.0) -> None:
        now = time.time()
        if concept in self._store:
            m = self._store[concept]
            m["last_access"]   = now
            m["access_count"] += 1
            m["salience"]      = min(1.0, m["salience"] + 0.05)
        else:
            if len(self._store) >= self.capacity:
                self._evict_one()
            self._store[concept] = {
                "first_seen":   now,
                "last_access":  now,
                "access_count": 1,
                "salience":     float(np.clip(salience, 0.01, 1.0)),
            }
            self._cold.pop(concept, None)

    def _evict_one(self) -> None:
        """Evict the concept with the lowest composite score."""
        now  = time.time()
        worst, worst_score = None, float("inf")
        for c, m in self._store.items():
            age   = now - m["last_access"]
            score = m["salience"] * m["access_count"] / (1.0 + math.log1p(age))
            if score < worst_score:
                worst_score, worst = score, c
        if worst:
            self._cold[worst] = self._store.pop(worst)
            if len(self._cold) > CFG["cold_cache_capacity"]:
                self._cold.popitem(last=False)

    def get_active(self, limit: Optional[int] = None) -> List[str]:
        keys = sorted(
            self._store.keys(),
            key=lambda c: self._store[c]["last_access"],
            reverse=True,
        )
        return keys[:limit] if limit else keys

    def get_salience(self, concept: str) -> float:
        if concept in self._store:
            return self._store[concept]["salience"]
        return 0.0

    def is_cold(self, concept: str) -> bool:
        return concept in self._cold

    def clear(self) -> None:
        self._store.clear()
        self._cold.clear()

    def apply_temporal_decay(self, factor: float = 0.97) -> None:
        """Tick-based salience decay for all active concepts."""
        for m in self._store.values():
            m["salience"] = max(0.01, m["salience"] * factor)


class EpisodicMemory:
    """
    Records of past reasoning sessions for retrospective learning.

    Supports:
    - consolidation (high-salience traces → semantic facts)
    - replay (re-run traces to reinforce graph edges)
    - abstraction compression (detect repeated patterns)
    """

    def __init__(self, capacity: int = CFG["episodic_memory_capacity"]) -> None:
        self.capacity = capacity
        self._traces: List[MemoryTrace] = []

    def record(self, trace: MemoryTrace) -> None:
        self._traces.append(trace)
        if len(self._traces) > self.capacity:
            # Evict lowest-salience trace
            self._traces.sort(key=lambda t: t.salience, reverse=True)
            self._traces = self._traces[:self.capacity]

    def get_recent(self, n: int = 10) -> List[MemoryTrace]:
        return sorted(self._traces, key=lambda t: t.timestamp, reverse=True)[:n]

    def get_by_concept(self, concept: str) -> List[MemoryTrace]:
        return [t for t in self._traces if t.concept == concept]

    def apply_salience_decay(self) -> None:
        for t in self._traces:
            t.decay_salience()

    def consolidate_patterns(self) -> List[Tuple[str, str, float]]:
        """
        Find concept-conclusion pairs that appear repeatedly with high salience.
        Returns list of (concept, conclusion, avg_confidence) to reinforce.
        """
        counts: Dict[Tuple[str, str], List[float]] = collections.defaultdict(list)
        for t in self._traces:
            if t.success and t.salience > 0.5:
                for conc, conf in t.conclusions[:1]:
                    counts[(t.concept, conc)].append(conf)
        result = []
        for (src, tgt), confs in counts.items():
            if len(confs) >= 2:
                result.append((src, tgt, float(np.mean(confs))))
        return result

    def to_dict(self) -> List[Dict]:
        return [dataclasses.asdict(t) for t in self._traces]

    def from_dict(self, data: List[Dict]) -> None:
        self._traces = [MemoryTrace(**d) for d in data]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SEMANTIC EMBEDDING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class SemanticEmbedder:
    """
    Lightweight neural-symbolic bridge using random projection embeddings.

    Each concept name gets a deterministic, reproducible embedding vector
    derived from character-level hashing + random projection.  This gives:
    - semantic proximity by string similarity (rough but consistent)
    - a low-dimensional latent space for clustering and analogy
    - zero external model dependencies

    For production use, swap `_hash_embed` with a proper sentence-transformer.
    """

    def __init__(self, dim: int = CFG["embed_dim"]) -> None:
        self.dim    = dim
        self._cache: Dict[str, np.ndarray] = {}
        # Reproducible random projection matrix seeded from dim
        rng = np.random.default_rng(seed=42 + dim)
        self._proj  = rng.standard_normal((256, dim)).astype(np.float32)

    def embed(self, concept: str) -> np.ndarray:
        if concept in self._cache:
            return self._cache[concept]
        vec = self._hash_embed(concept)
        self._cache[concept] = vec
        return vec

    def _hash_embed(self, text: str) -> np.ndarray:
        """Character-level frequency histogram projected into embed_dim space."""
        freq = np.zeros(256, dtype=np.float32)
        for ch in text.lower():
            freq[ord(ch) % 256] += 1.0
        # Also encode bigrams
        for i in range(len(text) - 1):
            bigram = (ord(text[i]) * 31 + ord(text[i + 1])) % 256
            freq[bigram] += 0.5
        norm = np.linalg.norm(freq)
        if norm > 1e-9:
            freq /= norm
        vec = freq @ self._proj
        n = np.linalg.norm(vec)
        return vec / n if n > 1e-9 else vec

    def similarity(self, a: str, b: str) -> float:
        ea, eb = self.embed(a), self.embed(b)
        return float(cosine_similarity(ea.reshape(1, -1), eb.reshape(1, -1))[0, 0])

    def nearest(self, concept: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        ec = self.embed(concept)
        scores = [(c, self.similarity(concept, c)) for c in candidates if c != concept]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ATTENTION & SALIENCE SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class AttentionSystem:
    """
    Dynamic cognitive attention that scores and ranks concepts for focus.

    Salience components:
    - novelty:              inverse of how often the concept has been accessed
    - goal_relevance:       semantic similarity to current goal target
    - contradiction_density: how many conflicting edges the concept has
    - temporal_recency:     recency of last access
    - predictive_importance: out-degree in graph (hub-like concepts matter more)
    """

    def __init__(self, embedder: SemanticEmbedder) -> None:
        self._embedder = embedder

    def score(
        self,
        concept: str,
        graph: nx.DiGraph,
        working_mem: WorkingMemory,
        goal: CognitiveGoal,
        goal_concept: Optional[str] = None,
        contradiction_map: Optional[Dict[str, float]] = None,
    ) -> float:
        now = time.time()

        # 1. Novelty (inverse frequency)
        access = working_mem._store.get(concept, {}).get("access_count", 0)
        novelty = 1.0 / (1.0 + math.log1p(access))

        # 2. Goal relevance
        goal_rel = 0.5
        if goal_concept and concept != goal_concept:
            goal_rel = self._embedder.similarity(concept, goal_concept)

        # 3. Contradiction density
        contrad = (contradiction_map or {}).get(concept, 0.0)

        # 4. Temporal recency
        last = working_mem._store.get(concept, {}).get("last_access", 0)
        age_sec = now - last
        recency = math.exp(-age_sec / 3600.0)   # half-life ~1 hour

        # 5. Hub importance (out-degree)
        hub = math.log1p(graph.out_degree(concept)) / (1.0 + math.log1p(graph.number_of_nodes()))

        # Goal-specific weighting
        weights = {
            CognitiveGoal.EXPLORE:    (0.35, 0.15, 0.10, 0.20, 0.20),
            CognitiveGoal.PRECISION:  (0.05, 0.40, 0.05, 0.30, 0.20),
            CognitiveGoal.CREATIVE:   (0.30, 0.20, 0.15, 0.15, 0.20),
            CognitiveGoal.SAFETY:     (0.05, 0.30, 0.30, 0.20, 0.15),
            CognitiveGoal.DISCOVERY:  (0.40, 0.20, 0.10, 0.10, 0.20),
            CognitiveGoal.RESOLVE:    (0.10, 0.10, 0.50, 0.10, 0.20),
            CognitiveGoal.ABSTRACT:   (0.20, 0.20, 0.05, 0.20, 0.35),
        }.get(goal, (0.20, 0.20, 0.20, 0.20, 0.20))

        score = (
            weights[0] * novelty +
            weights[1] * goal_rel +
            weights[2] * contrad +
            weights[3] * recency +
            weights[4] * hub
        )
        return float(np.clip(score, 0.0, 1.0))

    def rank(
        self,
        concepts: List[str],
        graph: nx.DiGraph,
        working_mem: WorkingMemory,
        goal: CognitiveGoal,
        goal_concept: Optional[str] = None,
        contradiction_map: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        scored = [
            (c, self.score(c, graph, working_mem, goal, goal_concept, contradiction_map))
            for c in concepts
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CONTRADICTION MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class ContradictionManager:
    """
    Maintains probabilistic coexistence of conflicting beliefs.

    Rather than rejecting contradictions, we:
    - compute contradiction pressure (CP) for each node
    - allow competing truths to coexist with a probability split
    - trigger reconciliation attempts when CP exceeds threshold
    - isolate high-CP regions for targeted reasoning
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self._graph = graph

    def compute_pressure(self, concept: str) -> float:
        """
        Contradiction pressure = sum of (confidence × weight) for all
        'opposite_of' or 'contradicts' edges touching this concept.
        """
        g = self._graph
        pressure = 0.0
        for u, v, d in g.out_edges(concept, data=True):
            if d.get("relation") in ("opposite_of", "contradicts"):
                pressure += float(d.get("confidence", 0)) * float(d.get("weight", 0))
        for u, v, d in g.in_edges(concept, data=True):
            if d.get("relation") in ("opposite_of", "contradicts"):
                pressure += float(d.get("confidence", 0)) * float(d.get("weight", 0))
        return float(np.clip(pressure, 0.0, 1.0))

    def all_pressures(self) -> Dict[str, float]:
        return {n: self.compute_pressure(n) for n in self._graph.nodes}

    def find_contradicted_pairs(self) -> List[Tuple[str, str, float]]:
        """Return (A, B, pressure) for all directly contradicted pairs."""
        result = []
        g = self._graph
        for u, v, d in g.edges(data=True):
            if d.get("relation") in ("opposite_of", "contradicts"):
                pressure = float(d.get("confidence", 0)) * float(d.get("weight", 0))
                result.append((u, v, pressure))
        result.sort(key=lambda x: x[2], reverse=True)
        return result

    def reconcile(self, a: str, b: str) -> Optional[str]:
        """
        Attempt to reconcile two contradicting concepts.
        Strategy: find a common ancestor that may subsume both.
        Returns the reconciling concept name, or None.
        """
        g = self._graph
        try:
            ug = g.to_undirected()
            ancestors_a = set(nx.single_source_shortest_path_length(ug, a, cutoff=3).keys())
            ancestors_b = set(nx.single_source_shortest_path_length(ug, b, cutoff=3).keys())
            common = (ancestors_a & ancestors_b) - {a, b}
            if common:
                # Prefer the one with highest confidence
                best = max(common, key=lambda n: g.nodes[n].get("confidence", 0))
                return best
        except Exception:
            pass
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TEMPORAL WORLD MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TemporalWorldModel:
    """
    Simulates causal chains and temporal state transitions.

    The model maintains:
    - event sequences (ordered concept activations)
    - causal propagation rules (if X then Y becomes p% more likely)
    - hypothetical branch simulation (branching futures)
    - state snapshots for counterfactual comparison
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self._graph     = graph
        self._timeline: List[Tuple[float, str, float]] = []  # (timestamp, concept, salience)
        self._snapshots: Dict[str, Dict[str, float]]  = {}   # snapshot_id → {concept: weight}

    def log_event(self, concept: str, salience: float = 1.0) -> None:
        self._timeline.append((time.time(), concept, salience))
        # Keep only the last 1000 events
        if len(self._timeline) > 1000:
            self._timeline = self._timeline[-1000:]

    def get_recent_sequence(self, n: int = 10) -> List[str]:
        return [c for _, c, _ in self._timeline[-n:]]

    def snapshot(self) -> str:
        """Take a snapshot of current edge weights. Returns snapshot_id."""
        snap_id = str(uuid.uuid4())[:8]
        self._snapshots[snap_id] = {
            f"{u}|{v}": float(d.get("weight", 0))
            for u, v, d in self._graph.edges(data=True)
        }
        return snap_id

    def simulate_propagation(
        self,
        trigger: str,
        steps: int = 3,
        decay: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """
        IF *trigger* occurs THEN propagate its activation forward.
        Returns list of (affected_concept, activation_strength).
        """
        if trigger not in self._graph:
            return []
        frontier  = {trigger: 1.0}
        activated = {}
        for _ in range(steps):
            next_frontier: Dict[str, float] = {}
            for concept, strength in frontier.items():
                for _, successor, d in self._graph.out_edges(concept, data=True):
                    w    = float(d.get("weight", 0.5)) * float(d.get("confidence", 0.5))
                    prop = strength * w * decay
                    if prop > 0.01:
                        next_frontier[successor] = max(
                            next_frontier.get(successor, 0.0), prop
                        )
                        activated[successor] = max(
                            activated.get(successor, 0.0), prop
                        )
            frontier = next_frontier
            if not frontier:
                break
        result = sorted(activated.items(), key=lambda x: x[1], reverse=True)
        return result

    def counterfactual(
        self, concept: str, negate: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Simulate 'what if *concept* were removed (or inverted)'.
        Returns concepts whose reachability changes most.
        """
        g = self._graph
        if concept not in g:
            return []
        # Baseline reachability
        base = dict(nx.single_source_shortest_path_length(g, concept, cutoff=4))
        # Remove node temporarily
        g_cf = nx.DiGraph(g)
        g_cf.remove_node(concept)
        changed = []
        for node in base:
            if node == concept:
                continue
            try:
                new_dist = nx.shortest_path_length(g_cf, concept if concept in g_cf else list(g_cf.nodes)[0], node)
            except (nx.NetworkXNoPath, nx.NodeNotFound, IndexError):
                new_dist = float("inf")
            old_dist = base[node]
            delta    = abs(new_dist - old_dist) if new_dist != float("inf") else 1.0
            if delta > 0:
                changed.append((node, float(np.clip(delta / 4.0, 0.0, 1.0))))
        changed.sort(key=lambda x: x[1], reverse=True)
        return changed[:10]

    def predict_future_states(
        self, start: str, horizon: int = 3
    ) -> List[Tuple[str, float, int]]:
        """
        Predict likely future concept activations from *start* within *horizon* hops.
        Returns (concept, probability, hop_distance).
        """
        result = []
        frontier: Dict[str, Tuple[float, int]] = {start: (1.0, 0)}
        visited: Set[str] = {start}
        for hop in range(1, horizon + 1):
            next_f: Dict[str, Tuple[float, int]] = {}
            for c, (p, _) in frontier.items():
                for _, succ, d in self._graph.out_edges(c, data=True):
                    if succ in visited:
                        continue
                    w    = float(d.get("weight", 0.5)) * float(d.get("confidence", 0.5))
                    newp = p * w
                    if newp > 0.01:
                        next_f[succ] = (max(next_f.get(succ, (0, hop))[0], newp), hop)
                        visited.add(succ)
            for c, (p, h) in next_f.items():
                result.append((c, p, h))
            frontier = next_f
        result.sort(key=lambda x: x[1], reverse=True)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — CURIOSITY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class CuriosityEngine:
    """
    Autonomously identifies knowledge gaps, weak regions, and contradictions,
    then generates self-directed questions and exploratory directions.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        contradiction_mgr: ContradictionManager,
        embedder: SemanticEmbedder,
    ) -> None:
        self._graph  = graph
        self._cm     = contradiction_mgr
        self._embed  = embedder

    def identify_gaps(self) -> List[str]:
        """Concepts with very low confidence or out-degree."""
        gaps = []
        for n, d in self._graph.nodes(data=True):
            conf    = float(d.get("confidence", 0.5))
            out_deg = self._graph.out_degree(n)
            if conf < CFG["curiosity_gap_threshold"] or out_deg == 0:
                gaps.append(n)
        return gaps

    def identify_weak_regions(self) -> List[Tuple[str, float]]:
        """Clusters of concepts connected only by low-weight edges."""
        weak = []
        for u, v, d in self._graph.edges(data=True):
            w = float(d.get("weight", 1.0)) * float(d.get("confidence", 1.0))
            if w < 0.25:
                weak.append((f"{u} → {v}", w))
        weak.sort(key=lambda x: x[1])
        return weak[:10]

    def generate_questions(self, focus: str) -> List[str]:
        """Generate exploratory questions about a concept."""
        questions = [
            f"What causes {focus}?",
            f"What does {focus} lead to?",
            f"What is {focus} similar to?",
            f"What contradicts {focus}?",
            f"What is a more abstract form of {focus}?",
        ]
        # Semantic neighbours add specificity
        neighbours = self._embed.nearest(focus, list(self._graph.nodes), top_k=3)
        for neighbour, sim in neighbours:
            if sim > 0.5:
                questions.append(f"How does {focus} relate to {neighbour}?")
        return questions

    def curiosity_signals(self, concept: str) -> List[str]:
        """Return actionable curiosity signals for a concept."""
        signals = []
        conf = float(self._graph.nodes.get(concept, {}).get("confidence", 1.0))
        if conf < CFG["curiosity_gap_threshold"]:
            signals.append(f"Low confidence on '{concept}' ({conf:.2f}) — needs more evidence.")
        pressure = self._cm.compute_pressure(concept)
        if pressure > CFG["curiosity_contradiction"]:
            signals.append(f"High contradiction pressure on '{concept}' ({pressure:.2f}) — needs reconciliation.")
        if self._graph.out_degree(concept) == 0:
            signals.append(f"'{concept}' is a dead-end — no outgoing relations known.")
        if self._graph.in_degree(concept) == 0:
            signals.append(f"'{concept}' has no known causes or predecessors.")
        return signals


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — EMERGENT CONCEPT SYNTHESISER
# ══════════════════════════════════════════════════════════════════════════════

class EmergentSynthesiser:
    """
    Discovers abstract parent concepts not yet in the graph by finding
    clusters of concepts that share multiple successors and similar embeddings.

    Example: 'Bird', 'Plane', 'Kite' all share successors 'flying', 'movement'
    → synthesises 'Aerial Entity'
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        embedder: SemanticEmbedder,
        working_mem: WorkingMemory,
    ) -> None:
        self._graph  = graph
        self._embed  = embedder
        self._wm     = working_mem

    def _shared_successors(self, a: str, b: str) -> Set[str]:
        return set(self._graph.successors(a)) & set(self._graph.successors(b))

    def _candidate_name(self, concepts: List[str]) -> str:
        """Derive a name for the emergent concept by hashing input concepts."""
        joined = "_".join(sorted(concepts))
        h      = hashlib.sha1(joined.encode()).hexdigest()[:6]
        return f"Emergent_{h}"

    def synthesise(self, active_concepts: List[str]) -> List[Tuple[str, List[str], float]]:
        """
        For each pair of active concepts, check if they share enough structure
        to warrant a new parent concept.

        Returns list of (new_concept_name, parent_of_list, initial_confidence).
        """
        results: List[Tuple[str, List[str], float]] = []
        seen: Set[FrozenSet] = set()

        for i, a in enumerate(active_concepts):
            for b in active_concepts[i + 1:]:
                key = frozenset({a, b})
                if key in seen:
                    continue
                seen.add(key)

                shared = self._shared_successors(a, b)
                if len(shared) < CFG["emergence_min_shared"]:
                    continue

                sim = self._embed.similarity(a, b)
                conf_a = float(self._graph.nodes.get(a, {}).get("confidence", 0))
                conf_b = float(self._graph.nodes.get(b, {}).get("confidence", 0))
                avg_conf = (conf_a + conf_b) / 2.0 * sim

                if avg_conf < CFG["emergence_min_conf"]:
                    continue

                name = self._candidate_name([a, b])
                if name not in self._graph:
                    results.append((name, [a, b], float(np.clip(avg_conf * 0.8, 0.1, 0.9))))

        return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CONTEXTUAL SEMANTIC COLLAPSE
# ══════════════════════════════════════════════════════════════════════════════

class SemanticCollapser:
    """
    Resolves ambiguous concepts into their most contextually appropriate meaning.

    A concept like 'light' has latent semantic states:
      physics | illumination | lightweight | spirituality

    The collapser selects the meaning whose domain best matches the active
    reasoning context, using semantic similarity as the tie-breaker.
    """

    def __init__(self, embedder: SemanticEmbedder, graph: nx.DiGraph) -> None:
        self._embed = embedder
        self._graph = graph
        # Registry: ambiguous_concept → {meaning: domain_label}
        self._ambiguity_registry: Dict[str, Dict[str, str]] = {}

    def register_ambiguous(self, concept: str, meanings: Dict[str, str]) -> None:
        """Register possible meanings for an ambiguous concept."""
        self._ambiguity_registry[concept] = meanings

    def collapse(
        self,
        concept: str,
        active_context: List[str],
        target_domain: str = "general",
    ) -> Tuple[str, float]:
        """
        Collapse *concept* into its most contextually relevant meaning.
        Returns (resolved_meaning, confidence).
        """
        if concept not in self._ambiguity_registry:
            return concept, 1.0

        meanings = self._ambiguity_registry[concept]
        if not active_context:
            # No context — return highest-confidence meaning in the graph
            best_m, best_c = concept, 0.0
            for m in meanings:
                conf = float(self._graph.nodes.get(m, {}).get("confidence", 0.3))
                if conf > best_c:
                    best_m, best_c = m, conf
            return best_m, best_c

        # Compute average similarity of each meaning to the active context
        scores = {}
        for meaning in meanings:
            sims = [self._embed.similarity(meaning, ctx) for ctx in active_context]
            scores[meaning] = float(np.mean(sims))

        # Domain bonus
        if target_domain != "general":
            for meaning, domain in meanings.items():
                if domain == target_domain:
                    scores[meaning] = scores.get(meaning, 0.0) + 0.2

        best_meaning = max(scores, key=scores.__getitem__)
        conf         = float(np.clip(scores[best_meaning], 0.0, 1.0))
        return best_meaning, conf

    def track_ambiguity(self, concept: str) -> float:
        """Return ambiguity score (0 = unambiguous, 1 = maximally ambiguous)."""
        if concept not in self._ambiguity_registry:
            return 0.0
        n = len(self._ambiguity_registry[concept])
        return float(1.0 - 1.0 / n) if n > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — META-COGNITIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class MetaCognition:
    """
    Self-reflective reasoning monitor.

    Tracks:
    - per-mode success rates
    - bias patterns (systematic over/under-confidence)
    - failed inference chains
    - uncertainty reliability (calibration)
    - confidence-about-confidence (meta-confidence)
    """

    def __init__(self) -> None:
        self._mode_history: Dict[str, List[bool]]       = collections.defaultdict(list)
        self._confidence_log: List[Tuple[float, bool]]  = []  # (predicted_conf, actual_success)
        self._failed_chains: List[Tuple[str, str]]      = []  # (start, failed_at)

    def record(self, mode: str, success: bool, confidence: float) -> None:
        self._mode_history[mode].append(success)
        self._confidence_log.append((confidence, success))

    def record_failure(self, start: str, failed_at: str) -> None:
        self._failed_chains.append((start, failed_at))
        if len(self._failed_chains) > 200:
            self._failed_chains = self._failed_chains[-200:]

    def mode_success_rate(self, mode: str) -> float:
        h = self._mode_history.get(mode, [])
        return float(np.mean(h)) if h else 0.5

    def best_mode(self, candidates: List[str]) -> str:
        return max(candidates, key=self.mode_success_rate)

    def meta_confidence(self, current_confidence: float) -> float:
        """
        Calibration-adjusted confidence (confidence-about-confidence).
        Uses Brier score on historical predictions.
        """
        if len(self._confidence_log) < 5:
            return current_confidence
        preds, actuals = zip(*self._confidence_log[-100:])
        brier = float(np.mean([(p - float(a)) ** 2 for p, a in zip(preds, actuals)]))
        # Lower Brier = better calibration → meta_conf closer to raw conf
        calibration = float(np.clip(1.0 - 2.0 * brier, 0.0, 1.0))
        return float(np.clip(current_confidence * calibration, 0.0, 1.0))

    def detect_biases(self) -> List[str]:
        biases = []
        if not self._confidence_log:
            return biases
        confs, acts = zip(*self._confidence_log)
        avg_conf = float(np.mean(confs))
        avg_acc  = float(np.mean([float(a) for a in acts]))
        if avg_conf - avg_acc > 0.15:
            biases.append(f"OVERCONFIDENCE bias detected (avg conf={avg_conf:.2f}, actual={avg_acc:.2f})")
        elif avg_acc - avg_conf > 0.15:
            biases.append(f"UNDERCONFIDENCE bias detected (avg conf={avg_conf:.2f}, actual={avg_acc:.2f})")
        return biases

    def reasoning_health(self) -> Dict[str, Any]:
        return {
            "mode_success_rates": {m: self.mode_success_rate(m) for m in self._mode_history},
            "total_sessions":     sum(len(v) for v in self._mode_history.values()),
            "failed_chains":      len(self._failed_chains),
            "biases":             self.detect_biases(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — QUANTUM-INSPIRED AMPLITUDE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AmplitudeEngine:
    """
    True quantum-inspired reasoning mechanics.

    Each reasoning path contributes a complex amplitude to a superposition
    state vector.  Paths can:

    - CONSTRUCTIVELY INTERFERE: paths leading to the same conclusion
      with the same phase → reinforce each other (amplitudes add).

    - DESTRUCTIVELY INTERFERE: paths leading to the same conclusion via
      contradictory intermediate concepts → partially cancel each other.

    - ENTANGLE concepts: concepts that frequently co-activate gain a shared
      amplitude component.

    The state vector is normalised and 'collapsed' by computing |amplitude|²
    (Born rule) to yield final probabilities.
    """

    def __init__(self, embedder: SemanticEmbedder) -> None:
        self._embed = embedder
        self._entanglement_cache: Dict[Tuple[str, str], float] = {}

    def _phase(self, path: List[str], graph: nx.DiGraph) -> float:
        """
        Compute the phase of a path.
        Phase is accumulated from the semantic direction of each hop.
        Paths through 'contradicts' or 'opposite_of' edges get a π phase shift.
        """
        phase = 0.0
        for i in range(len(path) - 1):
            e = graph.get_edge_data(path[i], path[i + 1], default={})
            relation = e.get("relation", "")
            if relation in ("opposite_of", "contradicts", "negates"):
                phase += math.pi          # destructive contribution
            elif relation in ("enables", "causes", "is_a", "encodes", "expresses"):
                phase += 0.0              # constructive
            else:
                # Semantic angle between consecutive concepts
                sim   = self._embed.similarity(path[i], path[i + 1])
                phase += (1.0 - sim) * math.pi * 0.5
        return phase

    def _path_amplitude(self, path: List[str], graph: nx.DiGraph) -> complex:
        """Compute the complex amplitude for a single path."""
        magnitude = 1.0
        for i in range(len(path) - 1):
            e   = graph.get_edge_data(path[i], path[i + 1], default={})
            w   = float(e.get("weight", 0.5))
            c   = float(e.get("confidence", 0.5))
            magnitude *= math.sqrt(w * c)   # amplitude ~ √probability
        phase = self._phase(path, graph)
        return complex(magnitude * math.cos(phase), magnitude * math.sin(phase))

    def _entanglement_boost(self, a: str, b: str) -> float:
        """
        Concepts that are semantically close get a small shared amplitude boost,
        simulating quantum entanglement of correlated concepts.
        """
        key = (min(a, b), max(a, b))
        if key not in self._entanglement_cache:
            self._entanglement_cache[key] = self._embed.similarity(a, b)
        return float(np.clip(self._entanglement_cache[key] * 0.15, 0.0, 0.15))

    def build_superposition(
        self,
        paths:          List[List[str]],
        concept_index:  Dict[str, int],
        n_qubits:       int,
        graph:          nx.DiGraph,
        add_noise:      bool = False,
        rng:            Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Build the superposition state vector across all paths.
        Returns a complex vector of length 2^n_qubits.
        """
        dim   = 2 ** n_qubits
        state = np.zeros(dim, dtype=np.complex128)

        for path in paths:
            amp      = self._path_amplitude(path, graph)
            basis_idx = 0
            for concept in path:
                q = concept_index.get(concept)
                if q is not None:
                    basis_idx |= (1 << q)

            # Entanglement boost between adjacent concepts in path
            for i in range(len(path) - 1):
                boost = self._entanglement_boost(path[i], path[i + 1])
                amp   = amp * complex(1.0 + boost, 0.0)

            state[basis_idx] += amp

        if add_noise and rng is not None:
            noise = rng.uniform(-CFG["noise_scale"], CFG["noise_scale"], dim)
            state += noise.astype(np.complex128)

        norm = np.linalg.norm(state)
        if norm < 1e-9:
            return state
        return state / norm

    def collapse(
        self,
        state_vector:  np.ndarray,
        index_to_concept: Dict[int, str],
        start_concept: str,
        n_qubits:      int,
    ) -> Dict[str, float]:
        """
        Apply Born rule: probability = |amplitude|²
        Decode each basis state into the concepts it represents.
        Map contributions back to destination concepts.
        """
        scores: Dict[str, float] = {}
        dim = 2 ** n_qubits

        for idx in range(dim):
            amp  = state_vector[idx]
            prob = float(abs(amp) ** 2)
            if prob < CFG["min_amplitude"] ** 2:
                continue

            active = [
                index_to_concept[j]
                for j in range(n_qubits)
                if (idx >> j) & 1 and j in index_to_concept
            ]
            if len(active) > 1 and start_concept in active:
                destination = active[-1]
                scores[destination] = scores.get(destination, 0.0) + prob

        return scores


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — MULTI-LAYER INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """
    Applies multiple reasoning strategies to a working graph and produces
    a unified set of volatile inferred links.

    Each strategy:
    - reads from the permanent graph
    - writes inferred links (flagged as volatile) to the returned list
    - never mutates the permanent graph
    """

    def __init__(self, graph: nx.DiGraph, embedder: SemanticEmbedder) -> None:
        self._g     = graph
        self._embed = embedder

    # ── Deductive (modus ponens chains) ────────────────────────────────────
    def deductive(self) -> List[Tuple]:
        """If A→B (is_a/causes) and B→C (is_a/causes), then A→C (deduced)."""
        FORWARD_RELS = {"is_a", "causes", "enables", "implies"}
        links = []
        for a in self._g.nodes:
            for b in self._g.successors(a):
                if self._g[a][b].get("relation") not in FORWARD_RELS:
                    continue
                for c in self._g.successors(b):
                    if c == a or self._g.has_edge(a, c):
                        continue
                    if self._g[b][c].get("relation") not in FORWARD_RELS:
                        continue
                    w  = math.sqrt(self._g[a][b].get("weight", 0) * self._g[b][c].get("weight", 0))
                    cf = self._g[a][b].get("confidence", 0) * self._g[b][c].get("confidence", 0)
                    if w >= 0.3 and cf >= 0.3:
                        links.append((a, c, "deduced", w, cf))
        return links

    # ── Inductive (generalisation from examples) ───────────────────────────
    def inductive(self, min_examples: int = 2) -> List[Tuple]:
        """
        If ≥ min_examples nodes share the same outgoing relation to the same
        target, infer a general 'generalises_to' link from their common
        predecessor to that target.
        """
        links = []
        target_sources: Dict[str, List[str]] = collections.defaultdict(list)
        for u, v, d in self._g.edges(data=True):
            if float(d.get("confidence", 0)) > 0.6:
                target_sources[v].append(u)
        for target, sources in target_sources.items():
            if len(sources) >= min_examples:
                for src in sources:
                    for other in sources:
                        if src != other and not self._g.has_edge(src, other):
                            sim = self._embed.similarity(src, other)
                            if sim > 0.4:
                                links.append((src, other, "co-instantiates", sim * 0.5, sim * 0.6))
        return links

    # ── Abductive (best explanation) ───────────────────────────────────────
    def abductive(self, target: str) -> List[Tuple]:
        """Find the most likely antecedent(s) for *target*."""
        links = []
        if target not in self._g:
            return links
        for pred in self._g.predecessors(target):
            w  = float(self._g[pred][target].get("weight", 0))
            cf = float(self._g[pred][target].get("confidence", 0))
            score = w * cf
            if score > 0.25:
                links.append((pred, target, "abduced", w * 0.75, cf * 0.75))
        links.sort(key=lambda x: x[3] * x[4], reverse=True)
        return links[:5]

    # ── Analogical (structural isomorphism) ────────────────────────────────
    def analogical(self) -> List[Tuple]:
        """If A→C and B→D and sim(C,D) high, infer A similar_to B."""
        links = []
        nodes = list(self._g.nodes)
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                if self._g.has_edge(a, b):
                    continue
                for c in self._g.successors(a):
                    for d in self._g.successors(b):
                        if c == d:
                            continue
                        sim = self._embed.similarity(c, d)
                        if sim > CFG["collapse_similarity_th"]:
                            w  = sim * 0.5
                            cf = sim * 0.55
                            links.append((a, b, "analogous_to", w, cf))
        return links

    # ── Causal (directed causal chains with strength propagation) ──────────
    def causal(self, concept: str, depth: int = 3) -> List[Tuple]:
        """Trace causal downstream effects from *concept*."""
        links = []
        CAUSAL_RELS = {"causes", "leads_to", "produces", "triggers", "enables"}
        frontier = {concept: 1.0}
        for _ in range(depth):
            nf: Dict[str, float] = {}
            for src, strength in frontier.items():
                for _, tgt, d in self._g.out_edges(src, data=True):
                    if d.get("relation") not in CAUSAL_RELS:
                        continue
                    w  = float(d.get("weight", 0.5))
                    cf = float(d.get("confidence", 0.5))
                    prop = strength * w * cf * 0.8
                    if prop > 0.05 and not self._g.has_edge(concept, tgt) and tgt != concept:
                        links.append((concept, tgt, "causally_leads_to", prop, cf))
                        nf[tgt] = max(nf.get(tgt, 0.0), prop)
            frontier = nf
        return links

    # ── Temporal (sequence-based prediction) ───────────────────────────────
    def temporal(self, recent_sequence: List[str]) -> List[Tuple]:
        """
        If a sequence A, B has been observed, predict C follows B based
        on known B→C edges.
        """
        links = []
        if len(recent_sequence) < 2:
            return links
        last = recent_sequence[-1]
        for _, succ, d in self._g.out_edges(last, data=True):
            w  = float(d.get("weight", 0.5))
            cf = float(d.get("confidence", 0.5))
            if w * cf > 0.2:
                links.append((last, succ, "temporally_follows", w * 0.9, cf * 0.9))
        return links

    # ── Counterfactual ─────────────────────────────────────────────────────
    def counterfactual(self, concept: str, temporal: TemporalWorldModel) -> List[Tuple]:
        """What concepts would be most affected if *concept* were removed?"""
        links = []
        changed = temporal.counterfactual(concept, negate=True)
        for affected, delta in changed[:5]:
            links.append((concept, affected, "counterfactually_affects", delta, delta * 0.8))
        return links

    # ── Meta-cognitive (reasoning about reasoning) ─────────────────────────
    def meta(self, start: str, meta_cog: MetaCognition) -> List[Tuple]:
        """
        Emit links that reflect meta-cognitive judgements:
        concepts whose reasoning history is weak get lower inferred weight.
        """
        links = []
        health = meta_cog.reasoning_health()
        for mode, rate in health["mode_success_rates"].items():
            if rate < 0.4:
                # Suppress inferences from poorly performing modes
                for _, v, d in self._g.out_edges(start, data=True):
                    if d.get("inferred_by") == mode:
                        links.append((start, v, "meta_suppressed", 0.1, 0.1))
        return links

    # ── Transitive (general chain completion) ──────────────────────────────
    def transitive(self) -> List[Tuple]:
        """Geometric-mean weight, product confidence chain completion."""
        links = []
        for a in self._g.nodes:
            for b in self._g.successors(a):
                w_ab = float(self._g[a][b].get("weight", 0))
                c_ab = float(self._g[a][b].get("confidence", 0))
                for c in self._g.successors(b):
                    if c == a or self._g.has_edge(a, c):
                        continue
                    w_bc = float(self._g[b][c].get("weight", 0))
                    c_bc = float(self._g[b][c].get("confidence", 0))
                    iw   = math.sqrt(w_ab * w_bc)
                    ic   = c_ab * c_bc
                    if iw >= 0.30 and ic >= 0.35:
                        links.append((a, c, "transitive", iw, ic))
        return links

    def run(
        self,
        start:           str,
        modes:           List[InferenceMode],
        temporal:        TemporalWorldModel,
        meta_cog:        MetaCognition,
        recent_sequence: List[str],
    ) -> List[Tuple]:
        """Dispatch selected inference modes and aggregate results."""
        links: List[Tuple] = []
        dedup:  Set[Tuple[str, str]] = set()

        dispatch = {
            InferenceMode.DEDUCTIVE:      lambda: self.deductive(),
            InferenceMode.INDUCTIVE:      lambda: self.inductive(),
            InferenceMode.ABDUCTIVE:      lambda: self.abductive(start),
            InferenceMode.ANALOGICAL:     lambda: self.analogical(),
            InferenceMode.CAUSAL:         lambda: self.causal(start),
            InferenceMode.TEMPORAL:       lambda: self.temporal(recent_sequence),
            InferenceMode.COUNTERFACTUAL: lambda: self.counterfactual(start, temporal),
            InferenceMode.META:           lambda: self.meta(start, meta_cog),
        }

        active_modes = list(dispatch.keys()) if InferenceMode.HYBRID in modes else modes

        for mode in active_modes:
            fn = dispatch.get(mode)
            if fn is None:
                continue
            try:
                for link in fn():
                    pair = (link[0], link[1])
                    if pair not in dedup:
                        dedup.add(pair)
                        links.append(link)
            except Exception as exc:
                pass  # individual mode failures are non-fatal

        return links


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — RECURSIVE THOUGHT SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class RecursiveThoughtSimulator:
    """
    Simulates branching thought chains:
    'If I conclude X, then Y becomes possible, which changes Z …'

    Builds a lightweight DAG of thought states and propagates belief revisions.
    """

    def __init__(self, graph: nx.DiGraph, max_depth: int = 4) -> None:
        self._graph    = graph
        self.max_depth = max_depth

    def simulate(
        self,
        start:          str,
        initial_beliefs: Dict[str, float],  # concept → confidence
    ) -> List[Dict]:
        """
        Run recursive simulation. Returns a list of thought-state dicts
        ordered by depth, each containing revised belief distributions.
        """
        states  = [{"depth": 0, "trigger": start, "beliefs": dict(initial_beliefs)}]
        current = dict(initial_beliefs)

        for depth in range(1, self.max_depth + 1):
            # Find the strongest belief not yet fully resolved
            trigger = max(current, key=current.__getitem__, default=None)
            if trigger is None:
                break

            revised = dict(current)
            changed = False

            # Propagate consequences
            for _, succ, d in self._graph.out_edges(trigger, data=True):
                w   = float(d.get("weight", 0.5))
                cf  = float(d.get("confidence", 0.5))
                rel = d.get("relation", "")

                old = revised.get(succ, 0.0)
                if rel in ("opposite_of", "contradicts", "negates"):
                    delta = -current[trigger] * w * cf * 0.5   # destabilise
                else:
                    delta = current[trigger] * w * cf * 0.6    # strengthen

                revised[succ] = float(np.clip(old + delta, 0.0, 1.0))
                if abs(delta) > 0.02:
                    changed = True

            if not changed:
                break

            states.append({"depth": depth, "trigger": trigger, "beliefs": revised})
            current = revised

        return states

    def detect_instabilities(self, states: List[Dict]) -> List[str]:
        """Find concepts whose belief flipped significantly across states."""
        unstable = []
        if len(states) < 2:
            return unstable
        first = states[0]["beliefs"]
        last  = states[-1]["beliefs"]
        for concept in set(first) | set(last):
            delta = abs(last.get(concept, 0.0) - first.get(concept, 0.0))
            if delta > 0.3:
                unstable.append(f"{concept} (Δ={delta:.2f})")
        return unstable


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — GOAL-DRIVEN COGNITION CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class GoalController:
    """
    Translates a CognitiveGoal into concrete reasoning parameters:
    - which inference modes to activate
    - search depth
    - novelty preference
    - path scoring bias
    """

    _GOAL_PARAMS = {
        CognitiveGoal.EXPLORE: {
            "modes":        [InferenceMode.ANALOGICAL, InferenceMode.INDUCTIVE, InferenceMode.HYBRID],
            "depth":        5,
            "budget":       600,
            "novelty_bias": 0.6,
        },
        CognitiveGoal.PRECISION: {
            "modes":        [InferenceMode.DEDUCTIVE, InferenceMode.CAUSAL],
            "depth":        3,
            "budget":       200,
            "novelty_bias": 0.1,
        },
        CognitiveGoal.CREATIVE: {
            "modes":        [InferenceMode.ANALOGICAL, InferenceMode.INDUCTIVE, InferenceMode.ABDUCTIVE],
            "depth":        5,
            "budget":       500,
            "novelty_bias": 0.8,
        },
        CognitiveGoal.SAFETY: {
            "modes":        [InferenceMode.DEDUCTIVE, InferenceMode.META],
            "depth":        2,
            "budget":       150,
            "novelty_bias": 0.05,
        },
        CognitiveGoal.DISCOVERY: {
            "modes":        [InferenceMode.HYBRID],
            "depth":        5,
            "budget":       700,
            "novelty_bias": 0.9,
        },
        CognitiveGoal.RESOLVE: {
            "modes":        [InferenceMode.ABDUCTIVE, InferenceMode.COUNTERFACTUAL, InferenceMode.META],
            "depth":        4,
            "budget":       400,
            "novelty_bias": 0.3,
        },
        CognitiveGoal.ABSTRACT: {
            "modes":        [InferenceMode.ANALOGICAL, InferenceMode.INDUCTIVE, InferenceMode.TRANSITIVE],
            "depth":        4,
            "budget":       500,
            "novelty_bias": 0.5,
        },
    }

    def get_params(self, goal: CognitiveGoal) -> Dict:
        return dict(self._GOAL_PARAMS.get(goal, self._GOAL_PARAMS[CognitiveGoal.DISCOVERY]))

    def select_goal(
        self,
        concept:        str,
        graph:          nx.DiGraph,
        contradiction:  ContradictionManager,
        meta:           MetaCognition,
    ) -> CognitiveGoal:
        """
        Autonomously select the most appropriate goal based on the current
        cognitive state of the concept being reasoned about.
        """
        pressure = contradiction.compute_pressure(concept)
        if pressure > 0.5:
            return CognitiveGoal.RESOLVE

        conf = float(graph.nodes.get(concept, {}).get("confidence", 0.5))
        if conf < 0.3:
            return CognitiveGoal.DISCOVERY

        health   = meta.reasoning_health()
        avg_rate = float(np.mean(list(health["mode_success_rates"].values()))) if health["mode_success_rates"] else 0.5
        if avg_rate < 0.35:
            return CognitiveGoal.EXPLORE

        return CognitiveGoal.PRECISION


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — KNOWLEDGE GRAPH MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class KnowledgeGraphManager:
    """
    Manages the persistent, self-organising knowledge graph.

    Extended edge schema:
        weight, confidence, relation, last_traversal,
        causality_strength, temporal_relevance,
        semantic_similarity, contradiction_pressure,
        reinforcement_count
    """

    # Node-name safety regex
    _SAFE_RE = re.compile(r"[^\w\s\-]")

    def __init__(self, graph_file: str) -> None:
        self.graph_file = graph_file
        self.graph: nx.DiGraph = self._load()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self) -> nx.DiGraph:
        if os.path.exists(self.graph_file):
            try:
                print(f"[KG] Loading from '{self.graph_file}' …")
                g = nx.read_graphml(self.graph_file)
                _FLOAT_NODE = ("confidence",)
                _FLOAT_EDGE = (
                    "weight", "confidence", "last_traversal",
                    "causality_strength", "temporal_relevance",
                    "semantic_similarity", "contradiction_pressure",
                    "reinforcement_count",
                )
                for _, d in g.nodes(data=True):
                    for k in _FLOAT_NODE:
                        if k in d:
                            d[k] = float(d[k])
                for _, _, d in g.edges(data=True):
                    for k in _FLOAT_EDGE:
                        if k in d:
                            d[k] = float(d[k])
                print(f"[KG] Loaded {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.")
                return g
            except Exception as exc:
                print(f"[KG] Load failed ({exc}). Starting fresh.")
        return nx.DiGraph()

    def save(self) -> None:
        try:
            nx.write_graphml(self.graph, self.graph_file)
        except Exception as exc:
            print(f"[KG] Save failed: {exc}")

    def reload(self) -> None:
        self.graph = self._load()

    # ── Validation ─────────────────────────────────────────────────────────

    @classmethod
    def _valid(cls, name) -> bool:
        return isinstance(name, str) and bool(name.strip())

    @classmethod
    def _clean(cls, name: str) -> str:
        return cls._SAFE_RE.sub("", name.strip())

    # ── Concept CRUD ───────────────────────────────────────────────────────

    def add_concept(
        self,
        name:       str,
        confidence: float = 1.0,
        domain:     str   = "general",
        info:       str   = "",
    ) -> bool:
        """Add or update a concept. Returns True if newly created."""
        if not self._valid(name):
            return False
        name = self._clean(name)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        if name not in self.graph:
            self.graph.add_node(name, confidence=confidence, domain=domain, info=info, creation_time=time.time())
            return True

        node = self.graph.nodes[name]
        node["confidence"] = float(np.clip(0.7 * node.get("confidence", confidence) + 0.3 * confidence, 0.0, 1.0))
        if domain != "general":
            node["domain"] = domain
        if info:
            node["info"] = info
        return False

    def add_edge(
        self,
        source:      str,
        target:      str,
        relation:    str   = "related",
        weight:      float = 1.0,
        confidence:  float = 1.0,
        causality:   float = 0.5,
        temporal:    float = 1.0,
        sem_sim:     float = 0.5,
        contradiction_pressure: float = 0.0,
        _graph:      Optional[nx.DiGraph] = None,
    ) -> bool:
        if not (self._valid(source) and self._valid(target)):
            return False
        if source == target:
            return False

        weight     = float(np.clip(weight, 0.01, 1.0))
        confidence = float(np.clip(confidence, 0.01, 1.0))

        self.add_concept(source)
        self.add_concept(target)

        g = _graph if _graph is not None else self.graph

        if g.has_edge(source, target):
            e = g[source][target]
            e["weight"]     = float(np.clip(0.6 * e.get("weight", weight)     + 0.4 * weight, 0.01, 1.0))
            e["confidence"] = float(np.clip(0.6 * e.get("confidence", confidence) + 0.4 * confidence, 0.01, 1.0))
            e["reinforcement_count"] = int(e.get("reinforcement_count", 0)) + 1
            e["last_traversal"] = time.time()
        else:
            g.add_edge(
                source, target,
                relation              = relation,
                weight                = weight,
                confidence            = confidence,
                causality_strength    = causality,
                temporal_relevance    = temporal,
                semantic_similarity   = sem_sim,
                contradiction_pressure= contradiction_pressure,
                reinforcement_count   = 1,
                last_traversal        = time.time(),
            )

        if g is self.graph:
            self.save()
        return True

    def reinforce(self, source: str, target: str, factor: float = CFG["reinforcement_factor"]) -> None:
        if not self.graph.has_edge(source, target):
            return
        factor = float(np.clip(factor, 0.0, 1.0))
        e = self.graph[source][target]
        e["weight"]     = float(np.clip(e.get("weight", 0.5)     + factor * (1 - e.get("weight", 0.5)), 0.01, 1.0))
        e["confidence"] = float(np.clip(e.get("confidence", 0.5) + factor * (1 - e.get("confidence", 0.5)), 0.01, 1.0))
        e["reinforcement_count"] = int(e.get("reinforcement_count", 0)) + 1
        e["last_traversal"] = time.time()
        self.save()

    def decay(self, factor: float = CFG["decay_factor"], threshold_s: float = CFG["decay_threshold_s"]) -> int:
        now = time.time()
        count = 0
        for _, _, d in self.graph.edges(data=True):
            last = float(d.get("last_traversal", 0))
            if now - last > threshold_s:
                d["weight"] = float(np.clip(d.get("weight", 1.0) * factor, 0.01, 1.0))
                count += 1
        if count:
            self.save()
        return count

    def merge(self, primary: str, redundant: str) -> None:
        g = self.graph
        if primary not in g or redundant not in g:
            return
        for src, _, d in list(g.in_edges(redundant, data=True)):
            if src != primary:
                self.add_edge(src, primary, d.get("relation", "related"), d.get("weight", 1.0), d.get("confidence", 1.0))
        for _, tgt, d in list(g.out_edges(redundant, data=True)):
            if tgt != primary:
                self.add_edge(primary, tgt, d.get("relation", "related"), d.get("weight", 1.0), d.get("confidence", 1.0))
        g.remove_node(redundant)
        self.save()
        print(f"[KG] Merged '{redundant}' → '{primary}'.")

    def split_concept(self, concept: str, specialisations: List[Tuple[str, str]]) -> None:
        """
        Split *concept* into multiple specialised sub-concepts.
        specialisations: [(new_name, domain), ...]
        """
        g = self.graph
        if concept not in g:
            return
        base_conf = float(g.nodes[concept].get("confidence", 0.5))
        for spec_name, spec_domain in specialisations:
            self.add_concept(spec_name, base_conf * 0.85, spec_domain)
            self.add_edge(concept, spec_name, "specialises_into", 0.9, 0.85)
        print(f"[KG] Split '{concept}' into {[s[0] for s in specialisations]}.")

    def stats(self) -> Dict:
        g = self.graph
        weights = [float(d.get("weight", 0)) for _, _, d in g.edges(data=True)]
        return {
            "nodes":            g.number_of_nodes(),
            "edges":            g.number_of_edges(),
            "density":          round(nx.density(g), 6),
            "avg_weight":       round(float(np.mean(weights)), 4) if weights else 0.0,
            "is_dag":           nx.is_directed_acyclic_graph(g),
            "weakly_connected": nx.number_weakly_connected_components(g),
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — QMIND v2 MAIN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class Qmind:
    """
    Qmind v2 — Proto-AGI Cognitive Architecture.

    Subsystems:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  KnowledgeGraphManager   │  self-organising persistent graph         │
    │  WorkingMemory           │  bounded LRU cognitive spotlight          │
    │  EpisodicMemory          │  session history & consolidation          │
    │  SemanticEmbedder        │  lightweight neural-symbolic bridge       │
    │  AttentionSystem         │  dynamic concept salience scoring         │
    │  ContradictionManager    │  probabilistic belief coexistence         │
    │  TemporalWorldModel      │  causal/temporal simulation               │
    │  CuriosityEngine         │  autonomous gap detection                 │
    │  EmergentSynthesiser     │  new concept formation                    │
    │  SemanticCollapser       │  contextual meaning resolution            │
    │  MetaCognition           │  self-reflective reasoning monitor        │
    │  AmplitudeEngine         │  quantum-inspired interference mechanics  │
    │  InferenceEngine         │  multi-layer inference strategies         │
    │  RecursiveThoughtSimulator│ branching belief simulation              │
    │  GoalController          │  goal-driven cognition adapter            │
    └──────────────────────────────────────────────────────────────────────┘
    """

    _UNSAFE_TERMS = frozenset({"harm", "violence", "weapon", "exploit", "malware"})
    _QUESTION_PREFIXES = (
        "what is", "what are", "who is", "who are",
        "how is", "how does", "how do", "explain", "describe",
        "define", "tell me about", "what causes", "why is",
    )

    def __init__(
        self,
        graph_file:    str   = CFG["graph_file"],
        episodic_file: str   = CFG["episodic_file"],
        wm_capacity:   int   = CFG["working_memory_capacity"],
        learning_rate: float = CFG["base_learning_rate"],
        seed:          Optional[int] = None,
    ) -> None:
        self._rng          = np.random.default_rng(seed)
        self.learning_rate = learning_rate

        # ── Subsystem initialisation ──────────────────────────────────────
        self.kg      = KnowledgeGraphManager(graph_file)
        self.wm      = WorkingMemory(capacity=wm_capacity)
        self.episodic= EpisodicMemory()
        self.embed   = SemanticEmbedder()
        self.attention= AttentionSystem(self.embed)
        self.contra  = ContradictionManager(self.kg.graph)
        self.temporal= TemporalWorldModel(self.kg.graph)
        self.curiosity= CuriosityEngine(self.kg.graph, self.contra, self.embed)
        self.synthesiser = EmergentSynthesiser(self.kg.graph, self.embed, self.wm)
        self.collapser= SemanticCollapser(self.embed, self.kg.graph)
        self.meta    = MetaCognition()
        self.amp_eng = AmplitudeEngine(self.embed)
        self.infer   = InferenceEngine(self.kg.graph, self.embed)
        self.rts     = RecursiveThoughtSimulator(self.kg.graph)
        self.goal_ctrl = GoalController()

        # Load episodic memory if it exists
        self._episodic_file = episodic_file
        self._load_episodic()

        print(f"[Qmind v2] Initialised. Graph: {self.kg.stats()}")

    # ── Persistence ────────────────────────────────────────────────────────

    def _load_episodic(self) -> None:
        if os.path.exists(self._episodic_file):
            try:
                with open(self._episodic_file) as f:
                    self.episodic.from_dict(json.load(f))
                print(f"[Qmind v2] Episodic memory loaded ({len(self.episodic._traces)} traces).")
            except Exception:
                pass

    def save_episodic(self) -> None:
        try:
            with open(self._episodic_file, "w") as f:
                json.dump(self.episodic.to_dict(), f, indent=2)
        except Exception as exc:
            print(f"[Qmind v2] Episodic save failed: {exc}")

    # ── Public graph API (thin wrappers) ───────────────────────────────────

    def add_concept(self, name: str, confidence: float = 1.0, domain: str = "general", info: str = "") -> bool:
        created = self.kg.add_concept(name, confidence, domain, info)
        self.wm.activate(name, salience=confidence)
        return created

    def link(
        self, source: str, target: str,
        relation: str = "related", weight: float = 1.0, confidence: float = 1.0,
        causality: float = 0.5,
    ) -> bool:
        ok = self.kg.add_edge(source, target, relation, weight, confidence, causality)
        if ok:
            self.wm.activate(source)
            self.wm.activate(target)
        return ok

    def reinforce(self, source: str, target: str, factor: float = CFG["reinforcement_factor"]) -> None:
        self.kg.reinforce(source, target, factor)
        self.wm.activate(source)
        self.wm.activate(target)

    def decay(self, **kwargs) -> int:
        return self.kg.decay(**kwargs)

    # ── Safety ─────────────────────────────────────────────────────────────

    def _safe(self, concept: str) -> str:
        if any(t in concept.lower() for t in self._UNSAFE_TERMS):
            return "[withheld:safety]"
        return concept

    # ── Concept extraction ─────────────────────────────────────────────────

    def _extract_concept(self, query: str) -> str:
        core = query.strip()
        for pfx in self._QUESTION_PREFIXES:
            if core.lower().startswith(pfx):
                core = core[len(pfx):].strip()
                break
        core = re.sub(r"[^\w\s\-]", "", core).strip()
        titled = core.title()
        return titled if titled in self.kg.graph else core

    # ── Path scoring with attention bias ───────────────────────────────────

    def _score_path(
        self,
        path:   List[str],
        graph:  nx.DiGraph,
        goal:   CognitiveGoal,
        goal_concept: Optional[str],
        contrad_map:  Dict[str, float],
    ) -> float:
        base = 1.0
        for i in range(len(path) - 1):
            e    = graph.get_edge_data(path[i], path[i + 1], default={})
            w    = float(e.get("weight", 0.5))
            cf   = float(e.get("confidence", 0.5))
            base *= math.sqrt(w * cf)

        # Attention bias: prefer paths through high-salience concepts
        attn = float(np.mean([
            self.attention.score(c, graph, self.wm, goal, goal_concept, contrad_map)
            for c in path
        ]))
        return base * (1.0 + 0.3 * attn)

    # ══════════════════════════════════════════════════════════════════════
    # CORE REASONING METHOD
    # ══════════════════════════════════════════════════════════════════════

    def reason(
        self,
        start_concept:  str,
        goal:           Optional[CognitiveGoal]   = None,
        depth:          Optional[int]              = None,
        budget:         Optional[int]              = None,
        add_noise:      bool                       = False,
        target_domain:  str                        = "general",
    ) -> ReasoningTrace:
        """
        Full reasoning session from *start_concept*.

        Returns a :class:`ReasoningTrace` containing conclusions, diagnostics,
        emergent concepts, curiosity signals, contradictions, and meta-confidence.
        """
        t0 = time.time()

        # ── Auto-select goal if not provided ──────────────────────────────
        if goal is None:
            goal = self.goal_ctrl.select_goal(
                start_concept, self.kg.graph, self.contra, self.meta
            )

        # ── Goal → reasoning parameters ───────────────────────────────────
        params       = self.goal_ctrl.get_params(goal)
        depth        = depth  or params["depth"]
        budget       = budget or params["budget"]
        modes        = params["modes"]
        novelty_bias = params["novelty_bias"]

        print(f"\n[Qmind] ▶ Reason: '{start_concept}' | goal={goal.name} depth={depth}")

        # ── Semantic collapse: resolve ambiguity in context ────────────────
        active_ctx = self.wm.get_active(limit=8)
        resolved, collapse_conf = self.collapser.collapse(
            start_concept, active_ctx, target_domain
        )
        if resolved != start_concept:
            print(f"[Qmind] Semantic collapse: '{start_concept}' → '{resolved}' (conf={collapse_conf:.2f})")
            start_concept = resolved

        # ── Activate in working memory & temporal log ─────────────────────
        self.wm.activate(start_concept, salience=1.0)
        self.temporal.log_event(start_concept)
        recent_seq = self.temporal.get_recent_sequence(6)

        # ── Contradiction analysis ─────────────────────────────────────────
        contrad_map   = self.contra.all_pressures()
        contrad_pairs = self.contra.find_contradicted_pairs()

        # ── Attention-ranked active concepts ──────────────────────────────
        active = self.wm.get_active()
        if start_concept not in active:
            self.wm.activate(start_concept)
            active = self.wm.get_active()

        ranked_active = self.attention.rank(
            active, self.kg.graph, self.wm, goal,
            goal_concept=start_concept, contradiction_map=contrad_map,
        )
        ordered_active = [c for c, _ in ranked_active]

        # ── Run inference engine ───────────────────────────────────────────
        inferred_links = self.infer.run(
            start_concept, modes, self.temporal, self.meta, recent_seq
        )

        # ── Build effective graph (permanent ∪ volatile inference) ─────────
        effective = nx.DiGraph(self.kg.graph)
        for a, c, rel, w, cf in inferred_links:
            if not effective.has_edge(a, c):
                effective.add_edge(
                    a, c,
                    relation=rel, weight=w, confidence=cf,
                    last_traversal=time.time(), volatile=True,
                )

        # ── Build qubit index ──────────────────────────────────────────────
        # Index covers all nodes in the effective graph so the amplitude
        # engine can encode and decode any reachable concept.
        all_effective_nodes = list(effective.nodes())
        MAX_QUBITS = 18
        if len(all_effective_nodes) > MAX_QUBITS:
            # Keep start_concept + highest-salience nodes
            sal_sorted = sorted(
                all_effective_nodes,
                key=lambda c: (c == start_concept,
                               self.attention.score(c, effective, self.wm, goal,
                                                    start_concept, contrad_map)),
                reverse=True,
            )[:MAX_QUBITS]
            all_effective_nodes = sal_sorted
        concept_to_q = {c: i for i, c in enumerate(all_effective_nodes)}
        q_to_concept = {i: c for i, c in enumerate(all_effective_nodes)}
        n_qubits     = len(all_effective_nodes)

        # ── Enumerate paths ────────────────────────────────────────────────
        # Traverse the full effective graph; working memory is for scoring only.
        try:
            reachable = set(nx.single_source_shortest_path_length(
                effective, start_concept, cutoff=depth
            ).keys()) - {start_concept}
        except Exception:
            reachable = set()

        all_paths: List[List[str]] = []
        for tgt in reachable:
            try:
                for p in nx.all_simple_paths(effective, start_concept, tgt, cutoff=depth):
                    all_paths.append(p)
            except nx.NetworkXError:
                pass

        # Domain filter
        if target_domain != "general":
            dom_paths = [
                p for p in all_paths
                if all(
                    self.kg.graph.nodes.get(c, {}).get("domain", "general")
                    in (target_domain, "general")
                    for c in p
                )
            ]
            all_paths = dom_paths if dom_paths else all_paths

        paths_explored = len(all_paths)

        # ── Score and select paths (attention + goal bias) ─────────────────
        scored_paths = [
            (p, self._score_path(p, effective, goal, start_concept, contrad_map))
            for p in all_paths
        ]

        # Novelty bias: prefer paths through cold/less-accessed concepts
        if novelty_bias > 0:
            for i, (p, s) in enumerate(scored_paths):
                novelty = float(np.mean([
                    1.0 / (1.0 + self.wm._store.get(c, {}).get("access_count", 0))
                    for c in p
                ]))
                scored_paths[i] = (p, s * (1.0 + novelty_bias * novelty))

        scored_paths.sort(key=lambda x: x[1], reverse=True)
        selected = [p for p, _ in scored_paths[:budget]]

        # ── Superposition & collapse ───────────────────────────────────────
        conclusion_scores: Dict[str, float] = {}

        if selected:
            state_vec = self.amp_eng.build_superposition(
                selected, concept_to_q, n_qubits, effective,
                add_noise=add_noise, rng=self._rng,
            )
            if np.linalg.norm(state_vec) > 1e-9:
                conclusion_scores = self.amp_eng.collapse(
                    state_vec, q_to_concept, start_concept, n_qubits
                )

        # ── Rank conclusions ───────────────────────────────────────────────
        ranked = sorted(conclusion_scores.items(), key=lambda x: x[1], reverse=True)
        conclusions = [(c, self._safe(c), p) for c, p in ranked]

        # ── Competing hypotheses (top-3 non-dominant) ─────────────────────
        competing = [(c, p) for c, _, p in conclusions[1:4]]

        # ── Uncertainty & meta-confidence ─────────────────────────────────
        probs = [p for _, _, p in conclusions]
        uncertainty = 0.0
        if len(probs) > 1:
            # Shannon entropy of probability distribution
            probs_arr = np.array(probs)
            probs_arr = probs_arr / probs_arr.sum()
            uncertainty = float(-np.sum(probs_arr * np.log2(probs_arr + 1e-12)))
            uncertainty = float(np.clip(uncertainty / math.log2(max(len(probs), 2)), 0.0, 1.0))

        top_conf     = conclusions[0][2] if conclusions else 0.0
        meta_conf    = self.meta.meta_confidence(top_conf)

        # ── Causal chains ──────────────────────────────────────────────────
        causal = self.temporal.simulate_propagation(start_concept, steps=3)
        causal_chains = [[start_concept] + [c for c, _ in causal[:4]]] if causal else []

        # ── Recursive thought simulation ───────────────────────────────────
        initial_beliefs = {c: p for c, _, p in conclusions[:6]}
        initial_beliefs[start_concept] = 1.0
        thought_states = self.rts.simulate(start_concept, initial_beliefs)
        instabilities  = self.rts.detect_instabilities(thought_states)

        # ── Emergent concept synthesis ─────────────────────────────────────
        new_concepts: List[str] = []
        if goal in (CognitiveGoal.ABSTRACT, CognitiveGoal.CREATIVE, CognitiveGoal.DISCOVERY):
            syntheses = self.synthesiser.synthesise(ordered_active[:10])
            for ec_name, parents, ec_conf in syntheses:
                self.kg.add_concept(ec_name, ec_conf, domain="emergent",
                                    info=f"Emergent concept from: {parents}")
                for parent in parents:
                    self.kg.add_edge(parent, ec_name, "abstracted_into", ec_conf, ec_conf * 0.9)
                new_concepts.append(ec_name)
                self.wm.activate(ec_name, salience=ec_conf)

        # ── Curiosity signals ──────────────────────────────────────────────
        curiosity_sigs = self.curiosity.curiosity_signals(start_concept)
        if instabilities:
            curiosity_sigs.append(f"Unstable belief chain: {instabilities[0]}")

        # ── Status determination ───────────────────────────────────────────
        if not conclusions:
            status = "NO_CONCLUSION"
        elif top_conf >= 0.5 and meta_conf >= 0.4:
            status = "SUCCESS"
        elif top_conf >= 0.2:
            status = "WEAK_CONFIDENCE"
        else:
            status = "SPECULATIVE"

        # ── Episodic recording ─────────────────────────────────────────────
        ep_trace = MemoryTrace(
            uid         = str(uuid.uuid4())[:8],
            concept     = start_concept,
            query       = start_concept,
            conclusions = [(c, p) for c, _, p in conclusions[:3]],
            confidence  = top_conf,
            timestamp   = time.time(),
            domain      = target_domain,
            goal        = goal.name,
            success     = status == "SUCCESS",
            salience    = top_conf,
        )
        self.episodic.record(ep_trace)

        # ── Meta-cognition logging ─────────────────────────────────────────
        for mode in (modes if InferenceMode.HYBRID not in modes else [m for m in InferenceMode if m != InferenceMode.HYBRID]):
            self.meta.record(mode.name, status == "SUCCESS", top_conf)

        # ── Consolidate episodic → graph reinforcement ─────────────────────
        for src, tgt, avg_conf in self.episodic.consolidate_patterns():
            if self.kg.graph.has_edge(src, tgt):
                self.kg.reinforce(src, tgt, factor=0.05)

        elapsed = time.time() - t0

        trace = ReasoningTrace(
            query           = start_concept,
            concept         = start_concept,
            goal            = goal,
            inference_modes = [m.name for m in (modes if InferenceMode.HYBRID not in modes else list(InferenceMode))],
            paths_explored  = paths_explored,
            paths_used      = len(selected),
            inferred_links  = len(inferred_links),
            conclusions     = conclusions,
            competing_hyps  = competing,
            contradictions  = [(a, b, p) for a, b, p in contrad_pairs[:5]],
            uncertainty     = uncertainty,
            reasoning_depth = depth,
            meta_confidence = meta_conf,
            causal_chains   = causal_chains,
            emergent_concepts=new_concepts,
            curiosity_signals=curiosity_sigs,
            elapsed_s       = elapsed,
            status          = status,
        )
        return trace

    # ══════════════════════════════════════════════════════════════════════
    # HIGH-LEVEL QUERY INTERFACE
    # ══════════════════════════════════════════════════════════════════════

    def answer(self, query: str, goal: Optional[CognitiveGoal] = None) -> Dict:
        """
        Answer a natural-language query. Returns a rich structured response.

        Response schema:
        {
          status, query, concept, answer, confidence, meta_confidence,
          uncertainty, conclusions, competing_hypotheses, contradictions,
          curiosity_signals, emergent_concepts, causal_chain,
          reasoning_trace (full ReasoningTrace object)
        }
        """
        concept = self._extract_concept(query)
        print(f"[Qmind] Query: '{query}' → concept: '{concept}'")

        if concept not in self.kg.graph:
            # Log unknown
            try:
                with open(CFG["unknown_log"], "a") as f:
                    f.write(concept + "\n")
            except OSError:
                pass
            # Generate curiosity signals even for unknowns
            gaps = self.curiosity.identify_gaps()
            return {
                "status":               "UNKNOWN",
                "query":                query,
                "concept":              concept,
                "answer":               f"No knowledge found for '{concept}'.",
                "confidence":           0.0,
                "meta_confidence":      0.0,
                "uncertainty":          1.0,
                "conclusions":          [],
                "competing_hypotheses": [],
                "contradictions":       [],
                "curiosity_signals":    [f"Knowledge gap: '{concept}' is unknown."],
                "emergent_concepts":    [],
                "causal_chain":         [],
                "reasoning_trace":      None,
            }

        domain = self.kg.graph.nodes[concept].get("domain", "general")
        trace  = self.reason(concept, goal=goal, target_domain=domain)

        top_answer = (
            f"'{concept}' is most strongly linked to "
            f"'{trace.conclusions[0][1]}' (confidence {trace.conclusions[0][2]:.3f})."
            if trace.conclusions else
            f"Reasoning about '{concept}' produced no confident conclusion."
        )

        return {
            "status":               trace.status,
            "query":                query,
            "concept":              concept,
            "answer":               top_answer,
            "confidence":           trace.conclusions[0][2] if trace.conclusions else 0.0,
            "meta_confidence":      trace.meta_confidence,
            "uncertainty":          trace.uncertainty,
            "conclusions":          trace.conclusions,
            "competing_hypotheses": trace.competing_hyps,
            "contradictions":       trace.contradictions,
            "curiosity_signals":    trace.curiosity_signals,
            "emergent_concepts":    trace.emergent_concepts,
            "causal_chain":         trace.causal_chains[0] if trace.causal_chains else [],
            "reasoning_trace":      trace,
        }

    # ── Parallel reasoning ─────────────────────────────────────────────────

    def reason_parallel(
        self,
        concepts:    List[str],
        goal:        Optional[CognitiveGoal] = None,
        max_workers: int = 4,
    ) -> Dict[str, ReasoningTrace]:
        results: Dict[str, ReasoningTrace] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.reason, c, goal): c for c in concepts}
            for future in concurrent.futures.as_completed(futures):
                c = futures[future]
                try:
                    results[c] = future.result()
                except Exception as exc:
                    print(f"[Qmind] Parallel error on '{c}': {exc}")
        return results

    # ── Curiosity-driven autonomous expansion ──────────────────────────────

    def autonomous_explore(self, n_steps: int = 3) -> List[str]:
        """
        Run n_steps of curiosity-driven reasoning without an external query.
        Returns list of concepts explored.
        """
        explored = []
        gaps = self.curiosity.identify_gaps()
        targets = gaps[:n_steps] if gaps else self.wm.get_active(limit=n_steps)
        for concept in targets:
            if concept in self.kg.graph:
                self.reason(concept, goal=CognitiveGoal.DISCOVERY)
                explored.append(concept)
        return explored

    # ── Accessors ──────────────────────────────────────────────────────────

    def graph_stats(self) -> Dict:
        return self.kg.stats()

    def cognitive_health(self) -> Dict:
        return {
            "graph":         self.kg.stats(),
            "working_memory": len(self.wm._store),
            "episodic_traces": len(self.episodic._traces),
            "meta":          self.meta.reasoning_health(),
            "biases":        self.meta.detect_biases(),
            "curiosity_gaps": len(self.curiosity.identify_gaps()),
            "contradiction_hotspots": len(self.contra.find_contradicted_pairs()),
        }


# ══════════════════════════════════════════════════════════════════════════════
# DEMO GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_demo_graph(kg: KnowledgeGraphManager) -> None:
    print("[Demo] Building knowledge graph …")
    now = time.time()

    nodes = [
        ("Quantum Computing",    "physics",  "Computation using quantum-mechanical phenomena."),
        ("Superposition",        "physics",  "Existing in multiple states simultaneously."),
        ("Quantum Entanglement", "physics",  "Non-local correlation between quantum particles."),
        ("Qubit",                "physics",  "The quantum analogue of a classical bit."),
        ("Wave Function",        "physics",  "Mathematical description of a quantum state."),
        ("Measurement",          "physics",  "Collapsing a quantum state to a definite value."),
        ("Biology",              "biology",  "Scientific study of living organisms."),
        ("Cell",                 "biology",  "Fundamental structural and functional unit of life."),
        ("DNA",                  "biology",  "Molecule encoding genetic instructions."),
        ("Gene",                 "biology",  "Segment of DNA encoding a protein."),
        ("Protein",              "biology",  "Biomolecule performing cellular functions."),
        ("Evolution",            "biology",  "Change in heritable characteristics over generations."),
        ("Consciousness",        "neuroscience", "Subjective experience and awareness."),
        ("Neuron",               "neuroscience", "Nerve cell transmitting electrical signals."),
        ("Cognition",            "neuroscience", "Mental processes of acquiring knowledge."),
        ("Intelligence",         "general",  "Capacity to acquire and apply knowledge."),
        ("Information",          "general",  "Data with meaning and context."),
        ("Entropy",              "physics",  "Measure of disorder or uncertainty in a system."),
        ("Emergence",            "general",  "Complex patterns arising from simple interactions."),
        ("Complexity",           "general",  "System exhibiting non-trivial, organised behaviour."),
    ]

    for name, domain, info in nodes:
        kg.add_concept(name, confidence=0.95, domain=domain, info=info)

    edges = [
        ("Quantum Computing",    "Superposition",         "uses",           0.95, 0.95, 0.8),
        ("Quantum Computing",    "Qubit",                 "operates_on",    0.90, 0.92, 0.7),
        ("Quantum Computing",    "Quantum Entanglement",  "exploits",       0.85, 0.88, 0.8),
        ("Superposition",        "Wave Function",         "described_by",   0.92, 0.90, 0.6),
        ("Wave Function",        "Measurement",           "collapses_via",  0.90, 0.88, 0.5),
        ("Qubit",                "Quantum Entanglement",  "exhibits",       0.88, 0.85, 0.7),
        ("Quantum Entanglement", "Information",           "encodes",        0.80, 0.82, 0.6),
        ("Biology",              "Cell",                  "studies",        0.95, 0.95, 0.5),
        ("Cell",                 "DNA",                   "contains",       0.98, 0.98, 0.5),
        ("DNA",                  "Gene",                  "encodes",        0.95, 0.95, 0.5),
        ("Gene",                 "Protein",               "expresses",      0.90, 0.92, 0.5),
        ("Protein",              "Cell",                  "maintains",      0.85, 0.88, 0.6),
        ("Evolution",            "Gene",                  "acts_on",        0.92, 0.90, 0.7),
        ("Cell",                 "Neuron",                "specialises_into",0.85, 0.82, 0.5),
        ("Neuron",               "Consciousness",         "enables",        0.75, 0.72, 0.8),
        ("Neuron",               "Cognition",             "produces",       0.88, 0.85, 0.7),
        ("Cognition",            "Intelligence",          "is_a",           0.90, 0.88, 0.6),
        ("Intelligence",         "Complexity",            "is_a",           0.80, 0.78, 0.5),
        ("Complexity",           "Emergence",             "produces",       0.85, 0.82, 0.7),
        ("Emergence",            "Consciousness",         "may_cause",      0.55, 0.50, 0.6),
        ("Entropy",              "Information",           "inverse_of",     0.88, 0.85, 0.5),
        ("Information",          "Cognition",             "feeds",          0.85, 0.82, 0.6),
        ("Quantum Computing",    "Information",           "processes",      0.88, 0.85, 0.5),
        ("DNA",                  "Information",           "is_a",           0.90, 0.88, 0.4),
    ]

    for src, tgt, rel, w, cf, caus in edges:
        kg.add_edge(src, tgt, rel, w, cf, causality=caus)

    print(f"[Demo] Graph ready: {kg.stats()}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    GRAPH_FILE    = CFG["graph_file"]
    EPISODIC_FILE = CFG["episodic_file"]

    # ── Build demo graph if needed ─────────────────────────────────────────
    kg_temp = KnowledgeGraphManager(GRAPH_FILE)
    if kg_temp.graph.number_of_nodes() < 5:
        _build_demo_graph(kg_temp)
    del kg_temp

    # ── Initialise engine ──────────────────────────────────────────────────
    qmind = Qmind(graph_file=GRAPH_FILE, episodic_file=EPISODIC_FILE, seed=42)

    SEP = "═" * 70

    # ── 1. Graph statistics ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  GRAPH STATISTICS")
    print(SEP)
    for k, v in qmind.graph_stats().items():
        print(f"  {k:<28}: {v}")

    # ── 2. Query suite ─────────────────────────────────────────────────────
    queries = [
        ("What is Quantum Computing?",  None),
        ("What is DNA?",                None),
        ("What is Consciousness?",      CognitiveGoal.EXPLORE),
        ("What causes Emergence?",      CognitiveGoal.RESOLVE),
        ("Explain Intelligence",        CognitiveGoal.ABSTRACT),
        ("What is Dark Matter?",        None),            # unknown concept
    ]

    for q, g in queries:
        print(f"\n{SEP}")
        result = qmind.answer(q, goal=g)
        print(result["reasoning_trace"].summary() if result["reasoning_trace"] else f"  Status: {result['status']}\n  {result['answer']}")

    # ── 3. Parallel reasoning ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PARALLEL REASONING: Quantum Computing + Biology + Cognition")
    print(SEP)
    par = qmind.reason_parallel(
        ["Quantum Computing", "Biology", "Cognition"],
        goal=CognitiveGoal.DISCOVERY,
    )
    for concept, trace in par.items():
        top = trace.conclusions[0] if trace.conclusions else None
        print(f"  {concept:<25} → {top[0] if top else 'None':25} p={top[2]:.3f}" if top else f"  {concept}: no conclusion")

    # ── 4. Temporal simulation ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TEMPORAL PROPAGATION: 'DNA' trigger")
    print(SEP)
    props = qmind.temporal.simulate_propagation("DNA", steps=3)
    for concept, strength in props[:6]:
        print(f"  {concept:<30} activation={strength:.3f}")

    # ── 5. Contradiction check ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  CONTRADICTION ANALYSIS")
    print(SEP)
    pairs = qmind.contra.find_contradicted_pairs()
    print(f"  Active contradiction pairs: {len(pairs)}")

    # ── 6. Autonomous curiosity exploration ────────────────────────────────
    print(f"\n{SEP}")
    print("  AUTONOMOUS CURIOSITY EXPLORATION")
    print(SEP)
    explored = qmind.autonomous_explore(n_steps=2)
    print(f"  Explored: {explored}")

    # ── 7. Counterfactual simulation ───────────────────────────────────────
    print(f"\n{SEP}")
    print("  COUNTERFACTUAL: What if 'DNA' were removed?")
    print(SEP)
    cf = qmind.temporal.counterfactual("DNA")
    for concept, impact in cf[:5]:
        print(f"  {concept:<30} impact={impact:.3f}")

    # ── 8. Cognitive health ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  COGNITIVE HEALTH REPORT")
    print(SEP)
    health = qmind.cognitive_health()
    for k, v in health.items():
        print(f"  {k:<28}: {v}")

    # ── 9. Reinforcement & decay ───────────────────────────────────────────
    print(f"\n{SEP}")
    print("  REINFORCEMENT + DECAY DEMO")
    print(SEP)
    qmind.reinforce("DNA", "Gene", factor=0.15)
    decayed = qmind.decay(factor=0.95, threshold_s=0)
    print(f"  Decayed {decayed} edge(s).")

    # ── 10. Save episodic memory ───────────────────────────────────────────
    qmind.save_episodic()
    print(f"\n{SEP}")
    print("  Qmind v2 demo complete.")
    print(SEP)
