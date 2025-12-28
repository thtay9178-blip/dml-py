"""
Dynamic Peer Selection Strategies for Collaborative Learning

This module implements intelligent peer selection mechanisms that allow networks
to dynamically choose which peers to learn from during training. This is a novel
extension beyond standard DML where all networks learn from all peers equally.

Strategies:
- All Peers: Standard DML approach (baseline)
- Best Performing: Learn from top-k performing peers
- Diverse Selection: Learn from diverse peers (different predictions)
- Curriculum-Based: Start with best peers, gradually include all
- Confidence-Weighted: Weight peers by prediction confidence
- Dynamic Tournament: Networks compete for peer slots

Author: DML-PY Team
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class PeerSelectionConfig:
    """Configuration for peer selection strategies."""
    strategy: str = "all"  # all, best, diverse, curriculum, confidence, tournament
    k_peers: int = 2  # Number of peers to select (for best/diverse strategies)
    curriculum_start_k: int = 1  # Starting number of peers for curriculum
    curriculum_end_epoch: int = 50  # When to reach full peer set
    confidence_threshold: float = 0.8  # Minimum confidence for confident peers
    diversity_metric: str = "kl"  # kl, l2, cosine
    tournament_window: int = 10  # Episodes for tournament evaluation
    temperature: float = 1.0  # Temperature for soft selection


class PeerSelector(ABC):
    """Abstract base class for peer selection strategies."""
    
    def __init__(self, config: PeerSelectionConfig):
        self.config = config
        self.current_epoch = 0
        self.history: Dict = {
            'performance': [],  # Per-model performance history
            'selections': [],   # Selection history
            'diversity': []     # Diversity metrics
        }
    
    @abstractmethod
    def select_peers(
        self,
        model_idx: int,
        outputs: List[torch.Tensor],
        targets: torch.Tensor,
        metrics: Optional[Dict[int, float]] = None
    ) -> List[int]:
        """
        Select peer indices for a given model to learn from.
        
        Args:
            model_idx: Index of the model selecting peers
            outputs: List of model outputs (logits or probabilities)
            targets: Ground truth labels
            metrics: Optional dict mapping model_idx -> performance metric
            
        Returns:
            List of peer indices to learn from
        """
        pass
    
    def update(self, epoch: int, metrics: Dict[int, float]):
        """Update strategy state based on epoch and metrics."""
        self.current_epoch = epoch
        self.history['performance'].append(metrics.copy())
    
    def get_selection_mask(self, model_idx: int, num_models: int, peer_indices: List[int]) -> torch.Tensor:
        """Create a binary mask for selected peers."""
        mask = torch.zeros(num_models, dtype=torch.bool)
        mask[peer_indices] = True
        mask[model_idx] = False  # Don't learn from self
        return mask
    
    def get_selection_weights(
        self,
        model_idx: int,
        outputs: List[torch.Tensor],
        peer_indices: List[int]
    ) -> torch.Tensor:
        """Get weights for selected peers (uniform by default)."""
        weights = torch.ones(len(peer_indices))
        return weights / weights.sum()


class AllPeersSelector(PeerSelector):
    """Standard DML: learn from all peers equally."""
    
    def select_peers(
        self,
        model_idx: int,
        outputs: List[torch.Tensor],
        targets: torch.Tensor,
        metrics: Optional[Dict[int, float]] = None
    ) -> List[int]:
        """Select all peers except self."""
        num_models = len(outputs)
        return [i for i in range(num_models) if i != model_idx]


class BestPeersSelector(PeerSelector):
    """Learn from top-k best performing peers."""
    
    def select_peers(
        self,
        model_idx: int,
        outputs: List[torch.Tensor],
        targets: torch.Tensor,
        metrics: Optional[Dict[int, float]] = None
    ) -> List[int]:
        """Select k best performing peers based on validation metrics."""
        if metrics is None or len(metrics) == 0:
            # Fall back to all peers if no metrics available
            return [i for i in range(len(outputs)) if i != model_idx]
        
        # Sort models by performance (higher is better)
        sorted_models = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
        
        # Select top-k, excluding self
        k = min(self.config.k_peers, len(outputs) - 1)
        selected = []
        for idx, _ in sorted_models:
            if idx != model_idx:
                selected.append(idx)
            if len(selected) >= k:
                break
        
        return selected


class DiversePeersSelector(PeerSelector):
    """Learn from diverse peers with different predictions."""
    
    def select_peers(
        self,
        model_idx: int,
        outputs: List[torch.Tensor],
        targets: torch.Tensor,
        metrics: Optional[Dict[int, float]] = None
    ) -> List[int]:
        """Select k most diverse peers based on prediction differences."""
        num_models = len(outputs)
        if num_models <= 2:
            return [i for i in range(num_models) if i != model_idx]
        
        # Compute diversity scores for each potential peer
        my_output = outputs[model_idx]
        diversity_scores = []
        
        for i in range(num_models):
            if i == model_idx:
                diversity_scores.append(-float('inf'))  # Don't select self
            else:
                diversity = self._compute_diversity(my_output, outputs[i])
                diversity_scores.append(diversity)
        
        # Select k most diverse peers
        k = min(self.config.k_peers, num_models - 1)
        diverse_indices = np.argsort(diversity_scores)[-k:].tolist()
        
        return diverse_indices
    
    def _compute_diversity(self, output1: torch.Tensor, output2: torch.Tensor) -> float:
        """Compute diversity between two model outputs."""
        metric = self.config.diversity_metric
        
        if metric == "kl":
            # KL divergence (higher = more diverse)
            p1 = torch.softmax(output1, dim=-1)
            p2 = torch.softmax(output2, dim=-1)
            kl = (p1 * (p1.log() - p2.log())).sum(dim=-1).mean().item()
            return kl
        
        elif metric == "l2":
            # L2 distance (higher = more diverse)
            return torch.norm(output1 - output2, p=2).item()
        
        elif metric == "cosine":
            # Cosine dissimilarity (higher = more diverse)
            cos_sim = torch.cosine_similarity(output1, output2, dim=-1).mean().item()
            return 1.0 - cos_sim
        
        else:
            raise ValueError(f"Unknown diversity metric: {metric}")


class CurriculumPeersSelector(PeerSelector):
    """Curriculum-based: start with best peers, gradually include all."""
    
    def select_peers(
        self,
        model_idx: int,
        outputs: List[torch.Tensor],
        targets: torch.Tensor,
        metrics: Optional[Dict[int, float]] = None
    ) -> List[int]:
        """Select peers based on curriculum schedule."""
        num_models = len(outputs)
        max_peers = num_models - 1  # All except self
        
        # Compute current k based on curriculum schedule
        start_k = self.config.curriculum_start_k
        end_epoch = self.config.curriculum_end_epoch
        
        if self.current_epoch >= end_epoch:
            current_k = max_peers  # Learn from all
        else:
            # Linear schedule from start_k to max_peers
            progress = self.current_epoch / end_epoch
            current_k = int(start_k + (max_peers - start_k) * progress)
            current_k = max(1, min(current_k, max_peers))
        
        # Select top-k best peers
        if metrics is None or len(metrics) == 0:
            # Random selection if no metrics
            candidates = [i for i in range(num_models) if i != model_idx]
            return np.random.choice(candidates, size=min(current_k, len(candidates)), replace=False).tolist()
        
        # Select best performing peers
        sorted_models = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
        selected = []
        for idx, _ in sorted_models:
            if idx != model_idx:
                selected.append(idx)
            if len(selected) >= current_k:
                break
        
        return selected


class ConfidenceWeightedSelector(PeerSelector):
    """Weight peers by prediction confidence."""
    
    def select_peers(
        self,
        model_idx: int,
        outputs: List[torch.Tensor],
        targets: torch.Tensor,
        metrics: Optional[Dict[int, float]] = None
    ) -> List[int]:
        """Select all peers (weighting done in get_selection_weights)."""
        num_models = len(outputs)
        return [i for i in range(num_models) if i != model_idx]
    
    def get_selection_weights(
        self,
        model_idx: int,
        outputs: List[torch.Tensor],
        peer_indices: List[int]
    ) -> torch.Tensor:
        """Weight peers by their prediction confidence."""
        weights = []
        
        for peer_idx in peer_indices:
            # Compute confidence as max probability
            peer_output = outputs[peer_idx]
            probs = torch.softmax(peer_output, dim=-1)
            confidence = probs.max(dim=-1)[0].mean().item()
            
            # Apply confidence threshold
            if confidence < self.config.confidence_threshold:
                weight = 0.1  # Low weight for unconfident peers
            else:
                weight = confidence
            
            weights.append(weight)
        
        weights = torch.tensor(weights)
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones_like(weights) / len(weights)
        
        return weights


class TournamentSelector(PeerSelector):
    """Dynamic tournament: networks compete for peer slots."""
    
    def __init__(self, config: PeerSelectionConfig):
        super().__init__(config)
        self.tournament_scores = {}  # Track recent performance
        self.window_results = []  # Sliding window of results
    
    def select_peers(
        self,
        model_idx: int,
        outputs: List[torch.Tensor],
        targets: torch.Tensor,
        metrics: Optional[Dict[int, float]] = None
    ) -> List[int]:
        """Select peers based on tournament rankings."""
        if metrics is None or len(metrics) == 0:
            # Fall back to all peers
            return [i for i in range(len(outputs)) if i != model_idx]
        
        # Update tournament scores
        self.window_results.append(metrics.copy())
        if len(self.window_results) > self.config.tournament_window:
            self.window_results.pop(0)
        
        # Compute average scores in window
        avg_scores = {}
        for result in self.window_results:
            for idx, score in result.items():
                if idx not in avg_scores:
                    avg_scores[idx] = []
                avg_scores[idx].append(score)
        
        for idx in avg_scores:
            avg_scores[idx] = np.mean(avg_scores[idx])
        
        # Select top performers
        sorted_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        k = min(self.config.k_peers, len(outputs) - 1)
        
        selected = []
        for idx, _ in sorted_models:
            if idx != model_idx:
                selected.append(idx)
            if len(selected) >= k:
                break
        
        return selected


def create_peer_selector(config: PeerSelectionConfig) -> PeerSelector:
    """Factory function to create peer selector based on config."""
    strategy_map = {
        'all': AllPeersSelector,
        'best': BestPeersSelector,
        'diverse': DiversePeersSelector,
        'curriculum': CurriculumPeersSelector,
        'confidence': ConfidenceWeightedSelector,
        'tournament': TournamentSelector,
    }
    
    if config.strategy not in strategy_map:
        raise ValueError(f"Unknown peer selection strategy: {config.strategy}")
    
    return strategy_map[config.strategy](config)


class PeerSelectionAnalyzer:
    """Tools for analyzing peer selection behavior."""
    
    @staticmethod
    def compute_selection_statistics(history: Dict) -> Dict:
        """Compute statistics from selection history."""
        stats = {
            'avg_peers_per_model': 0,
            'selection_diversity': 0,
            'selection_stability': 0,
        }
        
        if not history.get('selections'):
            return stats
        
        # Analyze selection patterns
        selections = history['selections']
        
        # Average number of peers selected
        total_peers = sum(len(s) for batch in selections for s in batch.values())
        total_selections = sum(len(batch) for batch in selections)
        if total_selections > 0:
            stats['avg_peers_per_model'] = total_peers / total_selections
        
        # Selection diversity (unique peer combinations)
        unique_combos = set()
        for batch in selections:
            for model_idx, peers in batch.items():
                unique_combos.add(tuple(sorted(peers)))
        stats['selection_diversity'] = len(unique_combos)
        
        # Selection stability (how often selections change)
        if len(selections) > 1:
            changes = 0
            for i in range(len(selections) - 1):
                for model_idx in selections[i]:
                    if model_idx in selections[i+1]:
                        if set(selections[i][model_idx]) != set(selections[i+1][model_idx]):
                            changes += 1
            stats['selection_stability'] = 1.0 - (changes / (len(selections) - 1))
        
        return stats
    
    @staticmethod
    def visualize_peer_network(
        selections: Dict[int, List[int]],
        metrics: Dict[int, float],
        save_path: Optional[str] = None
    ):
        """Visualize peer selection as a directed graph."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            print("matplotlib and networkx required for visualization")
            return
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes (models)
        for model_idx, performance in metrics.items():
            G.add_node(model_idx, performance=performance)
        
        # Add edges (peer selections)
        for model_idx, peers in selections.items():
            for peer_idx in peers:
                G.add_edge(model_idx, peer_idx)
        
        # Draw graph
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        
        # Color nodes by performance
        node_colors = [metrics.get(node, 0) for node in G.nodes()]
        
        nx.draw(
            G, pos,
            node_color=node_colors,
            cmap='RdYlGn',
            with_labels=True,
            node_size=1000,
            font_size=12,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            alpha=0.7
        )
        
        plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn'), label='Performance')
        plt.title('Peer Selection Network\n(Arrows: who learns from whom)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Peer Selection Strategies...")
    
    # Create dummy data
    num_models = 5
    batch_size = 32
    num_classes = 10
    
    outputs = [torch.randn(batch_size, num_classes) for _ in range(num_models)]
    targets = torch.randint(0, num_classes, (batch_size,))
    metrics = {i: np.random.rand() for i in range(num_models)}
    
    strategies = ['all', 'best', 'diverse', 'curriculum', 'confidence', 'tournament']
    
    for strategy_name in strategies:
        print(f"\n{'='*60}")
        print(f"Testing {strategy_name.upper()} strategy")
        print('='*60)
        
        config = PeerSelectionConfig(strategy=strategy_name, k_peers=2)
        selector = create_peer_selector(config)
        
        for model_idx in range(num_models):
            peers = selector.select_peers(model_idx, outputs, targets, metrics)
            weights = selector.get_selection_weights(model_idx, outputs, peers)
            
            print(f"Model {model_idx} (perf={metrics[model_idx]:.3f})")
            print(f"  Selected peers: {peers}")
            print(f"  Peer weights: {[f'{w:.3f}' for w in weights.tolist()]}")
    
    print("\n" + "="*60)
    print("All peer selection strategies tested successfully!")
    print("="*60)
