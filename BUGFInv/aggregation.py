"""
Gradient aggregation strategies module
"""

import torch
from typing import Tuple

class AggScheme:
    """
    Base class for gradient aggregation
    """
    def __init__(self):
        pass

    def aggregate(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class GaussianAgg(AggScheme):
    """
    Gaussian aggregation strategy: weight gradients by uncertainty (variance)
    """
    
    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks

    def aggregate(self, μ_g: torch.Tensor, Σ_g: torch.Tensor) -> torch.Tensor:
        """
        Aggregate across tasks assuming diagonal covariance
        
        Args:
            μ_g: Gradient mean [bs, num_tasks, dim]
            Σ_g: Gradient variance [bs, num_tasks, dim]
            
        Returns:
            Gradient of combined loss w.r.t shared hidden layer
        """
        Λ_g = 1 / Σ_g  # Inverse of variance (higher uncertainty → smaller weight)
        Λ_μ_g = Λ_g * μ_g  # Gradient mean × uncertainty weight

        bs = μ_g.shape[0]
        sum_inv_Λ_g = 1 / Λ_g.sum(dim=1)
        dL_dh = sum_inv_Λ_g * Λ_μ_g.sum(1) / bs  # Weighted sum

        return dL_dh