"""
Gradient moments computation module for regression
"""

import torch
from typing import Tuple


class Moments:
    """
    Base class for computing first and second moments of loss gradient w.r.t hidden layer
    """
    def __init__(self):
        pass

    def first_moment(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def second_moment(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def compute_moments(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class ExactMoments(Moments):
    """
    Closed-form moments computation for regression tasks with Gaussian parameter distribution
    """
    
    def __init__(self, obs_noise: float = 1.0, sqrt_power: float = 1.0):
        super().__init__()
        self.obs_noise = obs_noise
        self.sqrt_power = sqrt_power

    def first_moment(self, features: torch.Tensor, labels: torch.Tensor, 
                     E_w: torch.Tensor, E_ww: torch.Tensor) -> torch.Tensor:
        E_ww_h = torch.einsum('ode,bd->boe', E_ww, features)
        E_w_y = torch.einsum('bo,od->bod', labels, E_w)
        E_g = 2 * (E_ww_h - E_w_y)
        return E_g

    def second_moment(self, labels: torch.Tensor, E_ww: torch.Tensor,
                      E_wxww: torch.Tensor, E_wxwxww: torch.Tensor) -> torch.Tensor:
        E_ww_y_2 = torch.einsum('bo,ode->bode', labels ** 2, E_ww)
        E_wxww_y = torch.einsum('bo,bode->bode', labels, E_wxww)
        E_g_g = (2 ** 2) * (E_ww_y_2 - 2 * E_wxww_y + E_wxwxww)
        return E_g_g

    def compute_moments(self, features: torch.Tensor, labels: torch.Tensor,
                        p_t: torch.distributions) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assume independence between all outputs
        
        Args:
            features: Hidden layer representation [bs, dim]
            labels: Labels for each task and output [bs, n_tasks * n_outputs]
            p_t: Gaussian distribution over parameters [n_tasks * n_outputs, dim]
            
        Returns:
            Expected value and covariance of gradient
        """
        rep_dim = features.shape[-1]
        h_L = features.detach().clone()

        μ = p_t.mean[:, :rep_dim]
        Σ = p_t.covariance_matrix[:, :rep_dim, :rep_dim]

        μμ = torch.einsum('od,oe->ode', μ, μ)
        Σ_μμ = Σ + μμ
        Σ_μμ_neg = Σ - μμ

        E_w = μ
        E_ww = Σ_μμ

        μ_h = torch.einsum('od,be->obde', μ, h_L).permute(1, 0, 2, 3)
        μ_h_Σ_μμ = torch.einsum('bode,oef->bodf', μ_h, Σ_μμ)
        Σ_μμ_μ_h = torch.einsum('ode,bofe->bodf', Σ_μμ, μ_h)
        h_μ_Σ_μμ_neg = torch.einsum('bd,od,oef->boef', h_L, μ, Σ_μμ_neg)
        E_wxww = μ_h_Σ_μμ + Σ_μμ_μ_h + h_μ_Σ_μμ_neg

        hh = torch.einsum('bd,be->bde', h_L, h_L)
        hh_hh = hh + hh.permute(0, 2, 1)
        Σ_μμ_hh_hh_Σ_μμ = torch.einsum('ode,bef,ofg->bodg', Σ_μμ, hh_hh, Σ_μμ)
        μ_hh_μ_Σ_μμ_neg = torch.einsum('od,bde,oe,ofg->bofg', μ, hh, μ, Σ_μμ_neg)
        hhΣ = torch.einsum('bdf,ofe->bode', hh, Σ)
        tr_hhΣ = torch.diagonal(hhΣ, dim2=-2, dim1=-1).sum(-1)
        tr_hhΣ_Σ_μμ = torch.einsum('bo,ode->bode', tr_hhΣ, Σ_μμ)
        E_wxwxww = Σ_μμ_hh_hh_Σ_μμ + μ_hh_μ_Σ_μμ_neg + tr_hhΣ_Σ_μμ

        E_g = self.first_moment(h_L, labels, E_w, E_ww)
        E_g_g = self.second_moment(labels, E_ww, E_wxww, E_wxwxww)

        E_gE_g = torch.einsum('bod,boe->bode', E_g, E_g)
        Σ_g = E_g_g - E_gE_g
        Σ_g = torch.clamp(torch.diagonal(Σ_g.detach(), dim1=-2, dim2=-1), min=1e-8)
        Σ_g = Σ_g ** self.sqrt_power

        return (E_g, Σ_g)