"""
Posterior distribution computation module for regression
"""

import torch
from typing import List, Tuple, Union
from .utils import psd_safe_cholesky


class LastLayerPosterior:
    """
    Base class for obtaining posterior distribution over final layer parameters
    """
    def __init__(self, gamma: float):
        self.gamma = gamma

    def prior(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def posterior(self, *args, **kwargs) -> torch.distributions:
        raise NotImplementedError


class LastLayerPosteriorRegression(LastLayerPosterior):
    """
    Exact inference for regression tasks with Gaussian likelihood
    """
    
    def __init__(self, num_outputs: int, gamma: float = 0.001, obs_noise: float = 1.0):
        super().__init__(gamma)
        self.num_outputs = num_outputs
        self.obs_noise = obs_noise
        self.full_data_posterior = None

    def set_full_data_posterior(self, p_t: torch.distributions):
        self.full_data_posterior = p_t

    def prior(self, features: torch.Tensor, output_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rep_dim = features.shape[-1] + 1  # +1 for bias
        prior_mean = torch.zeros((output_dim, rep_dim), dtype=features.dtype, device=features.device)
        prior_precision = (1 / self.gamma) * torch.eye(rep_dim, dtype=features.dtype, device=features.device)

        return prior_mean, prior_precision

    def posterior(self, features: torch.Tensor, labels: torch.Tensor,
                  prior_mean: torch.Tensor, prior_precision: torch.Tensor) -> torch.distributions:
        """
        Obtain posterior distribution according to standard Gaussian parametric regression.
        Assume independence between classes and tasks and same observation noise for all.
        
        Args:
            features: [bs, dim]
            labels: [bs, n_tasks * n_outputs]
            prior_mean: [n_tasks * n_outputs, dim]
            prior_precision: [dim, dim]
            
        Returns:
            Posterior distribution over tasks * outputs (under independence assumption)
        """
        h_L = features.detach().clone()
        rep_dim = h_L.shape[-1]

        device = h_L.device
        prior_mean = prior_mean.to(device)
        prior_precision = prior_precision.to(device)
        labels = labels.to(device)

        Λ_prior = prior_precision[:rep_dim, :rep_dim]
        Λ = Λ_prior + (1 / self.obs_noise) * (h_L.t() @ h_L)

        scale_tri_Λ = psd_safe_cholesky(Λ)
        Σ = torch.cholesky_solve(torch.eye(scale_tri_Λ.shape[-1],
                                           dtype=h_L.dtype, device=h_L.device), scale_tri_Λ).unsqueeze(0)
        scale_tri_Σ = psd_safe_cholesky(Σ)

        μ_prior = prior_mean[:, :rep_dim, None]
        w_mean = Σ @ ((Λ_prior.unsqueeze(0) @ μ_prior).squeeze(-1) +
                      ((1 / self.obs_noise) * h_L.t() @ labels).t()).unsqueeze(-1)

        p_t = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=w_mean.squeeze(-1),
            scale_tril=scale_tri_Σ
        )

        return p_t

    def compute_posterior(self, last_layer_params: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
                          features: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
                          labels: torch.Tensor, full_train_features: torch.Tensor = None,
                          full_train_labels: torch.Tensor = None) -> torch.distributions:
        output_dim = self.num_outputs
        if full_train_features is not None:
            full_data_prior_mean, full_data_prior_precision = \
                self.prior(full_train_features, output_dim)
            full_data_posterior = self.posterior(full_train_features, full_train_labels,
                                                 full_data_prior_mean, full_data_prior_precision)
            self.full_data_posterior = full_data_posterior
        prior_mean, prior_precision = self.full_data_posterior.mean, \
            self.full_data_posterior.precision_matrix[0, ...]

        p_t = self.posterior(features, labels,
                             prior_mean=prior_mean, prior_precision=prior_precision)

        return p_t