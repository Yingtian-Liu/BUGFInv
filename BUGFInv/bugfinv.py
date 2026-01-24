"""
BUGFInv main class module for regression tasks
"""

import torch
from typing import Any, List, Tuple, Union, Dict
from .aggregation import GaussianAgg
from .moments import ExactMoments
from .posterior import LastLayerPosteriorRegression


class BUGFInv:
    """
    Main coordinator for multi-task Bayesian aggregation for regression
    """
    
    def __init__(self, num_tasks: int, n_outputs_per_task_group: Union[Tuple[int], List[int]],
                 agg_scheme_hps: dict = {}, reg_hps: dict = {}):
        """
        Main class for running Bayesian multi-task regression
        
        Args:
            num_tasks: Total number of regression tasks
            n_outputs_per_task_group: Sequence of number of outputs per task group
            agg_scheme_hps: Aggregation scheme hyperparameters
            reg_hps: Regression task hyperparameters
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.n_outputs_per_task_group = n_outputs_per_task_group
        
        # All tasks are regression tasks
        self.posterior_modules = []
        self.moments_modules = []
        
        for i in range(len(n_outputs_per_task_group)):
            self.posterior_modules.append(
                LastLayerPosteriorRegression(num_outputs=n_outputs_per_task_group[i],
                                             gamma=reg_hps.get('gamma', 0.001),
                                             obs_noise=reg_hps.get('obs_noise', 1.0))
            )
            self.moments_modules.append(
                ExactMoments(obs_noise=reg_hps.get('obs_noise', 1.0),
                             sqrt_power=reg_hps.get('sqrt_power', 1.0))
            )

        self.agg_scheme = GaussianAgg(num_tasks=self.num_tasks)

    @staticmethod
    def backward_last_layer(losses: torch.Tensor,
                            last_layer_params: Union[List[torch.nn.parameter.Parameter],
                                                     Tuple[torch.nn.parameter.Parameter]]):
        """
        Manually compute gradients for last layer parameters
        """
        grad = torch.autograd.grad(
            losses.sum(),
            last_layer_params,
            retain_graph=True,
            allow_unused=True
        )
        for p, g in zip(last_layer_params, grad):
            p.grad = g

    def backward(self, losses: torch.Tensor,
                 last_layer_params: Union[List[torch.nn.parameter.Parameter],
                                          Tuple[torch.nn.parameter.Parameter]],
                 representation: torch.Tensor, labels: Union[List[torch.Tensor], Tuple[torch.Tensor]],
                 **kwargs):
        """
        Set weighted gradient by task uncertainties w.r.t shared parameters
        
        Args:
            losses: Tensor of per-task loss
            last_layer_params: Parameters of last layer per task
            representation: Shared representation
            labels: Sequence of labels for regression tasks
        """
        self.backward_last_layer(losses, last_layer_params)
        
        task_probs = []
        features = torch.clone(representation).detach()
        E_g, Σ_g = [], []
        acc_num_out = 0
        
        for i, (post_module, moment_module) in enumerate(zip(self.posterior_modules, self.moments_modules)):
            num_output_items = self.n_outputs_per_task_group[i] * 2  # weights and biases for each output
            
            # Compute posterior distribution for this task
            p_t = post_module.compute_posterior(
                last_layer_params=last_layer_params[acc_num_out: acc_num_out + num_output_items],
                features=features,
                labels=labels[i],
                full_train_features=kwargs.get('full_train_features'),
                full_train_labels=kwargs.get('full_train_labels')[i] if isinstance(kwargs.get('full_train_labels'), list) else None
            )

            task_probs.append(p_t)
            
            # Compute moments for this task
            E_g_t, Σ_g_t = moment_module.compute_moments(features=features, labels=labels[i], p_t=p_t)

            E_g.append(E_g_t)
            Σ_g.append(Σ_g_t)
            acc_num_out += num_output_items

        μ_g = torch.cat(E_g, dim=1)
        Σ_g = torch.cat(Σ_g, dim=1)
        dL_dh = self.agg_scheme.aggregate(μ_g=μ_g, Σ_g=Σ_g)
        representation.backward(gradient=dL_dh.to(representation.dtype))