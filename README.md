# BUGFInv: Bayesian Uncertainty-Aware Gradient Fusion for 3D Subsurface Elastic Properties Inversion

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-red.svg)](https://pytorch.org/)
[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-green.svg)](https://www.python.org/)

BUGFInv is a Bayesian multi-task learning framework specifically designed for 3D subsurface elastic properties inversion. It implements uncertainty-aware gradient fusion that dynamically weights task contributions based on their predictive confidence, significantly improving inversion accuracy and robustness in multi-parameter estimation.

## 🔬 Conceptual Overview

BUGFInv implements a novel Bayesian uncertainty-aware gradient aggregation strategy for simultaneous estimation of three seismic parameters: P-wave velocity (Vp), S-wave velocity (Vs), and density (ρ). Unlike conventional fixed-weight or nonheuristic approaches, BUGFInv dynamically adjusts task contributions based on their predictive uncertainty, effectively resolving gradient conflicts and improving parameter estimation.

![Utility Function Exploration](https://github.com/Yingtian-Liu/BUGFInv/blob/main/Image/Conceptual%20comparison%20of%20multi-task%20learning%20gradient%20aggregation%20strategies%20in%20prestack%20three-parameter%20inversion.png)

### Key Features
- **Bayesian Uncertainty Quantification**: Estimates predictive uncertainty for each seismic parameter
- **Dynamic Gradient Fusion**: Automatically balances task contributions based on uncertainty
- **Closed-form Posterior Computation**: Efficient exact inference for Gaussian likelihood models
- **3D Subsurface Elastic Properties Inversion**: Optimized for simultaneous three-parameter estimation from seismic data

## 📦 Repository Structure
```
BUGFInv/
├── BUGFInv/                # Core package directory (source code)
│   ├── __init__.py         # Package initialization (expose core classes/functions for easy import)
│   ├── aggregation.py      # Core implementation of Bayesian uncertainty-aware gradient fusion algorithm
│   ├── bugfinv.py          # Main BUGFInv framework implementation (integrate all modules for end-to-end inversion)
│   ├── moments.py          # Calculation of first- and second-order moments for gradient distribution approximation via moment matching
│   ├── posterior.py        # Bayesian posterior inference for task-specific parameters (Gaussian prior/posterior computation)
│   ├── setup.py            # Subpackage installation configuration
│   └── util.py             # Auxiliary utilities (loss calculation, parameter validation, gradient processing helpers)
├── Image/                  # Project figures and conceptual diagrams (gradient aggregation comparison, framework structure, etc.)
└── README.md               # Project main documentation (installation, usage, technical details, citation)
```

## 🏗️ Installation

### Prerequisites
- Python 3.7 or higher
- CUDA-compatible GPU (optional, for accelerated training)
- Basic knowledge of seismic inversion and multi-task learning

### Create Conda Environment
```bash
conda create -n bugfinv python=3.8
conda activate bugfinv
```

### Install Dependencies
```bash
pip install torch==2.1.1 numpy==1.24.0 matplotlib==3.8.0 tqdm==4.67.1
pip install bruges==0.5.4 wget==3.2 python-dateutil==2.8.2
```

### Install BUGFInv
```bash
git clone https://github.com/Yingtian-Liu/BUGFInv.git
cd BUGFInv
pip install -e .
```

## 📋 Integration with Seismic Inversion Pipelines

### Basic Usage for 3D Elastic Properties Inversion

BUGFInv can be easily integrated into existing seismic inversion workflows for simultaneous estimation of Vp, Vs, and density:

```python
import torch
import numpy as np
from BUGFInv import BUGFInv

# Initialize BUGFInv for 3D subsurface elastic properties inversion
bugfinv = BUGFInv(
    num_tasks=3,  # Vp, Vs, Density
    n_outputs_per_task_group=[1, 1, 1],  # Each parameter has single output
    reg_hps={
        'gamma': 0.001,      # Prior precision (controls regularization strength)
        'sqrt_power': 1.0,   # Variance scaling parameter
        'obs_noise': 1.0     # Observation noise level
    }
)

# Example training loop for seismic inversion
for epoch in range(num_epochs):
    for seismic_batch, vp_true, vs_true, rhob_true in dataloader:
        # Forward pass through seismic inversion network
        shared_features = model.shared_encoder(seismic_batch)  # [batch, time, features]
        
        # Task-specific parameter estimation heads
        vp_pred = model.vp_head(shared_features)
        vs_pred = model.vs_head(shared_features)
        density_pred = model.density_head(shared_features)
        
        # Compute MSE losses for each parameter
        vp_loss = F.mse_loss(vp_pred, vp_true)
        vs_loss = F.mse_loss(vs_pred, vs_true)
        density_loss = F.mse_loss(density_pred, rhob_true)
        
        # Stack losses for multi-task optimization
        losses = torch.stack([vp_loss, vs_loss, density_loss])
        
        # Bayesian uncertainty-aware gradient fusion
        bugfinv.backward(
            losses=losses,
            last_layer_params=list(model.task_specific_parameters()),
            representation=shared_features,
            labels=[vp_true, vs_true, rhob_true],
            full_train_features=train_seismic_data,  # Optional: for full-data posterior
            full_train_labels=[train_vp, train_vs, train_rhob]  # Optional
        )
        
        # Update model parameters
        optimizer.step()
        optimizer.zero_grad()
```

### Integration with Existing Seismic Networks

If you're using the network architecture from the [Nash multi-task learning repository](https://github.com/Yingtian-Liu/Nash-multitask-learning-prestack-three-parameter-inversion), replace the Nash equilibrium gradient aggregation with BUGFInv:

**Before (Nash method):**
```python
# Conventional Nash equilibrium gradient aggregation
losses = torch.stack((vp_loss, vs_loss, density_loss))         
property_loss, _ = nash_method.backward(
    losses=losses,
    shared_parameters=list(model.shared_parameters()),
    task_specific_parameters=list(model.task_specific_parameters()),
    last_shared_parameters=list(model.last_shared_parameters()),
)
```

**After (BUGFInv method):**
```python
# Initialize BUGFInv for 3-parameter inversion
bugfinv = BUGFInv(
    num_tasks=3,
    n_outputs_per_task_group=[1, 1, 1],
    reg_hps={'gamma': 0.001, 'sqrt_power': 1.0, 'obs_noise': 1.0}
)

# Bayesian uncertainty-aware gradient fusion
losses = torch.stack([vp_loss, vs_loss, density_loss])
bugfinv.backward(
    losses=losses,
    last_layer_params=list(model.task_specific_parameters()),
    representation=shared_features,
    labels=[vp_true, vs_true, density_true],
    full_train_features=full_seismic_training_data,  # For better posterior estimation
    full_train_labels=[full_vp_labels, full_vs_labels, full_density_labels]
)
```

## 🎯 Performance and Validation

### Experimental Results

BUGFInv has been extensively validated on 3D synthetic datasets, demonstrating superior performance across all evaluation metrics:

### Quantitative Performance Comparison
The table below presents quantitative evaluation metrics comparing BUGFInv with three baseline methods for 3D subsurface elastic properties inversion:

| Method | PCC (Vp) | R² (Vs) | SSIM (ρ) | NRMSE (Overall) |
|--------|----------|---------|----------|-----------------|
| L-BFGS | 0.9768 | 0.9438 | 0.7967 | 0.1054 |
| CW | 0.9858 | 0.9774 | 0.9221 | 0.0596 |
| Nash | 0.9879 | 0.9840 | 0.9316 | 0.531 |
| **BUGFInv** | **0.9924** | **0.9895** | **0.9556** | **0.0348** |

## 🏭 Industrial Applications

BUGFInv is particularly suited for:

1. **Reservoir Characterization**: Simultaneous estimation of elastic parameters for lithology discrimination
2. **Fluid Detection**: Improved Vp/Vs ratio estimation for fluid identification
3. **Pore Pressure Prediction**: More accurate velocity models for geomechanical analysis
4. **4D Seismic Monitoring**: Robust parameter estimation for time-lapse analysis

## 📚 Technical Foundation

### Bayesian Uncertainty Estimation

For each seismic parameter θ ∈ {Vp, Vs, ρ}, BUGFInv computes the posterior distribution:

```
p(θ|D, X) = N(θ | μ_θ, Σ_θ)
```

Where:
- `μ_θ` is the posterior mean (point estimate)
- `Σ_θ` is the posterior covariance (uncertainty quantification)
- `D` represents seismic data
- `X` represents auxiliary features

### Gradient Fusion Mechanism

The uncertainty-aware gradient fusion follows:

```
Λ_t = Σ_t^{-1}           # Task-specific precision (inverse uncertainty)
w_t = Λ_t / Σ_{t'} Λ_t'  # Normalized task weights
∇L_total = Σ_t w_t ⊙ ∇L_t  # Fused gradient
```

This ensures that tasks with lower uncertainty (higher confidence) dominate the gradient updates, while uncertain tasks have reduced influence.

## 📝 Citation

If you use BUGFInv in your research or industrial applications, please cite:

```bibtex
@article{liu2026bugfinv,
  title={Bayesian Uncertainty-Aware Gradient Fusion for 3D Subsurface Elastic Properties Inversion},
  author={Yingtian Liu, Yong Li, Junheng Peng, Xiaowen Wang, Mingwei Wang, and Jianyong Xie
},
  journal={Arxiv},
  year={2026},
  note={Under Review}
}
```

## 🤝 Contributing

We welcome contributions from the geophysical and machine learning communities:

1. **Feature Requests**: Open an issue to discuss potential enhancements
2. **Bug Reports**: Help us improve reliability and performance
3. **Algorithm Extensions**: Extend BUGFInv to related geophysical inversion problems

### Development Workflow
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/improvement`)
3. Test your changes thoroughly
4. Submit a pull request with detailed documentation

## 📄 Copyright Notice

This code is the companion implementation of the research paper **"BUGFInv: Bayesian Uncertainty-Aware Gradient Fusion for 3D Subsurface Elastic Properties Inversion"**. The manuscript is currently under review in an academic journal.

Prior to the formal publication of the paper, the code is provided for academic research reference only. For any usage inquiries or collaboration interests, please contact the author through GitHub Issues.

## 📧 Contact and Support

For technical questions, collaboration opportunities, or consulting:

- **Primary Contact**: Yingtian Liu - [GitHub](https://github.com/Yingtian-Liu)
- **Repository**: [BUGFInv on GitHub](https://github.com/Yingtian-Liu/BUGFInv)
- **Documentation**: Detailed API documentation and tutorials available in the repository

## 🙏 Acknowledgments

- This research builds upon foundational work in Bayesian deep learning and seismic inversion
- We thank the open-source community for valuable tools and libraries
- Special appreciation to colleagues and reviewers for their constructive feedback
- The network architecture is based on our previous Nash multi-task learning framework: [GitHub repository](https://github.com/Yingtian-Liu/Nash-multitask-learning-prestack-three-parameter-inversion) | [Published paper](https://pubs.geoscienceworld.org/seg/geophysics/article-abstract/90/4/R175/654779/Nash-multitask-learning-semisupervised-temporal)

---

**Note**: The complete manuscript detailing the theoretical framework, algorithmic details, and comprehensive experimental validation is currently under review. The code provided here implements the core BUGFInv algorithm for Bayesian uncertainty-aware gradient fusion in 3D subsurface elastic properties inversion.




