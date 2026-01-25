# BUGFInv: Bayesian Uncertainty-Guided Feature Inversion for Multi-task Regression

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-green.svg)](https://www.python.org/)

BUGFInv is a Bayesian multi-task regression framework that dynamically aggregates gradients based on task uncertainty. It provides an efficient solution for seismic inversion tasks by weighting task contributions according to their predictive confidence, significantly improving inversion accuracy and robustness.

## 🔬 Conceptual Overview

BUGFInv implements a novel Bayesian uncertainty-aware gradient aggregation strategy for multi-task learning in seismic inversion. Unlike conventional fixed-weight or heuristic approaches, BUGFInv dynamically adjusts task contributions based on their predictive uncertainty.

![Utility Function Exploration](https://github.com/Yingtian-Liu/BUGFInv/blob/main/Image/Conceptual%20comparison%20of%20multi-task%20learning%20gradient%20aggregation%20strategies%20in%20prestack%20three-parameter%20inversion.png)

### Key Features
- **Bayesian Uncertainty Estimation**: Quantifies prediction uncertainty for each regression task
- **Dynamic Gradient Weighting**: Automatically adjusts task contributions based on uncertainty
- **Exact Posterior Computation**: Uses closed-form solutions for Gaussian likelihood models
- **Multi-task Integration**: Simultaneously handles multiple related regression tasks

## 🏗️ Installation

### Prerequisites
- Python 3.7 or higher
- CUDA-compatible GPU (optional, for accelerated training)

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

## 📋 Usage Example

### Basic Integration

BUGFInv can be easily integrated into existing multi-task learning pipelines. Here's an example of how to replace Nash equilibrium-based gradient aggregation with BUGFInv:

```python
import torch
from BUGFInv import BUGFInv

# Initialize BUGFInv for 3 regression tasks (e.g., P-wave velocity, S-wave velocity, density)
bugfinv = BUGFInv(
    num_tasks=3,
    n_outputs_per_task_group=[1, 1, 1],  # Each task has 1 output
    reg_hps={
        'gamma': 0.001,      # Prior precision
        'sqrt_power': 1.0,   # Variance scaling
        'obs_noise': 1.0     # Observation noise
    }
)

# During training loop
for batch in dataloader:
    # Forward pass through shared network
    features = model.shared_layers(batch['input'])
    
    # Task-specific predictions
    vp_pred = model.vp_head(features)
    vs_pred = model.vs_head(features)
    rhob_pred = model.rhob_head(features)
    
    # Compute losses
    vp_loss = loss_fn(vp_pred, batch['vp_label'])
    vs_loss = loss_fn(vs_pred, batch['vs_label'])
    rhob_loss = loss_fn(rhob_pred, batch['rhob_label'])
    
    # Stack losses
    losses = torch.stack([vp_loss, vs_loss, rhob_loss])
    
    # Use BUGFInv for gradient aggregation
    bugfinv.backward(
        losses=losses,
        last_layer_params=list(model.task_specific_parameters()),
        representation=features,
        labels=[batch['vp_label'], batch['vs_label'], batch['rhob_label']],
        full_train_features=train_features,  # Optional: for full-data posterior
        full_train_labels=[train_vp, train_vs, train_rhob]  # Optional
    )
    
    # Update model parameters
    optimizer.step()
    optimizer.zero_grad()
```

### Integration with Existing Codebase

If you're using the network architecture from the [Nash multi-task learning repository](https://github.com/Yingtian-Liu/Nash-multitask-learning-prestack-three-parameter-inversion), replace the Nash equilibrium gradient aggregation with BUGFInv:

**Before (Nash method):**
```python
# Nash Equilibrium
losses = torch.stack((vp_train, vs_train, rhob_train))         
property_loss, _ = method.backward(
    losses=losses,
    shared_parameters=list(model_seam.shared_parameters()),
    task_specific_parameters=list(model_seam.task_specific_parameters()),
    last_shared_parameters=list(model_seam.last_shared_parameters()),
)
```

**After (BUGFInv method):**
```python
# Initialize BUGFInv
bugfinv = BUGFInv(
    num_tasks=3,
    n_outputs_per_task_group=[1, 1, 1],
    reg_hps={'gamma': 0.001, 'sqrt_power': 1.0, 'obs_noise': 1.0}
)

# Use BUGFInv for gradient aggregation
losses = torch.stack((vp_train, vs_train, rhob_train))
bugfinv.backward(
    losses=losses,
    last_layer_params=list(model_seam.task_specific_parameters()),
    representation=shared_features,
    labels=[vp_labels, vs_labels, rhob_labels],
    full_train_features=full_train_features,  # If available
    full_train_labels=[full_vp, full_vs, full_rhob]  # If available
)
```

## 🎯 Performance

BUGFInv has demonstrated superior performance in seismic inversion tasks:

- **Higher Accuracy**: Improved parameter estimation compared to conventional methods
- **Better Robustness**: More stable performance with limited training data
- **Uncertainty Quantification**: Provides confidence intervals for predictions
- **Noise Resilience**: Maintains performance under low signal-to-noise ratios

### Key Results from Manuscript

The method has been extensively validated on both synthetic and field datasets:

1. **Synthetic Data Experiments**: BUGFInv achieves higher Pearson Correlation Coefficient (PCC), R² scores, and Structural Similarity Index (SSIM) compared to L-BFGS, Constant Weight (CW), and Nash methods.

2. **Field Data Application**: In real-world seismic data from the study area, BUGFInv provides more accurate estimates of P-wave velocity, S-wave velocity, and density parameters.

3. **Robustness Analysis**: The method maintains performance even with reduced training wells (from 16 to 4) and under varying noise conditions (SNR from 20 dB to 2 dB).

## 📚 Method Details

### Bayesian Uncertainty Estimation

BUGFInv computes the posterior distribution of last-layer parameters using a Gaussian likelihood model:

```
p(w|D) ∝ p(D|w) × p(w)
```

Where:
- `p(w|D)` is the posterior distribution of parameters
- `p(D|w)` is the Gaussian likelihood
- `p(w)` is the Gaussian prior

### Gradient Aggregation

The framework weights task gradients based on their uncertainty:

```
Λ_g = 1 / Σ_g  # Inverse variance (uncertainty weight)
Λ_μ_g = Λ_g × μ_g  # Weighted gradient mean
dL_dh = (Σ_t Λ_μ_g_t) / (Σ_t Λ_g_t)  # Aggregated gradient
```

Tasks with lower uncertainty (higher confidence) receive larger weights in the gradient aggregation.

## 📊 Citation

If you use BUGFInv in your research, please cite our work:

```bibtex
@article{bugfinv2024,
  title={BUGFInv: Bayesian Uncertainty-Guided Feature Inversion for Multi-task Seismic Parameter Estimation},
  author={Liu, Yingtian and Collaborators},
  journal={Under Review},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or collaboration opportunities, please contact:
- **Yingtian Liu** - [GitHub](https://github.com/Yingtian-Liu)
- **Project Repository** - [BUGFInv on GitHub](https://github.com/Yingtian-Liu/BUGFInv)

## 🙏 Acknowledgments

- Thanks to the seismic inversion research community for their valuable insights
- Appreciation to the open-source projects that made this work possible
- Special thanks to the reviewers for their constructive feedback

---

**Note**: The full manuscript detailing the theoretical foundations and experimental results is currently under review. The network architecture used in conjunction with BUGFInv can be found in our previous repository: [Nash-multitask-learning-prestack-three-parameter-inversion](https://github.com/Yingtian-Liu/Nash-multitask-learning-prestack-three-parameter-inversion).
