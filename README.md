# Physics-Informed Neural Networks (PINN) for PDE Solving

A comprehensive PyTorch-based implementation of Physics-Informed Neural Networks (PINNs) for solving partial differential equations (PDEs), with a focus on Laplace and Poisson equations.

## 📋 Overview

This repository provides a modular and flexible framework for training PINNs to solve various PDEs. The implementation includes:

- **Laplace Equation**: ∇²u = 0
- **Poisson Equation**: ∇²u = f(x, y)

The framework supports both forward problems (solving PDEs with known parameters) and inverse problems (parameter identification).

## ⚠️ Current Status & Development Roadmap

### Phase 1: Initial Implementation (Current) ✅

**Completed:**
- ✅ Basic PINN architecture implementation
- ✅ Laplace equation solver (working correctly)
- ✅ Comprehensive visualization system
- ✅ CSV-based configuration
- ✅ GPU acceleration support

**Known Issues:**
### Phase 2: Advanced Optimization & Paper Implementation (Current Target) 🎯

**Primary Objectives:**
1. **Fix Poisson Equation Convergence** 
2. **Complete Full Paper Implementation**
   - Implement all methodologies from the Ψ-NN paper
   - Replicate experimental results
   - Validate against paper benchmarks

**Active Research Tasks:**
- 🔬 **Investigate Poisson equation convergence issues**
  - Analyze loss landscape and gradient flow patterns
  - Profile gradient magnitudes across training
  - Test alternative activation functions (Swish, GELU, adaptive)
  - Experiment with different network architectures (deeper, wider, residual)
  - Study collocation point distribution effects
  
- 🔧 **Optimization Methods to Implement:**
  - **L-BFGS Optimizer**: Second-order optimization for better convergence
    - Two-stage training: Adam (warm-up) → L-BFGS (fine-tuning)
    - Full-batch requirement and memory management
  - **Adaptive Activation Functions**: Self-tuning activation strategies
    - Learnable activation parameters
    - Dynamic activation selection
  - **Multi-scale Training**: Progressive refinement approach
    - Coarse-to-fine grid refinement
    - Hierarchical collocation strategy
  - **Loss Balancing**: Dynamic weighting schemes
    - Gradient-based balancing
    - Adaptive loss coefficients
  - **Curriculum Learning**: Gradual increase in problem complexity
    - Progressive source term complexity
    - Staged boundary condition enforcement

### Phase 3: Advanced Models Implementation (Planned) 📅

**Target: Complete Ψ-NN Framework**

**Models to Implement:**
- **PINN-Post**: Post-processing enhancement methods
  - Solution refinement techniques
  - Error correction mechanisms
  - Multi-fidelity approaches
  
- **Ψ-NN Architecture**: Full framework from paper
  - Core Ψ-NN methodology
  - Novel loss formulations
  - Advanced training strategies
  
- **Ablation Studies**: Systematic


## 🚀 Features

- **Modular Architecture**: Clean separation between training, visualization, and neural network components
- **Automatic Differentiation**: Leverages PyTorch's autograd for computing physics-based loss functions
- **Comprehensive Visualization**: Generates detailed plots including:
  - Loss propagation curves
  - Ground truth vs. predictions
  - Absolute and relative error analysis
  - Error distribution histograms
- **Flexible Configuration**: CSV-based configuration for easy parameter tuning
- **GPU Support**: Automatic GPU detection and utilization
- **Adaptive Learning**: Problem-specific learning rate scheduling
- **Batch Sampling**: Configurable batch training for large-scale problems

## 📁 Repository Structure

```
Last-edition-PINN-1st/
├── Panel.py                 # Main execution script
├── requirements.txt         # Python dependencies
├── Config/                  # Configuration files
│   ├── Laplace_EXP.csv     # Laplace equation settings
│   └── Poisson_EXP.csv     # Poisson equation settings
└── Module/                  # Core modules
    ├── PINN.py             # Neural network architecture
    ├── Training.py         # Training logic and PDE residuals
    ├── SingleVis.py        # Single model visualization
    └── GroupVis.py         # Multi-model comparison
```

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Faraz-Ardeh-2004/Last-edition-PINN-1st.git
cd Last-edition-PINN-1st
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `numpy`
- `pandas`
- `torch`
- `matplotlib`

## 🎯 Quick Start

### Running Examples

Execute the main script to train PINN models:

```bash
python Panel.py
```

By default, this will:
1. Train a PINN for the Laplace equation
2. Train a PINN for the Poisson equation

### Configuration

Modify the `Panel.py` script to select specific problems:

```python
import Module.Training as Training
import torch

torch.random.manual_seed(1234)

# Train Laplace equation solver
task_1 = Training.model('Laplace', 'EXP')
task_1.train()

# Train Poisson equation solver
task_2 = Training.model('Poisson', 'EXP')
task_2.train()
```

## ⚙️ Configuration Files

Configuration files are stored in `Config/` directory with the naming convention: `{ProblemName}_{ExperimentID}.csv`

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `x_min`, `x_max` | Domain boundaries in x | -1, 1 |
| `y_min`, `y_max` | Domain boundaries in y | -1, 1 |
| `grid_node_num` | Grid resolution | 101 |
| `node_num` | Neurons per layer | 8 |
| `hidden_layers_group` | Layer configuration | "1,1,2" |
| `train_steps` | Training iterations | 80000 |
| `learning_rate` | Initial learning rate | 1e-4 (Laplace), 1e-3 (Poisson) |
| `batch_size` | Training batch size | 2000 |
| `regularization_state` | Enable L2 regularization | 0 or 1 |

## 🧮 Problem Formulations

### Laplace Equation

**PDE**: 

```
∂²u/∂x² + ∂²u/∂y² = 0
```

**Boundary Conditions**: 

```
u(x, y) = x³ - 3xy²  on all boundaries
```

**Domain**: [-1, 1] × [-1, 1]

**Analytical Solution**:
```
u(x, y) = x³ - 3xy²
```

**Status**: ✅ Working correctly with excellent convergence

### Poisson Equation

**PDE**: 

```
∇²u = f(x, y)
```

where the source term is:

```
f(x, y) = Σ(k=1 to 4) [1/2 · (-1)^(k+1) · k² · sin(kπx) · sin(kπy)]
```

**Boundary Conditions**: 

```
u = 0  on all boundaries
```

**Domain**: [-1, 1] × [-1, 1]

**Analytical Solution**:

```
u(x, y) = 1/(4π²) · Σ(k=1 to 4) [(-1)^(k+1)/k · sin(kπx) · sin(kπy)]
```

**Status**: ⚠️ Under investigation - convergence issues detected

## 📊 Results and Plots

After training, results are automatically saved to `Results/{ProblemName}_{ExperimentID}/`:

```
Results/
└── Laplace_EXP/
    ├── Models/              # Saved model checkpoints
    │   └── Laplace_EXP_PINN.pth
    ├── Loss/                # Training loss data
    │   ├── Laplace_EXP_loss_PINN.csv
    │   └── *.png (loss curves)
    ├── Figure/              # Visualization outputs
    │   ├── *_comprehensive_*.png  (8-panel detailed analysis)
    │   ├── *_simple_*.png         (2x2 comparison)
    │   └── *_figure_*.png         (solution plots)
    ├── Parameters/          # Inverse problem results
    │   └── *_paras_PINN.csv
    └── Clock time.csv       # Training time statistics
```

### Visualization Types

#### 1. Comprehensive Results (*_comprehensive_*.png)

Eight-panel visualization including:
- **Panel 1**: MSE propagation curve (training loss over iterations)
- **Panel 2**: Ground truth solution
- **Panel 3**: PINN prediction
- **Panel 4**: Absolute error heatmap
- **Panel 5**: Loss components (physics, boundary, data)
- **Panel 6**: Error distribution histogram
- **Panel 7**: Relative error heatmap
- **Panel 8**: Statistical error metrics

#### 2. Simple Comparison (*_simple_*.png)

Four-panel comparison:
- Loss curve
- Ground truth
- PINN prediction
- Absolute error map

#### 3. Loss Components

Individual plots for each loss term:
- **Total loss**: Overall training objective
- **Physics residual loss (loss_f)**: PDE satisfaction at collocation points
- **Boundary condition loss (loss_b)**: BC violation at domain boundaries
- **Data loss (loss_d)**: Supervised error (inverse problems only)
- **Regularization loss (loss_rgl)**: L2 weight penalty

## 🧪 Advanced Features

### Inverse Problems

For parameter identification, configure in CSV file:

```csv
para_ctrl_add,1
para_ctrl,"1.0"  # Initial guess for unknown parameter
```

The framework will:
- Add parameters as trainable variables
- Track parameter evolution during training
- Generate parameter convergence plots in `Results/Parameters/`

### Custom PDEs

To implement a new PDE:

1. **Create configuration file** in `Config/` (e.g., `NewPDE_EXP.csv`)

2. **Implement PDE residual** in `Training.py`:
   - Update `net_f()` method for physics loss
   - Update `net_b()` method for boundary conditions
   - Add analytical solution in `compute_ground_truth()` method

3. **Run training**:
   ```python
   task = Training.model('NewPDE', 'EXP')
   task.train()
   ```

### Multi-Model Comparison

Use `GroupVis.py` to compare different models:

```python
import Module.GroupVis as GroupVis

vis = GroupVis.Vis('Laplace', 'EXP', './Results/Laplace_EXP/')
vis.loss_read('PINN')
vis.loss_read('PINN_post')
vis.loss_vis()
```

This generates comparison plots for multiple training runs.

## 📈 Training Details

### Loss Function Components

**Total Loss**:

```
L_total = L_physics + L_boundary + L_data + L_regularization
```

Where:
- **L_physics**: PDE residual at collocation points
  ```
  L_physics = mean[(∇²u - f)²]
  ```

- **L_boundary**: Boundary condition violation
  ```
  L_boundary = mean[(u_predicted - u_boundary)²]
  ```

- **L_data**: Supervised loss for inverse problems
  ```
  L_data = mean[(u_predicted - u_measured)²]
  ```

- **L_regularization**: L2 weight penalty
  ```
  L_reg = λ · Σ||W||²
  ```

### Optimization Strategy

| Component | Laplace | Poisson |
|-----------|---------|---------|
| **Optimizer** | Adam | Adam |
| **Initial LR** | 1e-4 | 1e-3 |
| **Scheduler** | ReduceLROnPlateau | MultiStepLR |
| **Gradient Clipping** | Yes (max_norm=1.0) | No |
| **Batch Sampling** | Random (2000 points) | Full domain |
| **Loss Weighting** | Static | Adaptive (progressive) |

### Adaptive Loss Weighting (Poisson Only)

For Poisson equation, the framework uses progressive loss weighting:

```python
pde_weight = 1.0 + min(9.0, iteration / (total_steps * 0.1))
boundary_weight = max(1.0, 10.0 - iteration / (total_steps * 0.05))
```

This prioritizes boundary conditions early, then shifts focus to PDE satisfaction.

## 🔬 Experimental Notes

### Laplace Equation
- **Learning Rate**: Lower (1e-4) for smooth, stable convergence
- **Scheduler**: Adaptive (ReduceLROnPlateau) responds to loss plateaus
- **Gradient Clipping**: Prevents gradient explosion in smooth regions
- **Training Strategy**: Emphasizes smoothness and boundary accuracy
- **Results**: Excellent convergence with low error across the domain

### Poisson Equation (Current Issues)
- **Learning Rate**: Higher (1e-3) for faster convergence
- **Scheduler**: Scheduled decay (MultiStepLR) at milestones
- **Adaptive Weighting**: Progressive shift from BC to PDE
- **No Gradient Clipping**: Preserves full gradient information for source term
- **Training Strategy**: Balances source term fitting with boundary conditions
- **Known Issues**: 
  - Poor convergence observed
  - High training error persists
  - Requires further investigation

### Suggested Improvements for Poisson Equation

Based on PINN literature and best practices:

1. **L-BFGS Optimizer**:
   - Second-order optimization method
   - Better for smooth problems with source terms
   - Requires full-batch training

2. **Network Architecture**:
   - Deeper networks (4-5 hidden layers)
   - Wider layers (32-64 neurons)
   - Alternative activations (Swish, GELU)

3. **Training Strategy**:
   - Two-stage training (Adam → L-BFGS)
   - Curriculum learning (simple → complex)
   - Higher resolution collocation points

4. **Loss Balancing**:
   - Separate learning rates for boundary and PDE terms
   - Normalized gradients for each loss component
   - Attention-based weighting

## 🎓 Architecture Details

### Neural Network Structure

The PINN architecture uses a fully-connected feedforward network:

```python
Input Layer (2D coordinates: x, y)
    ↓
Hidden Layer 1 (node_num neurons) + Tanh activation
    ↓
Hidden Layer 2 (node_num neurons) + Tanh activation
    ↓
Hidden Layer 3 (2 × node_num neurons) + Tanh activation
    ↓
Output Layer (1 output: u)
```

Default configuration:
- **Input**: 2 (x, y coordinates)
- **Hidden layers**: [8, 8, 16] neurons
- **Output**: 1 (solution u)
- **Activation**: Tanh (smooth, differentiable)

### Collocation Points

Training points are sampled on a uniform grid:
- **Grid resolution**: 101 × 101 = 10,201 points (default)
- **Batch size**: 2000 points per iteration (Laplace)
- **Full batch**: All points used (Poisson)
- **Boundary points**: 1000 points per boundary edge

## 📊 Performance Metrics

The framework computes and visualizes:

1. **Mean Squared Error (MSE)**:
   ```
   MSE = mean[(u_predicted - u_true)²]
   ```

2. **Maximum Absolute Error**:
   ```
   Max Error = max|u_predicted - u_true|
   ```

3. **Mean Absolute Error**:
   ```
   MAE = mean|u_predicted - u_true|
   ```

4. **Relative Error**:
   ```
   Relative Error = |u_predicted - u_true| / (|u_true| + ε)
   ```

5. **Standard Deviation of Error**:
   ```
   Std Error = std(|u_predicted - u_true|)
   ```

All metrics are computed on a high-resolution grid (200 × 200 points by default) and displayed in the comprehensive visualization.

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `batch_size` in configuration
- Reduce `grid_node_num` for training
- Use CPU: Set `device = torch.device('cpu')` in `Training.py`

**2. Training Instability**
- Lower learning rate
- Enable regularization: Set `regularization_state = 1`
- Increase gradient clipping threshold

**3. Poor Convergence (Especially Poisson)**
- Increase `train_steps` (try 200k-500k iterations)
- Adjust learning rate schedule milestones
- Consider switching to L-BFGS optimizer
- Try deeper/wider networks
- Increase collocation points

**4. High Error at Boundaries**
- Increase `bun_node_num`
- Adjust boundary loss weight in `net_b()`
- Verify boundary condition formulation

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{pinn_implementation,
  author = {Faraz-Ardeh-2004},
  title = {Physics-Informed Neural Networks for PDE Solving},
  year = {2024},
  url = {https://github.com/Faraz-Ardeh-2004/Last-edition-PINN-1st}
}
```

## 📚 References

This implementation is based on the following works:

1. **Ψ-NN Framework**: Liu, Z., et al. (2025). "Ψ-NN: A physics-informed neural network framework." *Nature Communications*. [https://www.nature.com/articles/s41467-025-64624-3](https://www.nature.com/articles/s41467-025-64624-3)

2. **Original Ψ-NN Repository**: [https://github.com/ZitiLiu/Psi-NN](https://github.com/ZitiLiu/Psi-NN/tree/main)

3. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.

4. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). "Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations." *arXiv preprint arXiv:1711.10561*.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs via GitHub Issues
- Suggest new features
- Submit pull requests
- Improve documentation
- Help investigate Poisson equation convergence issues

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Priority Areas for Contribution

- 🔴 **High Priority**: Poisson equation convergence improvement
- 🟡 **Medium Priority**: L-BFGS optimizer implementation
- 🟢 **Low Priority**: Additional PDE examples

## 📄 License

This project is available for academic and research purposes.

## 🙏 Acknowledgments

- PyTorch team for the automatic differentiation framework
- Physics-Informed Machine Learning community
- Ψ-NN framework authors
- Contributors and users of this repository

## 📧 Contact

For questions, collaborations, or bug reports:
- Open an issue on GitHub
- Repository: https://github.com/Faraz-Ardeh-2004/Last-edition-PINN-1st

---

**Note**: This framework is currently in Phase 1 development. The Laplace equation solver is production-ready, while the Poisson equation solver requires further research and optimization. Contributions and suggestions are highly welcome!

## 🔄 Version History

- **v1.0** (2024): Initial release - Phase 1
  - ✅ Laplace equation solver (working)
  - ⚠️ Poisson equation solver (under development)
  - ✅ Comprehensive visualization system
  - ✅ CSV-based configuration
  - ✅ GPU acceleration support
