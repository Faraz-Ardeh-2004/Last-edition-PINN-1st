# Physics-Informed Neural Networks (PINN) for PDE Solving

A comprehensive PyTorch-based implementation of Physics-Informed Neural Networks (PINNs) for solving partial differential equations (PDEs), with a focus on Laplace and Poisson equations.

## 📋 Overview

This repository provides a modular and flexible framework for training PINNs to solve various PDEs. The implementation includes:

- **Laplace Equation**: \(\nabla^2 u = 0\)
- **Poisson Equation**: \(\nabla^2 u = f(x, y)\)

The framework supports both forward problems (solving PDEs with known parameters) and inverse problems (parameter identification).

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

**PDE**: \(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0\)

**Boundary Conditions**: \(u(x, y) = x^3 - 3xy^2\) on all boundaries

**Domain**: \([-1, 1] \times [-1, 1]\)

### Poisson Equation

**PDE**: \(\nabla^2 u = f(x, y)\)

where \(f(x, y) = \sum_{k=1}^{4} \frac{1}{2}(-1)^{k+1} k^2 \sin(k\pi x)\sin(k\pi y)\)

**Boundary Conditions**: \(u = 0\) on all boundaries

**Analytical Solution**:
\[
u(x, y) = \frac{1}{4\pi^2} \sum_{k=1}^{4} \frac{(-1)^{k+1}}{k} \sin(k\pi x)\sin(k\pi y)
\]

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
    │   ├── *_comprehensive_*.png  (4-panel detailed analysis)
    │   ├── *_simple_*.png         (2x2 comparison)
    │   └── *_figure_*.png         (solution plots)
    └── Clock time.csv       # Training time statistics
```

### Visualization Types

1. **Comprehensive Results** (`*_comprehensive_*.png`):
   - MSE propagation curve
   - Ground truth solution
   - PINN prediction
   - Absolute error heatmap
   - Error distribution histogram
   - Relative error analysis
   - Statistical metrics

2. **Simple Comparison** (`*_simple_*.png`):
   - Loss curve
   - Ground truth
   - Prediction
   - Error map

3. **Loss Components**:
   - Total loss
   - Physics residual loss (`loss_f`)
   - Boundary condition loss (`loss_b`)
   - Data loss (`loss_d`) for inverse problems

## 🧪 Advanced Features

### Inverse Problems

For parameter identification, set `para_ctrl_add` in the configuration:

```csv
para_ctrl_add,1
para_ctrl,"1.0"  # Initial guess
```

The framework will automatically:
- Add parameters as trainable variables
- Track parameter evolution
- Generate parameter convergence plots

### Custom PDEs

To implement a new PDE:

1. Add configuration file in `Config/`
2. Implement PDE residual in `Training.py`:
   - Update `net_f()` for physics loss
   - Update `net_b()` for boundary conditions
   - Add analytical solution in `compute_ground_truth()`

### Multi-Model Comparison

Use `GroupVis.py` to compare different models:

```python
import Module.GroupVis as GroupVis

vis = GroupVis.Vis('Laplace', 'EXP', './Results/Laplace_EXP/')
vis.loss_read('PINN')
vis.loss_read('PINN_post')
vis.loss_vis()
```

## 📈 Training Details

### Loss Function Components

**Total Loss**:
\[
\mathcal{L} = \mathcal{L}_f + \mathcal{L}_b + \mathcal{L}_d + \mathcal{L}_{reg}
\]

- **Physics Loss** (\(\mathcal{L}_f\)): PDE residual at collocation points
- **Boundary Loss** (\(\mathcal{L}_b\)): Boundary condition violation
- **Data Loss** (\(\mathcal{L}_d\)): Supervised loss (inverse problems only)
- **Regularization** (\(\mathcal{L}_{reg}\)): L2 weight penalty

### Optimization Strategy

- **Optimizer**: Adam
- **Scheduler**: 
  - Laplace: ReduceLROnPlateau (adaptive)
  - Poisson: MultiStepLR (scheduled decay)
- **Gradient Clipping**: Applied to Laplace for stability
- **Batch Sampling**: Random sampling each iteration (configurable)

## 🔬 Experimental Notes

### Laplace Equation
- Uses lower learning rate (1e-4) for smooth convergence
- Adaptive scheduler responds to loss plateaus
- Gradient clipping prevents instability

### Poisson Equation
- Higher learning rate (1e-3) for faster convergence
- Adaptive loss weighting: progressively increase PDE weight
- No gradient clipping to preserve gradient information

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

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is available for academic and research purposes.

## 🙏 Acknowledgments

This implementation is based on the pioneering work on Physics-Informed Neural Networks:

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Note**: This framework is designed for research and educational purposes. For production use, additional validation and optimization may be required.