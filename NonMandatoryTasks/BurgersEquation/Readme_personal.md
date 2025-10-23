## 🚀 2D Burgers' Equation Solver with a Physics-Informed Neural Network (PINN)

While my understanding of advanced topics like **Adversarial PINNs** and **PGINNs (Physics-Informed Graph Neural Networks)—which are particularly effective for modeling systems with complex relational structures, such as particle interaction physics, molecular dynamics, and computational fluid dynamics on unstructured meshes**—is basic, this repository focuses on a standard PINN architecture for a known analytical solution.

***

## ⚙️ Model and Setup

### Governing Equation

The notebook targets the **2D Burgers' Equation**, a system of coupled, non-linear PDEs (representing momentum conservation with advection and diffusion):

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$
$$
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
$$

The network is trained to approximate the two velocity components, $u(x, y, t, \nu)$ and $v(x, y, t, \nu)$, simultaneously.

### Dataset

The model uses a **synthetic dataset** generated from the **exact analytical solution** (derived via the Cole–Hopf transform) of the 2D Burgers' equation:
* **Input ($\mathbf{X}$):** A 4-dimensional vector: $(x, y, t, \nu)$.
* **Output ($\mathbf{Y}$):** A 2-dimensional vector: $(u, v)$.
* The training is performed using a single viscosity value ($\nu = 0.01$).

### PINN Architecture

* **Model:** A simple **Fully Connected Neural Network (FCNN)** with multiple hidden layers.
    * Input Layer: 4 neurons ($x, y, t, \nu$).
    * Hidden Layers: 2 layers with **64 neurons** each, using a $\tanh$ activation function.
    * Output Layer: 2 neurons ($u, v$).

***

## 🎯 Physics-Informed Loss

The total loss function guides the network by combining the standard mean squared error (MSE) on the training data with the residual of the PDE.

$$
\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Data}} + \lambda \mathcal{L}_{\text{PDE}}
$$

1.  **Data Loss ($\mathcal{L}_{\text{Data}}$):** Measures the error between the network's prediction ($\hat{u}, \hat{v}$) and the true values ($u, v$) from the synthetic dataset.
2.  **PDE Loss ($\mathcal{L}_{\text{PDE}}$):** Measures how well the predicted solution satisfies the governing PDEs. The required spatial and temporal derivatives (e.g., $\frac{\partial u}{\partial t}$, $\frac{\partial^2 u}{\partial x^2}$) are computed using **PyTorch's Automatic Differentiation (Autograd)**.
    * In the provided code, the PDE loss is weighted by $\lambda = 0.1$.

### Training Analysis (as seen in `image_815600.png`)

The training converged successfully over **500 epochs** using the **Adam optimizer**:

* **Initial Phase:** The PDE Loss starts very high ($\approx 0.006$) before quickly dropping.
* **Learning Phase:** The **Data Loss** steadily decreases, and the **PDE Loss** stabilizes at a low value.
* **Final State:** The **Total Loss** converges to a low value ($\approx 0.001$), primarily driven by the **PDE Loss** ($\approx 0.002$ in the latter stages), which is a common characteristic of PINNs balancing data fit with physics enforcement.

***

## 📈 Evaluation

The model was evaluated on a test set with an **untrained viscosity parameter ($\nu = 0.02$)**.

* **Prediction vs. True Data:** A visual comparison between the true data (for $\nu=0.01$) and the predicted data (for $\nu=0.02$) shows a significant **underestimation of the velocity amplitude** in the predictions.
* **Interpretation:** The observed under-prediction of amplitude in the $\nu=0.02$ test set, relative to the $\nu=0.01$ training data, is **physically consistent**. Higher viscosity ($\nu$) generally leads to more dissipation and, thus, a lower magnitude of the velocity field.
