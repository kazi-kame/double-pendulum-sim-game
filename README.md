Double Pendulum Chaos Simulator

A real-time physical simulation of a double pendulum system built with Python and Pygame. This application visualizes chaotic motion and quantifies chaos using a numerical estimation of the largest Lyapunov exponent.

Features

Physics Engine: Uses Lagrangian mechanics solved via 4th-order Runge-Kutta (RK4) integration for high numerical stability.

Chaos Analysis: Real-time calculation of the Largest Lyapunov Exponent (LLE) using the shadow trajectory method (Wolf algorithm).

Interactive Controls: Adjust lengths, masses, gravity, and damping coefficients on the fly.

Fluid Damping: Presets for simulating motion in Air, Water, Oil, and Honey.

Visualization:

Real-time animation of the pendulum system.

Phase space trajectory plotting.

Time-series graphs for angular displacement.

Prerequisites

Python 3.6+

pip (Python package manager)

Installation

Clone the repository (or download the source code):

git clone <repository-url>
cd double-pendulum-sim


Set up a virtual environment (Recommended):

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Usage

Run the main simulation script:

python main.py


Note: Replace main.py with whatever you named your script file.

Controls

The simulation starts in a Paused state to allow initial configuration.

Play/Pause: Toggle the simulation loop.

Reset: Stop simulation and reset angles/velocities to initial slider values.

Fluid Button: Cycle through damping presets (Air, Water, Oil, Honey, Custom).

Sliders:

L1/L2: Length of the rods (m).

M1/M2: Mass of the bobs (kg).

Gravity: Acceleration due to gravity ($m/s^2$).

Damping: Friction coefficient (only active in "Custom" fluid mode).

Init θ1/θ2: Initial release angles.

Physics Background

The system is modeled using the Euler-Lagrange equations for a double pendulum. Because the system is chaotic, slight deviations in initial conditions lead to exponentially diverging trajectories.

To calculate the Lyapunov exponent ($\lambda$), the simulation tracks two systems simultaneously:

The main system ($S_1$).

A shadow system ($S_2$) initialized with a perturbation of $10^{-8}$.

$\lambda$ is estimated by monitoring the rate of separation between these two trajectories over time. A positive $\lambda$ indicates chaotic behavior.
