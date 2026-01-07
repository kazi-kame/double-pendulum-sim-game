Double Pendulum Chaos Simulator
A real-time physics simulation of a double pendulum built with Python and Pygame. I made this to visualize chaotic motion and calculate the Lyapunov exponent to measure how chaotic the system actually is.
Features

Physics Engine: Implements Lagrangian mechanics with 4th-order Runge-Kutta (RK4) integration for accurate results
Chaos Analysis: Calculates the Largest Lyapunov Exponent in real-time using the Wolf algorithm with shadow trajectories
Interactive Controls: Live sliders to adjust rod lengths, masses, gravity, and damping
Fluid Damping: Built-in presets for Air, Water, Oil, and Honey environments
Visualization:

Live pendulum animation with motion trails
Phase space trajectory plot
Real-time angle vs. time graphs



Prerequisites

Python 3.6 or higher
pip package manager

Installation

Clone this repo:

bashgit clone <repository-url>
cd double-pendulum-sim

Create a virtual environment (optional but recommended):

bash# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install dependencies:

bashpip install -r requirements.txt
Usage
Just run:
bashpython main.py
The simulation starts paused so you can adjust the initial conditions first.
Controls

Play/Pause Button: Start or stop the simulation
Reset Button: Resets everything back to the slider values
Fluid Button: Cycles through different damping presets (Air → Water → Oil → Honey → Custom)
Sliders (only adjustable when paused):

L1/L2: Rod lengths in meters
M1/M2: Bob masses in kilograms
Gravity: Acceleration due to gravity (m/s²)
Damping: Friction coefficient (editable only in Custom mode)
Init θ1/θ2: Starting angles for each pendulum



How It Works
The double pendulum is a classic example of a chaotic system - tiny changes in starting conditions create wildly different outcomes. The physics is based on Euler-Lagrange equations, which I solve numerically using RK4 integration.
To measure chaos, I calculate the Lyapunov exponent (λ). Here's the approach:

Run the main simulation normally
Run a second "shadow" simulation with an initial difference of just 10⁻⁸ radians
Track how fast these two trajectories diverge over time

A positive λ means the system is chaotic - the bigger the value, the faster trajectories diverge. The "Lyapunov Time" shown in the app tells you roughly how long predictions stay accurate before chaos takes over.
What I Learned
This project helped me understand:

How chaotic systems work in practice
Numerical integration methods (RK4 is way better than Euler!)
The Wolf algorithm for calculating Lyapunov exponents
Building interactive physics simulations with Pygame