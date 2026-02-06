!pip install numpy pygame math scipy

import pygame
import math
import sys
import numpy as np
from scipy.optimize import curve_fit

# ============================================================================
# Constants & Configuration
# ============================================================================

WIDTH, HEIGHT = 1200, 800
FPS = 60

BG_COLOR = (249, 250, 251)
WHITE = (255, 255, 255)
GRAY_300 = (209, 213, 219)
GRAY_500 = (107, 114, 128)
GRAY_800 = (31, 41, 55)
BLUE_500 = (59, 130, 246)
BLUE_600 = (37, 99, 235)
RED_400 = (248, 113, 113)
RED_600 = (220, 38, 38)
GREEN_400 = (74, 222, 128)
GREEN_600 = (22, 163, 74)
PURPLE_400 = (192, 132, 252)
PURPLE_600 = (147, 51, 234)
ORANGE_400 = (251, 146, 60)
ORANGE_600 = (234, 88, 12)
CYAN_400 = (34, 211, 238)
CYAN_600 = (8, 145, 178)
TEXT_COLOR = (51, 51, 51)
GRID_COLOR = (224, 224, 224)

pygame.font.init()
FONT_SM = pygame.font.SysFont('Arial', 14)
FONT_MD = pygame.font.SysFont('Arial', 16, bold=True)
FONT_LG = pygame.font.SysFont('Arial', 20, bold=True)
FONT_XL = pygame.font.SysFont('Arial', 24, bold=True)

# ============================================================================
# Physics & Mathematical Functions
# ============================================================================

def get_derivatives(theta1, omega1, theta2, omega2, params):
    """Calculate angular accelerations using Lagrangian mechanics."""
    l1, l2 = params['l1'], params['l2']
    m1, m2 = params['m1'], params['m2']
    g = params['g']
    damping = params['damping']

    delta = theta2 - theta1
    sin_delta = math.sin(delta)
    cos_delta = math.cos(delta)

    denom1 = (m1 + m2) * l1 - m2 * l1 * cos_delta * cos_delta
    denom2 = (l2 / l1) * denom1

    term1 = m2 * l1 * omega1 * omega1 * sin_delta * cos_delta
    term2 = m2 * g * math.sin(theta2) * cos_delta
    term3 = m2 * l2 * omega2 * omega2 * sin_delta
    term4 = (m1 + m2) * g * math.sin(theta1)
    term5 = damping * omega1
    alpha1 = (term1 + term2 + term3 - term4 - term5) / denom1

    term1_2 = -m2 * l2 * omega2 * omega2 * sin_delta * cos_delta
    term2_2 = (m1 + m2) * g * math.sin(theta1) * cos_delta
    term3_2 = (m1 + m2) * l1 * omega1 * omega1 * sin_delta
    term4_2 = (m1 + m2) * g * math.sin(theta2)
    term5_2 = damping * omega2
    alpha2 = (term1_2 + term2_2 - term3_2 - term4_2 - term5_2) / denom2

    return {'dtheta1': omega1, 'domega1': alpha1, 'dtheta2': omega2, 'domega2': alpha2}

def rk4_step(theta1, omega1, theta2, omega2, dt, params):
    """Perform one step of 4th-order Runge-Kutta integration."""
    k1 = get_derivatives(theta1, omega1, theta2, omega2, params)
    k2 = get_derivatives(theta1 + k1['dtheta1']*dt/2, omega1 + k1['domega1']*dt/2, theta2 + k1['dtheta2']*dt/2, omega2 + k1['domega2']*dt/2, params)
    k3 = get_derivatives(theta1 + k2['dtheta1']*dt/2, omega1 + k2['domega1']*dt/2, theta2 + k2['dtheta2']*dt/2, omega2 + k2['domega2']*dt/2, params)
    k4 = get_derivatives(theta1 + k3['dtheta1']*dt, omega1 + k3['domega1']*dt, theta2 + k3['dtheta2']*dt, omega2 + k3['domega2']*dt, params)

    return {
        'theta1': theta1 + (k1['dtheta1'] + 2*k2['dtheta1'] + 2*k3['dtheta1'] + k4['dtheta1']) * dt / 6,
        'omega1': omega1 + (k1['domega1'] + 2*k2['domega1'] + 2*k3['domega1'] + k4['domega1']) * dt / 6,
        'theta2': theta2 + (k1['dtheta2'] + 2*k2['dtheta2'] + 2*k3['dtheta2'] + k4['dtheta2']) * dt / 6,
        'omega2': omega2 + (k1['domega2'] + 2*k2['domega2'] + 2*k3['domega2'] + k4['domega2']) * dt / 6
    }

def angles_to_cartesian(theta1, theta2, l1, l2):
    """Convert pendulum angles to Cartesian coordinates."""
    x1 = l1 * math.sin(theta1)
    y1 = l1 * math.cos(theta1)
    x2 = x1 + l2 * math.sin(theta2)
    y2 = y1 + l2 * math.cos(theta2)
    return x1, y1, x2, y2

def exponential_func(t, a, b, c):
    """Exponential function: δ(t) = a * exp(b * t) + c"""
    return a * np.exp(b * t) + c

def fit_exponential(time_data, delta_data):
    """Fit exponential curve to divergence data."""
    if len(time_data) < 10:
        return None, None
    
    try:
        t_array = np.array(time_data)
        d_array = np.array(delta_data)
        
        # Remove any zeros or negative values for better fitting
        valid_idx = d_array > 1e-10
        if np.sum(valid_idx) < 10:
            return None, None
        
        t_fit = t_array[valid_idx]
        d_fit = d_array[valid_idx]
        
        # Initial guess for parameters
        p0 = [d_fit[0], 0.1, 0]
        
        # Fit the curve
        params, _ = curve_fit(exponential_func, t_fit, d_fit, p0=p0, maxfev=5000)
        
        return params, (t_fit, d_fit)
    except:
        return None, None

# ============================================================================
# UI Components
# ============================================================================

class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.label = label
        self.dragging = False
        self.handle_radius = 8

    def draw(self, screen):
        val_str = f"{self.value:.2f}"
        if "deg" in self.label:
            val_str = f"{int(self.value * 180 / math.pi)}°"
        lbl_surf = FONT_SM.render(f"{self.label}: {val_str}", True, TEXT_COLOR)
        screen.blit(lbl_surf, (self.rect.x, self.rect.y - 20))
        pygame.draw.rect(screen, GRAY_300, self.rect, border_radius=4)
        handle_x = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
        pygame.draw.circle(screen, BLUE_600, (int(handle_x), self.rect.centery), self.handle_radius)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            handle_x = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
            if abs(mx - handle_x) < 15 and abs(my - self.rect.centery) < 15:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mx, _ = event.pos
            rel_x = max(0, min(mx - self.rect.x, self.rect.width))
            self.value = self.min_val + (rel_x / self.rect.width) * (self.max_val - self.min_val)
            return True
        return False

class Button:
    def __init__(self, x, y, w, h, text, color, hover_color, action):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.action = action
        self.is_hovered = False

    def draw(self, screen):
        c = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, c, self.rect, border_radius=6)
        txt_surf = FONT_MD.render(self.text, True, WHITE)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        screen.blit(txt_surf, txt_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and self.is_hovered:
            if self.action: self.action()

# ============================================================================
# Main Application
# ============================================================================

class DoublePendulumApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Double Pendulum - Trajectory Divergence")
        self.clock = pygame.time.Clock()
        
        self.L1, self.L2 = 1.0, 1.0
        self.M1, self.M2 = 1.0, 1.0
        self.G = 9.81
        self.damping = 0.0
        self.theta1_init = math.pi * 0.4
        self.theta2_init = math.pi * 0.5
        self.delta_theta1 = 0.01
        self.delta_theta2 = 0.0
        
        self.is_running = False
        self.time_elapsed = 0
        self.max_simulation_time = 30.0
        
        self.fluids = ['Air', 'Water', 'Oil', 'Honey', 'Custom']
        self.fluid_dampings = [0.0, 0.5, 1.2, 2.5, 0.0]
        self.current_fluid_idx = 0
        
        # Exponential fit parameters
        self.fit_params = None
        self.fit_update_counter = 0
        self.fit_update_interval = 30  # Update fit every 30 frames
        self.final_fit_params = None
        self.simulation_completed = False
        
        self.init_sim_state()
        self.create_ui()

    def init_sim_state(self):
        """Initialize simulation state with two trajectories."""
        self.state1 = {
            'theta1': self.theta1_init, 'omega1': 0,
            'theta2': self.theta2_init, 'omega2': 0,
            'trail': [],
            'timeData': [], 'theta1Data': [], 'theta2Data': [],
        }
        self.state2 = {
            'theta1': self.theta1_init + self.delta_theta1, 'omega1': 0,
            'theta2': self.theta2_init + self.delta_theta2, 'omega2': 0,
            'trail': [],
            'timeData': [], 'theta1Data': [], 'theta2Data': [],
        }
        self.divergence_data = {
            'timeData': [],
            'deltaData': []
        }
        self.time_elapsed = 0
        self.fit_params = None
        self.final_fit_params = None
        self.simulation_completed = False

    def create_ui(self):
        """Create UI controls: sliders and buttons."""
        col1_x, col_w, gap_y, start_y = 20, 250, 60, 80
        self.sliders = [
            Slider(col1_x, start_y, col_w, 10, 0.5, 2.0, self.L1, "Length 1 (m)"),
            Slider(col1_x, start_y + gap_y, col_w, 10, 0.5, 2.0, self.L2, "Length 2 (m)"),
            Slider(col1_x, start_y + gap_y*2, col_w, 10, 0.5, 3.0, self.M1, "Mass 1 (kg)"),
            Slider(col1_x, start_y + gap_y*3, col_w, 10, 0.5, 3.0, self.M2, "Mass 2 (kg)"),
            Slider(col1_x, start_y + gap_y*4, col_w, 10, 1.0, 20.0, self.G, "Gravity (m/s²)"),
            Slider(col1_x, start_y + gap_y*5, col_w, 10, 0.0, 5.0, self.damping, "Damping"),
            Slider(col1_x, start_y + gap_y*6, col_w, 10, 0, math.pi, self.theta1_init, "Init θ1 (deg)"),
            Slider(col1_x, start_y + gap_y*7, col_w, 10, 0, math.pi, self.theta2_init, "Init θ2 (deg)"),
            Slider(col1_x, start_y + gap_y*8, col_w, 10, 0, 0.5, self.delta_theta1, "Δθ1 (rad)"),
            Slider(col1_x, start_y + gap_y*9, col_w, 10, 0, 0.5, self.delta_theta2, "Δθ2 (rad)")
        ]
        self.btn_play = Button(col1_x, 700, 120, 40, "Play", BLUE_500, BLUE_600, self.toggle_play)
        self.btn_reset = Button(col1_x + 130, 700, 120, 40, "Reset", GRAY_500, GRAY_800, self.reset_sim)
        self.btn_fluid = Button(col1_x, 640, 250, 40, f"Fluid: {self.fluids[0]}", GRAY_300, GRAY_500, self.cycle_fluid)

    def toggle_play(self):
        self.is_running = not self.is_running
        self.btn_play.text = "Pause" if self.is_running else "Play"

    def reset_sim(self):
        self.is_running = False
        self.btn_play.text = "Play"
        self.init_sim_state()

    def cycle_fluid(self):
        self.current_fluid_idx = (self.current_fluid_idx + 1) % len(self.fluids)
        fluid_name = self.fluids[self.current_fluid_idx]
        self.btn_fluid.text = f"Fluid: {fluid_name}"
        if fluid_name != 'Custom':
            damp = self.fluid_dampings[self.current_fluid_idx]
            self.sliders[5].value = damp
            self.damping = damp

    def update(self):
        """Update simulation state and calculate trajectory divergence."""
        if not self.is_running:
            self.L1 = self.sliders[0].value
            self.L2 = self.sliders[1].value
            self.M1 = self.sliders[2].value
            self.M2 = self.sliders[3].value
            self.G = self.sliders[4].value
            self.theta1_init = self.sliders[6].value
            self.theta2_init = self.sliders[7].value
            self.delta_theta1 = self.sliders[8].value
            self.delta_theta2 = self.sliders[9].value
            
            self.state1['theta1'] = self.theta1_init
            self.state1['theta2'] = self.theta2_init
            self.state2['theta1'] = self.theta1_init + self.delta_theta1
            self.state2['theta2'] = self.theta2_init + self.delta_theta2
            
        if self.fluids[self.current_fluid_idx] == 'Custom':
             self.damping = self.sliders[5].value
        else:
             self.sliders[5].value = self.damping

        if self.is_running:
            dt = 0.02
            params = {'l1': self.L1, 'l2': self.L2, 'm1': self.M1, 'm2': self.M2, 'g': self.G, 'damping': self.damping}
            
            # Check if simulation time limit reached
            if self.time_elapsed >= self.max_simulation_time:
                if not self.simulation_completed:
                    # Calculate final exponential fit
                    self.final_fit_params, _ = fit_exponential(
                        self.divergence_data['timeData'], 
                        self.divergence_data['deltaData']
                    )
                    self.simulation_completed = True
                self.is_running = False
                self.btn_play.text = "Play"
                return
            
            # Update both trajectories
            new_state1 = rk4_step(self.state1['theta1'], self.state1['omega1'], self.state1['theta2'], self.state1['omega2'], dt, params)
            new_state2 = rk4_step(self.state2['theta1'], self.state2['omega1'], self.state2['theta2'], self.state2['omega2'], dt, params)
            
            # Calculate divergence in phase space
            delta = math.sqrt(
                (new_state1['theta1'] - new_state2['theta1'])**2 + 
                (new_state1['theta2'] - new_state2['theta2'])**2 + 
                (new_state1['omega1'] - new_state2['omega1'])**2 + 
                (new_state1['omega2'] - new_state2['omega2'])**2
            )
            
            self.time_elapsed += dt

            # Store divergence data (keep all data, don't limit to 500)
            self.divergence_data['timeData'].append(self.time_elapsed)
            self.divergence_data['deltaData'].append(delta)

            # Update exponential fit periodically
            self.fit_update_counter += 1
            if self.fit_update_counter >= self.fit_update_interval:
                self.fit_params, _ = fit_exponential(
                    self.divergence_data['timeData'], 
                    self.divergence_data['deltaData']
                )
                self.fit_update_counter = 0

            # Update trails for trajectory 1
            x1, y1, x2, y2 = angles_to_cartesian(new_state1['theta1'], new_state1['theta2'], self.L1, self.L2)
            self.state1['trail'].append((x2, y2))
            if len(self.state1['trail']) > 800: self.state1['trail'].pop(0)
            
            # Update trails for trajectory 2
            x1_2, y1_2, x2_2, y2_2 = angles_to_cartesian(new_state2['theta1'], new_state2['theta2'], self.L1, self.L2)
            self.state2['trail'].append((x2_2, y2_2))
            if len(self.state2['trail']) > 800: self.state2['trail'].pop(0)
            
            # Store angle data for trajectory 1
            self.state1['timeData'].append(self.time_elapsed)
            self.state1['theta1Data'].append(new_state1['theta1'])
            self.state1['theta2Data'].append(new_state1['theta2'])
            if len(self.state1['timeData']) > 500:
                self.state1['timeData'].pop(0)
                self.state1['theta1Data'].pop(0)
                self.state1['theta2Data'].pop(0)

            # Store angle data for trajectory 2
            self.state2['timeData'].append(self.time_elapsed)
            self.state2['theta1Data'].append(new_state2['theta1'])
            self.state2['theta2Data'].append(new_state2['theta2'])
            if len(self.state2['timeData']) > 500:
                self.state2['timeData'].pop(0)
                self.state2['theta1Data'].pop(0)
                self.state2['theta2Data'].pop(0)

            self.state1.update(new_state1)
            self.state2.update(new_state2)

    def draw_pendulum_window(self):
        """Render both pendulums with their trails."""
        panel_rect = pygame.Rect(300, 80, 500, 500)
        pygame.draw.rect(self.screen, WHITE, panel_rect)
        pygame.draw.rect(self.screen, GRAY_300, panel_rect, 1)
        cx, cy = panel_rect.centerx, panel_rect.centery - 50
        scale = 100
        
        # Draw grid
        for i in range(-2, 3):
            pygame.draw.line(self.screen, GRID_COLOR, (cx + i*scale, panel_rect.top), (cx + i*scale, panel_rect.bottom))
            pygame.draw.line(self.screen, GRID_COLOR, (panel_rect.left, cy + i*scale), (panel_rect.right, cy + i*scale))
        
        # Draw trail for pendulum 1 (green)
        if len(self.state1['trail']) > 1:
            points = [(cx + p[0]*scale, cy + p[1]*scale) for p in self.state1['trail']]
            pygame.draw.lines(self.screen, GREEN_400, False, points, 2)
        
        # Draw trail for pendulum 2 (orange)
        if len(self.state2['trail']) > 1:
            points = [(cx + p[0]*scale, cy + p[1]*scale) for p in self.state2['trail']]
            pygame.draw.lines(self.screen, ORANGE_400, False, points, 2)
        
        # Draw pendulum 1
        x1, y1, x2, y2 = angles_to_cartesian(self.state1['theta1'], self.state1['theta2'], self.L1, self.L2)
        pygame.draw.line(self.screen, (51, 51, 51), (cx, cy), (cx + x1*scale, cy + y1*scale), 3)
        pygame.draw.line(self.screen, (51, 51, 51), (cx + x1*scale, cy + y1*scale), (cx + x2*scale, cy + y2*scale), 3)
        r1 = 8 + self.M1 * 3
        r2 = 8 + self.M2 * 3
        pygame.draw.circle(self.screen, RED_400, (cx + x1*scale, cy + y1*scale), int(r1))
        pygame.draw.circle(self.screen, RED_600, (cx + x1*scale, cy + y1*scale), int(r1), 2)
        pygame.draw.circle(self.screen, GREEN_400, (cx + x2*scale, cy + y2*scale), int(r2))
        pygame.draw.circle(self.screen, GREEN_600, (cx + x2*scale, cy + y2*scale), int(r2), 2)
        
        # Draw pendulum 2 (semi-transparent)
        x1_2, y1_2, x2_2, y2_2 = angles_to_cartesian(self.state2['theta1'], self.state2['theta2'], self.L1, self.L2)
        pygame.draw.line(self.screen, (100, 100, 100), (cx, cy), (cx + x1_2*scale, cy + y1_2*scale), 2)
        pygame.draw.line(self.screen, (100, 100, 100), (cx + x1_2*scale, cy + y1_2*scale), (cx + x2_2*scale, cy + y2_2*scale), 2)
        pygame.draw.circle(self.screen, PURPLE_400, (cx + x1_2*scale, cy + y1_2*scale), int(r1))
        pygame.draw.circle(self.screen, PURPLE_600, (cx + x1_2*scale, cy + y1_2*scale), int(r1), 2)
        pygame.draw.circle(self.screen, ORANGE_400, (cx + x2_2*scale, cy + y2_2*scale), int(r2))
        pygame.draw.circle(self.screen, ORANGE_600, (cx + x2_2*scale, cy + y2_2*scale), int(r2), 2)
        
        # Draw pivot point
        pygame.draw.circle(self.screen, (102, 102, 102), (cx, cy), 6)
        
        # Display time
        time_surf = FONT_MD.render(f"Time: {self.time_elapsed:.2f} s / {self.max_simulation_time:.0f} s", True, TEXT_COLOR)
        self.screen.blit(time_surf, (panel_rect.left + 10, panel_rect.top + 10))

    def draw_divergence_window(self):
        """Display trajectory divergence δ(t) over time with exponential fit."""
        panel_rect = pygame.Rect(820, 80, 350, 500)
        pygame.draw.rect(self.screen, WHITE, panel_rect)
        pygame.draw.rect(self.screen, GRAY_300, panel_rect, 1)
        
        title = FONT_LG.render("Trajectory Divergence", True, GRAY_800)
        self.screen.blit(title, (panel_rect.left + 20, panel_rect.top + 15))
        
        # Current divergence value
        current_delta = self.divergence_data['deltaData'][-1] if self.divergence_data['deltaData'] else 0
        delta_box = pygame.Rect(panel_rect.left + 20, panel_rect.top + 50, 310, 70)
        pygame.draw.rect(self.screen, (243, 244, 246), delta_box, border_radius=5)
        lbl = FONT_SM.render("Current δ(t)", True, GRAY_500)
        val = FONT_XL.render(f"{current_delta:.6f}", True, PURPLE_600)
        self.screen.blit(lbl, (delta_box.x + 10, delta_box.y + 10))
        self.screen.blit(val, (delta_box.x + 10, delta_box.y + 35))
        
        # Display Lyapunov exponent (exponential growth rate)
        fit_to_display = self.final_fit_params if self.simulation_completed else self.fit_params
        if fit_to_display is not None:
            lyapunov = fit_to_display[1]
            fit_box = pygame.Rect(panel_rect.left + 20, panel_rect.top + 130, 310, 50)
            pygame.draw.rect(self.screen, (236, 254, 255), fit_box, border_radius=5)
            lbl2 = FONT_SM.render("Lyapunov Exp. (λ)", True, GRAY_500)
            val2 = FONT_LG.render(f"{lyapunov:.4f}", True, CYAN_600)
            self.screen.blit(lbl2, (fit_box.x + 10, fit_box.y + 5))
            self.screen.blit(val2, (fit_box.x + 10, fit_box.y + 25))
            
            # Show fit status
            status_text = "Final Fit" if self.simulation_completed else "Real-time Fit"
            status = FONT_SM.render(status_text, True, GRAY_500)
            self.screen.blit(status, (fit_box.right - 90, fit_box.y + 5))
        
        # Plot δ(t) vs time
        plot_y_offset = 200 if fit_to_display is not None else 140
        plot_rect = pygame.Rect(panel_rect.left + 25, panel_rect.top + plot_y_offset, 300, 280)
        pygame.draw.rect(self.screen, WHITE, plot_rect)
        pygame.draw.rect(self.screen, GRAY_300, plot_rect, 1)
        
        if len(self.divergence_data['timeData']) > 1:
            mx, my = 20, 20
            plot_w, plot_h = plot_rect.width - mx * 2, plot_rect.height - my * 2
            t_data, delta_data = self.divergence_data['timeData'], self.divergence_data['deltaData']
            
            # Always use full time range (0 to max_simulation_time)
            t_min, t_max = 0, self.max_simulation_time
            
            # Use logarithmic scale for better visibility when data varies greatly
            # Or add padding to show full range
            delta_max = max(delta_data) if delta_data else 1
            delta_min = min(delta_data) if delta_data else 0
            
            # Add 10% padding to top and bottom for better visibility
            delta_range = max(delta_max - delta_min, 0.001)
            delta_padding = delta_range * 0.1
            delta_min = max(0, delta_min - delta_padding)
            delta_max = delta_max + delta_padding
            delta_range = delta_max - delta_min
            
            def map_pt(t, d):
                x = plot_rect.left + mx + (t - t_min) / (t_max - t_min) * plot_w if t_max > t_min else plot_rect.left + mx
                y = plot_rect.top + my + plot_h - (d - delta_min) / delta_range * plot_h
                return (x, y)
            
            # Draw axes
            pygame.draw.line(self.screen, GRAY_300, (plot_rect.left + mx, plot_rect.bottom - my), 
                           (plot_rect.right - mx, plot_rect.bottom - my), 1)
            pygame.draw.line(self.screen, GRAY_300, (plot_rect.left + mx, plot_rect.top + my), 
                           (plot_rect.left + mx, plot_rect.bottom - my), 1)
            
            # Draw exponential fit curve
            fit_to_display = self.final_fit_params if self.simulation_completed else self.fit_params
            if fit_to_display is not None:
                a, b, c = fit_to_display
                t_fit_range = np.linspace(t_min, t_max, 100)
                delta_fit = exponential_func(t_fit_range, a, b, c)
                # Clip fit values to visible range
                fit_pts = []
                for t, d in zip(t_fit_range, delta_fit):
                    if delta_min <= d <= delta_max:
                        fit_pts.append(map_pt(t, d))
                if len(fit_pts) > 1:
                    pygame.draw.lines(self.screen, CYAN_400, False, fit_pts, 3)
            
            # Draw divergence data points
            pts = [map_pt(t, d) for t, d in zip(t_data, delta_data)]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, PURPLE_600, False, pts, 2)
            
            # Labels
            xlabel = FONT_SM.render("Time (s)", True, TEXT_COLOR)
            ylabel = FONT_SM.render("δ(t)", True, TEXT_COLOR)
            self.screen.blit(xlabel, (plot_rect.centerx - 30, plot_rect.bottom - 15))
            self.screen.blit(ylabel, (plot_rect.left + 5, plot_rect.top + 5))
            
            # Legend
            if fit_to_display is not None:
                leg_y = plot_rect.top + 10
                pygame.draw.line(self.screen, PURPLE_600, (plot_rect.right - 120, leg_y), (plot_rect.right - 100, leg_y), 2)
                self.screen.blit(FONT_SM.render("Data", True, TEXT_COLOR), (plot_rect.right - 95, leg_y - 7))
                pygame.draw.line(self.screen, CYAN_400, (plot_rect.right - 120, leg_y + 15), (plot_rect.right - 100, leg_y + 15), 3)
                self.screen.blit(FONT_SM.render("Exp Fit", True, TEXT_COLOR), (plot_rect.right - 95, leg_y + 8))

    def draw_timeseries(self):
        """Plot angle evolution over time for both trajectories."""
        rect = pygame.Rect(300, 600, 870, 180)
        pygame.draw.rect(self.screen, WHITE, rect)
        pygame.draw.rect(self.screen, GRAY_300, rect, 1)
        
        title = FONT_MD.render("Angle Evolution", True, GRAY_800)
        self.screen.blit(title, (rect.left + 10, rect.top + 5))
        
        if len(self.state1['timeData']) < 2: return
        
        mx, my = 40, 30
        plot_w, plot_h = rect.width - mx * 2, rect.height - my * 2
        t_data = self.state1['timeData']
        th1_traj1, th2_traj1 = self.state1['theta1Data'], self.state1['theta2Data']
        th1_traj2, th2_traj2 = self.state2['theta1Data'], self.state2['theta2Data']
        
        t_min, t_max = t_data[0], t_data[-1]
        all_th = th1_traj1 + th2_traj1 + th1_traj2 + th2_traj2
        th_min, th_max = min(all_th), max(all_th)
        th_range = max(th_max - th_min, 0.5)
        
        def map_pt(t, th):
            x = rect.left + mx + (t - t_min) / (t_max - t_min) * plot_w if t_max > t_min else rect.left + mx
            y = rect.top + my + plot_h - (th - th_min) / th_range * plot_h
            return (x, y)
        
        # Draw zero line
        zero_y = rect.top + my + plot_h - (0 - th_min) / th_range * plot_h
        if rect.top < zero_y < rect.bottom: 
            pygame.draw.line(self.screen, GRID_COLOR, (rect.left + mx, zero_y), (rect.right - mx, zero_y))
        
        # Draw trajectories
        # Trajectory 1
        pts1_th1 = [map_pt(t, th) for t, th in zip(t_data, th1_traj1)]
        pts1_th2 = [map_pt(t, th) for t, th in zip(t_data, th2_traj1)]
        pygame.draw.lines(self.screen, RED_400, False, pts1_th1, 2)
        pygame.draw.lines(self.screen, GREEN_400, False, pts1_th2, 2)
        
        # Trajectory 2 (dashed appearance with segments)
        pts2_th1 = [map_pt(t, th) for t, th in zip(t_data, th1_traj2)]
        pts2_th2 = [map_pt(t, th) for t, th in zip(t_data, th2_traj2)]
        for i in range(0, len(pts2_th1) - 1, 3):
            if i + 1 < len(pts2_th1):
                pygame.draw.line(self.screen, PURPLE_400, pts2_th1[i], pts2_th1[i + 1], 2)
        for i in range(0, len(pts2_th2) - 1, 3):
            if i + 1 < len(pts2_th2):
                pygame.draw.line(self.screen, ORANGE_400, pts2_th2[i], pts2_th2[i + 1], 2)
        
        # Legend
        leg_x, leg_y = rect.right - 280, rect.top + 8
        # Trajectory 1
        pygame.draw.line(self.screen, RED_400, (leg_x, leg_y + 5), (leg_x + 15, leg_y + 5), 3)
        self.screen.blit(FONT_SM.render("θ1 (Traj 1)", True, TEXT_COLOR), (leg_x + 20, leg_y))
        pygame.draw.line(self.screen, GREEN_400, (leg_x + 90, leg_y + 5), (leg_x + 105, leg_y + 5), 3)
        self.screen.blit(FONT_SM.render("θ2 (Traj 1)", True, TEXT_COLOR), (leg_x + 110, leg_y))
        # Trajectory 2
        pygame.draw.line(self.screen, PURPLE_400, (leg_x, leg_y + 20), (leg_x + 15, leg_y + 20), 3)
        self.screen.blit(FONT_SM.render("θ1 (Traj 2)", True, TEXT_COLOR), (leg_x + 20, leg_y + 15))
        pygame.draw.line(self.screen, ORANGE_400, (leg_x + 90, leg_y + 20), (leg_x + 105, leg_y + 20), 3)
        self.screen.blit(FONT_SM.render("θ2 (Traj 2)", True, TEXT_COLOR), (leg_x + 110, leg_y + 15))

    def run(self):
        """Main event loop."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                for s in self.sliders: 
                    if not self.is_running: s.handle_event(event)
                self.btn_play.handle_event(event)
                self.btn_reset.handle_event(event)
                self.btn_fluid.handle_event(event)
            self.update()
            self.screen.fill(BG_COLOR)
            title = FONT_XL.render("Double Pendulum - Trajectory Divergence", True, GRAY_800)
            self.screen.blit(title, (20, 20))
            for s in self.sliders: s.draw(self.screen)
            self.btn_play.draw(self.screen)
            self.btn_reset.draw(self.screen)
            self.btn_fluid.draw(self.screen)
            self.draw_pendulum_window()
            self.draw_divergence_window()
            self.draw_timeseries()
            pygame.display.flip()
            self.clock.tick(FPS)

if __name__ == "__main__":
    app = DoublePendulumApp()
    app.run()
