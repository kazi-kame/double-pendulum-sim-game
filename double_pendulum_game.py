import pygame
import math
import sys

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
        pygame.display.set_caption("Double Pendulum Simulator (Corrected Lyapunov)")
        self.clock = pygame.time.Clock()
        
        self.L1, self.L2 = 1.0, 1.0
        self.M1, self.M2 = 1.0, 1.0
        self.G = 9.81
        self.damping = 0.0
        self.theta1_init = math.pi * 0.4
        self.theta2_init = math.pi * 0.5
        
        self.is_running = False
        self.time_elapsed = 0
        self.lyapunov_exp = 0
        self.lyapunov_time = 0
        self.lyap_sum_log = 0.0
        
        self.fluids = ['Air', 'Water', 'Oil', 'Honey', 'Custom']
        self.fluid_dampings = [0.0, 0.5, 1.2, 2.5, 0.0]
        self.current_fluid_idx = 0
        
        self.init_sim_state()
        self.create_ui()

    def init_sim_state(self):
        """Initialize simulation state with main and reference trajectories."""
        epsilon = 1e-8
        self.state = {
            'theta1': self.theta1_init, 'omega1': 0,
            'theta2': self.theta2_init, 'omega2': 0,
            'theta1_ref': self.theta1_init + epsilon, 'omega1_ref': 0,
            'theta2_ref': self.theta2_init, 'omega2_ref': 0,
            'trail1': [], 'trail2': [],
            'timeData': [], 'theta1Data': [], 'theta2Data': [],
            'time': 0
        }
        self.time_elapsed = 0
        self.lyapunov_exp = 0
        self.lyap_sum_log = 0.0

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
            Slider(col1_x, start_y + gap_y*7, col_w, 10, 0, math.pi, self.theta2_init, "Init θ2 (deg)")
        ]
        self.btn_play = Button(col1_x, 600, 120, 40, "Play", BLUE_500, BLUE_600, self.toggle_play)
        self.btn_reset = Button(col1_x + 130, 600, 120, 40, "Reset", GRAY_500, GRAY_800, self.reset_sim)
        self.btn_fluid = Button(col1_x, 540, 250, 40, f"Fluid: {self.fluids[0]}", GRAY_300, GRAY_500, self.cycle_fluid)

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
        """Update simulation state and calculate Lyapunov exponent."""
        if not self.is_running:
            self.L1 = self.sliders[0].value
            self.L2 = self.sliders[1].value
            self.M1 = self.sliders[2].value
            self.M2 = self.sliders[3].value
            self.G = self.sliders[4].value
            self.theta1_init = self.sliders[6].value
            self.theta2_init = self.sliders[7].value
            
            self.state['theta1'] = self.theta1_init
            self.state['theta2'] = self.theta2_init
            self.state['theta1_ref'] = self.theta1_init + 1e-8
            self.state['theta2_ref'] = self.theta2_init
            
        if self.fluids[self.current_fluid_idx] == 'Custom':
             self.damping = self.sliders[5].value
        else:
             self.sliders[5].value = self.damping

        if self.is_running:
            dt = 0.02
            params = {'l1': self.L1, 'l2': self.L2, 'm1': self.M1, 'm2': self.M2, 'g': self.G, 'damping': self.damping}
            
            new_state = rk4_step(self.state['theta1'], self.state['omega1'], self.state['theta2'], self.state['omega2'], dt, params)
            new_ref = rk4_step(self.state['theta1_ref'], self.state['omega1_ref'], self.state['theta2_ref'], self.state['omega2_ref'], dt, params)
            
            # Calculate phase space distance between main and reference trajectories
            d = math.sqrt(
                (new_state['theta1'] - new_ref['theta1'])**2 + 
                (new_state['theta2'] - new_ref['theta2'])**2 + 
                (new_state['omega1'] - new_ref['omega1'])**2 + 
                (new_state['omega2'] - new_ref['omega2'])**2
            )
            d0 = 1e-8
            
            # Wolf algorithm: rescale reference trajectory to prevent overflow
            if d > d0 * 100:
                self.lyap_sum_log += math.log(d / d0)
                
                scale = d0 / d
                new_ref['theta1'] = new_state['theta1'] + (new_ref['theta1'] - new_state['theta1']) * scale
                new_ref['theta2'] = new_state['theta2'] + (new_ref['theta2'] - new_state['theta2']) * scale
                new_ref['omega1'] = new_state['omega1'] + (new_ref['omega1'] - new_state['omega1']) * scale
                new_ref['omega2'] = new_state['omega2'] + (new_ref['omega2'] - new_state['omega2']) * scale
                
                d = d0

            self.time_elapsed += dt

            # Compute Lyapunov exponent after warm-up period
            if self.time_elapsed > 1.0:
                current_divergence = math.log(d / d0) if d > 0 else 0
                self.lyapunov_exp = (self.lyap_sum_log + current_divergence) / self.time_elapsed
                
                if self.lyapunov_exp > 0.001:
                    self.lyapunov_time = 1 / self.lyapunov_exp
                else:
                    self.lyapunov_time = 0

            x1, y1, x2, y2 = angles_to_cartesian(new_state['theta1'], new_state['theta2'], self.L1, self.L2)
            self.state['trail1'].append((x1, y1))
            self.state['trail2'].append((x2, y2))
            if len(self.state['trail1']) > 800: self.state['trail1'].pop(0)
            if len(self.state['trail2']) > 800: self.state['trail2'].pop(0)
            
            self.state['timeData'].append(self.time_elapsed)
            self.state['theta1Data'].append(new_state['theta1'])
            self.state['theta2Data'].append(new_state['theta2'])
            if len(self.state['timeData']) > 500:
                self.state['timeData'].pop(0)
                self.state['theta1Data'].pop(0)
                self.state['theta2Data'].pop(0)

            self.state.update(new_state)
            self.state['theta1_ref'] = new_ref['theta1']
            self.state['omega1_ref'] = new_ref['omega1']
            self.state['theta2_ref'] = new_ref['theta2']
            self.state['omega2_ref'] = new_ref['omega2']

    def draw_pendulum_window(self):
        """Render the main pendulum visualization with trail."""
        panel_rect = pygame.Rect(300, 80, 500, 500)
        pygame.draw.rect(self.screen, WHITE, panel_rect)
        pygame.draw.rect(self.screen, GRAY_300, panel_rect, 1)
        cx, cy = panel_rect.centerx, panel_rect.centery - 50
        scale = 100
        for i in range(-2, 3):
            pygame.draw.line(self.screen, GRID_COLOR, (cx + i*scale, panel_rect.top), (cx + i*scale, panel_rect.bottom))
            pygame.draw.line(self.screen, GRID_COLOR, (panel_rect.left, cy + i*scale), (panel_rect.right, cy + i*scale))
        if len(self.state['trail2']) > 1:
            points = [(cx + p[0]*scale, cy + p[1]*scale) for p in self.state['trail2']]
            pygame.draw.lines(self.screen, (76, 175, 80), False, points, 2)
        x1, y1, x2, y2 = angles_to_cartesian(self.state['theta1'], self.state['theta2'], self.L1, self.L2)
        pygame.draw.line(self.screen, (51, 51, 51), (cx, cy), (cx + x1*scale, cy + y1*scale), 3)
        pygame.draw.line(self.screen, (51, 51, 51), (cx + x1*scale, cy + y1*scale), (cx + x2*scale, cy + y2*scale), 3)
        pygame.draw.circle(self.screen, (102, 102, 102), (cx, cy), 6)
        r1 = 8 + self.M1 * 3
        r2 = 8 + self.M2 * 3
        pygame.draw.circle(self.screen, RED_400, (cx + x1*scale, cy + y1*scale), int(r1))
        pygame.draw.circle(self.screen, RED_600, (cx + x1*scale, cy + y1*scale), int(r1), 2)
        pygame.draw.circle(self.screen, GREEN_400, (cx + x2*scale, cy + y2*scale), int(r2))
        pygame.draw.circle(self.screen, GREEN_600, (cx + x2*scale, cy + y2*scale), int(r2), 2)
        time_surf = FONT_MD.render(f"Time: {self.time_elapsed:.2f} s", True, TEXT_COLOR)
        self.screen.blit(time_surf, (panel_rect.left + 10, panel_rect.top + 10))

    def draw_analysis_window(self):
        """Display Lyapunov exponent, Lyapunov time, and phase space trajectory."""
        panel_rect = pygame.Rect(820, 80, 350, 500)
        pygame.draw.rect(self.screen, WHITE, panel_rect)
        pygame.draw.rect(self.screen, GRAY_300, panel_rect, 1)
        titles = ["Lyapunov Exponent", "Lyapunov Time", "Simulation Time"]
        values = [f"{self.lyapunov_exp:.4f} s⁻¹", f"{self.lyapunov_time:.2f} s" if self.lyapunov_time > 0 else "—", f"{self.time_elapsed:.2f} s"]
        colors = [BLUE_600, GREEN_600, (147, 51, 234)]
        y_off = panel_rect.top + 20
        for i in range(3):
            box = pygame.Rect(panel_rect.left + 20, y_off, 310, 70)
            pygame.draw.rect(self.screen, (243, 244, 246), box, border_radius=5)
            lbl = FONT_SM.render(titles[i], True, GRAY_500)
            val = FONT_XL.render(values[i], True, colors[i])
            self.screen.blit(lbl, (box.x + 10, box.y + 10))
            self.screen.blit(val, (box.x + 10, box.y + 30))
            y_off += 80
        traj_rect = pygame.Rect(panel_rect.left + 25, y_off + 20, 300, 200)
        pygame.draw.rect(self.screen, WHITE, traj_rect)
        pygame.draw.rect(self.screen, GRAY_300, traj_rect, 1)
        cx, cy = traj_rect.centerx, traj_rect.centery
        scale = 60
        pygame.draw.line(self.screen, GRID_COLOR, (cx, traj_rect.top), (cx, traj_rect.bottom))
        pygame.draw.line(self.screen, GRID_COLOR, (traj_rect.left, cy), (traj_rect.right, cy))
        if len(self.state['trail2']) > 1:
            points = [(cx + p[0]*scale, cy + (p[1]-self.L1)*scale) for p in self.state['trail2']]
            safe_points = []
            for p in points:
                if traj_rect.left < p[0] < traj_rect.right and traj_rect.top < p[1] < traj_rect.bottom:
                    safe_points.append(p)
                else:
                    if len(safe_points) > 1: pygame.draw.lines(self.screen, GREEN_600, False, safe_points, 2)
                    safe_points = []
            if len(safe_points) > 1: pygame.draw.lines(self.screen, GREEN_600, False, safe_points, 2)

    def draw_timeseries(self):
        """Plot angle evolution over time."""
        rect = pygame.Rect(300, 600, 870, 180)
        pygame.draw.rect(self.screen, WHITE, rect)
        pygame.draw.rect(self.screen, GRAY_300, rect, 1)
        if len(self.state['timeData']) < 2: return
        mx, my = 40, 20
        plot_w, plot_h = rect.width - mx * 2, rect.height - my * 2
        t_data, th1, th2 = self.state['timeData'], self.state['theta1Data'], self.state['theta2Data']
        t_min, t_max = t_data[0], t_data[-1]
        all_th = th1 + th2
        th_min, th_max = min(all_th), max(all_th)
        th_range = max(th_max - th_min, 0.5)
        def map_pt(t, th):
            x = rect.left + mx + (t - t_min) / (t_max - t_min) * plot_w if t_max > t_min else rect.left + mx
            y = rect.top + my + plot_h - (th - th_min) / th_range * plot_h
            return (x, y)
        zero_y = rect.top + my + plot_h - (0 - th_min) / th_range * plot_h
        if rect.top < zero_y < rect.bottom: pygame.draw.line(self.screen, GRAY_300, (rect.left + mx, zero_y), (rect.right - mx, zero_y))
        pts1, pts2 = [map_pt(t, th) for t, th in zip(t_data, th1)], [map_pt(t, th) for t, th in zip(t_data, th2)]
        pygame.draw.lines(self.screen, RED_400, False, pts1, 2)
        pygame.draw.lines(self.screen, GREEN_400, False, pts2, 2)
        leg_x, leg_y = rect.right - 150, rect.top + 10
        pygame.draw.rect(self.screen, RED_400, (leg_x, leg_y, 10, 10))
        self.screen.blit(FONT_SM.render("Theta 1", True, TEXT_COLOR), (leg_x + 15, leg_y - 2))
        pygame.draw.rect(self.screen, GREEN_400, (leg_x + 80, leg_y, 10, 10))
        self.screen.blit(FONT_SM.render("Theta 2", True, TEXT_COLOR), (leg_x + 95, leg_y - 2))

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
            title = FONT_XL.render("Double Pendulum Simulator", True, GRAY_800)
            self.screen.blit(title, (20, 20))
            for s in self.sliders: s.draw(self.screen)
            self.btn_play.draw(self.screen)
            self.btn_reset.draw(self.screen)
            self.btn_fluid.draw(self.screen)
            self.draw_pendulum_window()
            self.draw_analysis_window()
            self.draw_timeseries()
            pygame.display.flip()
            self.clock.tick(FPS)

if __name__ == "__main__":
    app = DoublePendulumApp()
    app.run()