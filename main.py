import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# physics params
def run_simulation(r0_disp):
    M = 2.0       # mass on table, kg
    m = 1.0       # hanging mass, kg
    g = 9.81      # grav, m/s^2
    r0 = 1.0      # equilibrium radius, m
    omega0 = np.sqrt(m * g / (M * r0))  # equilibrium angular speed, rad/s
    L_total = r0 + (m * g) / (M * omega0**2)  # total string length, m
    dt = 0.01
    t_max = 20
    times = np.arange(0, t_max, dt)
    n_steps = len(times)
    r = np.zeros(n_steps)
    theta = np.zeros(n_steps)
    r_prime = np.zeros(n_steps)
    # initial conditions, start a bit displaced from equilibrium
    r[0] = r0 * r0_disp
    theta[0] = 0.0
    r_prime[0] = 0.0
    # angular momentum = const
    Lz = M * r0**2 * omega0

    # euler integration
    for i in range(n_steps-1):
        omega = Lz / (M * r[i]**2)
        accel_r = (M * r[i] * omega**2 - m * g) / (M + m)
        r_prime[i+1] = r_prime[i] + accel_r * dt
        r[i+1] = r[i] + r_prime[i+1] * dt
        theta[i+1] = theta[i] + omega * dt

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z_hang = -(L_total - r)
    
    return x, y, z_hang, r, theta, n_steps, dt, times

# initial simulation
init_disp = 1.05
x, y, z_hang, r, theta, n_steps, dt, times = run_simulation(init_disp)

# subplots
fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 0.1])
ax3d = fig.add_subplot(gs[0], projection='3d')
ax2d = fig.add_subplot(gs[1])
# place the slider in a dedicated area below the subplots
slider_ax = fig.add_axes([0.2, 0.03, 0.6, 0.03])

plt.subplots_adjust(hspace=0.35, bottom=0.13, top=0.97)

# 3d plot 
ax3d.set_xlim(-1.5, 1.5)
ax3d.set_ylim(-1.5, 1.5)
ax3d.set_zlim(-1.2, 0.5)
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
xx, yy = np.meshgrid(np.linspace(-1.5,1.5,10), np.linspace(-1.5,1.5,10))
zz = np.zeros_like(xx)
ax3d.plot_surface(xx, yy, zz, color='lightgray', alpha=0.5)
mass_rot, = ax3d.plot([], [], [], 'ro', markersize=8, label='M')
mass_hang, = ax3d.plot([], [], [], 'bo', markersize=8, label='m')
string_line, = ax3d.plot([], [], [], 'k-', lw=1)
trajectory_line, = ax3d.plot([], [], [], 'g-', lw=1, alpha=0.7, label='Trajectory')

# 2d plot 
radial_line, = ax2d.plot([], [], 'b-', lw=2)
ax2d.set_xlim(0, times[-1])
ax2d.set_ylim(0.9 * np.min(r), 1.1 * np.max(r))
ax2d.set_xlabel('Time (s)')
ax2d.set_ylabel('Radial Distance r(t) (m)')
ax2d.set_title('Radial Distance vs Time')
ax2d.grid(True, alpha=0.3)

# slider for initial displacement
# use the dedicated slider_ax
disp_slider = Slider(slider_ax, 'Initial Radial Displacement', 0.95, 1.10, valinit=init_disp, valstep=0.001)

# animation functions
frame_data = {'x': x, 'y': y, 'z_hang': z_hang, 'r': r, 'times': times}

def init():
    mass_rot.set_data([], [])
    mass_rot.set_3d_properties([])
    mass_hang.set_data([], [])
    mass_hang.set_3d_properties([])
    string_line.set_data([], [])
    string_line.set_3d_properties([])
    trajectory_line.set_data([], [])
    trajectory_line.set_3d_properties([])
    radial_line.set_data([], [])
    return mass_rot, mass_hang, string_line, trajectory_line, radial_line

def update(frame):
    x = frame_data['x']
    y = frame_data['y']
    z_hang = frame_data['z_hang']
    r = frame_data['r']
    times = frame_data['times']
    xr, yr = x[frame], y[frame]
    zr = 0
    mass_rot.set_data([xr], [yr])
    mass_rot.set_3d_properties([zr])
    xh, yh = 0, 0
    zh = z_hang[frame]
    mass_hang.set_data([xh], [yh])
    mass_hang.set_3d_properties([zh])
    string_line.set_data([xh, xr], [yh, yr])
    string_line.set_3d_properties([zh, zr])
    # trajectory line
    trajectory_line.set_data(x[:frame+1], y[:frame+1])
    trajectory_line.set_3d_properties(np.zeros(frame+1))
    # update 2d plot up to current frame
    radial_line.set_data(times[:frame+1], r[:frame+1])
    ax2d.set_ylim(0.9 * np.min(r), 1.1 * np.max(r))
    return mass_rot, mass_hang, string_line, trajectory_line, radial_line

ani = FuncAnimation(fig, update, frames=n_steps, init_func=init,
                    blit=True, interval=dt*1000)

def slider_update(val):
    new_disp = disp_slider.val
    x, y, z_hang, r, theta, n_steps, dt, times = run_simulation(new_disp)
    frame_data['x'] = x
    frame_data['y'] = y
    frame_data['z_hang'] = z_hang
    frame_data['r'] = r
    frame_data['times'] = times
    ax2d.set_ylim(0.9 * np.min(r), 1.1 * np.max(r))
    ani.event_source.stop()
    ani.new_frame_seq()
    ani.event_source.start()

disp_slider.on_changed(slider_update)

ax3d.legend()
plt.show()
