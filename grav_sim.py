from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, fft
from scipy.optimize import curve_fit

class grav_obj:

    def __init__(self, mass, pos, vel, name):
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.name = name
        self.half_vel = np.zeros(3)
        self.accel_accumulator = np.zeros(3)
        self.pos_history = np.empty((N,3))

    def __repr__(self) -> str:
        return f"{self.name} grav object"

    def draw_obj(self, ax):
        pos = self.pos_history
        ax.plot(pos[:,0], pos[:,1])
        ax.scatter(pos[-1,0], pos[-1,1])
    
def grav_effect(obj1, obj2):
    """ return grav "force" between obj1 and obj2, with sign such that it acts on obj1 """
    R_vec = obj1.pos-obj2.pos
    R = np.linalg.norm(R_vec)
    f = -G*R_vec/R**3
    return f

def compute_pairs(objects):
    """ return list of unique pairs of objects"""
    elements = range(len(objects))
    pairs = list(combinations(elements, 2))
    return pairs



def vel_verlet(objects, pairs):
    
    for pair in pairs:
        
        obj1 = objects[pair[0]]
        obj2 = objects[pair[1]]
        
    
        accel_over_m = grav_effect(obj1, obj2)
    
    
        obj1.accel_accumulator = obj1.accel_accumulator + accel_over_m*obj2.mass
        obj2.accel_accumulator = obj2.accel_accumulator - accel_over_m*obj1.mass
    
    for obj in objects:
        obj.half_vel = obj.vel + 1/2*obj.accel_accumulator*dt
        obj.pos = obj.pos + obj.half_vel*dt
        obj.accel_accumulator = np.zeros(3)
        

    for pair in pairs:
        obj1 = objects[pair[0]]
        obj2 = objects[pair[1]]
        force = grav_effect(obj1, obj2)
        obj1.accel_accumulator = obj1.accel_accumulator + force*obj2.mass
        obj2.accel_accumulator = obj2.accel_accumulator - force*obj1.mass
    
    for obj in objects:
        obj.vel = obj.half_vel + 1/2*obj.accel_accumulator*dt
        obj.accel_accumulator = np.zeros(3)

def orb_vel(a, T):
    T = T*60**2*24
    return 2*np.pi*a/T

def CoM(objects):
    centre = np.zeros(3)
    M_total = 0
    for object in objects:
        M_total += object.mass
        centre = centre + object.pos*object.mass

    return centre/M_total

def fit(freqs, phases, A):
    def fit_func(t, norm):
        
        out = 0
        for i in range(len(freqs)):
            out += A[i]*norm*np.cos(np.pi*2/freqs[i]*t+phases[i])
        return out
    return fit_func

N = 10000
M_S = 1.32712440018e11
M_E = 3.986004418e5
AU = 1.496e+8 
G = 1 
dt = 3600*24/10 # s


solar_system = [grav_obj(M_S, np.array([0,0,0]), np.array([0,0,0]), "Sun"),
           grav_obj(M_E, np.array([AU,0,0]), np.array([0,29.8,0]), "Earth"),
           grav_obj(M_E*0.107, np.array([AU*1.523679,0,0]), np.array([0,24.1,0]), "Mars")]

trappist_1 = [grav_obj(M_S*0.0898, np.array([0,0,0]), np.array([0,0,0]), "Trappist 1a")] \
    +[grav_obj(0.93*M_E, np.array([0.0385*AU,0,0]), np.array([0, orb_vel(0.0385*AU, 9.207540), 0]), "Trappist 1b")] \
+[grav_obj(0.33*M_E, np.array([0.0619*AU,0,0]), np.array([0, orb_vel(0.0619*AU, 18.772866), 0]), "Trappist 1c")] \
+[grav_obj(1.15*M_E, np.array([0.0469*AU,0,0]), np.array([0, orb_vel(0.0469*AU, 12.352446), 0]), "Trappist 1d")]\
+[grav_obj(0.77*M_E, np.array([0.0293*AU,0,0]), np.array([0, orb_vel(0.0293*AU, 6.101013), 0]), "Trappist 1e")] \
+[grav_obj(1.02*M_E, np.array([0.0115*AU,0,0]), np.array([0, orb_vel(0.0115*AU, 1.510826), 0]), "Trappist 1f")] \
+[grav_obj(1.16*M_E, np.array([0.0158*AU,0,0]), np.array([0, orb_vel(0.0158*AU, 2.421937), 0]), "Trappist 1g")] \
+[grav_obj(0.3*M_E, np.array([0.0223*AU,0,0]), np.array([0, orb_vel(0.0223*AU, 4.049219), 0]), "Trappist 1h")]  \

objects = trappist_1



pairs = compute_pairs(objects)
star_vel = np.empty((N,3))
for i in range(N):
    vel_verlet(objects, pairs)
    for n, object in enumerate(objects):
        object.pos_history[i,:] = object.pos-CoM(objects)
        if n==0:
            star_vel[i,:] = object.vel

fig, ax = plt.subplots()

dT = dt/(60**2*24) # in days



fft_star = fft.rfft(star_vel[:,0]*1e3)
fft_star_real = np.abs(fft.rfft(star_vel[:,0]*1e3))

fac = N/1000
max_val = np.max([fft_star_real])
ax.vlines(1/9.207540, 0, np.max(max_val)*1.3, color="red", label="True Periods")
ax.vlines(1/18.772866, 0, np.max(max_val)*1.3, color="red")
ax.vlines(1/6.101013, 0, np.max(max_val)*1.3, color="red")
ax.vlines(1/12.352446, 0, np.max(max_val)*1.3, color="red")

ax.vlines(1/4.049219, 0, np.max(max_val)*1.3, color="red")
ax.vlines(1/2.421937, 0, np.max(max_val)*1.3, color="red")
ax.vlines(1/1.510826, 0, np.max(max_val)*1.3, color="red")

frequency_space = fft.rfftfreq(N, dt)*(3600*24)
ax.plot(frequency_space[:int(N/fac)], fft_star_real[:int(N/fac)], label="Star Velocity FFT")

peaks = signal.find_peaks(fft_star_real)[0][:7]
frequencies = []
phases = []
widths = []
amplitudes = []

for n, peak in enumerate(peaks):
    frequencies.append(1/(frequency_space[peak]))
    phases.append(np.angle(fft_star[peak]))
    amplitudes.append(fft_star_real[peak])


ax.legend(loc="best")
ax.set_xlabel("Frequency [days]")
popt, pcov = curve_fit(fit(frequencies, phases, amplitudes), dT*np.arange(N), star_vel[:,0])
fitted_curve = fit(frequencies, phases, amplitudes)


plt.show()