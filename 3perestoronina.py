import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp

A, K, Ms = 2e-11, 4e4, 1.7e6
mu0 = 4 * np.pi * 1e-7
d0 = np.sqrt(A / K)
HK = 2 * K / (mu0 * Ms)
L = 5 * np.pi

print(
    f'Delta0 = {d0*1e9:.2f} нм, delta0_form = {np.pi*d0*1e9:.2f} нм, HK ='
    f' {HK/1e3:.2f} кА/м'
)

def solve(h):
  x = np.linspace(-L, L, 1000)
  y0 = np.vstack((2 * np.arctan(np.exp(x)), 1 / np.cosh(x)))
  ode = lambda x, y: np.vstack((y[1], np.sin(y[0]) * np.cos(y[0]) - h * np.cos(y[0])))
  bc = lambda ya, yb: np.array([ya[0], yb[0] - np.pi])
  return solve_bvp(ode, bc, x, y0, tol=1e-6, max_nodes=10000)


def get_stats(sol):
  xf = np.linspace(-L, L, 20000)
  th = sol.sol(xf)[0]
  w = np.interp(0.9 * np.pi, th, xf) - np.interp(0.1 * np.pi, th, xf)
  c = np.interp(np.pi / 2, th, xf)
  return w, c


hs = [0.0, 0.05, 0.10, 0.20]
sols, ws, cs = [], [], []

for h in hs:
  s = solve(h)
  w, c = get_stats(s)
  sols.append(s)
  ws.append(w)
  cs.append(c)
  print(
      f'h={h:.2f}: w={w:.4f} d0 ({w*d0*1e9:.2f} нм), center={c:.2e}'
  )

xa = np.linspace(-L, L, 1000)
tha = 2 * np.arctan(np.exp(xa))

plt.figure(figsize=(8, 5))
plt.plot(xa, tha, 'k--', label='аналитика, h=0')
for h, s in zip(hs, sols):
  plt.plot(s.x, s.y[0], 'k-' if h == 0 else '-', label=f'h={h}')
plt.xlabel(r'$x/\Delta_0$')
plt.ylabel(r'$\theta$, рад')
plt.title('Профиль доменной границы')
plt.legend()
plt.tight_layout()
plt.savefig('theta_profiles.pdf')
plt.savefig('theta_profiles.png', dpi=200)
plt.close()

plt.figure(figsize=(12, 5))
for i, (fn, name) in enumerate([(np.cos, 'M_z'), (np.sin, 'M_x')], 1):
  plt.subplot(1, 2, i)
  plt.plot(xa, fn(tha), 'k--', label='аналитика, h=0')
  for h, s in zip(hs, sols):
    plt.plot(s.x, fn(s.y[0]), 'k-' if h == 0 else '-', label=f'h={h}')
  plt.xlabel(r'$x/\Delta_0$')
  plt.ylabel(rf'${name}/M_s$')
  plt.title(f'${name}$')
  plt.legend()
plt.tight_layout()
plt.savefig('M_components.pdf')
plt.savefig('M_components.png', dpi=200)
plt.close()

h_arr = np.array(hs)
ratio = np.array(ws) / ws[0]

plt.figure(figsize=(6, 4))
plt.plot(h_arr, ratio, 'o-')
plt.xlabel(r'$H_\perp/H_K$')
plt.ylabel(r'$\delta/\delta_0$')
plt.title('Ширина границы в поперечном поле')
plt.tight_layout()
plt.savefig('width_vs_field.pdf')
plt.savefig('width_vs_field.png', dpi=200)
plt.close()

hf = np.linspace(0, 0.21, 200)
C = np.sum((ratio - 1) * h_arr**2) / np.sum(h_arr**4)
p = np.polyfit(h_arr, ratio, 2)

plt.figure(figsize=(8, 5))
plt.scatter(h_arr, ratio, color='k', zorder=5)
plt.plot(
    hf, np.polyval(p, hf), label=f'${p[0]:.2f}h^2 + {p[1]:.2f}h + {p[2]:.3f}$'
)
plt.plot(hf, 1 + C * hf**2, 'r--', label=f'$1 + O(h^2)$: $1 + {C:.2f}h^2$')
plt.xlabel(r'$h = H_\perp / H_K$')
plt.ylabel(r'$\delta(H_\perp) / \delta_0$')
plt.legend()
plt.savefig('delta_vs_H.pdf')
plt.close()

print(f'\nC = {C:.4f}, poly = {p}')
