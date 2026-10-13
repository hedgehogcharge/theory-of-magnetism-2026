import numpy as np
import matplotlib.pyplot as plt
import time
sn, cs = np.sin, np.cos
pi = np.pi

def f(h, phi, theta):
    return 0.5 * sn(2 * phi) - h * sn(theta - phi)

def diff_f(h, theta, phi):
    return cs(2 * phi) + h * cs(theta - phi)

def newton(h, theta, a, b, fa, tol=1e-4, maxiter=50):

    x = 0.5 * (a + b)
    for _ in range(maxiter):
        fx = f(h, x, theta)
        dfx = diff_f(h, theta, x)
        dfx = np.where(np.abs(dfx) < 1e-12, 1e-12, dfx)
        xn = x - fx / dfx

        bad = (xn <= a) | (xn >= b)
        xn = np.where(bad, 0.5 * (a + b), xn)

        fn = f(h, xn, theta)
        same = fa * fn > 0
        a = np.where(same, xn, a)
        b = np.where(same, b, xn)
        fa = np.where(same, fn, fa)
        x = xn

    return x


def roots_of_f(h, theta, n=1000):
    phi = np.linspace(-pi, pi, n)
    fv = f(h, phi, theta)

    exact = phi[np.abs(fv) < 1e-10].tolist()

    idx = np.where(np.diff(np.sign(fv)) != 0)[0]
    if len(idx) == 0:
        return np.array(exact)

    a = phi[idx].copy()
    b = phi[idx + 1].copy()
    fa = fv[idx].copy()

    roots = newton(h, theta, a, b, fa)

    return np.array(exact + roots.tolist())


def equilibrium(h, theta, phi_prev):
    roots = roots_of_f(h, theta)
    if len(roots) == 0:
        return phi_prev
    d2 = diff_f(h, theta, roots)
    stable = roots[d2 > 0]
    if len(stable) == 0:
        return phi_prev
    d = np.abs(stable - phi_prev)
    d = np.minimum(d, 2 * pi - d)
    return float(stable[np.argmin(d)])

def loop(theta, hmax=1.5, n=400):
    hs_d = np.linspace(hmax, -hmax, n)
    m_d = np.empty(n)
    phi = theta
    for i, h in enumerate(hs_d):
        phi = equilibrium(h, theta, phi)
        m_d[i] = cs(theta - phi)

    hs_u = np.linspace(-hmax, hmax, n)
    m_u = np.empty(n)
    for i, h in enumerate(hs_u):
        phi = equilibrium(h, theta, phi)
        m_u[i] = cs(theta - phi)

    return hs_d, m_d, hs_u, m_u


def hc_num(theta, hmax=1.5, n=600):
    hs, m, _, _ = loop(theta, hmax, n)
    idx = np.where(np.diff(np.sign(m)) != 0)[0]
    if len(idx) == 0:
        return 0.0
    i = idx[0]
    h0, h1 = hs[i], hs[i + 1]
    m0, m1 = m[i], m[i + 1]
    if m1 == m0:
        return abs(h0)
    return abs(h0 - m0 * (h1 - h0) / (m1 - m0))


def astroid(n=1000):
    phi = np.linspace(0, 2 * pi, n)
    return -sn(phi) ** 3, cs(phi) ** 3


def hc(theta):
    return 1.0 / (abs(sn(theta)) ** (2 / 3) + abs(cs(theta)) ** (2 / 3)) ** 1.5


def plot_loops(thetas_deg=(0, 30, 60, 90), hmax=1.5, n=400,
               fname='loops.png'):
    plt.figure(figsize=(8, 6))
    for i, t in enumerate(thetas_deg):
        th = np.deg2rad(t)
        hs_d, m_d, hs_u, m_u = loop(th, hmax, n)
        plt.plot(hs_d, m_d, color=f'C{i}', label=rf'$\theta$={t}°')
        plt.plot(hs_u, m_u, color=f'C{i}')
    plt.xlabel('h')
    plt.ylabel('m')
    plt.title('Петли гистерезиса m(h)')
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def plot_astroid(fname='astroid.png'):
    hx, hz = astroid()
    plt.figure(figsize=(6, 6))
    plt.plot(hx, hz)
    plt.xlabel('$h_x$')
    plt.ylabel('$h_z$')
    plt.title('Астроида Стопера-Вольфарта')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def plot_hc(theta_deg, hc_vals, fname='hc_theta.png'):
    th_rad = np.deg2rad(theta_deg)
    hc_theory = hc(th_rad)

    plt.figure(figsize=(8, 5))
    plt.plot(theta_deg, hc_vals, '-', label='численный расчёт')
    plt.plot(theta_deg, hc_theory, '--', label=r'теория')
    plt.xlabel('θ, градусы')
    plt.ylabel('$h_c$')
    plt.title(r'Коэрцитивная сила $h_c(\theta)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

def main():
    t0 = time.time()
    plot_loops()
    plot_astroid()

    th_deg = np.linspace(0, 90, 91)
    hc = [hc_num(np.deg2rad(t), n=400) for t in th_deg]
    plot_hc(th_deg, hc)

    print('hc(0°)  =', round(hc[0], 4))
    print('hc(45°) =', round(hc[len(hc) // 2], 4))
    print('hc(90°) =', round(hc[-1], 4))
    print(f'время: {time.time() - t0:.2f} с')

if __name__ == '__main__':
    main()
