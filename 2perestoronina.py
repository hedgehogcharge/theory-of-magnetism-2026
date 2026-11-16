import time
from functools import partial

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import minimum_filter, label


def make_grid(N_theta, N_phi):
    theta = np.linspace(0, np.pi, N_theta)
    phi = np.linspace(0, 2*np.pi, N_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    sT = np.sin(T)
    return theta, T, P, sT*np.cos(P), sT*np.sin(P), np.cos(T)


theta, Theta, Phi, ax, ay, az = make_grid(120, 240)
theta_c, Tc, Pc, axc, ayc, azc = make_grid(60, 120)


def w_uniaxial(ax, ay, K1, K2=0.0):
    s2 = ax**2 + ay**2
    return K1*s2 + K2*s2**2


def w_cubic(ax, ay, az, K1c, K2c=0.0):
    return (K1c*(ax**2*ay**2 + ay**2*az**2 + az**2*ax**2)
            + K2c*ax**2*ay**2*az**2)


def w_combined(ax, ay, az, K1c, Kef, n, K2c=0.0):
    return (w_cubic(ax, ay, az, K1c, K2c)
            + Kef*(1 - (ax*n[0] + ay*n[1] + az*n[2])**2))


def w_total(ax, ay, az, w_an, h, hdir):
    return w_an - 2.0*h*(ax*hdir[0] + ay*hdir[1] + az*hdir[2])


W_uni = partial(w_uniaxial, ax, ay)
W_cub = partial(w_cubic, ax, ay, az)
W_combo = partial(w_combined, ax, ay, az)
W_tot = partial(w_total, ax, ay, az)

Wc_uni = partial(w_uniaxial, axc, ayc)
Wc_cub = partial(w_cubic, axc, ayc, azc)
Wc_combo = partial(w_combined, axc, ayc, azc)
Wc_tot = partial(w_total, axc, ayc, azc)


FS_TITLE = 18
FS_LABEL = 16
FS_TICK = 13
FS_CBAR = 15
FS_CBAR_TICK = 13
FS_SUPTITLE = 20
FS_2D_TITLE = 16
FS_2D_AXIS = 17
FS_2D_TICK = 15
FS_2D_LEGEND = 14


def surface_data(w, shift=0.15):
    w_min, w_max = w.min(), w.max()
    w_norm = ((w - w_min)/(w_max - w_min)
              if w_max - w_min > 1e-12 else np.zeros_like(w))
    r = w_norm + shift
    X = r*np.sin(Theta)*np.cos(Phi)
    Y = r*np.sin(Theta)*np.sin(Phi)
    Z = r*np.cos(Theta)
    return X, Y, Z, w_norm


def add_surface(ax3d, w, title='', cmap=cm.jet,
                cbar_label=r'$\tilde w$', lim=1.3):
    X, Y, Z, w_norm = surface_data(w)
    ax3d.plot_surface(X, Y, Z, facecolors=cmap(w_norm),
                      rstride=2, cstride=2, linewidth=0,
                      antialiased=False)
    ax3d.set_title(title, fontsize=FS_TITLE, pad=10)
    ax3d.view_init(elev=20, azim=45)
    ax3d.set_xlabel('X', fontsize=FS_LABEL)
    ax3d.set_ylabel('Y', fontsize=FS_LABEL)
    ax3d.set_zlabel('Z', fontsize=FS_LABEL)
    ax3d.tick_params(axis='both', labelsize=FS_TICK)
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)
    w_min, w_max = float(w.min()), float(w.max())
    m = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(w_min, w_max))
    m.set_array([])
    cb = ax3d.figure.colorbar(m, ax=ax3d, shrink=0.55, pad=0.08, aspect=15)
    cb.set_label(cbar_label, fontsize=FS_CBAR)
    cb.ax.tick_params(labelsize=FS_CBAR_TICK)


def count_min(w, theta, tol_rel=1e-4, size=5):
    Nt, Np = w.shape
    w_range = w.max() - w.min()
    if w_range < 1e-14:
        return 0
    tol = tol_rel * w_range

    pad = size // 2
    we = np.concatenate([w[:, -pad:], w, w[:, :pad]], axis=1)
    minf = minimum_filter(we, size=(size, size), mode='nearest')
    cand = ((we - minf) < tol)[:, pad:-pad].copy()

    pole_count = 0
    if theta[0] < 1e-8:
        if float(w[1, :].min()) > float(w[0, 0]) + 1e-12:
            pole_count += 1
        cand[0, :] = False
    if theta[-1] > np.pi - 1e-8:
        if float(w[-2, :].min()) > float(w[-1, 0]) + 1e-12:
            pole_count += 1
        cand[-1, :] = False

    if not cand.any():
        return pole_count

    padded = np.concatenate([cand[:, -1:], cand, cand[:, :1]], axis=1)
    labeled, _ = label(padded, structure=np.ones((3, 3), dtype=int))

    parent = list(range(int(labeled.max()) + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    L, R = labeled[:, 0], labeled[:, Np + 1]
    for i in range(Nt):
        if L[i] > 0 and R[i] > 0:
            union(int(L[i]), int(R[i]))
        if i + 1 < Nt:
            if L[i] > 0 and R[i + 1] > 0:
                union(int(L[i]), int(R[i + 1]))
            if L[i + 1] > 0 and R[i] > 0:
                union(int(L[i + 1]), int(R[i]))

    center = labeled[:, 1:Np + 1]
    roots = {find(int(v)) for v in np.unique(center) if v > 0}
    return len(roots) + pole_count


def min_curve(w0, hvals, hdir_x):
    return np.array([count_min(Wc_tot(w0, h, hdir_x), theta_c)
                     for h in hvals])


def fig_aniso_types():
    cases = [
        (W_uni(1.0), r'(а) Одноосная $K_1>0$'),
        (W_cub(1.0), r'(в) Кубическая $K_{1c}>0$'),
        (W_cub(-1.0), r'(г) Кубическая $K_{1c}<0$'),
    ]
    fig = plt.figure(figsize=(16, 14))
    for i, (w, title) in enumerate(cases):
        ax3d = fig.add_subplot(2, 2, i+1, projection='3d')
        add_surface(ax3d, w, title, cbar_label=r'$\tilde w_a$')
    fig.tight_layout()
    fig.savefig('fig_aniso_types.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig_combined():
    ratios = [0.1, 0.5, 1.0, 2.0]
    n_z = np.array([0.0, 0.0, 1.0])
    n_diag = np.array([1.0, 1.0, 1.0])/np.sqrt(3)
    for fname, n_vec, label in [
        ('fig_combined_z.png', n_z, r'$\mathbf{n}\parallel z$'),
        ('fig_combined_diag.png', n_diag, r'$\mathbf{n}\parallel(111)$'),
    ]:
        fig = plt.figure(figsize=(16, 14))
        for i, k in enumerate(ratios):
            w = W_combo(1.0, k, n_vec)
            ax3d = fig.add_subplot(2, 2, i+1, projection='3d')
            add_surface(ax3d, w, rf'$K_{{ef}}/K_{{1c}}={k}$, {label}',
                        cbar_label=r'$\tilde w_{ef}$')
        fig.tight_layout()
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)


def fig_field():
    hvals = [0.0, 0.5, 1.2, 4.0]
    hdir_x = np.array([1.0, 0.0, 0.0])
    n_z = np.array([0.0, 0.0, 1.0])

    cases = [
		
        ('fig_field_uniaxial.png',
         r'Одноосная $K_1>0$, $\mathbf{h}\parallel x$',
         W_uni(1.0)),
        ('fig_field_cubic_pos.png',
         r'Кубическая $K_{1c}>0$, $\mathbf{h}\parallel x$',
         W_cub(1.0)),
        ('fig_field_cubic_neg.png',
         r'Кубическая $K_{1c}<0$, $\mathbf{h}\parallel x$',
         W_cub(-1.0)),
        ('fig_field_combined.png',
         r'Комбинированная, $K_{ef}/K_{1c}=2$, $\mathbf{h}\parallel x$',
         W_combo(1.0, 2, n_z)),
    ]
    for fname, suptitle, w0 in cases:
        fig = plt.figure(figsize=(16, 14))
        for i, h in enumerate(hvals):
            w = W_tot(w0, h, hdir_x)
            ax3d = fig.add_subplot(2, 2, i+1, projection='3d')
            add_surface(ax3d, w, f'$h={h}$', cbar_label=r'$\tilde w_i$')
        fig.suptitle(suptitle, fontsize=FS_SUPTITLE)
        fig.tight_layout()
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)


def fig_min_cubic_uniaxial():
    hvals = np.linspace(0, 5, 51)
    hdir_x = np.array([1.0, 0.0, 0.0])
    n_uni_p = min_curve(Wc_uni(1.0), hvals, hdir_x)
    n_cub_p = min_curve(Wc_cub(1.0), hvals, hdir_x)
    n_cub_n = min_curve(Wc_cub(-1.0), hvals, hdir_x)

    fig, axp = plt.subplots(figsize=(11, 7))
    axp.plot(hvals, n_uni_p, 'o-', color='C0',
             label=r'Одноосная $K_1>0$', lw=2, ms=6)
    axp.plot(hvals, n_cub_p, 's-', color='C2',
             label=r'Кубическая $K_{1c}>0$', lw=2, ms=6)
    axp.plot(hvals, n_cub_n, 's--', color='C3',
             label=r'Кубическая $K_{1c}<0$', lw=2, ms=6)

    h_cub = 2.0/(3.0*np.sqrt(6.0))
    for hc, txt in [(h_cub, r'$h_c^{\rm cub}=2/(3\sqrt{6})$'),
                    (1.0, r'$h_c=1$')]:
        axp.axvline(hc, color='gray', ls=':', alpha=0.6)
        axp.text(hc + 0.03, 11.3, txt, fontsize=12, color='gray')

    axp.set_xlabel(r'$h = H/H_K$', fontsize=FS_2D_AXIS)
    axp.set_ylabel('Число минимумов', fontsize=FS_2D_AXIS)
    axp.set_title(r'Число минимумов от поля: одноосная $K_1>0$, '
                  r'кубическая $K_{1c}>0$ и $K_{1c}<0$ ($\mathbf{h}\parallel x$)',
                  fontsize=FS_2D_TITLE)
    axp.tick_params(labelsize=FS_2D_TICK)
    axp.set_yticks(range(0, 12))
    axp.set_ylim(-0.3, 11.5)
    axp.grid(True, alpha=0.3)
    axp.legend(fontsize=FS_2D_LEGEND, loc='upper right')
    fig.tight_layout()
    fig.savefig('fig_minima_cubic_uniaxial.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    print('  h   n_uni+ n_cub+ n_cub-')
    for h, a, b, c in zip(hvals, n_uni_p, n_cub_p, n_cub_n):
        print(f'{h:5.2f} {a:6d} {b:6d} {c:6d}')


def fig_min_combined():
    hvals = np.linspace(0, 5, 51)
    hdir_x = np.array([1.0, 0.0, 0.0])
    n_z = np.array([0.0, 0.0, 1.0])
    ratios = [0.1, 0.5, 1.0, 2.0]
    styles = ['o-', 's-', '^-', 'd-']

    n_combo = np.array([min_curve(Wc_combo(1.0, k, n_z), hvals, hdir_x)
                        for k in ratios])

    fig, axp = plt.subplots(figsize=(11, 7))
    for i, (k, style) in enumerate(zip(ratios, styles)):
        axp.plot(hvals, n_combo[i], style, color=f'C{i}', lw=2, ms=6,
                 label=rf'$K_{{ef}}/K_{{1c}}={k}$')

    axp.set_xlabel(r'$h = H/H_K$', fontsize=FS_2D_AXIS)
    axp.set_ylabel('Число минимумов', fontsize=FS_2D_AXIS)
    axp.set_title(r'Комбинированная анизотропия ($\mathbf{n}\parallel z$, '
                  r'$\mathbf{h}\parallel x$): число минимумов от поля',
                  fontsize=FS_2D_TITLE)
    axp.tick_params(labelsize=FS_2D_TICK)
    axp.set_yticks(range(0, 9))
    axp.set_ylim(-0.3, 8.5)
    axp.grid(True, alpha=0.3)
    axp.legend(fontsize=FS_2D_LEGEND, loc='upper right',
               title=r'$K_{ef}/K_{1c}$', title_fontsize=FS_2D_LEGEND)
    fig.tight_layout()
    fig.savefig('fig_minima_combined.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    print('   h  ' + ' '.join(f'k={k:<4}' for k in ratios))
    for j, h in enumerate(hvals):
        print(f'{h:5.2f} ' +
              ' '.join(f'{n_combo[i, j]:5d}' for i in range(len(ratios))))


if __name__ == '__main__':
    t0 = time.time()
    #fig_aniso_types()
    fig_combined()
    fig_field()
    #fig_min_cubic_uniaxial()
    fig_min_combined()
    print(f'Время: {time.time()-t0:.1f} с')
