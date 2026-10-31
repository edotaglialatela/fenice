"""Spectra of time series and of periodic fields, NumPy FFT only.

    from fenice import spectra
    f, P, df = spectra.psd(lift, dt)                  # power spectral density of a series, df = frequency resolution
    fpk = spectra.peaks(f, P, 3)                      # the three strongest peaks (frequency, power), interpolated
    St = spectra.strouhal(fpk[0][0], D=1.0, U=1.0)
    k, E = spectra.energy_spectrum(u, v, w=None, lengths=(2*np.pi,)*2)       # shell-averaged E(k)
    sol, dil = spectra.helmholtz(u, v)                # solenoidal / dilatational parts of a velocity field

Conventions.
Time series (D1): `resample_uniform` linearly interpolates a series sampled at a variable step onto a uniform grid (the error
of a linear interpolation is of the order (dt/T)^2 of the series' curvature; the function returns the largest step ratio so
the caller can judge). `psd` removes a linear (default) or constant trend from every segment, multiplies by a Hann window,
averages the periodograms of segments of `nperseg` points overlapping by `overlap`, and returns the one-sided density
(units^2 / frequency) so that  sum(P) df  equals the windowed mean square of the detrended segments, sum((w y)^2) / sum(w^2),
to roundoff for one segment (the Parseval identity of the module's check). The frequency resolution is df = 1 / (nperseg dt);
`peaks` interpolates a peak with a parabola through the log power of the three bins around it.
Fields (D2): a field is a periodic array on a uniform grid, axes ordered (z, y, x) in 3-D or (y, x) in 2-D, one array per
velocity component, domain lengths `lengths` ordered like the axes' reverse (Lx, Ly[, Lz]). The mean kinetic energy per unit
mass is 1/2 <|u|^2>. `energy_spectrum` assigns every Fourier mode to the shell n = round(|k| / k0), k0 = 2 pi / L_min is the
smallest wavenumber unit (all directions are assumed to share it: use equal lengths), and returns E(n) with
sum_n E(n) = 1/2 <|u|^2> exactly (Parseval: every mode, corners included, is in some shell). Shells above N/2 (the Nyquist
shell of the shortest axis) are the corner modes of a cube, they are returned but flagged by the `nyquist` value (k index
N/2) of the result dict: treat them as unresolved. With `rho` the density-weighted velocity sqrt(rho) u is transformed
(kinetic energy 1/2 rho |u|^2, mean over the domain). `helmholtz` splits u = u_sol + u_dil with u_dil = k (k . u_hat) / |k|^2
in Fourier space, the mean flow (k = 0) kept in the solenoidal part.
"""
import numpy as np


def resample_uniform(t, y, dt=None):
    """(t_u, y_u, step_ratio): the series linearly interpolated onto a uniform grid of step dt (default: the median step).
    step_ratio = largest original step / dt: above 1 the interpolation fills a gap with a straight line."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    steps = np.diff(t)
    if np.any(steps <= 0):
        raise ValueError("times must be increasing")
    dt = float(np.median(steps)) if dt is None else float(dt)
    n = int(np.floor((t[-1] - t[0]) / dt)) + 1
    tu = t[0] + dt * np.arange(n)
    return tu, np.interp(tu, t, y), float(steps.max() / dt)


def _detrend(seg, kind):
    if kind in (None, "none"):
        return seg
    if kind == "mean":
        return seg - seg.mean()
    x = np.arange(len(seg), dtype=float)
    p = np.polyfit(x, seg, 1)
    return seg - np.polyval(p, x)


def psd(y, dt, nperseg=None, overlap=0.5, detrend="linear"):
    """One-sided power spectral density by Welch's method with a Hann window: (f, P, df)."""
    y = np.asarray(y, float)
    n = len(y)
    nperseg = n if nperseg is None else min(int(nperseg), n)
    step = max(1, int(round(nperseg * (1 - overlap)))) if nperseg < n else nperseg
    w = np.hanning(nperseg)   # symmetric Hann, zero at both ends
    scale = 1.0 / ((1.0 / dt) * np.sum(w * w))
    acc, count = 0.0, 0
    for s in range(0, n - nperseg + 1, step):
        seg = _detrend(y[s:s + nperseg], detrend)
        Y = np.fft.rfft(w * seg)
        acc = acc + np.abs(Y) ** 2
        count += 1
    P = acc / count * scale
    P[1:] *= 2.0   # one-sided: fold the negative frequencies
    if nperseg % 2 == 0:
        P[-1] /= 2.0   # the Nyquist bin has no mirror
    f = np.fft.rfftfreq(nperseg, dt)
    return f, P, 1.0 / (nperseg * dt)


def peaks(f, P, count=3):
    """The `count` strongest local maxima of P as [(frequency, power)], parabola-interpolated, strongest first."""
    f, P = np.asarray(f), np.asarray(P)
    idx = [i for i in range(1, len(P) - 1) if P[i] > P[i - 1] and P[i] >= P[i + 1]]
    idx.sort(key=lambda i: -P[i])
    out = []
    for i in idx[:count]:
        if min(P[i - 1], P[i], P[i + 1]) <= 0:
            out.append((float(f[i]), float(P[i])))
            continue
        a, b, c = np.log(P[i - 1]), np.log(P[i]), np.log(P[i + 1])   # a Hann-windowed peak is close to a Gaussian in log power
        den = a - 2 * b + c
        d = 0.0 if den == 0 else 0.5 * (a - c) / den
        out.append((float(f[i] + d * (f[1] - f[0])), float(np.exp(b - 0.25 * (a - c) * d))))
    return out


def strouhal(frequency, D, U):
    """St = f D / U."""
    return frequency * D / U


def _wavevectors(shape, lengths):
    """Integer wavenumber index arrays (in units of 2 pi / L_min) for np.fft.fftn of an array of this shape; axes are
    (z, y, x) or (y, x), lengths = (Lx, Ly[, Lz])."""
    L = list(lengths)[::-1]   # to the axes' order
    kmin = 2 * np.pi / min(L)
    ks = [np.fft.fftfreq(n, d=1.0 / n) * (2 * np.pi / l) / kmin for n, l in zip(shape, L)]
    return np.meshgrid(*ks, indexing="ij")


def energy_spectrum(u, v, w=None, lengths=None, rho=None):
    """Shell-averaged kinetic-energy spectrum: {"k" (shell numbers), "E", "nyquist" (shell index N/2), "mean_energy"}.

    u, v, (w): arrays of one shape, (y, x) or (z, y, x). lengths: (Lx, Ly[, Lz]), default 2 pi each (then shell n is the
    integer wavenumber n). sum(E) == mean kinetic energy 1/2 <|u|^2> (times rho if given: 1/2 <rho |u|^2>)."""
    comps = [np.asarray(c, float) for c in (u, v, w) if c is not None]
    shape = comps[0].shape
    lengths = lengths or (2 * np.pi,) * len(shape)
    if rho is not None:
        comps = [np.sqrt(np.asarray(rho, float)) * c for c in comps]
    N = np.prod(shape)
    power = sum(np.abs(np.fft.fftn(c)) ** 2 for c in comps) / N ** 2   # mean of |c|^2 split over modes
    K = _wavevectors(shape, lengths)
    kk = np.sqrt(sum(k * k for k in K))
    shell = np.rint(kk).astype(int)
    E = np.bincount(shell.ravel(), weights=0.5 * power.ravel())
    return {"k": np.arange(len(E)), "E": E, "nyquist": min(shape) // 2,
            "mean_energy": float(0.5 * sum(np.mean(c * c) for c in comps))}


def helmholtz(u, v, w=None, lengths=None):
    """(solenoidal, dilatational): each a tuple of component arrays, from the Fourier-space projection of the velocity."""
    comps = [np.asarray(c, float) for c in (u, v, w) if c is not None]
    shape = comps[0].shape
    lengths = lengths or (2 * np.pi,) * len(shape)
    L = list(lengths)[::-1]
    ks = np.meshgrid(*[np.fft.fftfreq(n, d=1.0 / n) * (2 * np.pi / l) for n, l in zip(shape, L)], indexing="ij")
    # component i of the velocity varies along the axis that carries it: (u, v, w) <-> axes (x, y, z) = last, second to last, first
    kvec = [ks[len(shape) - 1 - i] for i in range(len(comps))]
    k2 = sum(k * k for k in kvec)
    k2[(k2 == 0)] = 1.0   # the mean flow: no dilatational part (k . u_hat = 0 as k = 0)
    hats = [np.fft.fftn(c) for c in comps]
    div = sum(k * h for k, h in zip(kvec, hats))
    dil_hat = [k * div / k2 for k in kvec]
    dil = tuple(np.real(np.fft.ifftn(d)) for d in dil_hat)
    sol = tuple(c - d for c, d in zip(comps, dil))
    return sol, dil
