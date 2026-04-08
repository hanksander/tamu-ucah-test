"""
Core gas dynamic relations for the scramjet cycle.

Kept minimal: only what the inlet/isolator/combustor/nozzle modules
actually call.  All functions take explicit γ so they work for both
air (γ ≈ 1.40) and combustion products (γ ≈ 1.20–1.30).
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq, minimize_scalar


# ── SHARED DATA STRUCTURE ─────────────────────────────────────────────────────

@dataclass
class FlowState:
    """Thermodynamic + kinematic state at a cycle station."""
    M:     float   # Mach number
    T:     float   # static temperature [K]
    P:     float   # static pressure [Pa]
    Pt:    float   # total pressure [Pa]
    Tt:    float   # total temperature [K]
    gamma: float   # ratio of specific heats
    R:     float   # specific gas constant [J/(kg·K)]

    @property
    def a(self) -> float:
        """Speed of sound [m/s]."""
        return (self.gamma * self.R * self.T) ** 0.5

    @property
    def V(self) -> float:
        """Flow velocity [m/s]."""
        return self.M * self.a

    @property
    def rho(self) -> float:
        """Density [kg/m³]."""
        return self.P / (self.R * self.T)

    def __str__(self) -> str:
        return (f"M={self.M:.3f}  T={self.T:.1f}K  Tt={self.Tt:.1f}K  "
                f"P={self.P/1e3:.2f}kPa  Pt={self.Pt/1e3:.2f}kPa  γ={self.gamma:.4f}")


def make_state(M: float, T: float, P: float,
               gamma: float = 1.4, R: float = 287.05) -> FlowState:
    """Construct a FlowState, computing Pt and Tt from isentropic relations."""
    fac = 1.0 + (gamma - 1.0) / 2.0 * M ** 2
    return FlowState(M=M, T=T, P=P,
                     Pt=P * fac ** (gamma / (gamma - 1.0)),
                     Tt=T * fac,
                     gamma=gamma, R=R)


# ── ISENTROPIC ────────────────────────────────────────────────────────────────

def isentropic_T(Tt: float, M: float, gam: float) -> float:
    return Tt / (1.0 + (gam - 1.0) / 2.0 * M ** 2)


def isentropic_P(Pt: float, M: float, gam: float) -> float:
    return Pt / (1.0 + (gam - 1.0) / 2.0 * M ** 2) ** (gam / (gam - 1.0))


def isentropic_M_from_Pt_P(Pt_over_P: float, gam: float) -> float:
    """Invert Pt/P ratio → M."""
    return np.sqrt(2.0 / (gam - 1.0) * (Pt_over_P ** ((gam - 1.0) / gam) - 1.0))


# ── NORMAL SHOCK ──────────────────────────────────────────────────────────────

def normal_shock(M1: float, gam: float) -> tuple[float, float, float, float]:
    """
    Rankine-Hugoniot relations.
    Returns (M2, P2/P1, T2/T1, Pt2/Pt1).
    """
    if M1 <= 1.0:
        return M1, 1.0, 1.0, 1.0
    g = gam
    M2    = np.sqrt(((g - 1.0) * M1**2 + 2.0) / (2.0 * g * M1**2 - (g - 1.0)))
    P2_P1 = (2.0 * g * M1**2 - (g - 1.0)) / (g + 1.0)
    T2_T1 = P2_P1 * (2.0 + (g - 1.0) * M1**2) / ((g + 1.0) * M1**2)
    fac1  = ((g + 1.0) / 2.0 * M1**2 / (1.0 + (g - 1.0) / 2.0 * M1**2)) ** (g / (g - 1.0))
    fac2  = (2.0 * g / (g + 1.0) * M1**2 - (g - 1.0) / (g + 1.0)) ** (-1.0 / (g - 1.0))
    return M2, P2_P1, T2_T1, fac1 * fac2


# ── OBLIQUE SHOCK ─────────────────────────────────────────────────────────────

def _theta_from_beta(beta_rad: float, M1: float, gam: float) -> float:
    Mn1_sq = (M1 * np.sin(beta_rad)) ** 2
    if Mn1_sq <= 1.0:
        return 0.0
    numer = 2.0 / np.tan(beta_rad) * (Mn1_sq - 1.0)
    denom = M1**2 * (gam + np.cos(2.0 * beta_rad)) + 2.0
    return np.arctan(numer / denom)


def beta_from_theta(theta_deg: float, M1: float, gam: float) -> float | None:
    """Weak-shock wave angle β [rad] for deflection θ [deg] at M1. None if detached."""
    theta = np.radians(theta_deg)
    b_min = np.arcsin(1.0 / M1) + 1e-6
    b_max = np.pi / 2.0 - 1e-6
    betas = np.linspace(b_min, b_max, 300)
    fvals = np.array([_theta_from_beta(b, M1, gam) - theta for b in betas])
    roots = [brentq(lambda b, i=i: _theta_from_beta(b, M1, gam) - theta,
                    betas[i], betas[i+1], xtol=1e-12)
             for i in range(len(betas)-1) if fvals[i]*fvals[i+1] < 0]
    return min(roots) if roots else None


def oblique_shock(
    M1: float, theta_deg: float, gam: float
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """
    Full oblique shock solution.  Returns (M2, P2/P1, T2/T1, Pt2/Pt1, β[rad]).
    All None if shock detaches (θ > θ_max).
    """
    beta = beta_from_theta(theta_deg, M1, gam)
    if beta is None:
        return None, None, None, None, None
    Mn1 = M1 * np.sin(beta)
    M2n, P2_P1, T2_T1, Pt2_Pt1 = normal_shock(Mn1, gam)
    M2  = M2n / np.sin(beta - np.radians(theta_deg))
    return M2, P2_P1, T2_T1, Pt2_Pt1, beta


# ── RAYLEIGH FLOW ─────────────────────────────────────────────────────────────

def _rayleigh_Tt_ratio(M: float, gam: float) -> float:
    """Tt/Tt* for Rayleigh flow at Mach M."""
    f = 1.0 + gam * M**2
    return 2.0 * (1.0 + gam) * M**2 * (1.0 + (gam-1.0)/2.0 * M**2) / f**2


def _rayleigh_Pt_ratio(M: float, gam: float) -> float:
    """Pt/Pt* for Rayleigh flow at Mach M."""
    f = 1.0 + gam * M**2
    return ((1.0 + gam) / f) * (
        (2.0/(gam+1.0)) * (1.0 + (gam-1.0)/2.0 * M**2)
    ) ** (gam/(gam-1.0))


def rayleigh_exit(M3: float, Tt4_Tt3: float, gam: float,
                  supersonic: bool = True) -> tuple[float, float, bool]:
    """
    Given combustor-entrance Mach M3 and total-temperature ratio Tt4/Tt3,
    return (M4, Pt4/Pt3, choked).

    supersonic=True  → scramjet (M > 1 branch)
    supersonic=False → ramjet  (M < 1 branch)
    """
    Tt3_Tts = _rayleigh_Tt_ratio(M3, gam)
    Tt4_Tts = Tt3_Tts * Tt4_Tt3

    if Tt4_Tts >= 1.0:
        M4     = 1.0
        choked = True
    else:
        lo, hi = (1.0 + 1e-7, 30.0) if supersonic else (1e-6, 1.0 - 1e-7)
        try:
            M4 = brentq(lambda M: _rayleigh_Tt_ratio(M, gam) - Tt4_Tts,
                        lo, hi, xtol=1e-12)
            choked = False
        except Exception:
            M4, choked = 1.0, True

    Pt4_Pt3 = _rayleigh_Pt_ratio(M4, gam) / _rayleigh_Pt_ratio(M3, gam)
    return M4, Pt4_Pt3, choked


# ── INLET UTILITIES ───────────────────────────────────────────────────────────

def pi_milspec(M0: float) -> float:
    """MIL-E-5007D inlet total-pressure recovery (fallback / validation benchmark)."""
    if M0 < 1.0:  return 1.0
    if M0 <= 5.0: return 1.0 - 0.075 * (M0 - 1.0) ** 1.35
    return 800.0 / (M0**4 + 935.0)


def kantrowitz_limit(M0: float, gam: float) -> float:
    """Maximum A_throat/A_capture for self-starting (Kantrowitz 1945)."""
    def area_ratio(M):
        return (1.0/M) * ((2.0/(gam+1.0))*(1.0+(gam-1.0)/2.0*M**2))**((gam+1.0)/(2.0*(gam-1.0)))
    M_ns, *_ = normal_shock(M0, gam)
    return area_ratio(M_ns) / area_ratio(M0)


if __name__ == '__main__':
    print("=== Normal shock at M=6.0, γ=1.40 ===")
    M2, PP, TT, PtPt = normal_shock(6.0, 1.40)
    print(f"  M2={M2:.4f}, P2/P1={PP:.3f}, T2/T1={TT:.3f}, Pt2/Pt1={PtPt:.5f}")

    print("\n=== Oblique shock: M0=6, θ=7° ===")
    M2, PP, TT, PtPt, beta = oblique_shock(6.0, 7.0, 1.36)
    print(f"  β={np.degrees(beta):.2f}°, M2={M2:.4f}, P2/P1={PP:.3f}, Pt2/Pt1={PtPt:.5f}")

    print("\n=== Rayleigh flow: M3=3.0, Tt4/Tt3=1.5 ===")
    M4, Pt4Pt3, choked = rayleigh_exit(3.0, 1.5, 1.25)
    print(f"  M4={M4:.4f}, Pt4/Pt3={Pt4Pt3:.5f}, choked={choked}")

    print("\n=== Kantrowitz limit M=6 ===")
    kl = kantrowitz_limit(6.0, 1.4)
    print(f"  A_t/A_c limit = {kl:.4f}")
