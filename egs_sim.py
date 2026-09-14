"""
egs_sim.py -- a fast 2-D thermal-hydraulic model of an Enhanced Geothermal System (EGS).

Captures the core design physics of Singh et al. (2025, URTeC 4245311, "Project Cape"):
heat mined from a hot fractured reservoir by circulating water between injectors and producers
across STACKED BENCHES, and the tradeoff between net power and thermal breakthrough as a
function of well spacing and bench spacing.

Physics (single-phase water, local thermal equilibrium, quasi-steady Darcy flow):
  pressure :  -div(k/mu grad p) = wells                        (TPFA div-grad operator)
  heat     :  (rho c)_bulk dT/dt + (rho c)_w div(v T) = lam grad^2 T
             -> the cold front advects at v_water / R  (thermal retardation -> decades)
  power    :  P = mdot * c_w * (T_prod - T_reinj) * eta        (net electric)

x = lateral (injector<->producer), z = depth (two stacked benches), thickness H into the page.
NOT a fracture-propagation simulator (ResFrac's job) -- this is the fast, transparent, interactive
twin/design layer on top.  Calibrated to the paper's ballpark (~129 kg/s, ~400F, ~9 MWe/producer).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

FT = 0.3048
DAY = 86400.0
YEAR = 365.25 * DAY
RHO_W, C_W = 1000.0, 4186.0
RHOC_ROCK = 2650.0 * 900.0
LAMBDA = 2.8


class EGS2D:
    def __init__(self, Lx=1400*FT, Lz=1700*FT, nx=44, nz=40, H=200*FT,
                 k_mD=50.0, phi=0.06, mu=3e-4,
                 T_top=196.0, geo_grad=0.03, T_inj=40.0, T_res=None,
                 eta=0.11, retard_boost=48.0):
        self.nx, self.nz, self.H = nx, nz, H
        self.Lx, self.Lz = Lx, Lz
        self.dx, self.dz = Lx/nx, Lz/nz
        self.N = nx*nz
        self.k = k_mD * 9.869e-16
        self.phi, self.mu, self.T_inj, self.eta = phi, mu, T_inj, eta
        z = (np.arange(nz)+0.5)*self.dz
        if T_res is not None:                                            # anchor window centre to reservoir T
            self.T_init = np.tile((T_res + geo_grad*(z - Lz/2))[:, None], (1, nx))
        else:
            self.T_init = np.tile((T_top + geo_grad*z)[:, None], (1, nx))     # deeper = hotter
        self.rhoc_bulk = ((1-phi)*RHOC_ROCK + phi*RHO_W*C_W) * retard_boost
        self.id = np.arange(self.N).reshape(nz, nx)
        self._build_operators()

    def _build_operators(self):
        nx, nz, H = self.nx, self.nz, self.H
        rows, cols, vals = [], [], []
        f = 0
        for j in range(nz):
            for i in range(nx-1):
                rows += [f, f]; cols += [self.id[j, i], self.id[j, i+1]]; vals += [-1, 1]; f += 1
        self.nxf = f
        for j in range(nz-1):
            for i in range(nx):
                rows += [f, f]; cols += [self.id[j, i], self.id[j+1, i]]; vals += [-1, 1]; f += 1
        self.G = sp.csr_matrix((vals, (rows, cols)), shape=(f, self.N))
        Tx = self.k/self.mu * (H*self.dz)/self.dx
        Tz = self.k/self.mu * (H*self.dx)/self.dz
        self.Tface = np.concatenate([np.full(self.nxf, Tx), np.full(f-self.nxf, Tz)])
        cx = LAMBDA*(H*self.dz)/self.dx; cz = LAMBDA*(H*self.dx)/self.dz
        self.condw = np.concatenate([np.full(self.nxf, cx), np.full(f-self.nxf, cz)])
        # face endpoints (L,R)
        self.aL = np.array([self.G.indices[self.G.indptr[r]]   for r in range(f)])
        self.aR = np.array([self.G.indices[self.G.indptr[r]+1] for r in range(f)])

    def _cell(self, x, z):
        return self.id[min(self.nz-1, max(0, int(z/self.dz))), min(self.nx-1, max(0, int(x/self.dx)))]

    def set_wells(self, well_spacing=500*FT, bench_spacing=700*FT, Q_bpd=70000.0):
        self.Qp = self.Qi = Q_bpd * 0.1589873 / DAY                       # bbl/day -> m^3/s (per well)
        xc, zc = self.Lx/2, self.Lz/2
        zu, zl = zc - bench_spacing/2, zc + bench_spacing/2
        xi, xp = xc - well_spacing/2, xc + well_spacing/2
        self.inj = [self._cell(xi, zu), self._cell(xi, zl)]
        self.prod = [self._cell(xp, zu), self._cell(xp, zl)]
        self.wells_xz = dict(inj=[(xi, zu), (xi, zl)], prod=[(xp, zu), (xp, zl)])

    def set_gunbarrel(self, n_inj=3, n_prod=5, bench_spacing=700*FT, Q_bpd_per_prod=70000.0,
                      inj_bench="lower", prod_bench="upper"):
        """Place n_inj injectors and n_prod producers across the (1-mile-wide) gunbarrel section.
        Injectors default to the lower bench, producers the upper bench.  Rates are mass-balanced:
        each producer makes Q_prod; injectors share the total (Q_inj = Q_prod * n_prod / n_inj)."""
        zc = self.Lz/2
        zu, zl = zc - bench_spacing/2, zc + bench_spacing/2
        z_inj = zl if inj_bench == "lower" else zu
        z_prod = zu if prod_bench == "upper" else zl
        xi = np.linspace(self.Lx/(2*n_inj), self.Lx - self.Lx/(2*n_inj), n_inj)
        xp = np.linspace(self.Lx/(2*n_prod), self.Lx - self.Lx/(2*n_prod), n_prod)
        self.Qp = Q_bpd_per_prod * 0.1589873 / DAY
        self.Qi = self.Qp * n_prod / n_inj                                # balance total inj = total prod
        self.inj = [self._cell(x, z_inj) for x in xi]
        self.prod = [self._cell(x, z_prod) for x in xp]
        self.wells_xz = dict(inj=[(x, z_inj) for x in xi], prod=[(x, z_prod) for x in xp])

    def solve_pressure(self):
        A = (self.G.T @ sp.diags(self.Tface) @ self.G).tolil()
        b = np.zeros(self.N)
        for c in self.inj:  b[c] += self.Qi
        for c in self.prod: b[c] -= self.Qp
        A[0, :] = 0; A[0, 0] = 1; b[0] = 0
        p = spsolve(A.tocsr(), b)
        self.Fface = -self.Tface * (self.G @ p)                           # Darcy: q = -T grad p (m^3/s, +dir)
        return p

    def run(self, years=30.0, n_snaps=16, cfl=0.4):
        self.solve_pressure()
        nz, nx = self.nz, self.nx
        T = self.T_init.ravel().copy()
        for c in self.inj: T[c] = self.T_inj
        Vcell = self.dx*self.dz*self.H
        rc_w = RHO_W*C_W
        F, aL, aR, condw = self.Fface, self.aL, self.aR, self.condw
        # per-cell throughput CFL (flux converges at wells -> single-face max is not enough)
        absF = np.abs(F)
        thru = np.bincount(aL, weights=absF, minlength=self.N) + np.bincount(aR, weights=absF, minlength=self.N)
        for c in self.prod: thru[c] += self.Qp                             # production sink adds outflow
        adv_dt = cfl*self.rhoc_bulk*Vcell/max((rc_w*thru).max(), 1e-30)
        con_dt = cfl*self.rhoc_bulk*Vcell/max(condw.max()*4, 1e-30)
        dt = min(adv_dt, con_dt)
        nsteps = int(np.ceil(years*YEAR/dt)); dt = years*YEAR/nsteps
        snap_at = set(np.linspace(0, nsteps, n_snaps).astype(int).tolist())
        times, Tprod_hist, power_hist, snaps = [], [], [], []
        for step in range(nsteps+1):
            if step in snap_at:
                snaps.append(T.reshape(nz, nx).copy())
            Tp = T[self.prod]
            power = float((rc_w*self.Qp*(Tp - self.T_inj)*self.eta).sum()/1e6)   # total, all producers
            times.append(step*dt/YEAR); Tprod_hist.append(float(Tp.mean())); power_hist.append(max(power, 0.0))
            if step == nsteps:
                break
            Tup = np.where(F >= 0, T[aL], T[aR])
            H = rc_w*F*Tup + condw*(T[aL]-T[aR])                            # advection (upwind) + conduction
            dE = np.bincount(aR, weights=H, minlength=self.N) - np.bincount(aL, weights=H, minlength=self.N)
            for c in self.prod: dE[c] -= rc_w*self.Qp*T[c]                  # produced fluid leaves at cell T
            T = T + dt/(self.rhoc_bulk*Vcell)*dE
            T[self.inj] = self.T_inj                                        # injectors held at cold T
        self.T_final = T.reshape(nz, nx)
        return dict(t=np.array(times), Tprod=np.array(Tprod_hist), power=np.array(power_hist),
                    snaps=snaps, T_init=self.T_init.copy(),
                    breakthrough=self._breakthrough(np.array(times), np.array(Tprod_hist)))

    @staticmethod
    def _breakthrough(t, Tp, frac=0.1):
        span = max(Tp[0]-Tp[-1], 1.0)
        below = np.where(Tp < Tp[0]-frac*span)[0]
        return float(t[below[0]]) if len(below) else float(t[-1])


if __name__ == "__main__":
    import time
    m = EGS2D(); m.set_wells(500*FT, 700*FT, 70000.0)
    t0 = time.time(); r = m.run(years=30.0, n_snaps=8); dt = time.time()-t0
    f2c = lambda c: c*9/5+32
    print("base case 500x700 ft, 70k bbl/d  [%.0fs, %d snaps]" % (dt, len(r['snaps'])))
    print("  produced T:  %.0fC->%.0fC   (%.0fF->%.0fF)" % (r['Tprod'][0], r['Tprod'][-1], f2c(r['Tprod'][0]), f2c(r['Tprod'][-1])))
    print("  net power :  %.1f -> %.1f MWe" % (r['power'][0], r['power'][-1]))
    print("  breakthrough onset ~ %.1f yr" % r['breakthrough'])
    for yr in [0, 2, 5, 15, 30]:
        k = int(np.argmin(np.abs(r['t']-yr)))
        print("   t=%2dyr  Tprod=%3.0fC  P=%.1f MWe" % (yr, r['Tprod'][k], r['power'][k]))
