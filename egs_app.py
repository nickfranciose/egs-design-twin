"""
egs_app.py -- interactive EGS design twin: a GUNBARREL well-pattern editor + live economics.

Design levers (sidebar, tabbed):
  Wells      : producers/mile (slider), injectors/mile, injector depth (TVD), bench separation
  Completion : lateral length, proppant lb/ft, fluid bbl/ft, pump rate bbl/min, producer rate
  Economics  : drilling $/ft, completion $ line items, power price
Visualizations (main, tabbed):
  Gunbarrel  : live thermal-hydraulic cross-section, injectors lower / producers upper bench
  Economics  : decomposed cost, single-well PV vs density, total DSU NPV vs density
  Performance: per-producer power & produced-temperature decline
  Sensitivities: 3-D NPV/IRR surfaces over any two levers (dropdown-selected)

Run:  reservoir_simulation/pysim/pyresvenv/bin/streamlit run egs_app.py     (from egs_twin/)
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from egs_sim import EGS2D, FT
from egs_econ import EGSEconomics

st.set_page_config(page_title="EGS Design Twin", layout="wide")
NX, NZ, MILE = 96, 44, 5280*FT
SNX, SNZ = 64, 36                          # coarse grid for sweeps/surfaces
T_SURF, GEO_C_PER_FT = 15.0, 0.0222        # reservoir T(depth): ~204C at 8500 ft TVD (Project Cape ballpark)
BASE_RETARD = 48.0                         # contacted-rock / thermal-retardation factor at the reference job
SAND_REF, FLUID_REF = 2000.0, 1500.0       # reference completion intensity (A-index = 1); fluid in gal/ft
LATERAL_REF = 5000.0                        # lateral (ft) the 2-D cross-section physics is calibrated to
def lat_gain(L):
    """Per-producer power scales ~linearly with lateral length: a longer lateral contacts proportionally
    more reservoir. The 2-D gunbarrel physics is a per-mile cross-section (no lateral dimension), so power
    is flat in L without this. With it, revenue grows with L while the FIXED vertical drilling cost is
    amortized over more feet — so more lateral dilutes the vertical capex and improves the economics."""
    return float(L)/LATERAL_REF
def f2c(c): return c*9/5+32
def T_res_of(depth_ft): return T_SURF + GEO_C_PER_FT*depth_ft
def nearest(arr, v):
    arr = np.asarray(arr); return arr[int(np.argmin(np.abs(arr - v)))]
DELIV_MAX, DELIV_H = 1.03, 3              # deliverability plateau (distance limit) and Hill sharpness
DELIV_K = DELIV_MAX - 1.0                  # so deliv=1 at the reference job (S=1)
def deliv_frac(sand_lb_ft, fluid_gal_ft):
    """Stimulation -> deliverable power, a sharply-saturating (Hill) diminishing-returns curve.
    The SINGLE channel from completion intensity to power: →0 at no stimulation (no surface area, no
    flow), =1 at the reference job, and nearly flat beyond it — the reference already fills the drainage
    volume, so DOUBLING a component buys only ~2-3% more power.  Because that thin, discounted benefit
    is overtaken by the (undiscounted, up-front) doubling cost, cumulative power keeps diminishing to the
    plateau while the economics PEAK just past the reference and then decline."""
    S = (fluid_gal_ft/FLUID_REF) * (sand_lb_ft/SAND_REF)           # =1 at the reference job (1500 gal/ft, 2000 lb/ft)
    return float(DELIV_MAX * S**DELIV_H / (S**DELIV_H + DELIV_K))  # DELIV_MAX·1/(1+K)=1 at S=1

SUPPORT_ALPHA, SUPPORT_REF_RATIO = 6.0, 0.6   # injection-support sharpness and the optimal injector:producer ratio
def inj_support(n_inj, n_prod):
    """Injection support on producer power vs the injector:producer ratio. Too few injectors → producers
    are under-supported (pressure/thermal recharge) → power falls; enough → saturates (=1 at the optimal
    ratio, base case). Over-injecting adds cost with ~no power gain. This puts the NPV ridge on a ratio
    line and, with the density parabola, makes the producers×injectors surface a peaked saddle."""
    return (1.0 - np.exp(-SUPPORT_ALPHA*n_inj/n_prod)) / (1.0 - np.exp(-SUPPORT_ALPHA*SUPPORT_REF_RATIO))

BENCH_CONN_SEP, BENCH_CONN_W = 800.0, 400.0   # bench sep (ft) where inter-bench connectivity begins to fail, and its width
def bench_conn(sep):
    """Inter-bench connectivity / fluid transfer vs bench separation. Full up to ~the fracture network's
    vertical reach, then falls off as the injector and producer benches get too far apart for the fractures
    to bridge. Combined with the sim's early-breakthrough at small separation, this makes bench separation a
    parabola: too close → fast thermal breakthrough, too far → inadequate connectivity."""
    return 1.0 / (1.0 + (max(0.0, sep - BENCH_CONN_SEP)/BENCH_CONN_W)**2)

INTERF_NTHR, INTERF_SLOPE = 6.0, 0.05      # wells/mile with no interference, then linear deliverability derate
def interf(n):
    """Well-interference deliverability factor: 1 up to ~paper spacing, then linear decline as
    producers pack tighter and share drainage.  Makes single-well PV fall ~linearly with density."""
    return float(np.clip(1.0 - INTERF_SLOPE*max(n - INTERF_NTHR, 0.0), 0.0, 1.0))

# make the tab groups read as discrete panels (applies to sidebar + main tabs)
st.markdown("""<style>
.stTabs [role="tablist"]{ gap:6px; }
.stTabs [data-testid="stTab"]{ border:1px solid #9ea5b0 !important; border-radius:6px 6px 2px 2px !important;
    padding:6px 12px !important; background:transparent !important; flex:1 1 0 !important;
    justify-content:center !important; }
.stTabs [data-testid="stTab"]:hover{ border-color:#6b7280 !important; }
.stTabs [data-testid="stTab"][aria-selected="true"]{ border-color:#6b7280 !important;
    background:light-dark(#ffffff, #3f414d) !important; font-weight:600 !important; }
.stTabs [data-testid="stTabPanel"]{ border:1px solid #cfd3da; border-radius:8px; padding:14px 16px; margin-top:6px; }
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ levers (sidebar, tabbed)
with st.sidebar:
    st.title("Design levers")
    tab_wells, tab_frac, tab_compl, tab_econ = st.tabs(["Wells", "Survey viz", "Completion", "Cost model"])
    with tab_wells:
        n_prod = st.slider("Producer wells / mile", 1, 20, 5)
        n_inj = st.slider("Injector wells / mile (lower bench)", 1, 10, 3)
        inj_depth = st.slider("Injector depth (ft TVD)", 6000, 11000, 8500, 250)
        sep = st.slider("Bench separation (ft)", 100, 1200, 300, 50)
    with tab_compl:
        lateral = st.slider("Lateral length (ft)", 3000, 8000, 5000, 250)
        sand = st.slider("Proppant intensity (lb/ft)", 500, 4000, 2000, 100)
        water_gal = st.slider("Fluid intensity (gal/ft)", 200, 3000, 1500, 25)
        pump = st.slider("Pump rate (bbl/min)", 20, 200, 100, 5)
        Qprod = st.slider("Producer rate (k bbl/d)", 30, 120, 70, 5) * 1000
    with tab_frac:
        stage_len = st.slider("Stage length (ft)", 100, 500, 250, 25)
        clus_per_stage = st.slider("Clusters / stage", 3, 12, 8)
        frac_xf = st.slider("Avg frac half-length (ft)", 50, 2000, 800, 10)
        frac_up = st.slider("Avg vertical height, up (ft)", 20, 400, 150, 10)
        frac_down = st.slider("Avg downward height (ft)", 20, 400, 100, 10)
        frac_noise = st.slider("Geometry noise (%)", 0, 100, 50, 5) / 100.0
        frac_asym = st.checkbox("Asymmetric (biwing) growth", value=True)
        frac_iso = st.checkbox("1:1 z : x/y scale (true geometry)", value=True)   # equal data units on all axes
        frac_ap = st.slider("Peak aperture (mm)", 1.0, 15.0, 6.0, 0.5)
        frac_pct = st.slider("Fracture outcome (percentile)", 10, 90, 50, 5)   # P10..P90 growth range (Bui Fig 5 a→c)
        frac_g = (frac_pct - 10) / 80.0                    # 0 at P10 .. 1 at P90
        frac_gf = 0.5 + frac_g                             # size factor: P10 0.5×, P50 1×, P90 1.5× the avg wing
        frac_prop = frac_g                                 # aperture morphology (a→c) rides the percentile
    with tab_econ:
        cvt = st.number_input("Drilling  $/vertical ft", 100, 900, 450, 25)
        chz = st.number_input("Drilling  $/horizontal ft", 100, 900, 550, 25)
        csand = st.number_input("$/sand ton", 100, 600, 300, 10)
        cwater = st.slider("$/bbl water", 0.5, 6.0, 2.5, 0.25)
        chhp = st.slider("HHP $/hr (k)", 4, 30, 12, 1) * 1000
        cspread = st.number_input("$/day frac spread (k)", 40, 400, 150, 10) * 1000
        coverstim = st.slider("Over-stim penalty (exp)", 1.0, 3.0, 2.0, 0.1)
        copex = st.slider("Opex ($k/mo/well)", 0, 50, 10, 1) * 1000
        price = st.slider("Power price ($/MWh)", 40, 140, 80, 5)
    st.caption(f"Reservoir ≈ {T_res_of(inj_depth):.0f} °C ({f2c(T_res_of(inj_depth)):.0f} °F) at {inj_depth:,} ft"
               f"  ·  deliverability ≈ {deliv_frac(sand, water_gal):.2f}× (stimulation → power, diminishing to {DELIV_MAX}×)")

# fingerprint of every lever -> cache key for econ sweeps / surfaces
_ek = (lateral, sand, water_gal, pump, cvt, chz, csand, cwater, chhp, cspread, coverstim, copex, price)
FP = (n_prod, n_inj, sep, inj_depth, Qprod) + _ek


def make_econ():
    return EGSEconomics(vertical_depth_ft=inj_depth, cost_vt_ft=cvt, cost_hz_ft=chz, cost_sand_ton=csand,
                        cost_water_gal=cwater/42.0, cost_hhp_hr=chhp, cost_day_spread=cspread, sand_lb_per_ft=sand,
                        water_gal_per_ft=water_gal, pump_rate_bpm=pump, lateral_ft=lateral, power_price=price,
                        overstim_exp=coverstim, opex_per_well_mo=copex)


@st.cache_data(show_spinner=False)
def run_design(n_inj, n_prod, sep, Qprod, inj_depth, retard):
    m = EGS2D(Lx=MILE, Lz=1700*FT, nx=NX, nz=NZ, T_res=T_res_of(inj_depth), retard_boost=retard)
    m.set_gunbarrel(n_inj, n_prod, sep*FT, Qprod)
    r = m.run(years=30, n_snaps=10)
    r["wells"] = m.wells_xz; r["Lx"] = m.Lx; r["Lz"] = m.Lz; r["Tfield"] = m.T_final
    return r


@st.cache_data(show_spinner=False)
def phys_run(n_inj, n_prod, sep, Qprod, inj_depth, retard):
    """Cheap thermal-hydraulic run -> (t, power). Cached, so repeated sweep points are free.
    `retard` is the fixed thermal-retardation (breakthrough physics); stimulation enters via deliverability."""
    m = EGS2D(Lx=MILE, Lz=1700*FT, nx=SNX, nz=SNZ, T_res=T_res_of(inj_depth), retard_boost=retard)
    m.set_gunbarrel(int(n_inj), int(n_prod), sep*FT, Qprod)
    r = m.run(years=30, n_snaps=2)
    return r["t"], r["power"]


@st.cache_data(show_spinner=False)
def density_sweep(n_inj, sep, Qprod, inj_depth, retard, dfac, ek):
    """Sweep producers/mile.
      single-well PV = per-producer revenue PV − that producer's OWN capex  (≈ linear, negative slope)
      DSU NPV       = single-well PV × producers − injector capital          (a true parabola)
    Low end starts negative (injectors are sunk before producers pay out); high end goes negative
    (well interference). Returns ns, single, dsu, kopt, injector-capital anchor."""
    econ = make_econ(); cap = econ.well_capex()["total"]; capMM = cap/1e6
    ns = np.arange(1, 21)                                    # cut at 20 producers/mile
    single = np.empty(len(ns))
    for i, n in enumerate(ns):
        t, power = phys_run(n_inj, n, sep, Qprod, inj_depth, retard)
        pp = power/n * interf(n) * dfac * inj_support(n_inj, n) * bench_conn(sep) * lat_gain(lateral)
        rev = econ.project(t, pp, 0.0, n_wells=1)["npv"]/1e6  # revenue PV per producer, net of its own opex
        single[i] = rev - capMM                              # net of the producer's own drill+frac
    inj_pv = econ.project(t, np.zeros_like(t), n_inj*cap, n_wells=n_inj)["npv"]/1e6   # injectors: −capex −opex, no revenue
    dsu = ns*single + inj_pv                                 # plus (negative) injector fixed PV -> parabola
    return ns, single, dsu, int(np.argmax(dsu)), -inj_pv


@st.cache_data(show_spinner=False)
def wells_1to1_sweep(sep, Qprod, inj_depth, retard, dfac, ek):
    """Vary producers = injectors 1:1 (paired wine-rack). Returns single-well PV (per producer, net of its
    own capex/opex) and total DSU NPV (all producers + injectors) vs the paired count."""
    econ = make_econ(); cap = econ.well_capex()["total"]; capMM = cap/1e6
    ws = np.arange(1, 21)
    single = np.empty(len(ws)); dsu = np.empty(len(ws))
    for i, w in enumerate(ws):
        t, power = phys_run(w, w, sep, Qprod, inj_depth, retard)
        pp = power/w * interf(w) * dfac * inj_support(w, w) * bench_conn(sep) * lat_gain(lateral)
        single[i] = econ.project(t, pp, 0.0, n_wells=1)["npv"]/1e6 - capMM       # per producer, net own capex/opex
        dsu[i] = econ.project(t, pp*w, (2*w)*cap, n_wells=2*w)["npv"]/1e6         # w producers + w injectors
    return ws, single, dsu, int(np.argmax(dsu))


# ------------------------------------------------------------------ sensitivity-surface machinery
def lever_registry():
    """name -> (is-physics, current value, grid) for the surface explorer."""
    return {
        "Producers / mile":       (True,  n_prod,     np.unique(np.round(np.linspace(1, 20, 9)).astype(int))),
        "Injectors / mile":       (True,  n_inj,      np.arange(1, 9)),
        "Wells/mile (1:1 inj:prod)": (True, n_prod,   np.arange(1, 11)),
        "Injector depth (ft)":    (True,  inj_depth,  np.linspace(6000, 11000, 9).astype(int)),
        "Bench separation (ft)":  (True,  sep,        np.linspace(300, 1200, 9).astype(int)),
        "Producer rate (kbbl/d)": (True,  Qprod/1e3,  np.linspace(30, 120, 9).astype(int)),
        "Lateral (ft)":           (False, lateral,    np.linspace(3000, 8000, 9).astype(int)),
        "Proppant (lb/ft)":       (False, sand,       np.linspace(500, 4000, 9).astype(int)),
        "Fluid (gal/ft)":         (False, water_gal,  np.linspace(300, 3000, 9).astype(int)),
        "Pump rate (bbl/min)":    (False, pump,       np.linspace(20, 200, 9).astype(int)),
        "$/vertical ft":          (False, cvt,        np.linspace(100, 900, 9).astype(int)),
        "$/horizontal ft":        (False, chz,        np.linspace(100, 900, 9).astype(int)),
        "Power price ($/MWh)":    (False, price,      np.linspace(40, 140, 9).astype(int)),
    }


L = lever_registry()


METRICS = ["NPV / DSU ($MM)", "IRR (%)", "NPV / well ($MM)", "Single-well PV ($MM)"]


def eval_point(ov):
    """Full economics + cumulative power at the current sidebar values overridden by `ov`. Breakthrough
    physics is fixed (retard = BASE); stimulation (fluid/proppant) enters ONLY through deliverability, so
    cumulative power vs fluid/proppant is the diminishing deliverability curve while the economics peak."""
    d = {k: v[1] for k, v in L.items()}
    d.update(ov)
    if "Wells/mile (1:1 inj:prod)" in ov:                          # paired sweep: producers & injectors together
        w = int(ov["Wells/mile (1:1 inj:prod)"]); d["Producers / mile"] = w; d["Injectors / mile"] = w
    dfac = deliv_frac(d["Proppant (lb/ft)"], d["Fluid (gal/ft)"])
    t, power = phys_run(d["Injectors / mile"], d["Producers / mile"], float(d["Bench separation (ft)"]),
                        float(d["Producer rate (kbbl/d)"])*1e3, float(d["Injector depth (ft)"]), BASE_RETARD)
    econ = EGSEconomics(vertical_depth_ft=d["Injector depth (ft)"], cost_vt_ft=d["$/vertical ft"],
                        cost_hz_ft=d["$/horizontal ft"], cost_sand_ton=csand, cost_water_gal=cwater/42.0,
                        cost_hhp_hr=chhp, cost_day_spread=cspread, sand_lb_per_ft=d["Proppant (lb/ft)"],
                        water_gal_per_ft=d["Fluid (gal/ft)"], pump_rate_bpm=d["Pump rate (bbl/min)"],
                        lateral_ft=d["Lateral (ft)"], power_price=d["Power price ($/MWh)"], overstim_exp=coverstim)
    cap = econ.well_capex()["total"]; capMM = cap/1e6
    nP, nI = int(d["Producers / mile"]), int(d["Injectors / mile"])
    ppp = (power/nP * interf(nP) * dfac * inj_support(nI, nP) * bench_conn(float(d["Bench separation (ft)"]))
           * lat_gain(d["Lateral (ft)"]))            # longer lateral → proportionally more power per well
    proj = econ.project(t, ppp*nP, (nP + nI) * cap, n_wells=nP + nI)
    single = econ.project(t, ppp, 0.0, n_wells=1)["npv"]/1e6 - capMM   # per-producer PV net of own capex & opex
    cumE = float(np.trapezoid(ppp, t)) * 8760 * econ.capacity_factor / 1e3   # lifetime GWh per producer
    irr = proj["irr"]*100 if np.isfinite(proj["irr"]) else np.nan
    return {"NPV / DSU ($MM)": proj["npv"]/1e6, "IRR (%)": irr, "NPV / well ($MM)": proj["npv"]/(nP + nI)/1e6,
            "Single-well PV ($MM)": single, "Cum power (GWh/well)": cumE}


@st.cache_data(show_spinner=False)
def surface(xname, yname, metric, fp):
    """Returns xg, yg, the chosen-metric grid Z, the single-well-PV grid SW, and the cumulative-power
    grid CUM (all from one physics run per point)."""
    xg, yg = L[xname][2], L[yname][2]
    Z = np.empty((len(yg), len(xg))); SW = np.empty_like(Z); CUM = np.empty_like(Z)
    for j, yv in enumerate(yg):
        for i, xv in enumerate(xg):
            pt = eval_point({xname: xv, yname: yv})
            Z[j, i] = pt[metric]; SW[j, i] = pt["Single-well PV ($MM)"]; CUM[j, i] = pt["Cum power (GWh/well)"]
    return xg, yg, Z, SW, CUM


# ------------------------------------------------------------------ compute
econ = make_econ()
retard = BASE_RETARD                                                # breakthrough physics decoupled from stimulation
dfac = deliv_frac(sand, water_gal)                                  # stimulation -> deliverable power (diminishing)
with st.spinner("running thermal-hydraulic model + economics…"):
    r = run_design(n_inj, n_prod, sep, Qprod, inj_depth, retard)
    ns, single_pv, dsu_npv, kopt, inj_cap = density_sweep(n_inj, sep, Qprod, inj_depth, retard, dfac, _ek)
    ws1, single1, dsu1, kopt1 = wells_1to1_sweep(sep, Qprod, inj_depth, retard, dfac, _ek)
cx = econ.well_capex()
total_capex = (n_prod + n_inj) * cx["total"]
sup = inj_support(n_inj, n_prod) * bench_conn(sep) * lat_gain(lateral)   # support × inter-bench conn × lateral gain
proj = econ.project(r["t"], r["power"]*interf(n_prod)*dfac*sup, total_capex, n_wells=n_prod+n_inj)
pw = r["power"]/n_prod * interf(n_prod) * dfac * sup
i15 = int(np.argmin(np.abs(r["t"]-15)))
here = min(max(n_prod, 1), ns[-1]) - 1                              # index of current density on the sweep
here1 = min(max(n_prod, 1), ws1[-1]) - 1                            # current producer count on the 1:1 axis (approx)

# ------------------------------------------------------------------ header + metrics (always visible)
st.title("EGS Design Twin")
st.caption("Injectors in the lower bench, producers in the upper; a live thermal-hydraulic model + "
           "decomposed cost model. Beyond the power-only study of Singh et al. (2025, URTeC 4245311): "
           "NPV / IRR per DSU, driven by every design & cost lever.")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("NPV / DSU (per mile)", f"${proj['npv']/1e6:.0f}MM")
c2.metric("IRR", f"{proj['irr']*100:.0f}%" if np.isfinite(proj['irr']) else "n/a")
c3.metric("Power / producer", f"{pw[0]:.1f} → {pw[i15]:.1f} MWe")
c4.metric("Thermal breakthrough", f"{r['breakthrough']:.0f} yr")
c5.metric("Well cost", f"${cx['total']/1e6:.1f}MM")
c6.metric("Pump time / well", f"{cx['job_days']:.1f} d")


# ------------------------------------------------------------------ figure builders
def fig_gunbarrel(height):
    T = r["Tfield"]; nz, nx = T.shape
    xc = (np.arange(nx)+0.5)*r["Lx"]/nx/FT; zc = (np.arange(nz)+0.5)*r["Lz"]/nz/FT
    fig = go.Figure(go.Heatmap(x=xc, y=zc, z=f2c(T), colorscale="RdYlBu_r", colorbar=dict(title="T (F)"),
                               hovertemplate="x=%{x:.0f}ft z=%{y:.0f}ft T=%{z:.0f}F<extra></extra>"))
    fig.add_trace(go.Scatter(x=[x/FT for x, z in r["wells"]["inj"]], y=[z/FT for x, z in r["wells"]["inj"]],
                  mode="markers", name="injectors", marker=dict(symbol="triangle-down", size=15, color="#1d4ed8",
                  line=dict(color="white", width=1.5))))
    fig.add_trace(go.Scatter(x=[x/FT for x, z in r["wells"]["prod"]], y=[z/FT for x, z in r["wells"]["prod"]],
                  mode="markers", name="producers", marker=dict(symbol="triangle-up", size=15, color="#b91c1c",
                  line=dict(color="white", width=1.5))))
    fig.update_layout(height=height, template="plotly_white",
                      title=f"Gunbarrel — {n_inj} injectors (lower) + {n_prod} producers (upper) / mile, at 30 yr",
                      xaxis=dict(title="lateral position across the mile (ft)"),
                      yaxis=dict(title="depth into window (ft)", autorange="reversed"),
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=54, l=55, r=20, b=40))
    return fig


def fig_cost(unit):
    perft = unit.startswith("$/hz")
    sc = (lambda x: x/float(lateral)) if perft else (lambda x: x/1e6)
    fmt = (lambda x: f"${x:,.0f}") if perft else (lambda x: f"${x:.2f}MM")
    Dv, Dh, Dr = sc(cx["drill_vertical"]), sc(cx["drill_horizontal"]), sc(cx["drilling"])
    Sa, Wa, Sp, Hp = sc(cx["sand"]), sc(cx["water"]), sc(cx["spread"]), sc(cx["horsepower"])
    Co, Tot = sc(cx["completion"]), sc(cx["total"])
    labels = ["vertical drill", "horizontal drill", "Drilling", "sand", "water", "spread", "horsepower",
              "Completion", "Well total"]
    measure = ["relative", "relative", "total", "relative", "relative", "relative", "relative", "relative", "total"]
    ydelta = [Dv, Dh, 0, Sa, Wa, Sp, Hp, 0, 0]                      # "Completion" is a placeholder (bar overlaid below)
    wtext = [fmt(Dv), fmt(Dh), fmt(Dr), fmt(Sa), fmt(Wa), fmt(Sp), fmt(Hp), "", fmt(Tot)]
    fig = go.Figure()
    fig.add_trace(go.Waterfall(orientation="v", x=labels, measure=measure, y=ydelta,
                  text=wtext, textposition="outside", connector=dict(line=dict(color="#9aa0aa", width=1)),
                  increasing=dict(marker=dict(color="#2563eb")), totals=dict(marker=dict(color="#0f172a"))))
    fig.add_trace(go.Bar(x=["Completion"], y=[Co], base=[Dr], width=0.6, marker_color="#b45309",
                  text=[fmt(Co)], textposition="outside", showlegend=False,
                  hovertemplate="Completion subtotal %{text}<extra></extra>"))
    unit_lab = "$ per horizontal foot" if perft else "$MM per well"
    fig.update_layout(height=380, template="plotly_white", barmode="overlay", showlegend=False,
                      title=f"Well-cost waterfall — drilling {fmt(Dr)} + completion {fmt(Co)} = {fmt(Tot)} "
                            f"({'per hz ft' if perft else 'per well'})",
                      yaxis=dict(title=unit_lab),
                      xaxis=dict(tickangle=-25, categoryorder="array", categoryarray=labels),
                      margin=dict(t=48, l=60, r=15, b=70))
    return fig


def _zero_crossings(x, y):
    s = np.sign(y); idx = np.where(np.diff(s) != 0)[0]
    xs = []
    for i in idx:
        x0, x1, y0, y1 = x[i], x[i+1], y[i], y[i+1]
        xs.append(x0 - y0*(x1-x0)/(y1-y0))                   # linear interpolation to y=0
    return xs


def fig_single(xa, ya, here_i, xlabel="producers per mile", title="Single-well PV vs producers/mile (net of own capex)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xa, y=ya, mode="lines+markers", line=dict(color="#b45309", width=3), name="single-well PV"))
    fig.add_hline(y=0, line=dict(color="#999", width=1, dash="dot"))
    for xc in _zero_crossings(xa, ya):
        fig.add_vline(x=xc, line=dict(color="#111", width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=[xa[here_i]], y=[ya[here_i]], mode="markers+text", text=["  you are here"],
                  textposition="top right", marker=dict(size=12, color="#ea580c", symbol="circle"), showlegend=False))
    fig.update_layout(height=360, template="plotly_white", title=title,
                      xaxis=dict(title=xlabel), yaxis=dict(title="PV per producer ($MM)"),
                      margin=dict(t=54, l=55, r=20, b=42))
    return fig


def fig_total(xa, ya, kopt_i, here_i, anchor, xlabel="producers per mile",
              title="Total DSU NPV = single-well × producers − injector capital"):
    xs = np.concatenate([[0], xa]); ys = np.concatenate([[anchor], ya])   # x=0 anchor (−injector capital, or 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color="#0f766e", width=3), name="total DSU NPV"))
    fig.add_hline(y=0, line=dict(color="#999", width=1, dash="dot"))
    for xc in _zero_crossings(xs, ys):
        fig.add_vline(x=xc, line=dict(color="#111", width=1, dash="dot"),
                      annotation_text=f"{xc:.0f}/mi", annotation_position="bottom")
    fig.add_trace(go.Scatter(x=[xa[kopt_i]], y=[ya[kopt_i]], mode="markers+text", text=[f"  opt {xa[kopt_i]}/mi"],
                  textposition="top center", marker=dict(size=13, color="#111", symbol="star"), showlegend=False))
    fig.add_trace(go.Scatter(x=[xa[here_i]], y=[ya[here_i]], mode="markers+text", text=["  you are here"],
                  textposition="bottom center", marker=dict(size=12, color="#ea580c", symbol="circle"), showlegend=False))
    fig.update_layout(height=360, template="plotly_white", title=title,
                      xaxis=dict(title=xlabel), yaxis=dict(title="NPV per DSU ($MM/mile)"),
                      margin=dict(t=54, l=55, r=20, b=42))
    return fig


def fig_decline():
    figp = go.Figure()
    figp.add_trace(go.Scatter(x=r["t"], y=pw, name="MWe / producer", line=dict(color="#b45309", width=3), yaxis="y1"))
    figp.add_trace(go.Scatter(x=r["t"], y=f2c(r["Tprod"]), name="produced T (F)",
                   line=dict(color="#b91c1c", width=2, dash="dot"), yaxis="y2"))
    figp.update_layout(height=430, template="plotly_white", title="Power & produced temperature decline",
                       xaxis=dict(title="years"), yaxis=dict(title="MWe / producer"),
                       yaxis2=dict(title="produced T (F)", overlaying="y", side="right", showgrid=False),
                       legend=dict(orientation="h", y=1.12, x=0), margin=dict(t=52, l=55, r=55, b=42))
    return figp


def fig_surface(xg, yg, Z, xname, yname, metric, cur):
    fig = go.Figure(go.Surface(x=xg, y=yg, z=Z, colorscale="Viridis", colorbar=dict(title=metric),
                               hovertemplate=f"{xname}=%{{x}}<br>{yname}=%{{y}}<br>{metric}=%{{z:.1f}}<extra></extra>"))
    if cur is not None and np.isfinite(cur[2]):
        fig.add_trace(go.Scatter3d(x=[cur[0]], y=[cur[1]], z=[cur[2]], mode="markers+text",
                      text=["  you are here"], textposition="top center",
                      marker=dict(size=6, color="#ea580c"), showlegend=False))
    fig.update_layout(height=640, template="plotly_white", title=f"{metric}   vs   {xname} × {yname}",
                      scene=dict(xaxis_title=xname, yaxis_title=yname, zaxis_title=metric,
                                 camera=dict(eye=dict(x=1.7, y=-1.7, z=1.1))),
                      margin=dict(t=48, l=0, r=0, b=0))
    return fig


def fig_slice(axis_vals, zvals, axis_name, metric, cur_val, slice_desc):
    fig = go.Figure(go.Scatter(x=axis_vals, y=zvals, mode="lines+markers",
                    line=dict(color="#0f766e", width=3), marker=dict(size=6)))
    fig.add_vline(x=cur_val, line=dict(color="#ea580c", width=2, dash="dot"),
                  annotation_text="you are here", annotation_position="top left")
    fig.update_layout(height=330, template="plotly_white", showlegend=False,
                      title=f"{metric} vs {axis_name}<br><sub>{slice_desc}</sub>",
                      xaxis=dict(title=axis_name), yaxis=dict(title=metric),
                      margin=dict(t=58, l=55, r=15, b=42))
    return fig


MAXFRAC = 34                               # frac ellipses drawn per well (display cap)


VIR = ["#440154", "#443983", "#31688e", "#21918c", "#35b779", "#90d743", "#fde725"]   # Viridis anchors
def _banded(vir):
    """Duplicate each stop -> hard colour steps (no blending): the 8-bit / discretized look."""
    nb = len(vir); cs = []
    for i, cc in enumerate(vir):
        cs += [[i/nb, cc], [(i+1)/nb, cc]]
    return cs
BANDED = _banded(VIR)

# fixed normalized biwing lattice, reused for every cluster's aperture sheet (u = half-width, v = height)
_WSTEP = 0.14
_WU = np.arange(-1.0, 1.0 + 1e-9, _WSTEP)
_WV = np.arange(-0.10, 2.10 + 1e-9, _WSTEP)
_WUU, _WVV = np.meshgrid(_WU, _WV)
_NV, _NU = _WUU.shape                                        # lattice dims (rows=height, cols=width)


def _biwing_field(t, asym, flip):
    """Full normalized biwing aperture (0..1) on the fixed lattice — same geometry as the aperture panel:
    two wide lobes off a flat bottom + a low central bridge + a notch cut from the top. u∈[-1,1] is
    half-width, v∈[0,2] is height (lobe centres at v=1). `t`∈[0,1] grows the lobes and opens the valley
    (Bui Fig 5 a→c); `asym` biases one wing larger, `flip` picks which side per frac. Returns (nv, nu)."""
    U, V = _WUU, _WVV
    Rx, Rz, zc = 0.42, 1.0, 1.0
    c = 0.30 + 0.35*t                                        # lobe separation opens with the outcome
    aR, aL, hR, hL, pR, pL = (1.18, 0.82, 1.10, 0.90, 1.00, 0.82) if asym else (1.,)*6
    if flip:
        aR, aL, hR, hL, pR, pL = aL, aR, hL, hR, pL, pR      # per-frac: which wing grew larger
    lR = pR*np.sqrt(np.clip(1.0 - ((U - c)/(Rx*aR))**2 - ((V - zc)/(Rz*hR))**2, 0.0, None))
    lL = pL*np.sqrt(np.clip(1.0 - ((U + c)/(Rx*aL))**2 - ((V - zc)/(Rz*hL))**2, 0.0, None))
    lobe = np.maximum(lR, lL)
    br = c + Rx*0.25
    bridge = 0.34*np.sqrt(np.clip(1.0 - (U/br)**2 - ((V - 0.55)/0.55)**2, 0.0, None))   # low bridge near bottom
    return np.maximum(lobe, bridge)


def _frac_mesh(xs, zbench_ft, ys, up_ft, dn_ft, xf_ft, asym, noise, rng):
    """Connected planar biwing fractures as ONE triangulated aperture mesh (for Mesh3d). Each cluster is
    a filled aperture SHEET in the x-z plane (two lobes + bridge, tapering to zero at the tips), not a
    voxel cloud. Each frac draws its OWN outcome from the P-band, so the geometry-noise slider spreads
    size/shape along the lateral. Returns vertex arrays X/Y/Z, per-vertex aperture C (mm), and triangle
    index arrays I/J/K into those vertices."""
    X, Y, Z, C = [], [], [], []
    I, J, K = [], [], []
    base = 0
    for xw in xs:
        for yy in ys:
            g = float(np.clip(frac_g + noise*rng.standard_normal()*0.6, 0.0, 1.0))   # per-frac outcome in the band
            gf = 0.5 + g                                     # its size factor (P10 0.5× … P90 1.5×)
            A = _biwing_field(g, asym, rng.random() < 0.5)   # (nv, nu) aperture 0..1
            up_i, dn_i = up_ft*gf, dn_ft*gf                  # this frac's up/down reach
            Xg = xw + _WUU*(xf_ft*gf)                         # half-width scales with the drawn outcome
            Zg = zbench_ft + dn_i - (_WVV/2.0)*(up_i + dn_i)  # v:0→+down (deeper) .. 2→−up (shallower)
            X += Xg.ravel().tolist(); Z += Zg.ravel().tolist()
            Y += [yy]*(_NV*_NU); C += (A.ravel()*frac_ap).tolist()
            inside = A > 0.06                                # footprint mask -> the biwing silhouette
            cell = inside[:-1, :-1] & inside[:-1, 1:] & inside[1:, :-1] & inside[1:, 1:]   # cells fully inside
            ii, jj = np.where(cell)
            a = base + ii*_NU + jj; b = a + 1; d = a + _NU; cc = d + 1   # quad corners (2 triangles each)
            I += a.tolist() + a.tolist(); J += b.tolist() + cc.tolist(); K += cc.tolist() + d.tolist()
            base += _NV*_NU
    return X, Y, Z, C, I, J, K


def fig_survey():
    Lft = float(lateral)
    xs_inj = [x/FT for x, z in r["wells"]["inj"]]; zl_ft = r["wells"]["inj"][0][1]/FT
    xs_prod = [x/FT for x, z in r["wells"]["prod"]]; zu_ft = r["wells"]["prod"][0][1]/FT
    csp = stage_len/clus_per_stage                                  # cluster spacing (ft)
    n_clusters = max(int(round(Lft/csp)), 1)
    ndraw = min(n_clusters, MAXFRAC)
    ys = np.linspace(csp/2, Lft-csp/2, ndraw)
    rng = np.random.default_rng(42)                                 # fixed seed -> stable across reruns
    fig = go.Figure()
    # wellbores (lines running along the lateral)
    for xs, zb, col, nm in [(xs_inj, zl_ft, "#1d4ed8", "injectors"), (xs_prod, zu_ft, "#b91c1c", "producers")]:
        wx, wy, wz = [], [], []
        for xw in xs:
            wx += [xw, xw, None]; wy += [0, Lft, None]; wz += [zb, zb, None]
        fig.add_trace(go.Scatter3d(x=wx, y=wy, z=wz, mode="lines", name=nm,
                      line=dict(color=col, width=6)))
    # transverse fractures as connected planar biwing aperture sheets (one triangulated Mesh3d)
    fX, fY, fZ, fC, fI, fJ, fK = [], [], [], [], [], [], []
    for xs, zb in [(xs_inj, zl_ft), (xs_prod, zu_ft)]:
        ax, ay, az, ac, ai, aj, ak = _frac_mesh(xs, zb, ys, frac_up, frac_down, frac_xf, frac_asym, frac_noise, rng)
        off = len(fX)                                         # re-base this bench's triangle indices
        fX += ax; fY += ay; fZ += az; fC += ac
        fI += [v + off for v in ai]; fJ += [v + off for v in aj]; fK += [v + off for v in ak]
    if fI:
        fig.add_trace(go.Mesh3d(x=fX, y=fY, z=fZ, i=fI, j=fJ, k=fK, intensity=fC, intensitymode="vertex",
                      colorscale="Viridis", cmin=0.0, cmax=frac_ap, showscale=True, flatshading=False,
                      name="fracture aperture", hoverinfo="skip",
                      lighting=dict(ambient=0.85, diffuse=0.4, specular=0.08, roughness=0.9),
                      lightposition=dict(x=0, y=0, z=50000),
                      colorbar=dict(title="aperture (mm)", len=0.55, x=1.02)))
    shown = f"{ndraw} of {n_clusters}" if n_clusters > ndraw else f"{n_clusters}"
    aspect = dict(aspectmode="data") if frac_iso else dict(aspectmode="manual", aspectratio=dict(x=1.4, y=1.6, z=0.9))
    fig.update_layout(height=650, template="plotly_white",
                      title=f"3-D survey — {len(xs_inj)} inj + {len(xs_prod)} prod / mile · "
                            f"{stage_len:.0f}ft stages × {clus_per_stage} clusters = {csp:.0f}ft cluster spacing · "
                            f"{'asymmetric biwing' if frac_asym else 'symmetric'} fracs ({shown}/well)"
                            f"{' · 1:1 scale' if frac_iso else ''}",
                      scene=dict(xaxis_title="across the mile (ft)", yaxis_title="along lateral (ft)",
                                 zaxis=dict(title="depth into window (ft)", autorange="reversed"),
                                 camera=dict(eye=dict(x=1.9, y=-1.7, z=1.0)), **aspect),
                      legend=dict(orientation="h", y=1.02, x=0), margin=dict(t=64, l=0, r=0, b=0))
    return fig


def fig_aperture():
    """Effective-aperture field of a single fracture (cf. Bui et al. 2021, Fig. 5). A WIDE, short footprint:
    two rounded lobes (yellow aperture cores) rising off a ~flat bottom, a notch cut from the top-centre, and
    a thin low-aperture bridge joining them along the bottom. `frac_prop` grows the lobes and pushes them apart
    — thinning/lengthening the bridge and deepening the notch (their T = 308 → 908 → 1200 s)."""
    t = frac_prop
    Lspan = frac_xf * frac_gf                             # width scales with the P10..P90 outcome
    Rx = Lspan * 0.40                                     # lobe x-radius (wide)
    Rz = 0.5*(frac_up + frac_down) * (0.32 + 0.22*t)      # lobe half-height (short, ~1.7:1 wide lobes)
    c = Lspan * (0.30 + 0.30*t)                           # lobe separation grows with the outcome (opens the valley)
    zc = Rz                                               # lobe centres up by Rz so their bottoms align at z=0
    aR, aL, hR, hL, pR, pL = (1.18, 0.82, 1.10, 0.90, 1.00, 0.82) if frac_asym else (1.,)*6   # optional wing asymmetry
    xmax = c + Rx*max(aR, aL)*1.06
    DCELL = 10.0                                          # 10-ft grid cells (the discretization the sim sees)
    xs = np.arange(0.0, 2*xmax + DCELL, DCELL) - xmax     # cell centres on a fixed 10-ft lattice
    zs = np.arange(-0.12*Rz, 2.20*Rz + DCELL, DCELL)
    X, Z = np.meshgrid(xs, zs)
    lR = pR*np.sqrt(np.clip(1.0 - ((X - c)/(Rx*aR))**2 - ((Z - zc)/(Rz*hR))**2, 0.0, None))   # right lobe
    lL = pL*np.sqrt(np.clip(1.0 - ((X + c)/(Rx*aL))**2 - ((Z - zc)/(Rz*hL))**2, 0.0, None))   # left lobe
    lobe = np.maximum(lL, lR)                            # peaks/extents differ when asymmetric
    br_span = c + Rx*0.25                                # low-aperture bridge along the centre, up to the wing midpoint
    bridge = 0.32 * np.sqrt(np.clip(1.0 - (X/br_span)**2 - ((Z - 0.50*Rz)/(0.50*Rz))**2, 0.0, None))
    field = np.maximum(lobe, bridge)                    # lobes + bridge; the gap above the bridge is the notch
    w = frac_ap * np.where(field > 0.05, field, np.nan)  # peak = frac_ap at lobe cores; NaN outside footprint
    # 8-bit look: hard-banded (stepped) Viridis + cell gaps that read as a pixel grid
    fig = go.Figure(go.Heatmap(x=xs, y=zs, z=w, colorscale=BANDED, zmin=0, zmax=frac_ap,
                    xgap=2, ygap=2, colorbar=dict(title="aperture (mm)"),
                    hovertemplate="aperture = %{z:.1f} mm<extra></extra>"))
    fig.update_layout(height=320, template="plotly_white", plot_bgcolor="#0c0c10",
                      title=f"Effective aperture — P{frac_pct} outcome{' · asymmetric' if frac_asym else ''} · 8-bit "
                            f"(peak {frac_ap:.1f} mm)",
                      xaxis=dict(title="along fracture (ft)", scaleanchor="y", scaleratio=1,
                                 showgrid=False, zeroline=False),
                      yaxis=dict(title="height (ft)", showticklabels=False, showgrid=False, zeroline=False),
                      margin=dict(t=48, l=30, r=15, b=42))
    return fig


# ------------------------------------------------------------------ visualizations (tabbed)
viz_gun, viz_survey, viz_econ, viz_perf, viz_surf = st.tabs(
    ["🛢 Gunbarrel", "🧭 3D Survey", "💰 Economics", "📉 Performance", "🌐 Sensitivities"])
with viz_gun:
    st.plotly_chart(fig_gunbarrel(560), use_container_width=True)
    st.caption("Injectors (▽) lower bench, producers (△) upper bench, separated by the bench-separation lever. "
               "**Injector depth** sets reservoir temperature (deeper = hotter); wells/mile set the pattern. "
               "Everything redraws live from the sidebar.")
with viz_survey:
    st.plotly_chart(fig_survey(), use_container_width=True)
    st.caption("The gunbarrel extruded along the lateral — wellbores run into the page; each cluster carries a "
               "**connected biwing aperture sheet** (the same field as the panel below, now a filled planar surface "
               "in the x–z plane, tapering to zero at the tips) shaded by aperture. **Survey viz** sidebar drives "
               "stage length, clusters/stage (→ cluster spacing), average size, and outcome percentile; **Geometry "
               "noise** spreads each frac's outcome across the P-band, so sheets vary in size and shape along the "
               "lateral. Drag to orbit.")
    st.plotly_chart(fig_aperture(), use_container_width=True)
    st.caption("Effective-aperture field of a single fracture (plan view, 10-ft cells), after Bui et al. (2021, "
               "Fig. 5): two **wide, flat lobes** joined by a low-aperture **bridge**, with a **valley from the top**. "
               "The **Fracture outcome (percentile)** on the Survey viz tab reads Fig. 5's panels as an uncertainty "
               "range — **P10** = small, near-single blob (their T = 308 s) → **P90** = large, two-lobe (T = 1200 s) — "
               "and that same percentile scales the wings in the 3-D survey above. **Asymmetric (biwing)** lets one "
               "wing outgrow the other; **Peak aperture** sets the magnitude.")
with viz_econ:
    cost_unit = st.selectbox("Cost basis", ["$/hz ft", "Total ($MM/well)"], key="cost_unit")
    st.plotly_chart(fig_cost(cost_unit), use_container_width=True)
    e1, e2 = st.columns(2)
    e1.plotly_chart(fig_single(ns, single_pv, here), use_container_width=True)
    e2.plotly_chart(fig_total(ns, dsu_npv, kopt, here, -inj_cap), use_container_width=True)
    st.caption("**Producers/mile** (injectors fixed at the sidebar count). **Single-well PV** (net of the producer's "
               "own drill+frac) falls as producers pack tighter and interfere, crossing zero at breakeven. **Total "
               "DSU NPV** = single-well × producers − injector capital: starts negative (injectors sunk first), "
               "crosses zero, peaks (★), and turns back down. Orange = today's design.")
    g1, g2 = st.columns(2)
    g1.plotly_chart(fig_single(ws1, single1, here1, "wells/mile (1:1 inj:prod)",
                    "Single-well PV vs wells/mile (1:1 inj:prod)"), use_container_width=True)
    g2.plotly_chart(fig_total(ws1, dsu1, kopt1, here1, 0.0, "wells/mile (1:1 inj:prod)",
                    "Total DSU NPV vs wells/mile (1:1 inj:prod)"), use_container_width=True)
    st.caption("**Paired 1:1** — producers AND injectors added together (matched wine-rack). Single-well PV and "
               "total DSU vs the paired count; DSU starts at 0 (no wells) and peaks (★) at the optimal paired density.")
with viz_perf:
    st.plotly_chart(fig_decline(), use_container_width=True)
    st.caption("Per-producer net electric power and produced temperature over 30 years. The cold front from the "
               "injectors lags the fluid (thermal retardation) → decade-scale breakthrough; tighter spacing or "
               "a shallower (cooler) reservoir pulls breakthrough earlier.")
with viz_surf:
    names = list(L.keys())
    s1, s2, s3 = st.columns(3)
    xname = s1.selectbox("X lever", names, index=names.index("Producers / mile"))
    yname = s2.selectbox("Y lever", names, index=names.index("Injectors / mile"))
    metric = s3.selectbox("Surface metric (Z)", METRICS)
    if xname == yname:
        st.warning("Pick two different levers for the X and Y axes.")
    else:
        heavy = L[xname][0] and L[yname][0]
        with st.spinner("computing sensitivity surface…" + (" (both axes drive the physics — a few seconds)" if heavy else "")):
            xg, yg, Z, SW, CUM = surface(xname, yname, metric, FP)
            cur = eval_point({})
        st.plotly_chart(fig_surface(xg, yg, Z, xname, yname, metric,
                                    (L[xname][1], L[yname][1], cur[metric])), use_container_width=True)
        st.caption(f"Covarying sensitivity of **{metric}** across **{xname}** and **{yname}**, every other lever "
                   "held at its sidebar value (orange = today's design). Fluid & proppant move the physics "
                   "(A√k contacted area **and** deliverability), so too little stimulation kills power, not just cost.")
        st.markdown("**2-D slices** — the slider under each column sets where the *other* axis is held. "
                    "Top row: chosen metric. Bottom row: the single-well PV underneath it.")
        cA, cB = st.columns(2)
        with cA:                                                    # metric vs X, at a chosen Y
            phA = st.empty()
            yopts = [int(v) for v in yg]
            yhold = st.select_slider(f"hold {yname} at", options=yopts,
                                     value=int(nearest(yg, L[yname][1])), key=f"hold_y_{xname}_{yname}")
            jj = yopts.index(yhold)
            phA.plotly_chart(fig_slice(xg, Z[jj, :], xname, metric, L[xname][1], f"at {yname} = {yhold}"),
                             use_container_width=True)
        with cB:                                                    # metric vs Y, at a chosen X
            phB = st.empty()
            xopts = [int(v) for v in xg]
            xhold = st.select_slider(f"hold {xname} at", options=xopts,
                                     value=int(nearest(xg, L[xname][1])), key=f"hold_x_{xname}_{yname}")
            ii = xopts.index(xhold)
            phB.plotly_chart(fig_slice(yg, Z[:, ii], yname, metric, L[yname][1], f"at {xname} = {xhold}"),
                             use_container_width=True)
        cC, cD = st.columns(2)                                       # single-well PV underneath (same holds)
        cC.plotly_chart(fig_slice(xg, SW[jj, :], xname, "Single-well PV ($MM)", L[xname][1],
                        f"at {yname} = {yhold}"), use_container_width=True)
        cD.plotly_chart(fig_slice(yg, SW[:, ii], yname, "Single-well PV ($MM)", L[yname][1],
                        f"at {xname} = {xhold}"), use_container_width=True)
        cE, cF = st.columns(2)                                       # cumulative power (physics, diminishing returns)
        cE.plotly_chart(fig_slice(xg, CUM[jj, :], xname, "Cum power (GWh/well)", L[xname][1],
                        f"at {yname} = {yhold}"), use_container_width=True)
        cF.plotly_chart(fig_slice(yg, CUM[:, ii], yname, "Cum power (GWh/well)", L[yname][1],
                        f"at {xname} = {xhold}"), use_container_width=True)
        st.caption("Rows: chosen metric · single-well PV · **cumulative power** (lifetime GWh/producer). "
                   "Cumulative power follows diminishing returns to the deliverability plateau; the economics "
                   "peak where marginal cost overtakes that flattening power — separating the physics from the money.")

st.caption("Physics is a fast single-phase thermal-hydraulic twin (not a frac-propagation sim) — the design & "
           "economics layer on top of ResFrac. Every number responds to the sidebar levers.")
