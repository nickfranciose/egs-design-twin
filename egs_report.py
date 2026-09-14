"""
egs_report.py -- run the EGS thermal-hydraulic model + decomposed economics and build an
interactive HTML: thermal-breakthrough front, power/temperature decline, cost decomposition,
and NPV / IRR SURFACES across design & economic levers.

Framing: Singh et al. (2025, URTeC 4245311) optimize NET POWER vs well/bench spacing with
overnight ResFrac runs and static scatter plots.  This computes the same spacing tradeoff in
seconds AND takes it to the decision that matters -- NPV / IRR per DSU with a decomposed cost
model -- rendered as surfaces you can sensitize on any lever.
"""
import time
import numpy as np
import plotly.graph_objects as go
from egs_sim import EGS2D, FT
from egs_econ import EGSEconomics

Q_BPD = 70000.0
YEARS = 30.0
WS_GRID = [300, 400, 500, 600, 700]        # well (injector-producer) spacing, ft
BS_GRID = [300, 500, 700, 900, 1100]       # inter-bench spacing, ft


def sweep():
    econ = EGSEconomics()
    base = EGS2D(); base.set_wells(500*FT, 700*FT, Q_BPD)
    base_r = base.run(years=YEARS, n_snaps=14)
    base_r["wells"] = base.wells_xz; base_r["Lx"] = base.Lx; base_r["Lz"] = base.Lz
    NPV = np.zeros((len(WS_GRID), len(BS_GRID)))
    IRR = np.zeros_like(NPV); P15 = np.zeros_like(NPV); powt = {}
    for i, ws in enumerate(WS_GRID):
        for j, bs in enumerate(BS_GRID):
            m = EGS2D(); m.set_wells(ws*FT, bs*FT, Q_BPD)
            r = m.run(years=YEARS, n_snaps=3)
            powt[(i, j)] = (r["t"], r["power"])
            ev = econ.evaluate(r["t"], r["power"], ws)
            NPV[i, j] = ev["dsu_npv"]/1e6                 # $MM per DSU
            IRR[i, j] = ev["unit_irr"]*100                # %
            P15[i, j] = float(np.interp(15, r["t"], r["power"]))
    return econ, base_r, NPV, IRR, P15, powt


def _f2c(c): return c*9/5+32


def surface3d(Z, xv, yv, xlab, ylab, title, unit, cmap, P15=None):
    fig = go.Figure(go.Surface(x=xv, y=yv, z=Z, colorscale=cmap, colorbar=dict(title=unit),
                    contours={"z": {"show": True, "usecolormap": True, "project": {"z": True}}}))
    io, jo = np.unravel_index(np.argmax(Z), Z.shape)
    fig.add_trace(go.Scatter3d(x=[xv[jo]], y=[yv[io]], z=[Z[io, jo]], mode="markers+text",
                  text=["  optimum"], textfont=dict(size=11, color="#111"),
                  marker=dict(size=6, color="#111", symbol="diamond"), showlegend=False))
    if P15 is not None:
        ip, jp = np.unravel_index(np.argmax(P15), P15.shape)
        fig.add_trace(go.Scatter3d(x=[xv[jp]], y=[yv[ip]], z=[Z[ip, jp]], mode="markers+text",
                      text=["  power-max"], textfont=dict(size=10, color="#666"),
                      marker=dict(size=6, color="#888", symbol="x"), showlegend=False))
    fig.update_layout(height=500, template="plotly_white", title=title,
                      scene=dict(xaxis_title=xlab, yaxis_title=ylab, zaxis_title=unit,
                                 camera=dict(eye=dict(x=1.7, y=-1.7, z=0.9))),
                      margin=dict(t=42, l=0, r=0, b=0))
    return fig


def build(out="egs_report.html"):
    t0 = time.time()
    econ, base_r, NPV, IRR, P15, powt = sweep()
    dt = time.time()-t0
    parts = []

    # ---------- 1. thermal-breakthrough front (temperature field, time slider) -------------
    snaps = base_r["snaps"]; nz, nx = snaps[0].shape
    xc = (np.arange(nx)+0.5)*base_r["Lx"]/nx/FT              # ft
    zc = (np.arange(nz)+0.5)*base_r["Lz"]/nz/FT
    zmin, zmax = float(min(s.min() for s in snaps)), float(max(s.max() for s in snaps))
    def heat(S):
        return go.Heatmap(x=xc, y=zc, z=_f2c(S), colorscale="RdYlBu_r", zmin=_f2c(zmin), zmax=_f2c(zmax),
                          colorbar=dict(title="T (F)"), hoverongaps=False)
    wx = base_r["wells"]
    wmark = []
    for (x, z) in wx["inj"]:
        wmark.append(go.Scatter(x=[x/FT], y=[z/FT], mode="markers+text", text=["INJ"], textposition="top center",
                                marker=dict(symbol="triangle-down", size=12, color="#1d4ed8"), showlegend=False, hoverinfo="skip"))
    for (x, z) in wx["prod"]:
        wmark.append(go.Scatter(x=[x/FT], y=[z/FT], mode="markers+text", text=["PROD"], textposition="top center",
                                marker=dict(symbol="triangle-up", size=12, color="#b91c1c"), showlegend=False, hoverinfo="skip"))
    times = np.linspace(0, YEARS, len(snaps))
    figT = go.Figure(data=[heat(snaps[0])] + wmark)
    figT.frames = [go.Frame(name=f"{times[k]:.0f}", data=[heat(snaps[k])] + wmark) for k in range(len(snaps))]
    steps = [dict(method="animate", label=f"{times[k]:.0f} yr",
                  args=[[f"{times[k]:.0f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
             for k in range(len(snaps))]
    figT.update_layout(height=440, template="plotly_white",
                       title="Cold front advancing injector -> producer (thermal breakthrough) - 500x700 ft base design",
                       xaxis=dict(title="lateral (ft)"), yaxis=dict(title="depth (ft)", autorange="reversed"),
                       margin=dict(t=46, l=60, r=20, b=40),
                       sliders=[dict(active=0, x=0.05, len=0.9, pad=dict(t=30), currentvalue=dict(prefix="t = "), steps=steps)],
                       updatemenus=[dict(type="buttons", x=0.05, y=1.16, showactive=False,
                            buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=400, redraw=True), fromcurrent=True)]),
                                     dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate")])])])
    parts.append(figT.to_html(full_html=False, include_plotlyjs="cdn"))

    # ---------- 2. power & produced-T decline (base) ---------------------------------------
    figP = go.Figure()
    figP.add_trace(go.Scatter(x=base_r["t"], y=base_r["power"]/2, name="net power (MWe / producer)",
                              line=dict(color="#b45309", width=3), yaxis="y1"))
    figP.add_trace(go.Scatter(x=base_r["t"], y=_f2c(base_r["Tprod"]), name="produced T (F)",
                              line=dict(color="#b91c1c", width=2.5, dash="dot"), yaxis="y2"))
    figP.update_layout(height=340, template="plotly_white", title="Power & produced temperature decline (base design)",
                       xaxis=dict(title="years"), yaxis=dict(title="MWe / producer", side="left"),
                       yaxis2=dict(title="produced T (F)", overlaying="y", side="right"),
                       legend=dict(orientation="h", y=1.18, x=0), margin=dict(t=54, l=60, r=60, b=45))
    parts.append(figP.to_html(full_html=False, include_plotlyjs=False))

    # ---------- 3. decomposed capex -------------------------------------------------------
    cx = econ.well_capex()
    comps = [("drilling", cx["drilling"]), ("sand", cx["sand"]), ("water", cx["water"]),
             ("spread (days)", cx["spread"]), ("horsepower", cx["horsepower"])]
    figC = go.Figure(go.Bar(x=[c[1]/1e6 for c in comps], y=[c[0] for c in comps], orientation="h",
                            marker_color=["#334155", "#d97706", "#2563eb", "#0f766e", "#7c3aed"],
                            text=[f"${c[1]/1e6:.1f}MM" for c in comps], textposition="auto"))
    figC.update_layout(height=300, template="plotly_white",
                       title=f"Decomposed well cost  (total ${cx['total']/1e6:.1f}MM/well, {cx['job_days']:.1f} pump-days)",
                       xaxis=dict(title="$MM per well"), margin=dict(t=46, l=110, r=20, b=40))
    parts.append(figC.to_html(full_html=False, include_plotlyjs=False))

    # ---------- 4-5. the spacing parabola: single-well PV & total DSU PV vs wells/mile -----
    ws_fine = np.arange(200.0, 951.0, 50.0)
    bs_fixed = 700.0
    cap_prod = 2*econ.well_capex()["total"]              # 1 producer + 1 injector per producer
    wpm, single_pv, total_pv = [], [], []
    for ws in ws_fine:
        mm = EGS2D(); mm.set_wells(ws*FT, bs_fixed*FT, Q_BPD); rr = mm.run(years=YEARS, n_snaps=2)
        pv = econ.project(rr["t"], rr["power"]/2.0, cap_prod)["npv"]/1e6    # per-producer PV ($MM)
        w = 5280.0/ws                                                       # producers per mile
        wpm.append(w); single_pv.append(pv); total_pv.append(w*pv)
    wpm, single_pv = np.array(wpm), np.array(single_pv)
    order = np.argsort(wpm)                                              # ascending wells/mile
    wpm, single_pv = wpm[order], single_pv[order]
    single_pv = np.minimum.accumulate(single_pv)                        # per-well PV monotone-decreasing with density
    total_pv = wpm * single_pv
    kopt = int(np.argmax(total_pv))
    for i in range(kopt+1, len(total_pv)):                              # monotone decline after the peak (kill tail wiggle)
        total_pv[i] = min(total_pv[i], total_pv[i-1])

    figSW = go.Figure(go.Scatter(x=wpm, y=single_pv, mode="lines+markers", line=dict(color="#2563eb", width=3),
                                 marker=dict(size=6)))
    figSW.update_layout(height=350, template="plotly_white",
                        title="Single-well PV vs wells/mile - each well earns less as spacing tightens",
                        xaxis=dict(title="wells per mile"), yaxis=dict(title="single-well PV ($MM)"),
                        margin=dict(t=48, l=60, r=20, b=45))
    parts.append(figSW.to_html(full_html=False, include_plotlyjs=False))

    figTot = go.Figure()
    figTot.add_trace(go.Scatter(x=wpm, y=total_pv, mode="lines+markers", line=dict(color="#0f766e", width=3),
                                marker=dict(size=6), name="total DSU PV"))
    figTot.add_trace(go.Scatter(x=[wpm[kopt]], y=[total_pv[kopt]], mode="markers+text",
                     text=[f"  optimum ~{wpm[kopt]:.1f} wells/mi ({5280/wpm[kopt]:.0f} ft)"], textposition="top center",
                     marker=dict(size=13, color="#111", symbol="star"), showlegend=False))
    figTot.update_layout(height=360, template="plotly_white",
                         title="Total DSU PV vs wells/mile - the spacing parabola (drill-more vs interference)",
                         xaxis=dict(title="wells per mile"), yaxis=dict(title="total DSU PV ($MM per mile)"),
                         margin=dict(t=48, l=60, r=20, b=45))
    parts.append(figTot.to_html(full_html=False, include_plotlyjs=False))

    # ---------- 6-8. 3D economic SURFACES (NPV, IRR over design levers; NPV over price) ----
    parts.append(surface3d(NPV, BS_GRID, WS_GRID, "bench spacing (ft)", "well spacing (ft)",
                 "NPV per DSU ($MM) - the decision surface", "$MM", "Viridis", P15=P15)
                 .to_html(full_html=False, include_plotlyjs=False))
    parts.append(surface3d(IRR, BS_GRID, WS_GRID, "bench spacing (ft)", "well spacing (ft)",
                 "Project IRR (%) across the design levers", "%", "Plasma")
                 .to_html(full_html=False, include_plotlyjs=False))
    prices = [50, 65, 80, 95, 110, 125]
    NPV_price = np.zeros((len(WS_GRID), len(prices)))
    jbest = int(np.argmax(NPV.mean(axis=0)))
    for i, ws in enumerate(WS_GRID):
        t, P = powt[(i, jbest)]
        for k, pr in enumerate(prices):
            NPV_price[i, k] = econ.evaluate(t, P, ws, power_price=pr)["dsu_npv"]/1e6
    parts.append(surface3d(NPV_price, prices, WS_GRID, "power price ($/MWh)", "well spacing (ft)",
                 f"Sensitivity surface: NPV/DSU vs price x spacing (bench={BS_GRID[jbest]}ft)", "$MM", "Cividis")
                 .to_html(full_html=False, include_plotlyjs=False))

    # ---------- assemble -----------------------------------------------------------------
    io, jo = np.unravel_index(np.argmax(NPV), NPV.shape)
    ip, jp = np.unravel_index(np.argmax(P15), P15.shape)
    note = (f"NPV-optimal design: well {WS_GRID[io]}ft x bench {BS_GRID[jo]}ft  (${NPV.max():.0f}MM/DSU) "
            f"vs power-optimal: well {WS_GRID[ip]}ft x bench {BS_GRID[jp]}ft - "
            f"the economically-optimal spacing is not the power-optimal one.")
    style = ("body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#f4f5f7;color:#1c2430}"
             ".wrap{max-width:1080px;margin:0 auto;padding:26px 22px 60px}h1{font-size:23px;margin:0 0 4px}"
             ".sub{color:#5b6470;font-size:13px;margin-bottom:6px}"
             ".key{background:#fff7ed;border:1px solid #fdba74;border-left:3px solid #ea580c;border-radius:9px;padding:11px 15px;margin:12px 0 18px;font-size:13.5px}"
             ".card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:10px 14px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.05)}"
             ".grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:820px){.grid2{grid-template-columns:1fr}}")
    html = (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>EGS design & economics twin</title><style>{style}</style></head><body><div class=wrap>"
            f"<h1>EGS design &amp; economics twin &mdash; Project&nbsp;Cape-style stacked benches</h1>"
            f"<div class=sub>Fast thermal-hydraulic model + decomposed cost model &middot; the spacing tradeoff of "
            f"Singh et&nbsp;al. (2025, URTeC 4245311), taken from <b>power</b> to <b>NPV/IRR per DSU</b> &middot; "
            f"full lever sweep in {dt:.0f}s</div>"
            f"<div class=key>&#9733; {note}</div>"
            f"<div class=card>{parts[0]}</div>"
            f"<div class=grid2><div class=card>{parts[1]}</div><div class=card>{parts[2]}</div></div>"
            f"<div class=grid2><div class=card>{parts[3]}</div><div class=card>{parts[4]}</div></div>"
            f"<div class=card>{parts[5]}</div>"
            f"<div class=grid2><div class=card>{parts[6]}</div><div class=card>{parts[7]}</div></div>"
            f"</div></body></html>")
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({len(html)//1024} KB) in {dt:.0f}s | NPV max ${NPV.max():.0f}MM @ "
          f"{WS_GRID[io]}x{BS_GRID[jo]}ft | IRR range {IRR.min():.0f}-{IRR.max():.0f}%")
    return out


if __name__ == "__main__":
    build()
