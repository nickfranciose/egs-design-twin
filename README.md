# EGS Design Twin

A fast, interactive design + economics twin for enhanced geothermal systems (EGS).
It reproduces the well-spacing / thermal-breakthrough tradeoff in seconds and extends
it from net power to **NPV / IRR per DSU** across every design and cost lever — the
economic layer on top of a ResFrac-grade physics picture.

Built with Streamlit. Benchmarked against published field results from Fervo's
Project Red / Blue Mountain demonstration (Norbeck & Latimer, 2023): reservoir
transmissibility, propped-fracture conductivity, SRV geometry, and the linear
lateral-length scaling.

## What's inside
- `egs_app.py` — the interactive app (sidebar design levers + gunbarrel / 3-D survey / economics / performance / sensitivity-surface tabs)
- `egs_sim.py` — 2-D thermal-hydraulic EGS model (steady Darcy + advective–conductive heat with thermal retardation)
- `egs_econ.py` — decomposed capex + NPV / IRR project economics
- `egs_report.py` — optional standalone HTML report generator (not used by the app)

## Run locally
```bash
pip install -r requirements.txt
streamlit run egs_app.py
```

## Deploy
Push to GitHub and deploy on [Streamlit Community Cloud](https://share.streamlit.io)
with `egs_app.py` as the entrypoint and Python 3.12.
