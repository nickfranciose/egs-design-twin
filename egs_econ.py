"""
egs_econ.py -- decomposed cost model + project economics (NPV / IRR per DSU) for an EGS design.

Turns the thermal cashflow from egs_sim into the decision that actually gets made: dollars.
Cost is decomposed the way an operator budgets it --

  DRILLING   :  vertical_depth * $/vt_ft  +  lateral * $/hz_ft
  COMPLETION :  per completed lateral foot, built up from
                  sand      = (sand_lb/ft / 2000) * $/sand_ton
                  water     = water_gal/ft * $/gal
                  spread    = job_days * $/day_spread   (job_days = pump_hours / (24 * pump_efficiency))
                  horsepower= $/HHP-hr * pump_hours     (pump_hours = fluid bbls / (pump_rate_bpm * 60))
  REVENUE    :  net power(t) [MWe] -> MWh * capacity_factor -> $ at $/MWh, less $/MWh opex
  -> NPV, IRR for a well unit, and NPV per DSU (well density scales with 1/spacing).

Every field is a lever you can sensitize on -> PV/IRR surfaces.
"""
import numpy as np


class EGSEconomics:
    def __init__(self,
                 # drilling ($/ft)
                 vertical_depth_ft=8000.0, cost_vt_ft=450.0, cost_hz_ft=550.0,
                 # completion decomposition (calibrated to ~$600/ft completed lateral at the reference job)
                 cost_sand_ton=300.0, cost_water_gal=0.06, cost_hhp_hr=12000.0, cost_day_spread=150000.0,
                 sand_lb_per_ft=2000.0, water_gal_per_ft=1500.0, pump_rate_bpm=100.0,
                 pump_efficiency=0.32,                              # fraction of the day actually pumping (NPT/stage swaps)
                 ref_fluid_gal_ft=1500.0, ref_sand_lb_ft=2000.0, overstim_exp=2.0,   # over-stimulation cost escalation
                 # project economics
                 power_price=80.0, capacity_factor=0.90, discount_rate=0.10,
                 opex_per_MWh=15.0, opex_per_well_mo=10000.0, years=30, lateral_ft=5000.0):
        self.__dict__.update(locals()); del self.self

    # ---- decomposed capex for one well ------------------------------------------------------
    def well_capex(self, lateral_ft=None):
        L = lateral_ft or self.lateral_ft
        drill_vertical = self.vertical_depth_ft*self.cost_vt_ft
        drill_horizontal = L*self.cost_hz_ft
        drilling = drill_vertical + drill_horizontal
        sand = (L*self.sand_lb_per_ft/2000.0) * self.cost_sand_ton
        water_gal = L*self.water_gal_per_ft
        water = water_gal * self.cost_water_gal
        job_bbls = water_gal/42.0
        pump_hours = job_bbls / (self.pump_rate_bpm*60.0)              # pure pumping time: bbls / (bbls/hr)
        job_days = pump_hours / (24.0*self.pump_efficiency)           # calendar days incl NPT / stage swaps
        spread = job_days * self.cost_day_spread
        horsepower = self.cost_hhp_hr * pump_hours                    # HHP fleet billed per pumping hour
        # over-stimulation escalation: mega-jobs get disproportionately expensive (NPT, screenouts,
        # water/proppant logistics, remediation) — a job-complexity premium on the whole completion.
        intensity = 0.5*(self.water_gal_per_ft/self.ref_fluid_gal_ft) + 0.5*(self.sand_lb_per_ft/self.ref_sand_lb_ft)
        esc = max(1.0, intensity)**self.overstim_exp                  # =1 at/below the reference job
        sand *= esc; water *= esc; spread *= esc; horsepower *= esc
        completion = sand + water + spread + horsepower
        return dict(total=drilling+completion, drilling=drilling, completion=completion,
                    drill_vertical=drill_vertical, drill_horizontal=drill_horizontal,
                    sand=sand, water=water, spread=spread, horsepower=horsepower, overstim_esc=esc,
                    job_days=job_days, pump_hours=pump_hours,
                    lateral_ft=L, completion_per_ft=completion/L, drilling_per_ft=drilling/L, total_per_ft=(drilling+completion)/L)

    # ---- project cashflow -> NPV, IRR -------------------------------------------------------
    def project(self, times_yr, power_MWe, capex, power_price=None, n_wells=0):
        price = self.power_price if power_price is None else power_price
        yrs = np.arange(0, self.years+1)
        P = np.interp(yrs, times_yr, power_MWe)                 # MWe at each year
        energy = P * self.capacity_factor * 8760.0             # MWh/yr
        revenue = energy * price
        opex = energy * self.opex_per_MWh + n_wells*self.opex_per_well_mo*12.0   # variable + fixed $/mo/well O&M
        cf = np.zeros(self.years+1)
        cf[0] = -capex
        cf[1:] += (revenue - opex)[1:]
        disc = (1+self.discount_rate)**yrs
        npv = float((cf/disc).sum())
        return dict(npv=npv, irr=self._irr(cf), cf=cf, energy=energy, revenue=revenue)

    @staticmethod
    def _irr(cf, lo=-0.5, hi=2.0):
        f = lambda r: np.sum(cf/(1+r)**np.arange(len(cf)))
        if f(lo)*f(hi) > 0:
            return float('nan')
        for _ in range(80):
            mid = 0.5*(lo+hi)
            if f(lo)*f(mid) <= 0: hi = mid
            else: lo = mid
        return 0.5*(lo+hi)

    # ---- one design: physics (power_t) -> unit economics + per-DSU -------------------------
    def evaluate(self, times_yr, power_MWe, well_spacing_ft, n_wells_per_unit=4,
                 dsu_width_ft=5000.0, power_price=None):
        capex = n_wells_per_unit * self.well_capex()['total']
        pr = self.project(times_yr, power_MWe, capex, power_price=power_price)
        n_units = max(dsu_width_ft/well_spacing_ft, 1e-9)      # more, tighter-spaced units per DSU
        return dict(unit_npv=pr['npv'], unit_irr=pr['irr'], unit_capex=capex,
                    dsu_npv=pr['npv']*n_units, dsu_capex=capex*n_units, n_units=n_units)
