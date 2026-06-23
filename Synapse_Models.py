
import torch
from Characterization import MODEL_CONFIGS,ModelCharac
from config import SYNAPSE_MODEL

charac_model=ModelCharac(SYNAPSE_MODEL)
base_params=charac_model()

class Ferroelectric:
    """
    This class models a ferroelectric device's behavior, incorporating parameters for its
    material properties, defect mask, and variations. It allows effective emulation of
    ferroelectric switching dynamics, combining physical characteristics with stochastic
    noise to simulate real-world conditions.

    The class utilizes key parameters such as switching rates, window functions, and defect
    masks to calculate weight updates based on input potentials. It ensures that weights
    are always clamped within set bounds and accommodates defects by masking certain weights.

    :ivar gamma_p: Positive switching exponent applied during calculations.
    :ivar gamma_d: Negative switching exponent applied during calculations.
    :ivar alpha_p: Positive switching rate factor that scales response to potential.
    :ivar alpha_d: Negative switching rate factor that scales response to potential.
    :ivar theta_p: Positive switching threshold, beyond which switching occurs.
    :ivar theta_d: Negative switching threshold, below which switching occurs.
    :ivar sf_p: Scaling factor for positive potential input.
    :ivar sf_d: Scaling factor for negative potential input.
    :ivar weight: The internal weight value, clamped within defined bounds.
    :ivar v_ref: Reference voltage used as a baseline for certain calculations.
    :ivar w_min: Minimum allowable weight value.
    :ivar w_max: Maximum allowable weight value.
    :ivar defect_mask: A boolean mask indicating the presence of defects in specific
                       components, which prevents weight updates at defective locations.
    :ivar cycle_vc_multiplicative: Multiplicative coefficient for cycle-to-cycle variation's
                                   noise contribution.
    :ivar cycle_vc_additive: Additive coefficient for cycle-to-cycle variation's noise contribution.
    :ivar c2c_additive_ref_scale: Scaling value for additive noise reference, derived from the
                                  weight bounds.
    """
    def __init__(self,
                 weight,
                 sf_p=1.0385,
                 sf_d = 1.30,
                 gamma_p = 1.62,
                 gamma_d = 1.79,
                 alpha_p = 0.67,
                 alpha_d = 0.38,
                 theta_p = -0.55,
                 theta_d = 0.47,
                 v_ref=1.02,
                 w_min=0.0, w_max=0.1,w_sat=10.0
                 ):
        self.gamma_p = gamma_p
        self.gamma_d = gamma_d
        self.alpha_p = alpha_p
        self.alpha_d = alpha_d
        self.theta_p = theta_p
        self.theta_d = theta_d
        self.sf_p = sf_p
        self.sf_d = sf_d
        self.weight=weight
        self.v_ref=v_ref
        self.w_min=w_min
        self.w_max=w_max
        self.weight = torch.clamp(self.weight, min=self.w_min, max=self.w_max)
        self.base_params=base_params

        self.defect_mask = ((self.gamma_p <= 1.0) |
                            (self.gamma_d <= 1.0) |
                            (self.alpha_p <= 0) |
                            (self.alpha_d <= 0) |
                            (self.theta_p >= 0) |
                            (self.theta_d <= 0))

        self.cycle_vc_multiplicative = MODEL_CONFIGS.get('Ferroelectric', {}).get("variations", {}).get(
            'cycle_to_cycle_variation_coefficient_multiplicative', 0.0
        )
        self.cycle_vc_additive = MODEL_CONFIGS.get('Ferroelectric', {}).get("variations", {}).get(
            'cycle_to_cycle_variation_coefficient_additive', 0.0
        )
        self.c2c_additive_ref_scale = self.w_max - self.w_min

    def __call__(self,potential,potential_rest,potential_threshold,cond_pot):
        v_p=(potential/potential_rest)*self.sf_p*self.base_params['theta_p']
        v_d=(self.v_ref-potential/potential_threshold)*self.sf_d*self.base_params['theta_d']
        fp,fd=self.f_v(v_p,v_d)
        gp,gd=self.g_w(v_p,v_d)
        ideal_delta_ltp = fp * gp
        ideal_delta_ltd = fd * gd

        noise_ltp = (
                torch.randn_like(ideal_delta_ltp) * ideal_delta_ltp * self.cycle_vc_multiplicative
                + torch.randn_like(ideal_delta_ltp) * self.c2c_additive_ref_scale * self.cycle_vc_additive
        )
        noise_ltd = (
                torch.randn_like(ideal_delta_ltd) * ideal_delta_ltd * self.cycle_vc_multiplicative
                + torch.randn_like(ideal_delta_ltd) * self.c2c_additive_ref_scale * self.cycle_vc_additive
        )

        delta_ltp = ideal_delta_ltp + noise_ltp
        delta_ltd = ideal_delta_ltd + noise_ltd

        # defects control
        if torch.any(self.defect_mask):
            delta_ltp[self.defect_mask] = 0.0
            delta_ltd[self.defect_mask] = 0.0
            self.weight[self.defect_mask] = 0.0

        signed_delta = torch.where(cond_pot, delta_ltp, -delta_ltd)
        new_weight = torch.clamp(self.weight + signed_delta, min=self.w_min, max=self.w_max)

        return new_weight - self.weight

    def f_v(self,v_p,v_d):
        ##switching rate function##
        fp=torch.where(v_p<self.theta_p,torch.exp(-self.alpha_p*(v_p-self.theta_p))-1,torch.zeros_like(v_p))
        fd=torch.where(v_d>self.theta_d,torch.exp(self.alpha_d*(v_d-self.theta_d))-1,torch.zeros_like(v_d))
        return fp,fd

    def g_w(self,v_p,v_d):
        ##window function##
        gp=torch.where(v_p<self.theta_p,torch.pow((self.w_max-self.weight),self.gamma_p),torch.zeros_like(v_p))
        gd=torch.where(v_d>self.theta_d,torch.pow(self.weight,self.gamma_d),torch.zeros_like(v_d))
        return gp,gd

# class Ferroelectric_Tanh:
#     """
#     Implementation of the Ferroelectric_Tanh class that models ferroelectric
#     behavior using a specialized hyperbolic tangent function.
#
#     This class is designed to simulate the dynamics of ferroelectric materials
#     and their interactions with external potentials, capturing phenomena such as
#     resistance changes, weight updates, and cycle variations.
#
#     :ivar sf_p: Scaling factor for positive polarization.
#     :ivar sf_d: Scaling factor for depolarization.
#     :ivar r_min: Minimum resistance value.
#     :ivar r_max: Maximum resistance value.
#     :ivar v0_up: Voltage parameter for upper transition shaping.
#     :ivar voff_up: Offset voltage for upper transition.
#     :ivar v0_low: Voltage parameter for lower transition shaping.
#     :ivar voff_low: Offset voltage for lower transition.
#     :ivar v_ref: Reference voltage level.
#     :ivar w_min: Minimum permissible weight value.
#     :ivar w_max: Maximum permissible weight value.
#     :ivar w_sat: Saturation weight value.
#     :ivar g_min: Minimum conductance derived from r_max.
#     :ivar g_max: Maximum conductance derived from r_min.
#     :ivar weight: Current weight value constrained between w_min and w_sat.
#     :ivar defect_mask: Boolean mask indicating invalid parameter configurations.
#     :ivar cycle_vc_multiplicative: Coefficient for multiplicative cycle-to-cycle variations.
#     :ivar cycle_vc_additive: Coefficient for additive cycle-to-cycle variations.
#     :ivar c2c_additive_ref_scale: Reference scale for additive cycle-to-cycle variations.
#     """
#     def __init__(self,
#                  weight,
#                  sf_p=1.0,
#                  sf_d=1.3,
#                  r_min=1.1e9,
#                  r_max=2.5e9,
#                  v0_up=0.45,
#                  voff_up=-1.0,
#                  v0_low=0.60,
#                  voff_low=1.6,
#                  v_ref=1.02,
#                  w_min=0.0,w_max=0.1,w_sat=1.0
#                  ):
#         self.sf_p = 1.0
#         self.sf_d = 1.0
#         self.r_min = r_min
#         self.r_max = r_max
#         self.v0_up = v0_up
#         self.voff_up = voff_up
#         self.v0_low = v0_low
#         self.voff_low = voff_low
#         self.v_ref = v_ref
#         self.w_min = w_min
#         self.w_max = w_max
#         self.w_sat = w_sat
#
#         self.base_params=base_params
#
#         self.v_max_p = self.base_params['voff_up']-self.base_params['v0_up']
#         self.v_max_d = self.base_params['voff_low']+self.base_params['v0_low']
#
#         self.g_min = 1.0 / self.r_max
#         self.g_max = 1.0 / self.r_min
#
#         self.weight = weight
#         self.weight = torch.clamp(self.weight, min=self.w_min, max=self.w_sat)
#
#         self.defect_mask = (self.r_max <= self.r_min) | (self.v0_up <= 0) | (self.v0_low <= 0) | (self.voff_up >= 0) | (self.voff_low <= 0)
#
#         self.cycle_vc_multiplicative = MODEL_CONFIGS.get('Ferroelectric_Tanh', {}).get("variations", {}).get(
#             'cycle_to_cycle_variation_coefficient_multiplicative', 0.0
#         )
#         self.cycle_vc_additive = MODEL_CONFIGS.get('Ferroelectric_Tanh', {}).get("variations", {}).get(
#             'cycle_to_cycle_variation_coefficient_additive', 0.0
#         )
#         self.c2c_additive_ref_scale = self.r_max - self.r_min
#
#     def __call__(self, potential, potential_rest,potential_threshold, cond_pot):
#         v_p = (potential / potential_rest) * self.sf_p * self.v_max_p
#         v_d = (self.v_ref - potential / potential_threshold) * self.sf_d * self.v_max_d
#
#
#         #r_initial = self.w_to_r(self.weight)
#         r_initial = self.weff_to_r(self.weight)
#
#         r_target_p = self.f_upper(v_p)
#         r_target_d = self.f_lower(v_d)
#         #print(v_p[cond_pot])
#         delta_r_p = r_target_p - r_initial
#         delta_r_d = r_target_d - r_initial
#
#         noise_p = (
#                 torch.randn_like(delta_r_p) * delta_r_p * self.cycle_vc_multiplicative
#                 + torch.randn_like(delta_r_p) * self.c2c_additive_ref_scale * self.cycle_vc_additive
#         )
#         noise_d = (
#                 torch.randn_like(delta_r_d) * delta_r_d * self.cycle_vc_multiplicative
#                 + torch.randn_like(delta_r_d) * self.c2c_additive_ref_scale * self.cycle_vc_additive
#         )
#         r_final_p = torch.clamp(r_target_p + noise_p,min=0.0)
#         r_final_d = torch.clamp(r_target_d + noise_d,min=0.0)
#
#         r_final_p = torch.min(r_initial, r_final_p)
#         r_final_d = torch.max(r_initial, r_final_d)
#
#         max_delta_ltp = self.r_to_weff(r_final_p) # >= 0
#         max_delta_ltd = self.r_to_weff(r_final_d) # <= 0
#
#         delta_weight_ltp = max_delta_ltp - self.weight
#         delta_weight_ltd = max_delta_ltd - self.weight
#
#         # defects control
#         if torch.any(self.defect_mask):
#             delta_weight_ltp[self.defect_mask] = 0.0
#             delta_weight_ltd[self.defect_mask] = 0.0
#             self.weight[self.defect_mask] = 0.0
#
#         eps = 1e-8
#         cond_ltp = cond_pot & (potential < -eps)
#         cond_ltd = (~cond_pot) & (potential > eps)
#
#         zero_update = torch.zeros_like(self.weight)
#         #print(delta_weight_ltp[cond_ltp])
#         return torch.where(
#             cond_ltp,
#             delta_weight_ltp,
#             torch.where(cond_ltd, delta_weight_ltd, zero_update)
#         )
#
#     def w_to_r(self, w):
#         w_norm = (w - self.w_min) / (self.w_max - self.w_min)
#         g = w_norm * (self.g_max - self.g_min) + self.g_min
#         return 1.0 / g
#
#     def r_to_w(self, r):
#         g = 1.0 / r
#         w_norm = (g - self.g_min) / (self.g_max - self.g_min)
#         return w_norm * (self.w_max - self.w_min) + self.w_min
#
#     def f_upper(self, v):
#         r_off = (self.r_max + self.r_min) / 2.0
#         r_s = (self.r_max - self.r_min) / 2.0
#         return r_off + r_s * torch.tanh((v - self.voff_up) / self.v0_up)
#
#     def f_lower(self, v):
#         r_off = (self.r_max + self.r_min) / 2.0
#         r_s = (self.r_max - self.r_min) / 2.0
#         return r_off + r_s * torch.tanh((v - self.voff_low) / self.v0_low)
#
#     def r_to_weff(self, r):
#         r_min = torch.as_tensor(self.r_min, dtype=r.dtype, device=r.device)
#         r_max = torch.as_tensor(self.r_max, dtype=r.dtype, device=r.device)
#
#         if r_min.ndim == 0:
#             r_min = torch.full_like(r, r_min)
#         if r_max.ndim == 0:
#             r_max = torch.full_like(r, r_max)
#
#         ratio = r_max / r_min
#         delta_w = self.w_max - self.w_min
#         h_short = self.w_sat - self.w_max
#
#         weff = torch.empty_like(r)
#
#         normal_mask = (r >= r_min) & (r <= r_max)
#         open_mask = r > r_max
#         short_mask = r < r_min
#
#         if normal_mask.any():
#             weff[normal_mask] = (
#                 self.w_min
#                 + delta_w * r_min[normal_mask] / r[normal_mask]
#             )
#
#         if open_mask.any():
#             y = r_max[open_mask] / r[open_mask]
#             weff[open_mask] = (
#                 self.w_min
#                 + (delta_w / ratio[open_mask]) * (6.0 * y**3 - 8.0 * y**4 + 3.0 * y**5)
#             )
#
#         if short_mask.any():
#             r_short = torch.clamp(r[short_mask], min=0.0)
#             x = r_short / r_min[short_mask]
#             weff[short_mask] = (
#                 self.w_sat
#                 + (-10.0 * h_short + 5.0 * delta_w) * x**3
#                 + (15.0 * h_short - 9.0 * delta_w) * x**4
#                 + (-6.0 * h_short + 4.0 * delta_w) * x**5
#             )
#
#         return weff
#
#     def weff_to_r(self, weff):
#         r_min = torch.as_tensor(self.r_min, dtype=weff.dtype, device=weff.device)
#         r_max = torch.as_tensor(self.r_max, dtype=weff.dtype, device=weff.device)
#
#         if r_min.ndim == 0:
#             r_min = torch.full_like(weff, r_min)
#         if r_max.ndim == 0:
#             r_max = torch.full_like(weff, r_max)
#
#         ratio = r_max / r_min
#         delta_w = self.w_max - self.w_min
#         w_open_max = self.w_min + delta_w / ratio
#
#         r = torch.empty_like(weff)
#
#         open_mask = weff < w_open_max
#         normal_mask = (weff >= w_open_max) & (weff <= self.w_max)
#         short_mask = weff > self.w_max
#
#         if normal_mask.any():
#             r[normal_mask] = (
#                 r_min[normal_mask] * delta_w / (weff[normal_mask] - self.w_min)
#             )
#
#         if open_mask.any():
#             r[open_mask] = self._invert_open_branch(
#                 weff[open_mask],
#                 r_max[open_mask],
#                 r_min[open_mask]
#             )
#
#         if short_mask.any():
#             weff_short = torch.clamp(weff[short_mask], max=self.w_sat)
#             r[short_mask] = self._invert_short_branch(
#                 weff_short,
#                 r_min[short_mask],
#                 r_max[short_mask]
#             )
#
#         return r
#
#     def _r_to_weff_local(self, r, r_min, r_max):
#         delta_w = self.w_max - self.w_min
#         h_short = self.w_sat - self.w_max
#         ratio = r_max / r_min
#
#         weff = torch.empty_like(r)
#
#         normal_mask = (r >= r_min) & (r <= r_max)
#         open_mask = r > r_max
#         short_mask = r < r_min
#
#         if normal_mask.any():
#             weff[normal_mask] = (
#                 self.w_min + delta_w * r_min[normal_mask] / r[normal_mask]
#             )
#
#         if open_mask.any():
#             y = r_max[open_mask] / r[open_mask]
#             weff[open_mask] = (
#                 self.w_min
#                 + (delta_w / ratio[open_mask]) * (6.0 * y**3 - 8.0 * y**4 + 3.0 * y**5)
#             )
#
#         if short_mask.any():
#             r_short = torch.clamp(r[short_mask], min=0.0)
#             x = r_short / r_min[short_mask]
#             weff[short_mask] = (
#                 self.w_sat
#                 + (-10.0 * h_short + 5.0 * delta_w) * x**3
#                 + (15.0 * h_short - 9.0 * delta_w) * x**4
#                 + (-6.0 * h_short + 4.0 * delta_w) * x**5
#             )
#
#         return weff
#
#     def _invert_open_branch(self, weff_target, r_max_target, r_min_target, max_iter=80):
#         lo = r_max_target.clone()
#         hi = r_max_target * 1e6
#
#         for _ in range(max_iter):
#             mid = 0.5 * (lo + hi)
#             val = self._r_to_weff_local(mid, r_min_target, r_max_target)
#             move_right = val > weff_target
#             lo = torch.where(move_right, mid, lo)
#             hi = torch.where(move_right, hi, mid)
#
#         return 0.5 * (lo + hi)
#
#     def _invert_short_branch(self, weff_target, r_min_target, r_max_target, max_iter=80):
#         lo = torch.zeros_like(weff_target)
#         hi = r_min_target.clone()
#
#         for _ in range(max_iter):
#             mid = 0.5 * (lo + hi)
#             val = self._r_to_weff_local(mid, r_min_target, r_max_target)
#             move_right = val > weff_target
#             lo = torch.where(move_right, mid, lo)
#             hi = torch.where(move_right, hi, mid)
#
#         return 0.5 * (lo + hi)
#
#     def plot_r_weff_mapping(self,
#                             num_points=2000,
#                             r_max_scale=5.0,
#                             save_path=None,
#                             show_boundary=True):
#         import matplotlib.pyplot as plt
#
#         device = self.weight.device if hasattr(self, "weight") else "cpu"
#         dtype = self.weight.dtype if hasattr(self, "weight") else torch.float32
#
#         r_min_val = float(torch.as_tensor(self.r_min).mean().item())
#         r_max_val = float(torch.as_tensor(self.r_max).mean().item())
#
#         r_short = torch.linspace(0.0, r_min_val * 0.999, num_points // 4, device=device, dtype=dtype)
#         r_nom = torch.linspace(r_min_val, r_max_val, num_points // 2, device=device, dtype=dtype)
#         r_open = torch.linspace(r_max_val * 1.001, r_max_val * r_max_scale, num_points // 4, device=device, dtype=dtype)
#
#         w_short = self.r_to_weff(r_short)
#         w_nom = self.r_to_weff(r_nom)
#         w_open = self.r_to_weff(r_open)
#
#         plt.figure(figsize=(8, 5))
#         plt.plot(r_short.detach().cpu().numpy(), w_short.detach().cpu().numpy(), linewidth=2, label="short")
#         plt.plot(r_nom.detach().cpu().numpy(), w_nom.detach().cpu().numpy(), linewidth=2, label="nominal")
#         plt.plot(r_open.detach().cpu().numpy(), w_open.detach().cpu().numpy(), linewidth=2, label="open")
#
#         if show_boundary:
#             plt.axvline(r_min_val, linestyle="--", linewidth=1.2, color="gray")
#             plt.axvline(r_max_val, linestyle="--", linewidth=1.2, color="gray")
#
#         plt.xlabel("Resistance r (Ohm)")
#         plt.ylabel(r"Effective weight $w_{\mathrm{eff}}$")
#         plt.title(r"Piecewise mapping from resistance to $w_{\mathrm{eff}}$")
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#         plt.tight_layout()
#
#         if save_path is not None:
#             plt.savefig(f"{save_path}/r_weff_curve.pdf", format="pdf", bbox_inches="tight")
#             plt.savefig(f"{save_path}/r_weff_curve.png", dpi=300, bbox_inches="tight",)
#
#         plt.show()

class Ferroelectric_Tanh:
    """
    Linear-only branch version of Ferroelectric_Tanh.
    Keeps the original tanh voltage-to-resistance target functions,
    but uses only the nominal linear resistance-weight mapping.
    Weight range remains [w_min, w_max], default [0.0, 0.1].
    """

    def __init__(self,
                 weight,
                 sf_p=1.0,
                 sf_d=1.3,
                 r_min=1.1e9,
                 r_max=2.5e9,
                 v0_up=0.45,
                 voff_up=-1.0,
                 v0_low=0.60,
                 voff_low=1.6,
                 v_ref=1.02,
                 w_min=0.0,
                 w_max=0.1,
                 w_sat=0):
        self.sf_p = 1.0
        self.sf_d = 1.0
        self.r_min = r_min
        self.r_max = r_max
        self.v0_up = v0_up
        self.voff_up = voff_up
        self.v0_low = v0_low
        self.voff_low = voff_low
        self.v_ref = v_ref
        self.w_min = w_min
        self.w_max = w_max

        self.base_params = base_params

        self.v_max_p = self.base_params['voff_up'] - self.base_params['v0_up']
        self.v_max_d = self.base_params['voff_low'] + self.base_params['v0_low']

        self.g_min = 1.0 / self.r_max
        self.g_max = 1.0 / self.r_min

        self.weight = weight
        self.weight = torch.clamp(self.weight, min=self.w_min, max=self.w_max)

        self.defect_mask = (
                (self.r_max <= self.r_min) |
                (self.v0_up <= 0) |
                (self.v0_low <= 0) |
                (self.voff_up >= 0) |
                (self.voff_low <= 0)
        )

        self.cycle_vc_multiplicative = MODEL_CONFIGS.get(
            'Ferroelectric_Tanh_Linear', {}
        ).get("variations", {}).get(
            'cycle_to_cycle_variation_coefficient_multiplicative', 0.0
        )

        self.cycle_vc_additive = MODEL_CONFIGS.get(
            'Ferroelectric_Tanh_Linear', {}
        ).get("variations", {}).get(
            'cycle_to_cycle_variation_coefficient_additive', 0.0
        )

        self.c2c_additive_ref_scale = self.r_max - self.r_min

    def _clamp_r(self, r):
        r_min = torch.as_tensor(self.r_min, dtype=r.dtype, device=r.device)
        r_max = torch.as_tensor(self.r_max, dtype=r.dtype, device=r.device)

        if r_min.ndim == 0:
            r_min = torch.full_like(r, r_min)
        if r_max.ndim == 0:
            r_max = torch.full_like(r, r_max)

        return torch.clamp(r, min=r_min, max=r_max)

    def __call__(self, potential, potential_rest, potential_threshold, cond_pot):
        v_p = (potential / potential_rest) * self.sf_p * self.v_max_p
        v_d = (self.v_ref - potential / potential_threshold) * self.sf_d * self.v_max_d

        r_initial = self.w_to_r(self.weight)

        r_target_p = self._clamp_r(self.f_upper(v_p))
        r_target_d = self._clamp_r(self.f_lower(v_d))

        delta_r_p = r_target_p - r_initial
        delta_r_d = r_target_d - r_initial

        noise_p = (
                torch.randn_like(delta_r_p) * delta_r_p * self.cycle_vc_multiplicative
                + torch.randn_like(delta_r_p) * self.c2c_additive_ref_scale * self.cycle_vc_additive
        )
        noise_d = (
                torch.randn_like(delta_r_d) * delta_r_d * self.cycle_vc_multiplicative
                + torch.randn_like(delta_r_d) * self.c2c_additive_ref_scale * self.cycle_vc_additive
        )

        r_final_p = self._clamp_r(r_target_p + noise_p)
        r_final_d = self._clamp_r(r_target_d + noise_d)

        r_final_p = torch.min(r_initial, r_final_p)
        r_final_d = torch.max(r_initial, r_final_d)

        new_weight_ltp = self.r_to_w(r_final_p)
        new_weight_ltd = self.r_to_w(r_final_d)

        delta_weight_ltp = new_weight_ltp - self.weight
        delta_weight_ltd = new_weight_ltd - self.weight

        if torch.any(self.defect_mask):
            delta_weight_ltp[self.defect_mask] = 0.0
            delta_weight_ltd[self.defect_mask] = 0.0
            self.weight[self.defect_mask] = 0.0

        eps = 1e-8
        cond_ltp = cond_pot & (potential < -eps)
        cond_ltd = (~cond_pot) & (potential > eps)

        zero_update = torch.zeros_like(self.weight)

        return torch.where(
            cond_ltp,
            delta_weight_ltp,
            torch.where(cond_ltd, delta_weight_ltd, zero_update)
        )

    def w_to_r(self, w):
        w = torch.clamp(w, min=self.w_min, max=self.w_max)
        w_norm = (w - self.w_min) / (self.w_max - self.w_min)
        g = w_norm * (self.g_max - self.g_min) + self.g_min
        return 1.0 / g

    def r_to_w(self, r):
        r = self._clamp_r(r)
        g = 1.0 / r
        w_norm = (g - self.g_min) / (self.g_max - self.g_min)
        w = w_norm * (self.w_max - self.w_min) + self.w_min
        return torch.clamp(w, min=self.w_min, max=self.w_max)

    def f_upper(self, v):
        r_off = (self.r_max + self.r_min) / 2.0
        r_s = (self.r_max - self.r_min) / 2.0
        return r_off + r_s * torch.tanh((v - self.voff_up) / self.v0_up)

    def f_lower(self, v):
        r_off = (self.r_max + self.r_min) / 2.0
        r_s = (self.r_max - self.r_min) / 2.0
        return r_off + r_s * torch.tanh((v - self.voff_low) / self.v0_low)

if __name__ == "__main__":
    model = Ferroelectric_Tanh(torch.tensor([0.1, 0.8]))
    model.plot_r_weff_mapping(save_path="figures")

