import os
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model
from pathlib import Path
from lmfit import Minimizer, Parameters, report_fit
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit

"""
Run this file to get characterization figures of the memristor devices. Fig.4.3 $ 4.5 & 4.6
change SYNAPSE_MODEL in config.py
"""

save_path = "C:/Users/28218/PycharmProjects/CSNN/figures"
RAW_DEVICE_DATA_PATH = "data/ABS_03_summary.dat"
MODEL_CONFIGS = {
    'Ferroelectric': {
        "params":{
            'gamma': {'value': 1, 'min': 0.01, 'max': 5},
            'gamma_pn': {'value': 1.0, 'min': 0.01, 'max': 2},
            'vth': {'value': 0.35, 'min': 0.1, 'max': 1},
            'vth_pn': {'value': 1.0, 'min': 0.5, 'max': 2},
            'alpha': {'value': 1.5, 'min': 0.01, 'max': 3},
            'alpha_pn': {'value': 1.0, 'min': 0.5, 'max': 2}
        },
        'output_mapping': {
            'gamma_p': lambda p: p['gamma'],
            'gamma_d': lambda p: p['gamma'] * p['gamma_pn'],
            'theta_p': lambda p: -p['vth'],
            'theta_d': lambda p: p['vth'] * p['vth_pn'],
            'alpha_p': lambda p: p['alpha'],
            'alpha_d': lambda p: p['alpha'] * p['alpha_pn'],
        },
        "variations": {
            # This defines the std (value*coefficient) of the noise added to the measurements.(0.05=5% variations)
            "device_to_device_variation_coefficient": {
                "gamma_p": 0.0,
                "gamma_d": 0.0,
                "theta_p": 0.0,
                "theta_d": 0.0,
                "alpha_p": 0.0,
                "alpha_d": 0.0
            },
            "cycle_to_cycle_variation_coefficient_multiplicative": 0.0,
            "cycle_to_cycle_variation_coefficient_additive": 0.0

        }
    },
    'Ferroelectric_Tanh': {
        "params_upper": {
            'r_min': {'value': 1.1e9, 'vary':False},
            'r_max': {'value': 2.5e9, 'vary':False},
            'v0_up': {'value': 0.45, 'min': 0.1, 'max': 2.0},
            'voff_up': {'value': -1.0, 'min': -3.0, 'max': 0.0}
        },
        "params_lower": {
            'r_min': {'value': 1.1e9, 'vary':False},
            'r_max': {'value': 2.5e9, 'vary':False},
            'v0_low': {'value': 0.60, 'min': 0.1, 'max': 2.0},
            'voff_low': {'value': 1.6, 'min': 0.0, 'max': 3.0}
        },
        "variations":{  # This defines the std (value*coefficient) of the noise added to the measurements.(0.05=5% variations)
            "device_to_device_variation_coefficient":{
                "r_min":0.2,
                "r_max":0.2,
                "v0_up":0.2,
                "voff_up":0.2,
                "v0_low":0.2,
                "voff_low":0.2
            },
            "cycle_to_cycle_variation_coefficient_multiplicative": 0.0,
            "cycle_to_cycle_variation_coefficient_additive": 0.0

        }
    },
}
# Define the model function. ignore the default values, they will be set by the fit function.

def memristor_Ferroelectric(w, vmem,
                   gamma=1.0,
                   gamma_pn=1.0,
                   alpha=1.0,
                   alpha_pn=1.0,
                   vth=0.1,
                   vth_pn=1
                   ):
    alpha_p = alpha
    alpha_n = alpha * alpha_pn
    vth_p = vth
    vth_n = vth * vth_pn
    gamma_p = gamma
    gamma_n = gamma * gamma_pn

    f_p = np.power(1 - w, gamma_p)
    f_n = np.power(w, gamma_n)

    cond_p = vmem < -vth_p
    cond_n = vmem > vth_n

    g_p = np.abs(np.exp(-alpha_p * (vmem + vth_p)) - 1)
    g_n = np.abs(np.exp(alpha_n * (vmem - vth_n)) - 1)

    dW = (cond_p * f_p * g_p) - (cond_n * f_n * g_n)

    W = np.clip(w + dW, 0, 1)

    dW = W - w

    return dW

def tanh_envelope_upper(V, r_min, r_max, v0_up, voff_up):
    r_off = (r_max + r_min) / 2.0
    r_s = (r_max - r_min) / 2.0
    return r_off + r_s * np.tanh((V - voff_up) / v0_up)

def tanh_envelope_lower(V, r_min, r_max, v0_low, voff_low):
    r_off = (r_max + r_min) / 2.0
    r_s = (r_max - r_min) / 2.0
    return r_off + r_s * np.tanh((V - voff_low) / v0_low)

class ModelCharac:
    """
    Represents a class for modeling and characterizing memristor devices.

    This class facilitates various operations related to memristor modeling,
    including fitting data using specific methods for different types of memristors.
    The characterization can compute parameters from raw data using predefined
    fitting methods and save them for later use.

    :ivar force_recompute: Flag to force recomputation of the characterization (default: False).
    :type force_recompute: bool
    :ivar memristor_type: Type of the memristor device being modeled.
    :type memristor_type: str
    :ivar save_path: Path where the computed parameters for the memristor will be saved.
    :type save_path: str
    :ivar data_df: DataFrame containing the raw input data for the modeling process.
    :type data_df: pd.DataFrame
    :ivar fit_methods_dictionary: Dictionary mapping memristor types to their respective fitting functions.
    :type fit_methods_dictionary: dict[str, Callable]
    """
    def __init__(self, memristor_type, force_recompute=False):
        self.force_recompute = force_recompute
        self.memristor_type = memristor_type

        data_path = self.normalize_data(RAW_DEVICE_DATA_PATH)
        self.save_path = f"{str(Path(data_path).parent)}/params_{memristor_type}.csv"

        self.data_df = pd.read_csv(data_path)


        self.fit_methods_dictionary = {
            'Ferroelectric': self.fit_ferroelectric_vdsp,
            'Ferroelectric_Tanh': self.fit_ferroelectric_tanh
        }

    def __call__(self):
        if os.path.exists(self.save_path) and not self.force_recompute:
            print(f">>> Found existing parameters for {self.memristor_type}, loading...")
            df = pd.read_csv(self.save_path)
            print(df)
            return df.to_dict(orient='records')[0]

        print(f">>> Characterizing {self.memristor_type} directly from raw data...")

        fit_function = self.fit_methods_dictionary.get(self.memristor_type)

        if fit_function:
            final_params = fit_function()
            pd.DataFrame([final_params]).to_csv(self.save_path, index=False)
            print(f">>> Parameters saved as: {self.save_path}")
            return final_params
        else:
            print(
                "Model not found or configuration not provided. Probably the model does not need to be characterized.")
            return None

    def fit_ferroelectric_vdsp(self):
        config = MODEL_CONFIGS['Ferroelectric']
        params = Parameters()
        for p_name, p_setting in config['params'].items():
            params.add(p_name, **p_setting)

        model = Model(memristor_Ferroelectric, independent_vars=['vmem', 'w'])
        fit = model.fit(self.data_df['dw'].values, params, vmem=self.data_df['V'].values, w=self.data_df['w'].values)

        raw_params = {name: p.value for name, p in fit.params.items()}

        v_exp = self.data_df['V'].values
        w_exp = self.data_df['w'].values
        dw_exp = self.data_df['dw'].values

        dw_model = memristor_Ferroelectric(w_exp, v_exp, **raw_params)

        # --- Derive nominal output params for D2D shadow ---
        output_mapping = config['output_mapping']
        nominal_output = {key: func(raw_params) for key, func in output_mapping.items()}

        d2d_vc = config["variations"]["device_to_device_variation_coefficient"]
        rng = np.random.default_rng(42)
        n_shadow = 30
        v_grid = np.linspace(-3.5, 3.5, 300)
        w_rep_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # Left: measurements
        sc1 = axes[0].scatter(v_exp, dw_exp, c=w_exp, cmap='plasma', s=10, alpha=0.8, zorder=3)
        axes[0].set_title('Measurements', fontsize=14)
        axes[0].set_xlabel('$V_{write}$ (V)', fontsize=12)
        axes[0].set_ylabel(r'$\Delta W$', fontsize=12)
        axes[0].set_ylim(-1.1, 1.1)
        axes[0].set_xlim(-3.5, 3.5)

        # Right: nominal model + D2D shadow
        sc2 = axes[1].scatter(v_exp, dw_model, c=w_exp, cmap='plasma', s=10, alpha=0.6, zorder=3)

        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sc1 = ax1.scatter(v_exp, dw_exp, c=w_exp, cmap='plasma', s=10, alpha=0.8, zorder=3)
        ax1.set_title('Measurements', fontsize=14)
        ax1.set_xlabel('$V_{write}$ (V)', fontsize=12)
        ax1.set_ylabel(r'$\Delta W$', fontsize=12)
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_xlim(-3.5, 3.5)
        cbar1 = fig1.colorbar(sc1, ax=ax1, pad=0.02)
        cbar1.set_label('$W_0$', rotation=0, labelpad=15, fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_path}/exponential_fit_measurements.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(f"{save_path}/exponential_fit_measurements.png", dpi=300, bbox_inches="tight")
        plt.show()

        fig2, ax2 = plt.subplots(figsize=(6, 5))

        sc2 = ax2.scatter(v_exp, dw_model, c=w_exp, cmap='plasma', s=10, alpha=0.6, zorder=3)

        # shadow fill_between
        from collections import defaultdict
        shadow_curves = defaultdict(list)
        for i in range(n_shadow):
            gp_s = nominal_output['gamma_p'] * (1 + rng.normal(0, d2d_vc['gamma_p']))
            gd_s = nominal_output['gamma_d'] * (1 + rng.normal(0, d2d_vc['gamma_d']))
            tp_s = abs(nominal_output['theta_p']) * (1 + rng.normal(0, d2d_vc['theta_p']))
            td_s = nominal_output['theta_d'] * (1 + rng.normal(0, d2d_vc['theta_d']))
            ap_s = nominal_output['alpha_p'] * (1 + rng.normal(0, d2d_vc['alpha_p']))
            ad_s = nominal_output['alpha_d'] * (1 + rng.normal(0, d2d_vc['alpha_d']))
            for w_val in w_rep_values:
                w_grid = np.full_like(v_grid, w_val)
                dw_shadow = memristor_Ferroelectric(w_grid, v_grid,
                                                    gamma=gp_s, gamma_pn=gd_s / gp_s,
                                                    alpha=ap_s, alpha_pn=ad_s / ap_s,
                                                    vth=tp_s, vth_pn=td_s / tp_s
                                                    )
                shadow_curves[w_val].append(dw_shadow)

        for w_val in w_rep_values:
            stack = np.array(shadow_curves[w_val])
            lo = np.min(stack, axis=0)
            hi = np.max(stack, axis=0)
            color = plt.cm.plasma(w_val)
            ax2.fill_between(v_grid, lo, hi, color=color, alpha=0.5)

        for w_val in w_rep_values:
            w_grid = np.full_like(v_grid, w_val)
            dw_nom = memristor_Ferroelectric(w_grid, v_grid,
                                             gamma=nominal_output['gamma_p'],
                                             gamma_pn=nominal_output['gamma_d'] / nominal_output['gamma_p'],
                                             alpha=nominal_output['alpha_p'],
                                             alpha_pn=nominal_output['alpha_d'] / nominal_output['alpha_p'],
                                             vth=abs(nominal_output['theta_p']),
                                             vth_pn=nominal_output['theta_d'] / abs(nominal_output['theta_p'])
                                             )
            ax2.plot(v_grid, dw_nom, color=plt.cm.plasma(w_val), lw=2.0, zorder=4)

        ax2.set_title('Model + D2D Variation', fontsize=14)
        ax2.set_xlabel('$V_{write}$ (V)', fontsize=12)
        ax2.set_ylabel(r'$\Delta W_{est.}$', fontsize=12)
        ax2.set_xlim(-3.5, 3.5)
        ax2.set_ylim(-1.1, 1.1)
        cbar2 = fig2.colorbar(sc2, ax=ax2, pad=0.02)
        cbar2.set_label('$W_0$', rotation=0, labelpad=15, fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_path}/exponential_fit_{d2d_vc["theta_p"]}.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(f"{save_path}/exponential_fit_{d2d_vc["theta_p"]}.png", dpi=300, bbox_inches="tight")
        plt.show()

        return {key: func(raw_params) for key, func in config['output_mapping'].items()}

    def fit_ferroelectric_tanh(self):
        config = MODEL_CONFIGS['Ferroelectric_Tanh']

        n_points = 300
        steady_data = self.data_df.iloc[-n_points:]

        V = steady_data['V'].values
        R = steady_data['Rfinal'].values

        Nbins = 50
        Venv_all, _, _ = binned_statistic(V, V, statistic='mean', bins=Nbins)
        R_low_all, _, _ = binned_statistic(V, R, statistic='min', bins=Nbins)
        R_high_all, _, _ = binned_statistic(V, R, statistic='max', bins=Nbins)

        valid = ~np.isnan(Venv_all) & ~np.isnan(R_low_all) & ~np.isnan(R_high_all)
        Venv = Venv_all[valid]
        R_low = R_low_all[valid]
        R_high = R_high_all[valid]

        def tanh_full_model(x, Rs, V0, Voff, Roff):
            return Rs * np.tanh((x - Voff) / V0) + Roff

        p0_L = [
            (np.max(R_low) - np.min(R_low)) / 2,
            config['params_lower']['v0_low']['value'],
            config['params_lower']['voff_low']['value'],
            np.mean(R_low)
        ]

        p_low_full, _ = curve_fit(tanh_full_model, Venv, R_low, p0=p0_L, maxfev=10000)
        Rs_L, V0_L, Voff_L, Roff_L = p_low_full

        r_min_derived = Roff_L - Rs_L
        r_max_derived = Roff_L + Rs_L

        def tanh_constrained_model(x, V0, Voff):
            return Rs_L * np.tanh((x - Voff) / V0) + Roff_L

        p0_H = [config['params_upper']['v0_up']['value'], config['params_upper']['voff_up']['value']]
        p_high_res, _ = curve_fit(tanh_constrained_model, Venv, R_high, p0=p0_H, maxfev=10000)
        V0_H, Voff_H = p_high_res

        print(f"--- Derived from Lower: Rmin={r_min_derived:.3e}, Rmax={r_max_derived:.3e}")
        print(f"--- Fit Result: Low(V0={V0_L:.3f}, Voff={Voff_L:.3f}) | High(V0={V0_H:.3f}, Voff={Voff_H:.3f})")

        # ---------------------------------------------------------

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_df = self.data_df.copy()

        scatter = ax.scatter(
            plot_df['V'], plot_df['Rfinal'] / 1e9,
            c=plot_df['dR'] / 1e9,
            cmap='RdBu_r', vmin=-2.0, vmax=2.0,
            alpha=0.2, s=8
        )
        plt.colorbar(scatter, ax=ax).set_label(r'$\Delta R$ (G$\Omega$)')

        ax.scatter(Venv, R_low / 1e9, c='g', s=25, label='Lower Envelope (Min points)')
        ax.scatter(Venv, R_high / 1e9, c='m', s=25, label='Upper Envelope (Max points)')

        v_plot = np.linspace(V.min(), V.max(), 600)

        lower_main = tanh_full_model(v_plot, *p_low_full)
        upper_main = tanh_constrained_model(v_plot, *p_high_res)

        d2d_vc = config["variations"]["device_to_device_variation_coefficient"]
        rng = np.random.default_rng(42)
        n_shadow = 30

        for _ in range(n_shadow):
            r_min_s = r_min_derived * (1 + rng.normal(0, d2d_vc["r_min"]))
            r_max_s = r_max_derived * (1 + rng.normal(0, d2d_vc["r_max"]))
            v0_low_s = V0_L * (1 + rng.normal(0, d2d_vc["v0_low"]))
            voff_low_s = Voff_L * (1 + rng.normal(0, d2d_vc["voff_low"]))
            v0_up_s = V0_H * (1 + rng.normal(0, d2d_vc["v0_up"]))
            voff_up_s = Voff_H * (1 + rng.normal(0, d2d_vc["voff_up"]))

            lower_shadow = tanh_envelope_lower(v_plot, r_min_s, r_max_s, v0_low_s, voff_low_s)
            upper_shadow = tanh_envelope_upper(v_plot, r_min_s, r_max_s, v0_up_s, voff_up_s)

            ax.plot(v_plot, lower_shadow / 1e9, color='red', alpha=0.2, lw=1)
            ax.plot(v_plot, upper_shadow / 1e9, color='black', alpha=0.2, lw=1)

        ax.plot(v_plot, lower_main / 1e9, 'r', lw=2.5, label='Lower Fit')
        ax.plot(v_plot, upper_main / 1e9, 'k', lw=2.5, label='Upper Fit (Constrained R)')

        ax.set_xlabel('Pulse Amplitude (V)')
        ax.set_ylabel('Read Resistance (GΩ)')
        ax.set_title('Tanh Fit with D2D Shadow Curves')
        ax.legend(loc='best', frameon=False)
        plt.savefig(f"{save_path}/Tanh_fit_{d2d_vc["r_min"]}.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(f"{save_path}/Tanh_fit_{d2d_vc["r_min"]}.png", dpi=300, bbox_inches="tight", )
        plt.show()

        return {
            'r_min': r_min_derived,
            'r_max': r_max_derived,
            'v0_up': V0_H,
            'voff_up': Voff_H,
            'v0_low': V0_L,
            'voff_low': Voff_L
        }

    def normalize_data(self, file_path):
        save_name = Path(file_path).with_name(Path(file_path).stem + "_normalized.csv")
        if os.path.exists(save_name) and not self.force_recompute:
            return str(save_name)

        data = pd.read_csv(file_path, sep=',')
        V = data['pulseAmplitude']
        dR = data['deltaRpos(measured at +80mV)']
        Rinit = data['RposInitial']
        Rfinal = Rinit + dR

        Ginitial, Gfinal = 1 / Rinit, 1 / Rfinal
        Gmin = min(Ginitial.min(), Gfinal.min())
        Gmax = max(Ginitial.max(), Gfinal.max())

        winit = (Ginitial - Gmin) / (Gmax - Gmin)
        wfinal = (Gfinal - Gmin) / (Gmax - Gmin)

        data_normalized = pd.DataFrame({
            'V': V, 'Rinit': Rinit, 'Rfinal': Rfinal, 'dR': dR,
            'dw': wfinal - winit, 'w': winit
        })
        data_normalized.to_csv(save_name, index=False)
        return str(save_name)


if __name__ == "__main__":
    from config import *
    model_charac = ModelCharac(SYNAPSE_MODEL, force_recompute=True)
    model_charac()