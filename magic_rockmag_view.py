"""
Lecture des donnees rock-mag (K-T et hysteresis) d'une contribution MagIC
avec les outils PmagPy - demande explicite utilisateur ("mettre les outils
PmagPy dans Stereo_Utils pour lire les donnees de rockmag (KT et
Hysteresis) des contributions Magic"). Sert a relire ce qui a ete depose
(par kt_magic.py / hyst_magic.py ou par d'autres) tel que PmagPy le voit.

Les calculs sont ceux de PmagPy (pas les notres) :
- K-T (LP-X-T) : rockmag.prepare_thermomag_branches (separation chauffe/
  refroidissement, meas_temp en K -> degC) et
  rockmag.curie_temperature_estimates (inflexion, courbure max, Curie-
  Weiss) ;
- hysteresis (LP-HYS) : rockmag.process_hyst_loop (Jackson & Solheid :
  grille, centrage, derive, pente haut champ, tests de fermeture/
  saturation) ; courbe de remanence LP-IRM-DCD (champ dans treat_dc_field,
  convention PmagPy) ;
- Day plot : pmagplotlib.plot_day (avec la courbe de melange SD-MD de
  Dunlop 2002), sur les parametres de la table specimens.

Les fonctions de trace de PmagPy dessinent dans des figures pyplot
(plt.figure(num)) ou en bokeh - pas dans la Figure embarquee de l'appli :
les traces sont donc refaits ici sur `fig` a partir des tableaux renvoyes
par PmagPy, et pour plot_day les lignes de la figure pyplot temporaire
sont recopiees.
"""

import contextlib
import io
import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MagicRockmag:
    path: str
    measurements: pd.DataFrame
    specimens: Optional[pd.DataFrame]


def _read_magic_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=1, low_memory=False)


def load_magic_contribution(path: str) -> MagicRockmag:
    """Fichier de contribution combine (telecharge de MagIC ou ecrit par
    l'export, plusieurs tables separees par >>>>>>>>>>) -> decoupe par
    ipmag.download_magic dans un dossier temporaire ; sinon un
    measurements.txt seul (specimens.txt lu s'il est dans le meme
    dossier)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = f.read(200000)
    if ">>>>>>>>>>" in head or head.split("\n", 1)[0].split("\t")[-1].strip() != "measurements":
        from pmagpy import ipmag
        tmp = tempfile.mkdtemp(prefix="magic_rockmag_")
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            ipmag.download_magic(os.path.basename(path), dir_path=tmp,
                                 input_dir_path=os.path.dirname(path) or ".",
                                 print_progress=False)
        meas_path = os.path.join(tmp, "measurements.txt")
        spec_path = os.path.join(tmp, "specimens.txt")
    else:
        meas_path = path
        spec_path = os.path.join(os.path.dirname(path), "specimens.txt")
    if not os.path.exists(meas_path):
        raise ValueError("no measurements table in this file")
    meas = _read_magic_table(meas_path)
    for col in ("method_codes", "experiment", "specimen"):
        if col not in meas.columns:
            raise ValueError(f"measurements table has no '{col}' column")
    specs = _read_magic_table(spec_path) if os.path.exists(spec_path) else None
    return MagicRockmag(path, meas, specs)


def _has_code(meas: pd.DataFrame, code: str) -> pd.Series:
    return meas["method_codes"].fillna("").str.split(":").apply(
        lambda codes: code in [c.strip() for c in codes])


def _first_filled(df: pd.DataFrame, columns: List[str]) -> Optional[str]:
    for c in columns:
        if c in df.columns and df[c].notna().any():
            return c
    return None


# ----------------------------------------------------------------------
# K-T
# ----------------------------------------------------------------------

KT_COLUMNS = ["susc_chi_mass", "susc_chi_volume", "susc_chi_qdr_mass", "susc_chi_qdr_volume",
              "magn_mass", "magn_moment"]


def kt_experiments(data: MagicRockmag) -> List[Tuple[str, str, int]]:
    """(experiment, specimen, n) des experiences LP-X-T."""
    m = data.measurements[_has_code(data.measurements, "LP-X-T")]
    g = m.groupby(["experiment", "specimen"], sort=False).size()
    return [(e, s, int(n)) for (e, s), n in g.items()]


def kt_experiment(data: MagicRockmag, experiment: str) -> Tuple[pd.DataFrame, Optional[str]]:
    exp = data.measurements[data.measurements["experiment"] == experiment]
    exp = exp.sort_values("treat_step_num") if "treat_step_num" in exp.columns else exp
    return exp.reset_index(drop=True), _first_filled(exp, KT_COLUMNS)


def build_magic_kt_figure(fig, experiment: pd.DataFrame, column: str, title: str,
                          smooth_window: float = 0.0) -> pd.DataFrame:
    """Courbe K-T (chauffe/refroidissement, degC) + derivee, et tableau
    des Tc de rockmag.curie_temperature_estimates. remove_holder=False :
    PmagPy retranche sinon le minimum de chaque branche (hypothese d'un
    signal nul au-dessus de Tc), ce qui decale les courbes affichees."""
    from pmagpy import rockmag
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        br = rockmag.prepare_thermomag_branches(
            experiment, magnetic_column=column, smooth_window=smooth_window, remove_holder=False)
        est = rockmag.curie_temperature_estimates(
            experiment, magnetic_column=column, smooth_window=smooth_window, remove_holder=False)
    fig.clf()
    ax = fig.add_subplot(211)
    axd = fig.add_subplot(212, sharex=ax)
    colors = {"heating": "tab:red", "cooling": "tab:blue"}
    for name in ("heating", "cooling"):
        b = br.get(name)
        if not b:
            continue
        ax.plot(b["raw_T"], b["raw_y"], ".", ms=2.5, color=colors[name], alpha=0.5)
        ax.plot(b["T"], b["y"], "-", lw=1, color=colors[name], label=name)
        if len(b["T"]) > 2:
            axd.plot(b["T"], np.gradient(b["y"], b["T"]), "-", lw=1, color=colors[name])
    if est is not None and len(est):
        for _, r in est[est["method"] == "inflection"].iterrows():
            if np.isfinite(r["curie_temp"]):
                ax.axvline(r["curie_temp"], color=colors.get(r["branch"], "k"), ls=":", lw=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(column, fontsize=8)
    ax.legend(fontsize=7)
    axd.set_xlabel("T (degC)", fontsize=8)
    axd.set_ylabel("d/dT", fontsize=8)
    for a in (ax, axd):
        a.tick_params(labelsize=7)
        a.grid(alpha=0.3)
    fig.tight_layout()
    return est


def format_kt_estimates(est: pd.DataFrame, column: str, smooth_window: float) -> str:
    lines = [f" PmagPy rockmag.curie_temperature_estimates ({column}, "
             f"smoothing {smooth_window:g} degC):"]
    if est is None or not len(est):
        return lines[0] + " none\n"
    for _, r in est.iterrows():
        err = r.get("curie_temp_stderr")
        err_s = f" +/- {err:.1f}" if err is not None and np.isfinite(err) else ""
        tc = r["curie_temp"]
        lines.append(f"  {r['branch']:<8} {r['method']:<24} "
                     f"{(f'{tc:7.1f}' if np.isfinite(tc) else '    n/a')}{err_s} degC")
    lines.append("  (inverse_susceptibility = paramagnetic Curie temperature, >= Tc; "
                 "an estimate at the first/last temperature is an edge artefact)")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Hysteresis
# ----------------------------------------------------------------------

@dataclass
class MagicHystSpecimen:
    specimen: str
    loop_experiment: Optional[str]
    dcd_experiment: Optional[str]


def hyst_specimens(data: MagicRockmag) -> List[MagicHystSpecimen]:
    m = data.measurements
    loops = m[_has_code(m, "LP-HYS")].groupby("specimen", sort=False)["experiment"].first()
    dcds = m[_has_code(m, "LP-IRM-DCD")].groupby("specimen", sort=False)["experiment"].first()
    names = list(dict.fromkeys(list(loops.index) + list(dcds.index)))
    return [MagicHystSpecimen(s, loops.get(s), dcds.get(s)) for s in names]


def _bcr_zero_crossing(field: np.ndarray, magn: np.ndarray) -> float:
    """Champ (positif, T) ou la remanence DCD s'annule - interpolation
    lineaire au premier changement de signe."""
    order = np.argsort(np.abs(field))
    f, mm = np.abs(field[order]), magn[order]
    for i in range(1, len(mm)):
        if mm[i - 1] > 0 >= mm[i]:
            return float(f[i - 1] + (f[i] - f[i - 1]) * mm[i - 1] / (mm[i - 1] - mm[i]))
    return float("nan")


def build_magic_hyst_figure(fig, data: MagicRockmag, spec: MagicHystSpecimen) -> dict:
    """Boucle brute + boucle corrigee (pente haut champ) par
    rockmag.process_hyst_loop, et courbe DCD avec Bcr. Retourne le dict de
    process_hyst_loop + 'Bcr_dcd', 'column', '_warnings'."""
    from pmagpy import rockmag
    m = data.measurements
    out = {"_warnings": []}
    fig.clf()
    ax = fig.add_subplot(211 if spec.dcd_experiment else 111)
    if spec.loop_experiment:
        loop = m[m["experiment"] == spec.loop_experiment]
        loop = loop.sort_values("treat_step_num") if "treat_step_num" in loop.columns else loop
        col = _first_filled(loop, ["magn_mass", "magn_moment", "magn_volume"])
        H = loop["meas_field_dc"].to_numpy(float)
        M = loop[col].to_numpy(float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = rockmag.process_hyst_loop(H, M, specimen_name=spec.specimen,
                                            show_results_table=False, show_plot=False)
        out.update(res)
        out["_warnings"] = [str(w.message) for w in caught]
        out["column"] = col
        ax.plot(H * 1e3, M, "-", color="0.6", lw=0.8, label="measured")
        if res.get("slope_corrected_M") is not None:
            ax.plot(np.asarray(res["centered_H"]) * 1e3, res["slope_corrected_M"], "-",
                    color="tab:blue", lw=1, label="PmagPy slope-corrected")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("B (mT)", fontsize=8)
        ax.set_ylabel(col, fontsize=8)
        ax.legend(fontsize=7)
    ax.set_title(spec.specimen, fontsize=9)
    if spec.dcd_experiment:
        dcd = m[m["experiment"] == spec.dcd_experiment]
        dcd = dcd.sort_values("treat_step_num") if "treat_step_num" in dcd.columns else dcd
        dcol = _first_filled(dcd, ["magn_mass", "magn_moment", "magn_volume"])
        B = dcd["treat_dc_field"].to_numpy(float)
        R = dcd[dcol].to_numpy(float)
        bcr = _bcr_zero_crossing(B, R)
        out["Bcr_dcd"] = bcr
        ax2 = fig.add_subplot(212)
        ax2.plot(B * 1e3, R, ".-", color="tab:green", ms=3, lw=1)
        ax2.axhline(0, color="k", lw=0.5)
        if np.isfinite(bcr):
            ax2.axvline(-bcr * 1e3, color="tab:red", ls=":", lw=0.8)
            ax2.text(0.02, 0.9, f"Bcr {bcr * 1e3:.1f} mT", transform=ax2.transAxes, fontsize=8)
        ax2.set_xlabel("B (mT) - IRM-DCD", fontsize=8)
        ax2.set_ylabel(dcol, fontsize=8)
        ax2.tick_params(labelsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return out


def format_magic_hyst(data: MagicRockmag, spec: MagicHystSpecimen, res: dict) -> str:
    def num(key, scale=1.0, unit="", fmt=".4g"):
        v = res.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not np.isfinite(v) else f"{v * scale:{fmt}} {unit}".strip()

    col = res.get("column", "")
    u = "Am2/kg" if col == "magn_mass" else "Am2"
    lines = [f" {spec.specimen} - PmagPy rockmag.process_hyst_loop ({col}):"]
    if spec.loop_experiment:
        lines += [
            f"  Ms {num('Ms', unit=u)}   Mr {num('Mr', unit=u)}   Bc {num('Bc', 1e3, 'mT')}   "
            f"Brh {num('Brh', 1e3, 'mT')}   chi_HF {num('chi_HF')}",
            f"  closed: {res.get('loop_is_closed')}   saturated: {res.get('loop_is_saturated')}   "
            f"linear: {res.get('loop_is_linear')}   Q {num('Q', fmt='.2f')}",
        ]
    if spec.dcd_experiment:
        lines.append(f"  Bcr (IRM-DCD zero crossing) {num('Bcr_dcd', 1e3, 'mT')}")
    for w in res.get("_warnings", []):
        lines.append(f"  PmagPy warning: {w}")
    sp = data.specimens
    if sp is not None and "specimen" in sp.columns:
        row = sp[sp["specimen"] == spec.specimen]
        cols = [c for c in ("hyst_ms_mass", "hyst_mr_mass", "hyst_ms_moment", "hyst_mr_moment",
                            "hyst_bc", "hyst_bcr", "hyst_mr_ms", "hyst_bcr_bc", "hyst_xhf_mass",
                            "hyst_xhf") if c in row.columns and row[c].notna().any()]
        if len(row) and cols:
            r = row.iloc[0]
            lines.append("  specimens table of the contribution: " + "  ".join(
                f"{c}={r[c]:.4g}" if c not in ("hyst_bc", "hyst_bcr") else f"{c}={r[c] * 1e3:.3g} mT"
                for c in cols))
    return "\n".join(lines) + "\n"


def day_plot_values(data: MagicRockmag) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(Bcr/Bc, Mr/Ms, specimens) depuis la table specimens : ratios
    hyst_bcr_bc/hyst_mr_ms s'ils sont la, sinon recalcules depuis les
    colonnes _moment ou _mass."""
    sp = data.specimens
    if sp is None or "specimen" not in sp.columns:
        return np.array([]), np.array([]), []

    def col(name):
        return pd.to_numeric(sp[name], errors="coerce") if name in sp.columns else pd.Series(np.nan, index=sp.index)

    bcrbc = col("hyst_bcr_bc").fillna(col("hyst_bcr") / col("hyst_bc"))
    mrms = col("hyst_mr_ms").fillna(col("hyst_mr_moment") / col("hyst_ms_moment")).fillna(
        col("hyst_mr_mass") / col("hyst_ms_mass"))
    ok = bcrbc.notna() & mrms.notna() & np.isfinite(bcrbc) & np.isfinite(mrms)
    return bcrbc[ok].to_numpy(), mrms[ok].to_numpy(), list(sp.loc[ok, "specimen"])


def build_magic_day_figure(fig, data: MagicRockmag, xmax: Optional[float] = None) -> int:
    """Day plot de pmagplotlib.plot_day (figure pyplot temporaire, lignes
    recopiees dans `fig`). `xmax` : borne de l'axe Bcr/Bc (None = 6, celle
    de PmagPy) - demande explicite utilisateur ("l'echelle en X peut elle
    etre changee, si necessaire"). Retourne le nombre de specimens."""
    import matplotlib.pyplot as plt
    from pmagpy import pmagplotlib
    x, y, names = day_plot_values(data)
    fig.clf()
    ax = fig.add_subplot(111)
    num = plt.figure().number
    try:
        pmagplotlib.plot_day(num, x, y, "bs")
        src = plt.figure(num).axes[0]
        for ln in src.get_lines():
            # axhline/axvline : coordonnees axes (0..1) sur une dimension.
            if ln.get_transform() is not src.transData:
                xd, yd = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
                if np.ptp(yd) == 0:
                    ax.axhline(yd[0], color=ln.get_color(), lw=ln.get_linewidth())
                else:
                    ax.axvline(xd[0], color=ln.get_color(), lw=ln.get_linewidth())
                continue
            ax.plot(ln.get_xdata(), ln.get_ydata(), ls=ln.get_linestyle(), marker=ln.get_marker(),
                    color=ln.get_color(), lw=ln.get_linewidth(), ms=ln.get_markersize())
        for t in src.texts:
            ax.text(*t.get_position(), t.get_text(), fontsize=7, color=t.get_color())
        ax.set_xlim(0, xmax if xmax else src.get_xlim()[1])
        ax.set_ylim(src.get_ylim())
    finally:
        plt.close(num)
    off = int(np.sum(x > ax.get_xlim()[1]))
    ax.set_xlabel("Bcr/Bc", fontsize=8)
    ax.set_ylabel("Mr/Ms", fontsize=8)
    ax.set_title(f"Day plot (PmagPy) - {len(x)} specimens"
                 + (f", {off} beyond Bcr/Bc = {ax.get_xlim()[1]:g}" if off else ""), fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return len(x)
