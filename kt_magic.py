"""
Export MagIC des courbes K-T (susceptibilite en fonction de la
temperature) - demande explicite utilisateur ("est-ce compliqué d'ajouter
des données K-T et Hysteresis dans les fichiers Magic?", puis "le nom du
sample est donné dans le nom du fichier. Le plus simple est alors de
donner le nom du fichier .cur comme nom du spécimen. la normalisation se
faisant par la susceptibilité (SI) de l'échantillon de paleomag").

Meme pipeline, entree par entree, que K-T > Select File from List...
(app._kt_process_list_entry) : lecture du .CUR/.CLW, correction four
(constante `zerodia` ou vase vide complet si 999), normalisation (volume
si `suscep` != 0 - susceptibilite SI de l'echantillon paleomag -, sinon
masse), point de Curie par la derivee seconde (curie_point_second_
derivative). Rien n'est recalcule differemment ici : seule l'ecriture
MagIC est nouvelle.

Sortie : UN fichier MagIC combine (tables specimens + measurements,
separees par ">>>>>>>>>>", meme format qu'un telechargement MagIC) -
fusionnable dans l'export MagIC de STARpaleomag_Py ou televersable tel
quel en complement d'une contribution existante.
- specimen = nom du fichier sans extension + "_cur" (ex. 11TU5906_cur) -
  demande explicite utilisateur ("pour le specimen ajouter _cur sinon le
  specimen = sample") ; sample = nom du fichier, ou ce nom sans sa
  derniere lettre s'il n'existe qu'ainsi dans le .prmag (13CB0503C ->
  13CB0503, 11TU2805B -> 11TU2805) (= nom d'echantillon des .prmag Rennes, specimens 11TU5906A/B) ;
  le .prmag optionnel sert a verifier que l'echantillon existe et a
  restreindre l'export a UNE etude (la meme liste K-T melange Beagle et
  Terre de Feu) - les entrees absentes du .prmag sont ecartees et listees.
- measurements : un point par ligne, method_codes LP-X-T, meas_temp en K,
  susc_chi_volume (SI) ou susc_chi_mass (m3/kg) ; description "heating"/
  "cooling" (separation au maximum de temperature, meme convention que
  curie_point_second_derivative).
- specimens : critical_temp (K) / critical_temp_type "Curie" UNIQUEMENT
  a partir de temperatures validees par l'utilisateur (`curie_temps`,
  fichier "specimen Tc_degC") - le maximum de la derivee seconde n'est
  PAS ecrit comme critical_temp : verifie sur les 60 courbes de
  Patagonie, il tombe parfois sur une autre inflexion (11TU6102 163.6,
  12TU9002 33.7, 12TU7421 124.1 degC) ; il reste seulement dans
  `description`, explicitement qualifie d'estimation automatique, avec
  les corrections appliquees.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from curie_kt import (
    KTListEntry, apply_furnace_correction, apply_vessel_correction,
    curie_point_second_derivative, minmax_temperature, normalize_by_mass,
    normalize_by_volume, read_cur_file, read_kt_sample_list,
)

_KELVIN = 273.15

MEASUREMENTS_HEADER = [
    "measurement", "experiment", "specimen", "quality", "standard", "treat_step_num",
    "meas_temp", "susc_chi_volume", "susc_chi_mass", "method_codes",
    "instrument_codes", "analysts", "citations", "description",
]
SPECIMENS_HEADER = [
    "specimen", "sample", "method_codes", "critical_temp", "critical_temp_type",
    "instrument_codes", "analysts", "citations", "description",
]


@dataclass
class KTMagicExport:
    specimens: List[Dict[str, str]] = field(default_factory=list)
    measurements: List[Dict[str, str]] = field(default_factory=list)
    report: List[str] = field(default_factory=list)


def read_prmag_samples(path: str) -> Dict[str, str]:
    """sample -> site d'un .prmag (lignes "specimen: ...\tsample: ...\t
    site: ..."). Lecture minimale volontaire : StereoUtils_Py ne depend pas
    de STARpaleomag_Py (testlect)."""
    samples: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("specimen:"):
                continue
            kv = dict(
                (m.group(1), m.group(2).strip())
                for m in re.finditer(r"(\w+):\s*([^\t]*)", line)
            )
            sample = kv.get("sample", "")
            if sample and sample != "n.d":
                samples.setdefault(sample, kv.get("site", ""))
    return samples


def read_curie_temps(path: str) -> Dict[str, float]:
    """Fichier "specimen Tc_degC" (espaces/tabulations, '#' = commentaire)
    - temperatures de Curie retenues par l'utilisateur."""
    temps: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split("#", 1)[0].split()
            if len(parts) >= 2:
                try:
                    name = os.path.splitext(parts[0])[0]
                    temps[name if name.endswith("_cur") else f"{name}_cur"] = float(parts[1])
                except ValueError:
                    continue
    return temps


def _fmt(x: float, sig: int = 6) -> str:
    return f"{x:.{sig}g}"


def _correct_and_normalize(entry: KTListEntry, base_dir: str):
    """Meme enchainement que app._kt_process_list_entry. Retourne
    (courbe, texte_correction) ou (None, message_erreur)."""
    path = os.path.join(base_dir, entry.filename)
    if not os.path.exists(path):
        return None, "file not found"
    curve = read_cur_file(path)
    if entry.zerodia == 999.0:
        if not entry.empty_vessel:
            return None, "code 999 but no empty-vessel file in the list"
        vessel_path = os.path.join(base_dir, entry.empty_vessel)
        if not os.path.exists(vessel_path):
            return None, f"empty-vessel file not found: {entry.empty_vessel}"
        curve = apply_vessel_correction(curve, read_cur_file(vessel_path))
        corr = f"furnace correction: empty-vessel curve {entry.empty_vessel}"
    else:
        curve = apply_furnace_correction(curve, entry.zerodia)
        corr = f"furnace correction: {entry.zerodia:g} subtracted"
    if entry.suscep:
        is_cur = os.path.splitext(path)[1].upper() == ".CUR"
        curve = normalize_by_volume(curve, entry.suscep, is_cur=is_cur)
        anchor = "first" if is_cur else "last"
        corr += f"; normalized to the bulk susceptibility {entry.suscep:g} SI ({anchor} point)"
    elif entry.masse:
        curve = normalize_by_mass(curve, entry.masse)
        corr += f"; normalized by mass ({entry.masse:g} mg)"
    if curve.normalized is None:
        return None, "no susceptibility nor mass in the list (curve cannot be normalized)"
    return curve, corr


def build_kt_magic(
    list_path: str,
    prmag_path: Optional[str] = None,
    instrument: str = "AGICO KLY-3 CS-3",
    analysts: str = "",
    citations: str = "This study",
    curie_temps: Optional[Dict[str, float]] = None,
) -> KTMagicExport:
    out = KTMagicExport()
    entries = read_kt_sample_list(list_path)
    base_dir = os.path.dirname(list_path)
    prmag_samples = read_prmag_samples(prmag_path) if prmag_path else None

    for entry in entries.values():
        stem = os.path.splitext(entry.filename)[0]
        specimen = f"{stem}_cur"
        sample = stem
        if prmag_samples is not None:
            # Nom de fichier avec lettre de specimen ou de reprise
            # (13CB0503C, 11TU2805B, 12TU8007b) : echantillon sans la lettre.
            if sample not in prmag_samples and sample[-1:].isalpha() and sample[:-1] in prmag_samples:
                sample = sample[:-1]
            if sample not in prmag_samples:
                out.report.append(f"{entry.filename}: sample {stem} not in the .prmag - skipped")
                continue
        curve, corr = _correct_and_normalize(entry, base_dir)
        if curve is None:
            out.report.append(f"{entry.filename}: {corr} - skipped")
            continue

        t_heat, t_cool = curie_point_second_derivative(curve)
        _imin, itmax = minmax_temperature(curve)
        experiment = f"{specimen}-LP-X-T"
        column = "susc_chi_volume" if curve.normalized == "volume" else "susc_chi_mass"
        # normalize_by_mass : 1e-6 m3/kg (voir curie_kt.normalize_by_mass).
        scale = 1.0 if curve.normalized == "volume" else 1.0e-6
        for i, (t, k) in enumerate(zip(curve.temp, curve.tsusc), start=1):
            out.measurements.append({
                "measurement": f"{experiment}-{i}",
                "experiment": experiment,
                "specimen": specimen,
                "quality": "g",
                "standard": "u",
                "treat_step_num": str(i),
                "meas_temp": f"{t + _KELVIN:.2f}",
                column: _fmt(k * scale),
                "method_codes": "LP-X-T",
                "instrument_codes": instrument,
                "analysts": analysts,
                "citations": citations,
                "description": "heating" if i - 1 <= itmax else "cooling",
            })

        heat = f"{t_heat:.1f}" if t_heat is not None else "n/a"
        cool = f"{t_cool:.1f}" if t_cool is not None else "n/a"
        tc = (curie_temps or {}).get(specimen)
        out.specimens.append({
            "specimen": specimen,
            "sample": sample,
            "method_codes": "LP-X-T",
            "critical_temp": f"{tc + _KELVIN:.1f}" if tc is not None else "",
            "critical_temp_type": "Curie" if tc is not None else "",
            "instrument_codes": instrument,
            "analysts": analysts,
            "citations": citations,
            "description": (
                f"K-T curve, Tmax {float(np.max(curve.temp)):.0f} degC; {corr}; "
                f"automatic estimate of the Curie temperature (maximum of the 2nd "
                f"derivative, not validated): heating {heat} degC, cooling {cool} degC"
            ),
        })
    return out


def write_magic_combined(path: str, tables: List[Tuple[str, List[str], List[Dict[str, str]]]]) -> None:
    """Fichier MagIC combine (meme format qu'un telechargement MagIC) -
    colonnes vides sur toutes les lignes retirees."""
    blocks = []
    for name, header, rows in tables:
        cols = [c for c in header if any(r.get(c, "") for r in rows)]
        lines = [f"tab delimited\t{name}", "\t".join(cols)]
        lines += ["\t".join(r.get(c, "") for c in cols) for r in rows]
        blocks.append("\n".join(lines))
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n>>>>>>>>>>\n".join(blocks) + "\n")


def export_kt_magic(list_path: str, out_path: str, prmag_path: Optional[str] = None,
                    instrument: str = "AGICO KLY-3 CS-3", analysts: str = "",
                    citations: str = "This study",
                    curie_temps: Optional[Dict[str, float]] = None) -> KTMagicExport:
    result = build_kt_magic(list_path, prmag_path, instrument, analysts, citations, curie_temps)
    write_magic_combined(out_path, [
        ("specimens", SPECIMENS_HEADER, result.specimens),
        ("measurements", MEASUREMENTS_HEADER, result.measurements),
    ])
    return result
