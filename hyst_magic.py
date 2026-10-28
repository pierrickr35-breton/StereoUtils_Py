"""
Export MagIC des cycles d'hysteresis AGM - demande explicite utilisateur
("peut on faire la même chose pour les hysteresis?", apres l'export K-T de
kt_magic.py). Meme pipeline que Hysteresis > Select File from List...
(app._hyst_process_entry, format AGM) : boucle <id> + courbe de remanence
<id>-r, parametres par compute_hysteresis (correction paramagnetique au-dela
de `valsat_frac`). Seule l'ecriture MagIC est nouvelle.

Conventions MagIC reprises de PmagPy (convert_2_magic.agm) :
- boucle : LP-HYS, champ dans meas_field_dc (T), moment dans magn_moment
  (Am2, brut) - magn_mass (Am2/kg) en plus quand la masse est connue ;
- courbe de remanence : LP-IRM-DCD, champ dans treat_dc_field (T),
  meas_field_dc = 0.
- specimens : hyst_ms/hyst_mr/hyst_xhf en _moment toujours, et aussi
  en _mass si masse connue (dayplot_magic ne lit que _moment) ; hyst_bc,
  hyst_bcr (T), hyst_mr_ms, hyst_bcr_bc.

Masse : la liste AGM (Liste_Fernando.txt) porte 1 partout, valeur
provisoire - les vraies masses sont dans une table de pesees ("nom tare
total net", 00_HistFPG.txt, net = total - tare en mg, verifie sur toutes
les lignes sauf deux fautes de frappe dans le TOTAL, le net restant
coherent) passee via `mass_table`. Une masse <= 1 mg sans pesee est
consideree inconnue : grandeurs ecrites en moment (Am2), sans _mass.

Noms : les fichiers AGM n'ont pas le prefixe d'annee des echantillons
(cb0105 -> 13CB0105, tu7421b -> 12TU7421 + "b", tf0207 -> 97TF0207) : un
fichier est rattache a l'UNIQUE echantillon du .prmag "<2 chiffres><2
lettres><4 chiffres>" de memes lettres/chiffres ; specimen = echantillon +
reste du nom de fichier + "_hys" (13CB0105_hys, 12TU7421b_hys) - meme
logique que "_cur" pour le K-T.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from curie_hyst import compute_hysteresis, read_agm_file, read_agm_sample_list
from kt_magic import read_prmag_samples, write_magic_combined

MEASUREMENTS_HEADER = [
    "measurement", "experiment", "specimen", "quality", "standard", "treat_step_num",
    "treat_dc_field", "meas_field_dc", "magn_moment", "magn_mass", "method_codes",
    "instrument_codes", "analysts", "citations", "description",
]
SPECIMENS_HEADER = [
    "specimen", "sample", "method_codes", "weight", "hyst_ms_mass", "hyst_mr_mass",
    "hyst_ms_moment", "hyst_mr_moment", "hyst_mr_ms", "hyst_bc", "hyst_bcr",
    "hyst_bcr_bc", "hyst_xhf_mass", "hyst_xhf", "instrument_codes", "analysts",
    "citations", "description",
]

_NAME_RE = re.compile(r"^([A-Za-z]{2})-?(\d{4})(.*)$")


@dataclass
class HystMagicExport:
    specimens: List[Dict[str, str]] = field(default_factory=list)
    measurements: List[Dict[str, str]] = field(default_factory=list)
    report: List[str] = field(default_factory=list)


def read_mass_table(path: str) -> Dict[str, Optional[float]]:
    """Table de pesees "nom ... masse_mg" (derniere colonne = masse nette,
    ex. 00_HistFPG.txt "cb0105 1.6 27.9 26.3"). Un nom pese deux fois avec
    des masses differentes (tu7414 : 21.5 et 34.4 mg) -> None (ambigu) ;
    une masse <= 0 (tu6720 : 0, non pese) -> None."""
    masses: Dict[str, Optional[float]] = {}
    with open(path, "r", encoding="iso-8859-1", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                m = float(parts[-1])
            except ValueError:
                continue
            k = parts[0].lower()
            m = m if m > 0 else None
            masses[k] = None if k in masses and masses[k] != m else m
    return masses


def lookup_mass(name: str, masses: Dict[str, Optional[float]]) -> Optional[float]:
    """Masse du nom EXACT (casse ignoree, tiret conserve : tu-7414 et
    tu7414 sont deux mesures distinctes) - pas de rapprochement
    approximatif : tu6902 aurait pris la masse de tu6902b, un autre
    morceau. Les noms a corriger dans la table sont signales."""
    return masses.get(name.lower())


def _physical(res) -> bool:
    """Bc > 0 et, si la courbe de remanence existe, Bcr >= Bc - sinon le
    calcul automatique a echoue (echantillon faible/paramagnetique :
    11TU5506 Bc < 0, 11TU6008 Bc = 0.55 T > Bcr)."""
    if not np.isfinite(res.hc_mt) or res.hc_mt <= 0:
        return False
    return not (np.isfinite(res.hcr_mt) and res.hcr_mt < res.hc_mt)


def match_sample(name: str, samples: Dict[str, str]) -> Tuple[Optional[str], str]:
    """(echantillon du .prmag, reste du nom) - None si aucun ou plusieurs
    echantillons correspondent."""
    m = _NAME_RE.match(name)
    if not m:
        return None, ""
    letters, digits, rest = m.group(1).upper(), m.group(2), m.group(3)
    hits = [s for s in samples if re.fullmatch(rf"\d{{2}}{letters}{digits}", s.upper())]
    return (hits[0], rest) if len(hits) == 1 else (None, rest)


def _backfield_path(loop_path: str) -> Optional[str]:
    # cb1807 n'a qu'un "-r2" (remesure).
    for suffix in ("-r", "-r2", "-r1"):
        if os.path.exists(loop_path + suffix):
            return loop_path + suffix
    return None


def _g(x: float, sig: int = 6) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:.{sig}g}"


def build_hyst_magic(
    list_paths: Union[str, List[str]],
    prmag_paths: List[str],
    mass_table: Optional[Dict[str, Optional[float]]] = None,
    valsat_frac: float = 0.7,
    instrument: str = "Princeton MicroMag AGM 2900",
    analysts: str = "",
    citations: str = "This study",
) -> HystMagicExport:
    out = HystMagicExport()
    if isinstance(list_paths, str):
        list_paths = [list_paths]
    used: set = set()
    samples: Dict[str, str] = {}
    for p in prmag_paths:
        samples.update(read_prmag_samples(p))

    entries = [(os.path.dirname(lp), e) for lp in list_paths
               for e in read_agm_sample_list(lp).values()]
    for base_dir, entry in entries:
        name = entry.filename
        sample, rest = match_sample(name, samples)
        if sample is None:
            out.report.append(f"{name}: no single matching sample in the .prmag - skipped")
            continue
        loop_path = os.path.join(base_dir, name)
        if not os.path.exists(loop_path):
            # Liste "TU-8606A"/"tu-9503", fichiers "TU8606A"/"tu9503".
            loop_path = os.path.join(base_dir, name.replace("-", ""))
        if not os.path.exists(loop_path):
            out.report.append(f"{name}: loop file not found - skipped")
            continue
        bf_path = _backfield_path(loop_path)
        loop = read_agm_file(loop_path)
        backfield = read_agm_file(bf_path) if bf_path else None

        mass_mg = lookup_mass(os.path.basename(loop_path), mass_table or {})
        if mass_mg is None and entry.masse > 1.0:
            mass_mg = entry.masse
        # Masse inconnue : 1 kg fictif -> grandeurs "normalisees" = moments (Am2).
        res = compute_hysteresis(loop, backfield, mass_mg if mass_mg else 1.0e6, valsat_frac)
        if res is None:
            out.report.append(f"{name}: hysteresis parameters could not be computed - skipped")
            continue

        # tu-7414 et tu7414 : deux mesures distinctes du meme echantillon.
        base = f"{sample}{rest}"
        specimen, n = f"{base}_hys", 2
        while specimen in used:
            specimen, n = f"{base}_{n}_hys", n + 1
        used.add(specimen)
        ok = _physical(res)
        mass_kg = mass_mg * 1.0e-6 if mass_mg else None
        common = {"specimen": specimen, "quality": "g", "standard": "u",
                  "instrument_codes": instrument, "analysts": analysts, "citations": citations}

        exp = f"{specimen}-LP-HYS"
        for i, (b, m) in enumerate(zip(loop.field, loop.moment), start=1):
            out.measurements.append({
                **common, "measurement": f"{exp}-{i}", "experiment": exp, "treat_step_num": str(i),
                "treat_dc_field": "0", "meas_field_dc": _g(b),
                "magn_moment": _g(m), "magn_mass": _g(m / mass_kg) if mass_kg else "",
                "method_codes": "LP-HYS",
            })
        if backfield is not None:
            exp = f"{specimen}-LP-IRM-DCD"
            for i, (b, m) in enumerate(zip(backfield.field, backfield.moment), start=1):
                out.measurements.append({
                    **common, "measurement": f"{exp}-{i}", "experiment": exp, "treat_step_num": str(i),
                    "treat_dc_field": _g(b), "meas_field_dc": "0",
                    "magn_moment": _g(m), "magn_mass": _g(m / mass_kg) if mass_kg else "",
                    "method_codes": "LP-IRM-DCD",
                })

        # susc_para_si = pente (Am2/kg/T, ou Am2/T si masse inconnue) x mu0.
        # Masse connue : moments ecrits aussi (x masse) - ipmag.dayplot_magic
        # ne lit que hyst_mr_moment/hyst_ms_moment.
        k = mass_kg or 1.0
        params = {
            "hyst_ms_moment": _g(res.js_ferro * k), "hyst_mr_moment": _g(res.jrs * k),
            "hyst_xhf": _g(res.susc_para_si * k),
            "hyst_mr_ms": _g(res.jrs_over_jsferro), "hyst_bc": _g(res.hc_mt * 1.0e-3),
            "hyst_bcr": _g(res.hcr_mt * 1.0e-3), "hyst_bcr_bc": _g(res.hcr_over_hc),
        } if ok else {}
        if ok and mass_kg:
            params.update({"hyst_ms_mass": _g(res.js_ferro), "hyst_mr_mass": _g(res.jrs),
                           "hyst_xhf_mass": _g(res.susc_para_si)})
        row = {
            **common, "sample": sample,
            "method_codes": "LP-HYS" + (":LP-IRM-DCD" if backfield is not None else ""),
            "weight": _g(mass_kg) if mass_kg else "",
            **params,
            "description": (
                f"AGM loop {name}" + (f" + backfield {os.path.basename(bf_path)}" if bf_path else "")
                + f"; paramagnetic slope fitted above {valsat_frac:g} x max field"
                + (f"; mass {mass_mg:g} mg" if mass_mg else "; mass unknown (moments)")
                + ("" if ok else "; automatic parameters unphysical (Bc <= 0 or Bcr < Bc), "
                   "not reported - curves only")
            ),
        }
        for k in ("quality", "standard"):
            row.pop(k)
        out.specimens.append(row)
        if specimen != f"{base}_hys":
            out.report.append(f"{name}: specimen name {base}_hys already used -> {specimen}")
        if not ok:
            out.report.append(
                f"{name}: unphysical parameters (Bc {res.hc_mt:.3g} mT, Bcr {res.hcr_mt:.3g} mT) "
                f"- curves exported, parameters not written")
        if ok and not mass_mg:
            out.report.append(f"{name}: no mass - parameters written as moments (Am2)")
    return out


def export_hyst_magic(list_paths: Union[str, List[str]], out_path: str, prmag_paths: List[str],
                      mass_table: Optional[Dict[str, float]] = None, valsat_frac: float = 0.7,
                      instrument: str = "Princeton MicroMag AGM 2900", analysts: str = "",
                      citations: str = "This study") -> HystMagicExport:
    result = build_hyst_magic(list_paths, prmag_paths, mass_table, valsat_frac,
                              instrument, analysts, citations)
    write_magic_combined(out_path, [
        ("specimens", SPECIMENS_HEADER, result.specimens),
        ("measurements", MEASUREMENTS_HEADER, result.measurements),
    ])
    return result
