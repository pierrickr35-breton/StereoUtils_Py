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

Table de correspondance (AGM, VSM, VFTB) - demande explicite utilisateur
("l'utilisateur pourrait aussi faire une liste nom_fichier nom_specimen et
nom_sample, avec nom_sample etant obligatoirement un sample d'un .prmag") :
colonnes file, specimen, sample [, mass_mg] [, note]. C'est elle qui fait
foi ; les regles automatiques ne servent qu'a la PRE-REMPLIR
(write_correspondence_template) : nom identique a un echantillon ->
echantillon + suffixe (le plus long) -> regle historique sans prefixe
d'annee ci-dessus -> sinon "?". Un fichier absent de la table n'est pas
exporte ; un "?" ou un echantillon absent du .prmag bloque l'export.
VFTB : le .coe (backfield sur une AUTRE poudre, autre masse) a sa propre
ligne ; specimen distinct (defaut <sample>_bf, mesures a sa masse) ou
meme specimen que la boucle (courbe remise a la masse de la boucle).
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from curie_hyst import (
    attach_vftb_backfield, compute_hysteresis, find_vftb_backfield, read_agm_file,
    read_agm_sample_list, read_vftb_file, read_vsm_csv, read_vsm_sample_list, vftb_sample_key,
    vsm_paths_for,
)
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


DEFAULT_INSTRUMENT = {
    "AGM": "Princeton MicroMag AGM 2900",
    "VSM": "Lake Shore 8600 VSM",
    "VFTB": "Petersen Instruments VFTB",
}


@dataclass
class HystItem:
    """Une mesure a exporter : boucle (+ courbe de remanence). `key` =
    valeur de la colonne "file" de la table de correspondance."""
    instrument: str                       # AGM | VSM | VFTB
    key: str
    name: str                             # nom servant au rapprochement automatique
    loop_path: str
    bf_path: Optional[str] = None
    list_mass_mg: Optional[float] = None  # masse de la liste / de l'entete VFTB
    bf_key: str = ""                      # VFTB : ligne propre du .coe
    bf_mass_mg: Optional[float] = None    # VFTB : masse de l'entete du .coe


def collect_agm_items(list_paths: List[str], report: List[str]) -> List[HystItem]:
    items = []
    for lp in list_paths:
        base_dir = os.path.dirname(lp)
        for entry in read_agm_sample_list(lp).values():
            loop_path = os.path.join(base_dir, entry.filename)
            if not os.path.exists(loop_path):
                # Liste "TU-8606A"/"tu-9503", fichiers "TU8606A"/"tu9503".
                loop_path = os.path.join(base_dir, entry.filename.replace("-", ""))
            if not os.path.exists(loop_path):
                report.append(f"{entry.filename}: loop file not found - skipped")
                continue
            items.append(HystItem("AGM", entry.filename, entry.filename, loop_path,
                                  _backfield_path(loop_path), entry.masse))
    return items


def collect_vsm_items(list_paths: List[str], report: List[str]) -> List[HystItem]:
    items = []
    for lp in list_paths:
        base_dir = os.path.dirname(lp)
        for entry in read_vsm_sample_list(lp).values():
            hyst, bf = vsm_paths_for(base_dir, entry.filename)
            if not os.path.exists(hyst):
                report.append(f"{entry.filename}: loop file \"{os.path.basename(hyst)}\" not found - skipped")
                continue
            items.append(HystItem("VSM", entry.filename, entry.filename, hyst,
                                  bf if os.path.exists(bf) else None, entry.masse))
    return items


def collect_vftb_items(folder: str, report: List[str]) -> List[HystItem]:
    names = sorted(os.listdir(folder))
    items, paired = [], set()
    for f in names:
        if not f.lower().endswith(".hys"):
            continue
        path = os.path.join(folder, f)
        coe = find_vftb_backfield(path)
        coe_mass = None
        if coe:
            paired.add(os.path.basename(coe))
            coe_mass = read_vftb_file(coe)[1]
        items.append(HystItem("VFTB", f, vftb_sample_key(path).upper(), path, coe,
                              read_vftb_file(path)[1],
                              os.path.basename(coe) if coe else "", coe_mass))
    for f in names:
        if f.lower().endswith(".coe") and f not in paired:
            report.append(f"{f}: no .hys of the same sample (or several .coe) - not exported")
    return items


def _norm(s: str) -> str:
    return re.sub(r"[-_\s]", "", s).lower()


def guess_sample(name: str, samples: Dict[str, str]) -> Tuple[Optional[str], str, str]:
    """(echantillon, suffixe, regle) - pre-remplissage de la table seulement."""
    clean = re.sub(r"[-_\s]", "", name)
    by_norm: Dict[str, List[str]] = {}
    for s in samples:
        by_norm.setdefault(_norm(s), []).append(s)
    hit = by_norm.get(clean.lower())
    if hit and len(hit) == 1:
        return hit[0], "", "same name"
    pref = sorted((k for k in by_norm if len(k) >= 4 and clean.lower().startswith(k)), key=len, reverse=True)
    if pref and len(by_norm[pref[0]]) == 1:
        return by_norm[pref[0]][0], clean[len(pref[0]):], "sample + suffix"
    sample, rest = match_sample(name, samples)
    if sample:
        return sample, rest, "sample without the 2-digit year prefix"
    return None, "", ""


def _unique(name: str, used: Set[str], tail: str) -> str:
    spec, n = f"{name}{tail}", 2
    while spec.lower() in used:
        spec, n = f"{name}_{n}{tail}", n + 1
    used.add(spec.lower())
    return spec


def write_correspondence_template(path: str, items: List[HystItem],
                                  samples: Dict[str, str]) -> Tuple[int, int]:
    """Table pre-remplie par les regles automatiques -> (lignes trouvees,
    lignes '?')."""
    used: Set[str] = set()
    lines = [
        "# Hysteresis -> MagIC correspondence table: one line per measurement file.",
        "# file = file name (AGM/VFTB) or list id (VSM); sample MUST be a sample of the .prmag;",
        "# specimen = MagIC specimen name (empty -> <sample>_hys); mass_mg optional (overrides",
        "# the list / file / weighing-table mass). Replace every '?', delete a line to leave",
        "# a file out. The note column is ignored.",
        "file\tspecimen\tsample\tmass_mg\tnote",
    ]
    n_ok = n_bad = 0
    for it in items:
        sample, rest, rule = guess_sample(it.name, samples)
        if sample is None:
            n_bad += 1 + bool(it.bf_key)
            lines.append(f"{it.key}\t?\t?\t\tno sample of the .prmag found")
            if it.bf_key:
                lines.append(f"{it.bf_key}\t?\t?\t\tbackfield of {it.key} (other powder)")
            continue
        n_ok += 1 + bool(it.bf_key)
        lines.append(f"{it.key}\t{_unique(sample + rest, used, '_hys')}\t{sample}\t\t{rule}")
        if it.bf_key:
            lines.append(f"{it.bf_key}\t{_unique(sample + rest, used, '_bf')}\t{sample}\t\t"
                         f"backfield of {it.key} (other powder, own specimen)")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return n_ok, n_bad


@dataclass
class TableRow:
    specimen: str
    sample: str
    mass_mg: Optional[float]
    line: int


def read_correspondence_table(path: str) -> Dict[str, TableRow]:
    """file -> TableRow (cle en minuscules). En-tete obligatoire avec au
    moins 'file' et 'sample' ; tabulation, ';', ',' ou espaces."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        raw = [(i, ln.rstrip("\r\n")) for i, ln in enumerate(f, 1)]
    raw = [(i, ln) for i, ln in raw if ln.strip() and not ln.lstrip().startswith("#")]
    if not raw:
        raise ValueError(f"{os.path.basename(path)}: empty correspondence table")
    head = raw[0][1]
    sep = "\t" if "\t" in head else ";" if ";" in head else "," if "," in head else None
    split = (lambda s: [c.strip() for c in s.split(sep)]) if sep else (lambda s: s.split())
    cols = [c.lower() for c in split(head)]
    if "file" not in cols or "sample" not in cols:
        raise ValueError(f"{os.path.basename(path)}: the header needs the columns file, specimen, sample "
                         f"(read: {' | '.join(split(head))})")
    ic = {k: (cols.index(k) if k in cols else None) for k in ("file", "specimen", "sample", "mass_mg")}
    table: Dict[str, TableRow] = {}
    errors = []
    for i, ln in raw[1:]:
        r = split(ln)
        get = lambda k: r[ic[k]].strip() if ic[k] is not None and ic[k] < len(r) else ""
        key = get("file")
        if not key:
            continue
        mass = None
        if get("mass_mg"):
            try:
                mass = float(get("mass_mg").replace(",", "."))
            except ValueError:
                errors.append(f"line {i}: mass_mg '{get('mass_mg')}' is not a number")
        if key.lower() in table:
            errors.append(f"line {i}: {key} already on line {table[key.lower()].line}")
        table[key.lower()] = TableRow(get("specimen"), get("sample"), mass, i)
    if errors:
        raise ValueError("correspondence table: " + "; ".join(errors))
    return table


def read_prmag_specimens(path: str) -> Set[str]:
    out = set()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("specimen:"):
                out.add(line.split("\t", 1)[0].split(":", 1)[1].strip().lower())
    return out


def _resolve(items: List[HystItem], table: Dict[str, TableRow], samples: Dict[str, str],
             report: List[str]) -> Dict[str, Tuple[str, str, Optional[float]]]:
    """key (minuscules) -> (specimen, sample, masse de la table) ; leve
    ValueError (rien n'est ecrit) au moindre probleme."""
    errors = []
    sample_ci = {s.lower(): s for s in samples}
    keys = {it.key.lower() for it in items} | {it.bf_key.lower() for it in items if it.bf_key}
    for k, row in table.items():
        if k not in keys:
            report.append(f"table line {row.line}: '{k}' matches no file of the selection - ignored")
    out: Dict[str, Tuple[str, str, Optional[float]]] = {}
    spec_sample: Dict[str, Tuple[str, int]] = {}
    for it in items:
        for key, tail in ((it.key, "_hys"), (it.bf_key, "_bf")):
            if not key:
                continue
            row = table.get(key.lower())
            if row is None:
                if key == it.key:
                    report.append(f"{key}: not in the correspondence table - not exported")
                else:
                    errors.append(f"{key} (backfield of {it.key}) is missing from the table: "
                                  "give it a line (own specimen, or the specimen of the loop)")
                continue
            if not row.sample or row.sample == "?":
                errors.append(f"line {row.line} ({key}): sample not given")
                continue
            sample = sample_ci.get(row.sample.lower())
            if sample is None:
                errors.append(f"line {row.line} ({key}): sample '{row.sample}' is not in the .prmag")
                continue
            spec = row.specimen if row.specimen and row.specimen != "?" else f"{sample}{tail}"
            prev = spec_sample.get(spec.lower())
            if prev and prev[0] != sample:
                errors.append(f"line {row.line}: specimen {spec} already belongs to sample {prev[0]} "
                              f"(line {prev[1]})")
            spec_sample.setdefault(spec.lower(), (sample, row.line))
            out[key.lower()] = (spec, sample, row.mass_mg)
    # Deux boucles (ou deux backfields) sur un meme specimen = ambigu.
    seen: Dict[str, str] = {}
    for it in items:
        for key in (it.key, it.bf_key):
            if key and key.lower() in out:
                kind = "loop" if key == it.key else "backfield"
                tag = out[key.lower()][0].lower() + "|" + kind
                if tag in seen:
                    errors.append(f"{key} and {seen[tag]}: two {kind}s on specimen {out[key.lower()][0]}")
                seen[tag] = key
    if errors:
        raise ValueError("\n".join(errors))
    return out


def _measurement_rows(common: Dict[str, str], specimen: str, curve, method: str,
                      mass_kg: Optional[float]) -> List[Dict[str, str]]:
    exp = f"{specimen}-{method}"
    rows = []
    for i, (b, m) in enumerate(zip(curve.field, curve.moment), start=1):
        rows.append({
            **common, "specimen": specimen, "measurement": f"{exp}-{i}", "experiment": exp,
            "treat_step_num": str(i),
            "treat_dc_field": _g(b) if method == "LP-IRM-DCD" else "0",
            "meas_field_dc": "0" if method == "LP-IRM-DCD" else _g(b),
            "magn_moment": _g(m), "magn_mass": _g(m / mass_kg) if mass_kg else "",
            "method_codes": method,
        })
    return rows


def build_hyst_magic(
    items: List[HystItem],
    prmag_paths: List[str],
    table: Dict[str, TableRow],
    mass_table: Optional[Dict[str, Optional[float]]] = None,
    valsat_frac: float = 0.7,
    instrument: str = "",
    analysts: str = "",
    citations: str = "This study",
    report: Optional[List[str]] = None,
) -> HystMagicExport:
    out = HystMagicExport(report=list(report or []))
    samples: Dict[str, str] = {}
    prmag_specs: Set[str] = set()
    for p in prmag_paths:
        samples.update(read_prmag_samples(p))
        prmag_specs |= read_prmag_specimens(p)
    resolved = _resolve(items, table, samples, out.report)

    for it in items:
        if it.key.lower() not in resolved:
            continue
        specimen, sample, t_mass = resolved[it.key.lower()]
        name = it.key
        instr = instrument or DEFAULT_INSTRUMENT[it.instrument]
        common = {"quality": "g", "standard": "u",
                  "instrument_codes": instr, "analysts": analysts, "citations": citations}
        bf_spec = bf_mass_kg = None
        bf_raw = None
        coe_mass = None
        if it.instrument == "VFTB":
            _n, h_mass, loops = read_vftb_file(it.loop_path)
            mass_mg = t_mass or h_mass
            if not loops or not mass_mg:
                out.report.append(f"{name}: no data block or no mass - skipped")
                continue
            loop = loops[0]
            # "mag" en emu/g : moment brut = mag x masse (voir read_vftb_file)
            loop.moment = loop.moment * (mass_mg / h_mass if h_mass else mass_mg * 1.0e-6)
            backfield = loops[1] if len(loops) > 1 else None
            if it.bf_path and backfield is None:
                bf_spec, _s, bt_mass = resolved[it.bf_key.lower()]
                _n, c_mass, cl = read_vftb_file(it.bf_path)
                coe_mass = bt_mass or c_mass
                if cl and len(cl[0].field) and coe_mass:
                    bf_raw = cl[0]
                    bf_raw.moment = bf_raw.moment * (coe_mass / c_mass if c_mass else coe_mass * 1.0e-6)
                    backfield = type(bf_raw)(bf_raw.sample, bf_raw.path, bf_raw.field,
                                             bf_raw.moment * (mass_mg / coe_mass))
                    bf_mass_kg = coe_mass * 1.0e-6
                else:
                    out.report.append(f"{it.bf_key}: unreadable or no mass - backfield not used")
                    bf_spec = None
        else:
            loop = read_agm_file(it.loop_path) if it.instrument == "AGM" else read_vsm_csv(it.loop_path)
            backfield = None
            if it.bf_path:
                backfield = read_agm_file(it.bf_path) if it.instrument == "AGM" else read_vsm_csv(it.bf_path)
            mass_mg = t_mass
            if mass_mg is None and it.instrument == "AGM":
                mass_mg = lookup_mass(os.path.basename(it.loop_path), mass_table or {})
            if mass_mg is None and it.list_mass_mg and it.list_mass_mg > 1.0:
                mass_mg = it.list_mass_mg

        # Masse inconnue : 1 kg fictif -> grandeurs "normalisees" = moments (Am2).
        res = compute_hysteresis(loop, backfield, mass_mg if mass_mg else 1.0e6, valsat_frac)
        if res is None:
            out.report.append(f"{name}: hysteresis parameters could not be computed - skipped")
            continue
        if bf_raw is not None:
            attach_vftb_backfield(res, it.bf_path, coe_mass)
        same_spec = bf_spec is not None and bf_spec.lower() == specimen.lower()
        ok = _physical(res)
        mass_kg = mass_mg * 1.0e-6 if mass_mg else None

        out.measurements += _measurement_rows(common, specimen, loop, "LP-HYS", mass_kg)
        if backfield is not None and (bf_raw is None or same_spec):
            # AGM/VSM, ou .coe range sur le specimen de la boucle (remis a sa masse)
            out.measurements += _measurement_rows(common, specimen, backfield, "LP-IRM-DCD", mass_kg)
        elif bf_raw is not None:
            out.measurements += _measurement_rows(common, bf_spec, bf_raw, "LP-IRM-DCD", bf_mass_kg)

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
        if it.instrument == "VFTB":
            src = f"VFTB loop {name}"
            if bf_raw is not None and same_spec:
                src += (f" + backfield {it.bf_key} (other powder, {coe_mass:g} mg; curve rescaled "
                        f"to {mass_mg:g} mg)")
            elif bf_raw is not None:
                src += f"; Bcr from the backfield of specimen {bf_spec} ({it.bf_key}, other powder)"
            if bf_raw is not None:
                src += "; Mr from the loop"
        else:
            src = (f"{it.instrument} loop {name}"
                   + (f" + backfield {os.path.basename(it.bf_path)}" if it.bf_path else ""))
        row = {
            "instrument_codes": instr, "analysts": analysts, "citations": citations,
            "specimen": specimen, "sample": sample,
            "method_codes": "LP-HYS" + (":LP-IRM-DCD" if backfield is not None and
                                        (bf_raw is None or same_spec) else ""),
            "weight": _g(mass_kg) if mass_kg else "",
            **params,
            "description": (
                src + f"; paramagnetic slope fitted above {valsat_frac:g} x max field"
                + (f"; mass {mass_mg:g} mg" if mass_mg else "; mass unknown (moments)")
                + ("" if ok else "; automatic parameters unphysical (Bc <= 0 or Bcr < Bc), "
                   "not reported - curves only")
            ),
        }
        out.specimens.append(row)
        if bf_raw is not None and not same_spec:
            out.specimens.append({
                "instrument_codes": instr, "analysts": analysts, "citations": citations,
                "specimen": bf_spec, "sample": sample, "method_codes": "LP-IRM-DCD",
                "weight": _g(bf_mass_kg),
                "hyst_bcr": _g(res.hcr_mt * 1.0e-3) if ok else "",
                "description": (f"VFTB backfield {it.bf_key}, other powder than the loop "
                                f"(specimen {specimen}); mass {coe_mass:g} mg"),
            })
        for s in {specimen, bf_spec} - {None}:
            if s.lower() in prmag_specs:
                out.report.append(f"{name}: specimen {s} is also a paleomagnetic specimen of the "
                                  ".prmag - check that this is intended")
        if not ok:
            out.report.append(
                f"{name}: unphysical parameters (Bc {res.hc_mt:.3g} mT, Bcr {res.hcr_mt:.3g} mT) "
                f"- curves exported, parameters not written")
        if ok and not mass_mg:
            out.report.append(f"{name}: no mass - parameters written as moments (Am2)")
    return out


def export_hyst_magic(items: List[HystItem], out_path: str, prmag_paths: List[str],
                      table: Dict[str, TableRow],
                      mass_table: Optional[Dict[str, Optional[float]]] = None,
                      valsat_frac: float = 0.7, instrument: str = "", analysts: str = "",
                      citations: str = "This study",
                      report: Optional[List[str]] = None) -> HystMagicExport:
    result = build_hyst_magic(items, prmag_paths, table, mass_table, valsat_frac,
                              instrument, analysts, citations, report)
    write_magic_combined(out_path, [
        ("specimens", SPECIMENS_HEADER, result.specimens),
        ("measurements", MEASUREMENTS_HEADER, result.measurements),
    ])
    return result
