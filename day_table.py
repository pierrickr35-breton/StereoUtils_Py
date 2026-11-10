"""
Lecture d'un tableau de parametres d'hysteresis (texte, une ligne par
echantillon) pour le Day plot seul - demande explicite utilisateur ("un
menu Hysteresis pour juste le Day Plot", exemple parametros_DayPlot.txt :
Codigo_muestra, Lito, Js_corregida_Am2/kg, Mr_Am2/kg, Hc_mT, Hcr_mT,
Mr/Js, Hcr/Hc - tabulations, BOM UTF-8, fins de ligne CRLF).

Colonnes reconnues par leur en-tete (casse et unites ignorees) :
  - rapports : Mr/Ms (ou Mr/Js, Mrs/Ms) et Bcr/Bc (ou Hcr/Hc) ;
  - sinon parametres : Mr, Ms (ou Js), Bc (ou Hc), Bcr (ou Hcr), les
    rapports etant alors recalcules (memes unites dans chaque paire).
Premiere colonne = nom de l'echantillon. Colonne de groupe (symboles
differents) : premiere autre colonne non numerique, p.ex. la lithologie.
Separateur : tabulation, point-virgule, virgule ou espaces ; virgule
decimale acceptee quand le separateur n'est pas la virgule.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class DayTable:
    names: List[str]
    bcr_bc: np.ndarray
    mr_ms: np.ndarray
    groups: Optional[List[str]] = None
    group_column: str = ""
    source: str = ""                      # comment les rapports ont ete obtenus
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _key(header: str) -> str:
    """En-tete normalise : minuscules, sans unites ni separateurs
    ('Js_corregida_Am2/kg' -> 'js', 'Mr/Js' -> 'mr/js', 'Hcr_mT' -> 'hcr')."""
    h = header.strip().lower()
    h = re.sub(r"\(.*?\)|\[.*?\]", "", h)
    if "/" in h:
        parts = [re.split(r"[_\s.\-]", p.strip())[0] for p in h.split("/")]
        if len(parts) == 2 and all(re.fullmatch(r"[a-z]+", p or "") for p in parts):
            # 'mr_am2/kg' n'est pas un rapport : 'kg' est une unite
            if parts[1] not in ("kg", "g", "m3", "m", "cm3", "cc"):
                return "/".join(parts)
    return re.split(r"[_\s.\-/]", h)[0]


RATIO_MRMS = ("mr/ms", "mr/js", "mrs/ms", "mrs/js")
RATIO_BCRBC = ("bcr/bc", "hcr/hc")
COL_MR = ("mr", "mrs")
COL_MS = ("ms", "js")
COL_BC = ("bc", "hc")
COL_BCR = ("bcr", "hcr")


def _num(s: str, decimal_comma: bool) -> Optional[float]:
    s = s.strip()
    if decimal_comma:
        s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def read_day_table(path: str) -> DayTable:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = [ln.rstrip("\r\n") for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:
        raise ValueError("the file needs a header line and at least one data line")
    head = lines[0]
    sep = "\t" if "\t" in head else ";" if ";" in head else "," if "," in head else None
    split = (lambda s: s.split(sep)) if sep else (lambda s: s.split())
    headers = [h.strip() for h in split(head)]
    keys = [_key(h) for h in headers]
    rows = [split(ln) for ln in lines[1:]]
    decimal_comma = sep != "," and any(re.fullmatch(r"-?\d+,\d+", c.strip()) for r in rows for c in r)

    def find(cands):
        for c in cands:
            if c in keys:
                return keys.index(c)
        return None

    i_mrms, i_bb = find(RATIO_MRMS), find(RATIO_BCRBC)
    i_mr, i_ms, i_bc, i_bcr = find(COL_MR), find(COL_MS), find(COL_BC), find(COL_BCR)
    have_params = None not in (i_mr, i_ms, i_bc, i_bcr)
    if (i_mrms is None or i_bb is None) and not have_params:
        raise ValueError(
            "columns not found: the header needs Mr/Ms (or Mr/Js) and Bcr/Bc (or Hcr/Hc), "
            "or Mr, Ms (or Js), Bc (or Hc) and Bcr (or Hcr).\nHeader read: " + " | ".join(headers))

    numeric = [all(_num(r[k], decimal_comma) is not None for r in rows if k < len(r) and r[k].strip())
               for k in range(len(headers))]
    i_group = next((k for k in range(1, len(headers)) if not numeric[k]
                    and any(k < len(r) and r[k].strip() for r in rows)), None)

    names, x, y, groups, skipped = [], [], [], [], []
    n_mismatch = []
    for r in rows:
        r = r + [""] * (len(headers) - len(r))
        name = r[0].strip()
        g = lambda k: None if k is None else _num(r[k], decimal_comma)
        mrms, bb = g(i_mrms), g(i_bb)
        if have_params:
            mr, ms, bc, bcr = g(i_mr), g(i_ms), g(i_bc), g(i_bcr)
            if None not in (mr, ms, bc, bcr) and ms > 0 and bc > 0:
                c_mrms, c_bb = mr / ms, bcr / bc
                if mrms is not None and bb is not None and (
                        abs(c_mrms - mrms) > 0.02 * abs(mrms) or abs(c_bb - bb) > 0.02 * abs(bb)):
                    n_mismatch.append(f"{name} (file {bb:.3g}/{mrms:.3g}, recomputed {c_bb:.3g}/{c_mrms:.3g})")
                if mrms is None or bb is None:
                    mrms, bb = c_mrms, c_bb
        if mrms is None or bb is None or mrms <= 0 or bb <= 0:
            skipped.append(name or "(no name)")
            continue
        names.append(name)
        x.append(bb)
        y.append(mrms)
        groups.append(r[i_group].strip() or "?" if i_group is not None else "")

    if i_mrms is not None and i_bb is not None:
        source = f"ratios read from columns '{headers[i_mrms]}' and '{headers[i_bb]}'"
    else:
        source = (f"ratios recomputed from '{headers[i_mr]}'/'{headers[i_ms]}' and "
                  f"'{headers[i_bcr]}'/'{headers[i_bc]}'")
    warns = []
    if n_mismatch:
        warns.append(f"{len(n_mismatch)} line(s) where the ratio columns differ by more than 2% "
                     "from the parameters (the ratio columns are used): " + "; ".join(n_mismatch))
    return DayTable(names=names, bcr_bc=np.array(x), mr_ms=np.array(y),
                    groups=groups if i_group is not None else None,
                    group_column=headers[i_group] if i_group is not None else "",
                    source=source, skipped=skipped, warnings=warns)
