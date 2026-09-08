"""
Menu "Statistics" (hors Bootstrap ellipse, cf. memoire projet
`project_stereoutils_pmagoutils_bugs.md`) : port de `pmagoutils.f` (Fisher
Statistics=ANGU, Fisher Dir+GC=FISHGC, Fisher recursive=angurecur, St Dev
dipole=STDDIP, Fisher Strati=STRATI, progressive Fold test=FOLDTEST, Mean
Inclination only=MEANINC, smallcircles Intersect=intersect/intersect2,
Reversal angle=reversalangle, difference angle=difangle).

Decisions utilisateur (2026, cf. memoire projet) sur les bugs Fortran
confirmes qui touchent ce sous-ensemble :
- ANGU : la branche NR>NN de la combinaison mode1+mode2-inverse lisait
  PN(I) hors-limites -> CORRIGEE ici (un seul chemin de code, base sur la
  branche "else" correcte du source, plus de branche asymetrique).
- FISHGC : NPRIM=M+(IMM/2) tronquait en division entiere -> CORRIGEE (
  division reelle M+IMM/2).
- STRATI : passait l'inclinaison directement comme argument colatitude a
  FISHER/MODES (au lieu de 90-inclinaison partout ailleurs) -> CORRIGEE
  (convention standard 90-inclinaison).

Tous les autres quirks/bugs confirmes (ANGU: filtrage gate NR==0/NN==0
asymetrique, angurecur: ecriture de code mort sur variable de boucle
perimee, STRATI: mutation en place de D() +90) sont REPLIQUES tels quels
(pas de demande explicite de l'utilisateur sur ceux-la).

`intersect` (menu "smallcircles Intersect") : seul le coeur bien defini de
`intersect2` (recherche d'intersection 2 ou 3 cercles a95, geometrie
McFadden) est porte ici - l'enveloppe `intersect` du source (boucle de
raffinement de 200 iterations qui ne depend pas de son indice et appelle
`angu2`/`angu` en cascade, sous-programme non retrouve/lu) est trop
fragile et sous-documentee pour etre reprise telle quelle ; voir
`project_stereoutils_pmagoutils_bugs.md` bug #4."""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from stereo_geometry import _angle, circle
from stereo_selection import read_text_lines, split_header

RAD = math.radians
DEG = math.degrees


# ---------------------------------------------------------------------------
# Noyau partage : FISHER/CPOLAR/POLERE/MODES/FISHE (pmagoutils.f)
# ---------------------------------------------------------------------------

def _cpolar(x: float, y: float, z: float) -> Tuple[float, float]:
    """Port de `CPOLAR` (pmagoutils.f:2375) : (x,y,z) -> (colatitude,
    azimuth) en RADIANS. Equivalent formule a `atan2` standard."""
    t = math.acos(max(-1.0, min(1.0, z)))
    p = math.atan2(y, x)
    if p < 0.0:
        p += 2.0 * math.pi
    return t, p


def fisher_stats(directions: Sequence[Tuple[float, float]]) -> dict:
    """Port de `FISHER` (pmagoutils.f:2311) : `directions`=(dec,inc) en
    degres. Retourne dec/inc/a95/k/n/stv (degres)."""
    n = len(directions)
    xsum = ysum = zsum = 0.0
    for dec, inc in directions:
        colat = RAD(90.0 - inc)
        phi = RAD(dec)
        xsum += math.sin(colat) * math.cos(phi)
        ysum += math.sin(colat) * math.sin(phi)
        zsum += math.cos(colat)
    r = math.sqrt(xsum * xsum + ysum * ysum + zsum * zsum)
    tbar, pbar = _cpolar(xsum / r, ysum / r, zsum / r)
    if (n - r) > 0:
        k = (n - 1) / (n - r)
    else:
        k = 1.0
    fac = 1.0 / (float(n) - 1.0)
    p = 20.0 ** fac - 1.0
    a = 1.0 - ((n - r) / r) * p
    a95 = DEG(math.acos(a)) if abs(a) < 1.0 else 90.0
    bx, by, bz = math.cos(pbar) * math.sin(tbar), math.sin(pbar) * math.sin(tbar), math.cos(tbar)
    tet = 0.0
    for dec, inc in directions:
        colat = RAD(90.0 - inc)
        phi = RAD(dec)
        x, y, z = math.sin(colat) * math.cos(phi), math.sin(colat) * math.sin(phi), math.cos(colat)
        stv_i = math.sqrt((x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2)
        tet += (2.0 * math.asin(min(1.0, stv_i / 2.0))) ** 2
    stv = DEG(math.sqrt(tet / (n - 1)))
    return {"dec": DEG(pbar), "inc": 90.0 - DEG(tbar), "a95": a95, "k": k, "n": n, "stv": stv,
            "pbar": pbar, "tbar": tbar}


def modes_split(
    directions: Sequence[Tuple[float, float]],
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Port de `MODES` (pmagoutils.f:3025, via `CALCT`+`QL`/EISPACK+
    `ROTAT`) : separe `directions` (dec,inc deg) en mode "normal" et mode
    "reverse" par analyse en axes principaux de la matrice d'orientation.
    `numpy.linalg.eigh` remplace TRED2/TQL2 (memes conventions : valeurs
    propres ascendantes, vecteurs propres orthonormes) - seul l'axe
    principal (plus grande valeur propre) importe pour la classification,
    le signe global du matrice n'affectant que ce vecteur pour ce calcul."""
    n = len(directions)
    vecs = []
    t = np.zeros((3, 3))
    for dec, inc in directions:
        colat = RAD(90.0 - inc)
        phi = RAD(dec)
        v = np.array([
            math.sin(colat) * math.cos(phi),
            math.sin(colat) * math.sin(phi),
            math.cos(colat),
        ])
        vecs.append(v)
        t += np.outer(v, v)
    t /= n
    eigvals, eigvecs = np.linalg.eigh(t)
    principal = eigvecs[:, -1]
    if principal[2] < 0.0:
        principal = -principal
    normal, reverse = [], []
    for (dec, inc), v in zip(directions, vecs):
        if float(np.dot(principal, v)) < 0.0:
            reverse.append((dec, inc))
        else:
            normal.append((dec, inc))
    return normal, reverse


def _fishe_core(directions: Sequence[Tuple[float, float]]) -> dict:
    """Port du calcul de `FISHE`/`FISHE2` (pmagoutils.f:286/5063), SANS
    l'effet de bord d'origine (append automatique dans DIC/INC/ALPH a
    chaque appel - `FOLDTEST` appelle ceci des dizaines de milliers de
    fois pendant sa recherche, un append systematique ferait exploser la
    liste des moyennes ; seul le resultat final de `FOLDTEST` est
    ajoute explicitement aux moyennes par l'appelant)."""
    m = len(directions)
    sx = sy = sz = 0.0
    for dec, inc in directions:
        d, i = RAD(dec), RAD(inc)
        sx += math.cos(i) * math.cos(d)
        sy += math.cos(i) * math.sin(d)
        sz += math.sin(i)
    bx, by, bz = sx, sy, sz
    r, dec_m, inc_m = _polere(sx, sy, sz)
    if m == 2:
        return {"dec": dec_m, "inc": inc_m, "r": r, "k": 0.0, "a95": 90.0}
    # FISHE (contrairement a FISHER) n'a pas de garde-fou M==R -> division
    # par zero possible (donnees parfaitement groupees) ; meme repli que
    # FISHER (pmagoutils.f:2335-2339) applique ici pour eviter un crash.
    k = (m - 1) / (m - r) if (m - r) > 0 else 1.0
    tet = 0.0
    for dec, inc in directions:
        d, i = RAD(dec), RAD(inc)
        x, y, z = math.cos(i) * math.cos(d), math.cos(i) * math.sin(d), math.sin(i)
        stv_i = math.sqrt((x - bx / r) ** 2 + (y - by / r) ** 2 + (z - bz / r) ** 2)
        tet += (2.0 * math.asin(min(1.0, stv_i / 2.0))) ** 2
    p = 0.05
    a2 = 1.0 - ((m - r) / r) * ((1.0 / p) ** (1.0 / (m - 1)) - 1.0)
    a2 = min(abs(a2), 1.0) * (1.0 if a2 >= 0 else -1.0)
    a95 = DEG(math.atan2(math.sqrt(max(0.0, 1.0 - a2 * a2)), a2)) if abs(a2) <= 1.0 else 90.0
    return {"dec": dec_m, "inc": inc_m, "r": r, "k": k, "a95": a95}


def _polere(x: float, y: float, z: float) -> Tuple[float, float, float]:
    from stereo_geometry import polere
    return polere(x, y, z)


# ---------------------------------------------------------------------------
# FISHQQ : test d'adequation Fisherien (Fisher et al. 1987)
# ---------------------------------------------------------------------------

def _fishqq(directions: Sequence[Tuple[float, float]], pbar: float, tbar: float) -> bool:
    """Port de `FISHQQ`/`DIROT`/`CRUNCH`/`QSEXP`/`DSTAT` (pmagoutils.f:
    3098-3357) : True si le mode est jugee Fisherien (test de Kuiper/K-S,
    seuils 5% de Fisher et al. 1987 : Mu<=1.207, Me<=1.094)."""
    n = len(directions)
    if n < 3:
        return True
    rot = np.array([
        [math.cos(tbar) * math.cos(pbar), math.cos(tbar) * math.sin(pbar), -math.sin(tbar)],
        [-math.sin(pbar), math.cos(pbar), 0.0],
        [math.sin(tbar) * math.cos(pbar), math.sin(tbar) * math.sin(pbar), math.cos(tbar)],
    ])
    x1, x2 = [], []
    for dec, inc in directions:
        colat, phi = RAD(90.0 - inc), RAD(dec)
        v = np.array([math.sin(colat) * math.cos(phi), math.sin(colat) * math.sin(phi), math.cos(colat)])
        rv = rot @ v
        trot, prot = _cpolar(rv[0], rv[1], rv[2])
        x2.append(1.0 - math.cos(trot))
        x1.append(prot / (2.0 * math.pi))
    x1.sort()
    x2.sort()
    en = float(n)
    xsum = sum(x2)
    kappa = (en - 1.0) / xsum if xsum else 0.0
    dpos = dneg = 0.0
    for i, v in enumerate(x2, start=1):
        f = 1.0 - math.exp(-kappa * v)
        dpos = max(dpos, i / en - f)
        dneg = max(dneg, f - (i - 1) / en)
    d = max(dneg, dpos)
    me = (d - 0.2 / en) * (math.sqrt(en) + 0.26 + 0.5 / math.sqrt(en))
    dpos = dneg = 0.0
    for i, v in enumerate(x1, start=1):
        dpos = max(dpos, i / en - v)
        dneg = max(dneg, v - (i - 1) / en)
    vv = dneg + dpos
    mu = vv * (math.sqrt(en) - 0.567 + 1.623 / math.sqrt(en))
    return not (mu > 1.207 or me > 1.094)


# ---------------------------------------------------------------------------
# Fisher Statistics (ANGU)
# ---------------------------------------------------------------------------

def angu(
    directions: Sequence[Tuple[float, float]], rfilt: float = 0.0,
    return_last_stats: bool = False,
):
    """Port de `ANGU` (pmagoutils.f:2-281, menu Statistics > Fisher
    Statistics) : separe les donnees en 1 ou 2 modes (via `MODES`), calcule
    la moyenne de Fisher de chaque mode puis (si les deux existent) du
    mode 1 + mode 2 inverse. `rfilt>0` : re-filtre et recalcule les points
    a moins de `rfilt` deg de la moyenne (seulement pour le mode qui n'a
    pas d'autre mode co-existant, meme asymetrie que le source pour les
    modes 1/2 seuls ; le filtre du mode combine, lui, est inconditionnel,
    comme le source). Retourne (rapport texte, nouvelles moyennes a
    ajouter a self.means) ; si `return_last_stats`, un 3e element (le
    dict de stats complet du DERNIER bloc calcule - equivalent de
    `dic(im)/nfish/rk` apres l'appel Fortran, utilise par `fisherproject`)."""
    if len(directions) < 2:
        empty = ("not enough data (need >= 2 directions)\n", [])
        return empty + (None,) if return_last_stats else empty
    lines: List[str] = []
    new_means: List[Tuple[float, float, float, str]] = []
    last_stats = {}

    def _mode_block(title: str, pts: List[Tuple[float, float]], allow_filter: bool) -> dict:
        stats = fisher_stats(pts)
        result_stats = stats
        lines.append(f"{title}")
        lines.append(" DEC,    INC,      A95,     N, KAPPA,  STV")
        lines.append(f" {stats['dec']:6.1f}  {stats['inc']:6.1f}  {stats['a95']:6.1f}  {stats['n']:5d}  {stats['k']:6.1f}  {stats['stv']:6.1f}")
        new_means.append((stats["dec"], stats["inc"], stats["a95"], "e"))
        filtered = []
        for dec, inc in pts:
            del_, _ang = _angle(dec - stats["dec"], 90.0 - stats["inc"], 90.0 - inc)
            del_ = abs(del_)
            mark = " >" if (del_ - 2 * stats["stv"]) > 0.0 else "< "
            lines.append(f" DEC={dec:7.2f}  INC={inc:7.2f} ADM={del_:7.2f}   {mark}")
            if allow_filter and rfilt > 0.0 and del_ < rfilt:
                filtered.append((dec, inc))
        if allow_filter and rfilt > 0.0 and len(filtered) >= 2:
            fstats = fisher_stats(filtered)
            lines.append(f"after filtering all data at more than {rfilt} from the mean")
            lines.append(f" {title} (filtered):")
            lines.append(f" {fstats['dec']:6.1f}  {fstats['inc']:6.1f}  {fstats['a95']:6.1f}  {fstats['n']:5d}  {fstats['k']:6.1f}  {fstats['stv']:6.1f}")
            new_means.append((fstats["dec"], fstats["inc"], fstats["a95"], "e"))
            result_stats = fstats
        if len(pts) > 2:
            verdict = "FISHERIAN" if _fishqq(pts, RAD(stats["dec"]), RAD(90.0 - stats["inc"])) else "NON-FISHERIAN"
            lines.append(f"MODE IS {verdict}")
        return result_stats

    normal, reverse = modes_split(directions)
    if len(normal) >= 2:
        last_stats = _mode_block(" MEAN OF FIRST MODE IS:", normal, allow_filter=(len(reverse) == 0))
    if len(reverse) >= 2:
        last_stats = _mode_block(" MEAN OF SECOND MODE IS:", reverse, allow_filter=(len(normal) == 0))
    if normal and reverse:
        combined = list(normal) + [((d + 180.0) % 360.0, -i) for d, i in reverse]
        if len(combined) >= 2:
            last_stats = _mode_block(" MEAN OF FIRST MODE + INVERTED SECOND MODE IS:", combined, allow_filter=True)
    lines.append("-" * 20)
    text = "\n".join(lines) + "\n"
    if return_last_stats:
        return text, new_means, (last_stats or None)
    return text, new_means


# ---------------------------------------------------------------------------
# Fisher recursive (angurecur)
# ---------------------------------------------------------------------------

def angurecur_mode(pts: Sequence[Tuple[float, float]], cutoff: float, title: str) -> Tuple[str, Optional[Tuple[float, float, float, str]]]:
    """Port du coeur iteratif de `angurecur` (pmagoutils.f:4941-5060) pour
    UN mode : elimine, un a la fois, le point le plus deviant qui depasse
    `cutoff` degres de la moyenne courante, recalcule, jusqu'a
    convergence."""
    lines = [title]
    pts = list(pts)
    if len(pts) < 3:
        return "\n".join(lines) + " (need >= 3 points)\n", None
    dec0 = inc0 = None
    while True:
        stats = fisher_stats(pts)
        dec0, inc0 = stats["dec"], stats["inc"]
        delmax, inmax = -100.0, -1
        for idx, (dec, inc) in enumerate(pts):
            del_, _ang = _angle(dec - stats["dec"], 90.0 - stats["inc"], 90.0 - inc)
            del_ = abs(del_)
            if del_ > delmax and del_ > cutoff:
                delmax, inmax = del_, idx
        if inmax < 0:
            lines.append(f" {stats['dec']:6.1f}  {stats['inc']:6.1f}  {stats['a95']:6.1f}  {stats['n']:5d}  {stats['k']:6.1f}  {stats['stv']:6.1f}")
            verdict = "FISHERIAN" if _fishqq(pts, RAD(stats["dec"]), RAD(90.0 - stats["inc"])) else "NON-FISHERIAN"
            lines.append(f"MODE IS {verdict}")
            return "\n".join(lines) + "\n", (dec0, inc0, stats["a95"], "e")
        removed = pts[inmax]
        lines.append(f" removed: {removed[0]:6.1f}  {removed[1]:6.1f}")
        del pts[inmax]


def angurecur(directions: Sequence[Tuple[float, float]], cutoff: float) -> Tuple[str, List[Tuple[float, float, float, str]]]:
    """Port de `angurecur` (menu Statistics > Fisher recursive), pour les
    deux modes (normal puis reverse), sans le "type return" manuel du
    source entre les deux (les deux sont traites automatiquement ici)."""
    normal, reverse = modes_split(directions)
    text1, mean1 = angurecur_mode(normal, cutoff, "recursive fisher first mode")
    text2, mean2 = angurecur_mode(reverse, cutoff, "recursive fisher second mode")
    means = [m for m in (mean1, mean2) if m is not None]
    return text1 + "\n" + text2, means


# ---------------------------------------------------------------------------
# St Dev dipole (STDDIP)
# ---------------------------------------------------------------------------

def stddip(directions: Sequence[Tuple[float, float]], ref_dec: float, ref_inc: float) -> str:
    """Port de `STDDIP` (pmagoutils.f:355-404, menu Statistics > St Dev
    dipole) : ecart-type angulaire des donnees par rapport a une direction
    de reference FIXE (pas la moyenne de Fisher des donnees)."""
    if not directions:
        return "no data\n"
    m = len(directions)
    bx = math.cos(RAD(ref_dec)) * math.cos(RAD(ref_inc))
    by = math.sin(RAD(ref_dec)) * math.cos(RAD(ref_inc))
    bz = math.sin(RAD(ref_inc))
    tet = 0.0
    for dec, inc in directions:
        x = math.cos(RAD(dec)) * math.cos(RAD(inc))
        y = math.sin(RAD(dec)) * math.cos(RAD(inc))
        z = math.sin(RAD(inc))
        stv_i = math.sqrt((x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2)
        tet += (2.0 * math.asin(min(1.0, stv_i / 2.0))) ** 2
    stv = DEG(math.sqrt(tet / (m - 1))) if m > 1 else 0.0
    lines = [f"STV={stv:6.1f}"]
    for dec, inc in directions:
        del_, _ang = _angle(dec - ref_dec, 90.0 - ref_inc, 90.0 - inc)
        del_ = abs(del_)
        mark = " >" if (del_ - 2 * stv) > 0.0 else "< "
        lines.append(f" DEC={dec:7.2f}  INC={inc:7.2f} ADM={del_:7.2f}   {mark}")
    lines.append("-" * 20)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Mean Inclination only (MEANINC)
# ---------------------------------------------------------------------------

def mean_inclination_native(inclinations: Sequence[float]) -> Optional[dict]:
    """Port de `MEANINC` (pmagoutils.f:416-487, menu Statistics > Mean
    Inclination only) : estimateur du maximum de vraisemblance de McFadden
    & Reid (1982) par Newton-Raphson natif - DIFFERENT du menu Pmag_Python
    > Mean Inclination (qui appelle `pmag.doincfish`, une implementation
    PmagPy independante)."""
    k = len(inclinations)
    if k < 2:
        return None
    conv = math.pi / 180.0
    som = som1 = som2 = som3 = 0.0
    for inc in inclinations:
        som += inc
        dd = (90.0 - inc) * conv
        som1 += math.cos(dd)
        som2 += math.sin(dd)
        som3 += dd
    amar = som / k
    par = som3 / k
    eps = 0.0001
    for _ in range(200):
        a1, a2 = math.cos(par) ** 2, math.sin(par) ** 2
        ff = k * math.cos(par) + (a2 - a1) * som1 - 2.0 * math.sin(par) * math.cos(par) * som2
        ffp = -k * math.sin(par) + 2.0 * som1 * math.sin(2.0 * par) - 2.0 * som2 * math.cos(2.0 * par)
        xx = par - ff / ffp
        if abs(xx - par) < eps:
            par = xx
            break
        par = xx
    xx = par
    c = math.cos(xx) * som1 + math.sin(xx) * som2
    s = math.sin(xx) * som1 - math.cos(xx) * som2
    biais = 180.0 * s / (math.pi * c)
    prec = (k - 1.0) / (2.0 * (k - c))
    ainc = 90.0 - xx / conv + biais
    aai = 90.0 - xx / conv
    res = ((prec - 1.0) * k + 1.0) / prec
    toto = (0.05 ** (-1.0 / (k - 1.0)) - 1.0) * (k - res) / res
    toto = min(2.0, max(0.0, toto))
    alp95 = DEG(math.acos(1.0 - toto))
    return {
        "arithmetic_mean": amar, "biased_arason": aai, "unbiased_mcfadden": ainc,
        "precision": prec, "alpha95": alp95, "alpha95_pos": alp95 + biais, "alpha95_neg": alp95 - biais,
    }


# ---------------------------------------------------------------------------
# Fisher Strati (STRATI)
# ---------------------------------------------------------------------------

def strati(
    directions: Sequence[Tuple[float, float, str]],
) -> Tuple[str, List[Tuple[float, float, str]], List[Tuple[float, float, float, float, float, float]]]:
    """Port de `STRATI` (pmagoutils.f:3688-3745, menu Statistics > Fisher
    Strati, donnees pole-a-litage) : convertit chaque direction en pole de
    grand cercle (decalage azimutal -90, convention colatitude STANDARD
    90-inclinaison - corrigee sur decision utilisateur, le source original
    utilisait l'inclinaison directement) puis separe/calcule les moyennes
    de Fisher par mode. Mute `directions` (dec+90, meme effet de bord que
    le source - appeler deux fois decale deux fois) et REMPLACE la liste
    des grands cercles par les poles ainsi calcules (meme effet de bord
    que `IMM=M` dans le source)."""
    if len(directions) < 3:
        return "not enough data (need >= 3 directions)\n", list(directions), []
    lines = []
    poles = []
    new_directions = []
    gc_dirs = []
    for dec, inc, sym in directions:
        glat = 90.0 - inc
        glon = dec - 90.0
        poles.append((glon, glat))
        gc_dirs.append((glon, glat))
        new_directions.append(((dec + 90.0) % 360.0, inc, sym))
    great_circles = [(glon, glat, 0.0, 0.0, 0.0, 0.0) for glon, glat in poles]

    normal, reverse = modes_split(gc_dirs)
    if len(normal) >= 3:
        s = fisher_stats(normal)
        lines.append(" MEAN OF FIRST MODE IS:")
        lines.append(" DEC,    INC,      A95,     N, KAPPA,  STV")
        lines.append(f" {90.0 + s['dec']:6.1f}  {s['inc']:6.1f}  {s['a95']:6.1f}  {s['n']:5d}  {s['k']:6.1f}  {s['stv']:6.1f}")
    if len(reverse) >= 3:
        s = fisher_stats(reverse)
        lines.append(" MEAN OF SECOND MODE IS:")
        lines.append(" DEC,    INC,      A95,     N, KAPPA,  STV")
        lines.append(f" {90.0 + s['dec']:6.1f}  {s['inc']:6.1f}  {s['a95']:6.1f}  {s['n']:5d}  {s['k']:6.1f}  {s['stv']:6.1f}")
    lines.append("-" * 20)
    return "\n".join(lines) + "\n", new_directions, great_circles


# ---------------------------------------------------------------------------
# progressive Fold test (FOLDTEST)
# ---------------------------------------------------------------------------

def fold_test_native(
    rows: Sequence[Tuple[str, float, float, float, float]],
) -> Tuple[str, List[Tuple[float, float, str]]]:
    """Port de `FOLDTEST` (pmagoutils.f:3812-3999, menu Statistics >
    progressive Fold test) - DIFFERENT du menu Pmag_Python > Fold Test
    (qui appelle `ipmag.bootstrap_fold_test`). `rows` = (site,dec,inc,
    azimuth,dip_percent_or_full) une ligne par site (dip en degres, PAS en
    fraction - la division /100 du source correspondait a un stockage en
    pourcentage historique, refaite ici en interne).

    Stage 1 : cherche le pourcentage de deplissement (JJ=-20..100%, pas de
    1) qui maximise kappa sur l'ensemble des sites (deplissement uniforme).
    Stage 2 : optimisation site-par-site (Gauss-Seidel, jusqu'a 100 passes)
    qui laisse chaque site choisir independamment son propre pourcentage
    de deplissement (0-100%, pas de 1%) pour maximiser kappa combine."""
    sites = [(site, dec, inc, az, dip) for site, dec, inc, az, dip in rows]
    n = len(sites)
    if n < 2:
        return "not enough sites (need >= 2)\n", []
    lines = []

    def _tilt(dec, inc, dip_percent, az):
        rj = (dip_percent / 100.0) if False else dip_percent
        from stereo_geometry import corpen, polere
        z = math.sin(RAD(inc))
        x = math.cos(RAD(inc)) * math.cos(RAD(dec))
        y = math.cos(RAD(inc)) * math.sin(RAD(dec))
        xx, yy, zz = corpen(x, y, z, dip_percent, az)
        _r, yy2, zz2 = polere(xx, yy, zz)
        return yy2, zz2

    # Stage 1 : recherche globale du % de deplissement optimal (-20..120%)
    ak0, iopti = -1.0, 0
    for jj in range(-20, 121):
        current = []
        for _site, dec, inc, az, dip in sites:
            rj = (dip / 100.0) * jj
            d, i = _tilt(dec, inc, rj, az)
            current.append((d, i))
        s = _fishe_core(current)
        if s["k"] > ak0:
            ak0, iopti = s["k"], jj
    iopti = min(100, max(0, iopti))
    lines.append(f" optimal global unfolding : {iopti}%  (k={ak0:.1f})")

    # Profils individuels de deplissement (0-100%, pas de 1) par site
    profiles = []
    for _site, dec, inc, az, dip in sites:
        prof = [(dec, inc)]
        for j in range(1, 101):
            rj = dip * j / 100.0
            if rj != 0.0:
                d, i = _tilt(dec, inc, rj, az)
            else:
                d, i = dec, inc
            prof.append((d, i))
        profiles.append(prof)

    current_best = [profiles[i][iopti] for i in range(n)]
    s = _fishe_core(current_best)
    lines.append(f" mean direction at {iopti}% unfolding : dec={s['dec']:.1f}  inc={s['inc']:.1f}  a95={s['a95']:.1f}  k={s['k']:.1f}")

    # Stage 2 : optimisation non-standard, site par site (Gauss-Seidel)
    lines.append("")
    lines.append(" non standard fold test")
    isite = [0] * n
    ak00 = ak0
    for _outer in range(100):
        for i in range(n):
            ak0_local = -1.0
            best_j = isite[i]
            for j in range(101):
                trial = list(current_best)
                trial[i] = profiles[i][j]
                s = _fishe_core(trial)
                if s["k"] > ak0_local:
                    ak0_local, best_j = s["k"], j
            current_best[i] = profiles[i][best_j]
            isite[i] = best_j
        s = _fishe_core(current_best)
        if s["k"] > ak00:
            ak00 = s["k"]
        else:
            break

    lines.append(" site    numline    % folding   strike  dip     dip_optimum    dec      inc")
    for i, (site, _dec, _inc, az, dip) in enumerate(sites):
        d, inc = current_best[i]
        lines.append(f" {site:<10s} {i + 1:3d}  {isite[i]:3d}  {az:6.1f}  {dip:6.1f}  {dip * isite[i] / 100.0:6.1f}  {d:6.1f}  {inc:6.1f}")
    s = _fishe_core(current_best)
    lines.append(f" optimal mean direction: dec={s['dec']:.1f}  inc={s['inc']:.1f}  a95={s['a95']:.1f}  k={s['k']:.1f}")

    new_directions = [(d, i, "c") for d, i in current_best]
    return "\n".join(lines) + "\n", new_directions


# ---------------------------------------------------------------------------
# smallcircles Intersect (coeur de intersect2)
# ---------------------------------------------------------------------------

def _di2xyz(dec: float, inc: float) -> Tuple[float, float, float]:
    z = math.sin(RAD(inc))
    x = math.cos(RAD(inc)) * math.cos(RAD(dec))
    y = math.cos(RAD(inc)) * math.sin(RAD(dec))
    return x, y, z


def _circle_boundary(a95: float, inc: float, dec: float) -> List[Tuple[float, float]]:
    ic = max(1, round(3600.0 * a95 / 90.0))
    ph0 = 360.0 / ic
    return [circle(a95, inc, dec, i * ph0) for i in range(1, ic + 1)]  # (ei,ed) pairs


def _closest_pair(
    b1: List[Tuple[float, float]], b2: List[Tuple[float, float]],
) -> Tuple[float, float, float, float, bool]:
    """Port du coeur de `intersect2` pour UNE paire de cercles : cherche la
    paire de points-echantillon la plus proche (`del0`), et si des paires a
    moins de 0.3 deg existent, choisit la plus proche (solution 1) puis la
    plus eloignee d'elle parmi ces paires proches (solution 2) - sinon,
    replie sur le milieu (moyenne vectorielle) des deux points les plus
    proches et signale `intersecting=False`."""
    del0 = 180.0
    sol1 = sol2 = None
    near = []
    for ei, ed in b1:
        for ei2, ed2 in b2:
            del_, _ang = _angle(ed - ed2, 90.0 - ei2, 90.0 - ei)
            del_ = abs(del_)
            if del_ < del0:
                del0 = del_
                sol1, sol2 = (ed, ei), (ed2, ei2)
            if del_ < 0.3:
                near.append(((ed, ei), del_))
    if del0 < 0.3 and near:
        imin = min(range(len(near)), key=lambda k: near[k][1])
        s1 = near[imin][0]
        delmin, jmax = 0.0, imin
        for k, (pt, _d) in enumerate(near):
            del_, _ang = _angle(s1[0] - pt[0], 90.0 - pt[1], 90.0 - s1[1])
            del_ = abs(del_)
            if del_ > delmin:
                delmin, jmax = del_, k
        s2 = near[jmax][0]
        return s1[0], s1[1], s2[0], s2[1], True
    x1, y1, z1 = _di2xyz(*sol1)
    x2, y2, z2 = _di2xyz(*sol2)
    from stereo_geometry import polere
    _r, dec, inc = polere(x1 + x2, y1 + y2, z1 + z2)
    return dec, inc, dec, inc, False


def intersect_smallcircles(
    means: Sequence[Tuple[float, float, float, str]],
) -> Tuple[str, List[Tuple[float, float, str]]]:
    """Port du coeur geometrique de `intersect2` (pmagoutils.f:4171-4692,
    menu Statistics > smallcircles Intersect) : intersection de 2 ou 3
    cercles de confiance a95, en utilisant TOUJOURS les 3 premieres
    moyennes en memoire (`means[0]`,`means[1]`,`means[2]`) - meme
    limitation que le source (pas de choix combinatoire pour >3 cercles,
    la partie du code qui l'aurait fait est commentee dans le Fortran).
    Retourne (rapport texte, points d'intersection a ajouter a
    self.directions, symbole 's')."""
    if len(means) < 2:
        return "need at least 2 mean directions (D-I-a95) in memory\n", []
    c1 = means[0]
    c2 = means[1]
    c3 = means[2] if len(means) >= 3 else None

    b1 = _circle_boundary(c1[2], c1[1], c1[0])
    b2 = _circle_boundary(c2[2], c2[1], c2[0])
    d1, i1, d2, i2, ok12 = _closest_pair(b1, b2)
    lines = [("intersection c1 et c2" if ok12 else "pas d'intersection entre c1 et c2 (approche la plus proche)")]
    lines.append(f"P1: {d1:6.1f} {i1:5.1f}   P2: {d2:6.1f} {i2:5.1f}")

    if c3 is None:
        return "\n".join(lines) + "\n", []

    b3 = _circle_boundary(c3[2], c3[1], c3[0])
    d1b, i1b, d2b, i2b, ok13 = _closest_pair(b1, b3)
    lines.append(("pas d'intersection entre c1 et c3" if not ok13 else "intersection c1 et c3"))
    lines.append(f"P1: {d1b:6.1f} {i1b:5.1f}   P2: {d2b:6.1f} {i2b:5.1f}")

    d1c, i1c, d2c, i2c, ok23 = _closest_pair(b2, b3)
    lines.append(("pas d'intersection entre c2 et c3" if not ok23 else "intersection c2 et c3"))
    lines.append(f"P1: {d1c:6.1f} {i1c:5.1f}   P2: {d2c:6.1f} {i2c:5.1f}")

    candidates_12 = [(d1, i1), (d2, i2)]
    candidates_13 = [(d1b, i1b), (d2b, i2b)]
    candidates_23 = [(d1c, i1c), (d2c, i2c)]
    best_r, best = -1.0, None
    for p1 in candidates_12:
        for p2 in candidates_13:
            for p3 in candidates_23:
                x1, y1, z1 = _di2xyz(*p1)
                x2, y2, z2 = _di2xyz(*p2)
                x3, y3, z3 = _di2xyz(*p3)
                r = math.sqrt((x1 + x2 + x3) ** 2 + (y1 + y2 + y3) ** 2 + (z1 + z2 + z3) ** 2)
                if r > best_r:
                    best_r, best = r, (p1, p2, p3)
    p1, p2, p3 = best
    lines.append(f"  D={p1[0]:6.1f} I={p1[1]:5.1f}  D={p2[0]:6.1f} I={p2[1]:5.1f}  D={p3[0]:6.1f} I={p3[1]:5.1f}")
    new_points = [(p1[0], p1[1], "s"), (p2[0], p2[1], "s"), (p3[0], p3[1], "s")]
    return "\n".join(lines) + "\n", new_points


# ---------------------------------------------------------------------------
# Reversal angle / difference angle
# ---------------------------------------------------------------------------

def reversal_angle(directions: Sequence[Tuple[float, float]], d_expect: float, r_expect: float) -> str:
    """Port de `reversalangle` (pmagoutils.f:5477-5495)."""
    if not directions:
        return "no data\n"
    lines = []
    for dec, inc in directions:
        del_, _ang = _angle(d_expect - dec, 90.0 - r_expect, 90.0 - inc)
        lines.append(f"{dec:7.1f}  {inc:7.1f}  {abs(del_):7.1f}")
    return "\n".join(lines) + "\n"


def diff_angle_file(path: str) -> str:
    """Port de `difangle` (pmagoutils.f:5496-5519) : fichier de paires de
    directions (D0,I0,D1,I1) par ligne, angle entre chaque paire -
    colonnes reperees par en-tete si present (ex. "#d0 i0 d1 i1"), sinon
    par position comme avant."""
    lines = read_text_lines(path)
    data_lines, idx = split_header(lines, "d0", "i0", "d1", "i1")
    if all(k in idx for k in ("d0", "i0", "d1", "i1")):
        i0, j0, i1, j1 = idx["d0"], idx["i0"], idx["d1"], idx["i1"]
    else:
        i0, j0, i1, j1 = 0, 1, 2, 3
    out = []
    for line in data_lines:
        line = line.strip()
        if not line or line.startswith("!") or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) <= max(i0, j0, i1, j1):
            continue
        try:
            d_expect, r_expect, yy, zz = float(parts[i0]), float(parts[j0]), float(parts[i1]), float(parts[j1])
        except ValueError:
            continue
        del_, _ang = _angle(d_expect - yy, 90.0 - zz, 90.0 - r_expect)
        out.append(f"{d_expect:6.1f}  {r_expect:6.1f}  {yy:6.1f}  {zz:6.1f}  :  {abs(del_):6.1f}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Fisher Dir+GC (FISHGC)
# ---------------------------------------------------------------------------

def fishgc(
    directions: Sequence[Tuple[float, float]],
    great_circles: Sequence[Tuple[float, float, float, float, float, float]],
    use_sectors: bool,
    start_dec: float = 0.0,
    start_inc: float = 0.0,
) -> Tuple[str, Optional[Tuple[float, float, float, str]]]:
    """Port de `FISHGC` (pmagoutils.f:3398-3684, menu Statistics > Fisher
    Dir+GC, McFadden & McElhinny 1988) : direction moyenne combinant
    directions Fisheriennes ET contraintes de grand cercle, avec secteurs
    optionnels. `start_dec/start_inc` : direction de depart si aucune
    direction Fisherienne n'est chargee (invite "DIRECTION DE DEPART"
    du source)."""
    m, imm = len(directions), len(great_circles)
    if (m + imm) < 3:
        return "not enough data (directions + great circles >= 3 required)\n", None

    def _project(u0, v0, w0, glon, glat):
        p, q, r = math.cos(RAD(glon)) * math.cos(RAD(glat)), math.sin(RAD(glon)) * math.cos(RAD(glat)), math.sin(RAD(glat))
        tau = u0 * p + v0 * q + w0 * r
        ro = math.sqrt(max(1e-12, 1.0 - tau * tau))
        return (u0 - tau * p) / ro, (v0 - tau * q) / ro, (w0 - tau * r) / ro

    sectors = []
    if use_sectors:
        for glon, glat, ad1, ai1, ad2, ai2 in great_circles:
            if (ad1, ai1, ad2, ai2) == (0.0, 0.0, 0.0, 0.0):
                sectors.append(None)
                continue
            x1, y1, z1 = _project(math.cos(RAD(ad1)) * math.cos(RAD(ai1)), math.sin(RAD(ad1)) * math.cos(RAD(ai1)), math.sin(RAD(ai1)), glon, glat)
            x2, y2, z2 = _project(math.cos(RAD(ad2)) * math.cos(RAD(ai2)), math.sin(RAD(ad2)) * math.cos(RAD(ai2)), math.sin(RAD(ai2)), glon, glat)
            sectors.append(((x1, y1, z1), (x2, y2, z2)))
    else:
        sectors = [None] * imm

    if m == 0:
        u0, v0, w0 = math.cos(RAD(start_dec)) * math.cos(RAD(start_inc)), math.sin(RAD(start_dec)) * math.cos(RAD(start_inc)), math.sin(RAD(start_inc))
        sx = sy = sz = 0.0
    else:
        sx = sy = sz = 0.0
        for dec, inc in directions:
            sx += math.cos(RAD(inc)) * math.cos(RAD(dec))
            sy += math.cos(RAD(inc)) * math.sin(RAD(dec))
            sz += math.sin(RAD(inc))
        norm = math.sqrt(sx * sx + sy * sy + sz * sz)
        u0, v0, w0 = sx / norm, sy / norm, sz / norm

    def _clamp_sector(xg, yg, zg, sector):
        if sector is None:
            return xg, yg, zg
        (x1, y1, z1), (x2, y2, z2) = sector
        betta = math.acos(max(-1.0, min(1.0, x1 * x2 + y1 * y2 + z1 * z2)))
        betta1 = math.acos(max(-1.0, min(1.0, x1 * xg + y1 * yg + z1 * zg)))
        betta2 = math.acos(max(-1.0, min(1.0, xg * x2 + yg * y2 + zg * z2)))
        if betta2 > betta:
            return (x1, y1, z1) if betta1 < betta2 else (x2, y2, z2)
        if betta1 > betta:
            return (x2, y2, z2) if betta2 < betta1 else (x1, y1, z1)
        return xg, yg, zg

    xg_list = [None] * imm
    for i, (glon, glat, *_r) in enumerate(great_circles):
        xg, yg, zg = _project(u0, v0, w0, glon, glat)
        xg, yg, zg = _clamp_sector(xg, yg, zg, sectors[i])
        xg_list[i] = (xg, yg, zg)
        sx, sy, sz = sx + xg, sy + yg, sz + zg
        norm = math.sqrt(sx * sx + sy * sy + sz * sz)
        u0, v0, w0 = sx / norm, sy / norm, sz / norm
    r0 = math.sqrt(sx * sx + sy * sy + sz * sz)

    itr = 0
    for itr in range(1, 101):
        for j, (glon, glat, *_r) in enumerate(great_circles):
            xg, yg, zg = xg_list[j]
            sx, sy, sz = sx - xg, sy - yg, sz - zg
            norm = math.sqrt(sx * sx + sy * sy + sz * sz)
            u0, v0, w0 = sx / norm, sy / norm, sz / norm
            xg, yg, zg = _project(u0, v0, w0, glon, glat)
            xg, yg, zg = _clamp_sector(xg, yg, zg, sectors[j])
            xg_list[j] = (xg, yg, zg)
            sx, sy, sz = sx + xg, sy + yg, sz + zg
            norm = math.sqrt(sx * sx + sy * sy + sz * sz)
            u0, v0, w0 = sx / norm, sy / norm, sz / norm
        r1 = math.sqrt(sx * sx + sy * sy + sz * sz)
        if r1 < (r0 + 1e-6 * r0) and itr > 15:
            break
        r0 = r1
    r1 = math.sqrt(sx * sx + sy * sy + sz * sz)

    lines = [f"iterations: {itr}"]
    for ij, (xg, yg, zg) in enumerate(xg_list, start=1):
        _r, gy, gz = _polere(xg, yg, zg)
        lines.append(f"{ij:3d}: DEC={gy:6.1f}  INC={gz:6.1f}")
    lines.append(f"R = {r1:6.1f}")
    _r, decli, rincli = _polere(sx, sy, sz)
    lines.append(f"DEC={decli:6.1f}  INC={rincli:6.1f}")
    rk = (2 * m + imm - 2) / (2 * (m + imm - r1))
    lines.append(f"K={rk:6.1f}")
    nprim = m + imm / 2.0  # correction utilisateur : division reelle (pas //)
    alpha95 = 1.0 - ((nprim - 1) * (20 ** (1.0 / (nprim - 1)) - 1) / (rk * r1))
    alpha95 = DEG(math.acos(alpha95)) if abs(alpha95) < 1.0 else 90.0
    lines.append(f"ALPHA95={alpha95:6.1f}")
    return "\n".join(lines) + "\n", (decli, rincli, alpha95, "c")
