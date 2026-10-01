"""
Incertitude des modeles de champ a coefficients de Gauss (CALS3k.4b,
CALS10k.1b) - demande explicite utilisateur ("dans le dossier
Field_models, il y a des vieilles sources en Fortran pour calculer des
incertitudes... est-ce possible de voir si une partie peut etre
integree").

Format de fichier (B-splines cubiques, Monika Korte - voir
Field_Models/cals3k-4/fielduncert3k.f et Field_Models/calsk_PhD/
CALS10kfield.f, memes 2 subroutines cles - errfdz/magfdz - juste des
tailles differentes, nsplt=402 pour CALS3k.4b, 303 pour CALS10k.1b) :
  ligne 1 : tstart tend <texte libre ignore>
  ligne 2+ : lm nm nspl tknts(nspl+4) gt(n,nspl) dgt(n,nspl)
     (n = lm*(lm+2) = 120 pour lm=10 ; gt = coefficients de Gauss par
     spline - la COURBE du modele, degre 3/de Boor ; dgt = INCERTITUDE
     de chaque coefficient, MEME representation en spline - procedure
     d'estimation "MAST", Korte 2010/2011)

ORDRE MEMOIRE Fortran (dimension gt(n,nsplt)) : COLONNE-MAJEUR, le
PREMIER indice (n, coefficient de Gauss) varie le plus vite - piege reel
rencontre en ecrivant ce module (un reshape naif ligne-majeur donne des
valeurs ~400x trop grandes et sans rapport avec le vrai champ ; corrige
en reshape(nspl, n).T).

VERIFIE CONTRE LE FORTRAN REEL (fielduncert3k.f recompile avec gfortran,
execute sur CALS3k.4b, site lat=-39/lon=-72, annee 500) : correspondance
EXACTE (D=1.10 I=-60.10 F=45.0uT dD=5.72 dI=4.31 dF=4.08uT, 2 decimales)
avec ce module. Egalement verifie que CALS3k.4b/CALS10k.1b sont bien les
MEMES modeles que pmagpy.coefficients.get_cals3k()/get_cals10k()
(coefficients de Gauss identiques a l'epoque commune testee, g10 a 8
chiffres significatifs) - l'incertitude est donc bien appariee au BON
modele deja utilise par stereo_pmagpy.predicted_field_curve, pas a une
version differente.

Modeles SANS incertitude disponible localement (`field_uncertainty_at`
retourne alors None) : IGRF14, GUFM1, ARCH3k, PFM9k, HFM.OL1.A1,
SHA.DIF.14k, SHAWQ2k, SHAWQ-Iberia, CALS10k.2 (ce dernier partage
pourtant la meme famille que CALS10k.1b, mais le fichier CALS10k.2
fourni ne porte qu'UN SEUL bloc de coefficients, pas de bloc
d'incertitude - verifie par comptage de mots du fichier avant d'ecrire
ce module).

Technique de propagation (voir errfdz du Fortran) : chaque composante
cartesienne (X,Y,Z) est LINEAIRE en les coefficients de Gauss, donc sa
derivee partielle par rapport a un coefficient g_k est simplement la
synthese du champ avec un vecteur "impulsion unite" (tout a zero sauf
g_k=1) - calculee ici en appelant DIRECTEMENT `pmag.magsyn` (deja
verifie/utilise ailleurs dans ce projet pour GUFM1) plutot que de porter
a la main la formule des derivees spheriques-harmoniques de errfdz (qui
recalcule les memes derivees via les fonctions de Legendre associees -
plus de risque d'erreur d'indexation pour un gain nul, magsyn EST deja
cette synthese, deja verifiee). L'incertitude quadratique
sigma_X = sqrt(sum_k (dg_k * dX/dg_k)^2) est ensuite combinee en
dD/dI/dF exactement comme errfdz (memes 4 formules scalaires,
fielduncert3k.f:128-131, reprises telles quelles - pure algebre de
propagation d'erreur, pas de nouvelle geometrie)."""

import os
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import BSpline

from pmagpy import pmag

_FIELD_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "field_models")

# mod (voir stereo_pmagpy._FIELD_MODEL_SPECS) -> fichier d'incertitude
# bundle avec l'appli - voir docstring de module pour ce qui a ete
# verifie/exclu.
_UNCERTAINTY_MODEL_FILES = {
    "cals3k": os.path.join(_FIELD_MODELS_DIR, "CALS3k.4b"),
    "cals10k": os.path.join(_FIELD_MODELS_DIR, "CALS10k.1b"),
}


class _BSplineModel:
    __slots__ = ("tstart", "tend", "lmax", "nspl", "tknts", "gt", "dgt")

    def __init__(self, tstart, tend, lmax, nspl, tknts, gt, dgt):
        self.tstart, self.tend = tstart, tend
        self.lmax, self.nspl = lmax, nspl
        self.tknts, self.gt, self.dgt = tknts, gt, dgt


@lru_cache(maxsize=None)
def _load_model(path: str) -> _BSplineModel:
    """Lit un fichier modele B-spline avec bloc d'incertitude (voir
    docstring de module pour le format exact et le piege colonne-majeur
    Fortran). Mis en cache - un seul chargement par fichier et par
    session, le fichier ne change jamais en cours d'execution."""
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        rest = f.read()
    tstart, tend = (float(x) for x in first_line.split()[:2])
    tokens = rest.split()
    lmax, _nm, nspl = int(tokens[0]), int(tokens[1]), int(tokens[2])
    pos = 3
    tknts = np.array(tokens[pos:pos + nspl + 4], dtype=float)
    pos += nspl + 4
    n = lmax * (lmax + 2)
    gt = np.array(tokens[pos:pos + n * nspl], dtype=float).reshape(nspl, n).T
    pos += n * nspl
    dgt = np.array(tokens[pos:pos + n * nspl], dtype=float).reshape(nspl, n).T
    return _BSplineModel(tstart, tend, lmax, nspl, tknts, gt, dgt)


def _eval_coeffs(date: float, arr: np.ndarray, tknts: np.ndarray) -> Optional[np.ndarray]:
    """g(date) ou dg(date) - None si `date` est hors de la plage REELLE
    couverte par les noeuds de spline (pas seulement tstart/tend du
    fichier, qui laisse une marge de part et d'autre - voir `interv` du
    Fortran, qui retourne silencieusement sans rien faire hors de
    [tknts(4), tknts(nspl+1)], laissant alors le champ a zero. Remplace
    ici par un None explicite plutot que de laisser deviner un champ
    nul."""
    n = arr.shape[0]
    out = np.empty(n)
    for k in range(n):
        spl = BSpline(tknts, arr[k], 3, extrapolate=False)
        v = spl(date)
        if np.isnan(v):
            return None
        out[k] = v
    return out


def has_uncertainty_model(mod: str) -> bool:
    path = _UNCERTAINTY_MODEL_FILES.get(mod)
    return path is not None and os.path.isfile(path)


def field_uncertainty_at(
    lat: float, lon: float, alt_km: float, date: float, mod: str,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """(dec, inc, f_nT, ddec, dinc, df_nT) a (lat,lon,alt_km,date) pour
    `mod` - None si ce modele n'a pas d'incertitude disponible localement
    (voir _UNCERTAINTY_MODEL_FILES) ou si `date` est hors de la plage
    reellement couverte. f_nT/df_nT en nT (meme convention que
    pmag.magsyn/ipmag.igrf - la conversion en microTesla est a la charge
    de l'appelant, voir stereo_pmagpy.predicted_field_curve)."""
    path = _UNCERTAINTY_MODEL_FILES.get(mod)
    if path is None or not os.path.isfile(path):
        return None
    model = _load_model(path)
    g = _eval_coeffs(date, model.gt, model.tknts)
    if g is None:
        return None
    dg = _eval_coeffs(date, model.dgt, model.tknts)
    n = len(g)
    colat = 90.0 - lat
    lon360 = lon % 360.0
    zero_sv = [0.0] * n

    x, y, z, f = pmag.magsyn(list(g), zero_sv, date, date, 1, alt_km, colat, lon360)
    h = (x ** 2 + y ** 2) ** 0.5
    dec = float(np.degrees(np.arctan2(y, x)) % 360.0)
    inc = float(np.degrees(np.arctan2(z, h)))

    # d(X,Y,Z)/dg_k = synthese avec un vecteur "impulsion unite" (voir
    # docstring de module) - X,Y,Z sont lineaires en g, c'est exactement
    # la derivee partielle recherchee, sans nouvelle formule.
    ex2 = ey2 = ez2 = 0.0
    for k in range(n):
        e = [0.0] * n
        e[k] = 1.0
        xk, yk, zk, _fk = pmag.magsyn(e, zero_sv, date, date, 1, alt_km, colat, lon360)
        dgk = dg[k]
        ex2 += (dgk * xk) ** 2
        ey2 += (dgk * yk) ** 2
        ez2 += (dgk * zk) ** 2
    ex, ey, ez = ex2 ** 0.5, ey2 ** 0.5, ez2 ** 0.5

    # memes 4 formules scalaires que errfdz/fielduncert3k.f:128-131 -
    # pure algebre de propagation d'erreur, aucune nouvelle geometrie.
    eh = ((1 / h ** 2) * ((x * ex) ** 2 + (y * ey) ** 2)) ** 0.5
    ef = ((1 / f ** 2) * ((x * ex) ** 2 + (y * ey) ** 2 + (z * ez) ** 2)) ** 0.5
    ei = ((1 / (1 + (z / h) ** 2)) ** 2 * ((ez / h) ** 2 + ((z * eh) / h ** 2) ** 2)) ** 0.5
    ed = ((1 / (1 + (y / x) ** 2)) ** 2 * ((ey / x) ** 2 + ((y * ex) / x ** 2) ** 2)) ** 0.5

    return dec, inc, float(f), float(np.degrees(ed)), float(np.degrees(ei)), float(ef)
