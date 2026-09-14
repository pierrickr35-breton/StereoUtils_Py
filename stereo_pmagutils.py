"""
Menu "Pmag Utilities" (pmagoutils.f + calcrota.f), hors `drillcore` et
`polarity` ("Echelle polarite") - demande explicite de l'utilisateur.
Couvre : VGP conversions (VGPC1-4), tests de rotation/aplatissement
(PALEOTEC/PALEODEC/PALEOTEC1), paleolatitude (paleolati/paleolati2),
rotation vers GMT (ROTA2GMT), corrections core/bedding (CORE1/CORE2/
invbedding), fold plunge (FOLDPLUNGE/FOLDFILE), rotation Euler (ROTMAG/
vgp_plate_rotation), paleointensite (MEANPAL/VIDIMO/relocate/relocatevar),
IGRF (igrfOSX, via le paquet `ppigrf`), conversion d'unites (convunit),
overprint/flatten.

Bugs Fortran confirmes dans ce sous-ensemble, REPLIQUES (pas silencieusement
"corriges", sauf mention contraire) - voir memoire projet
`project_stereoutils_pmagoutils_bugs.md` :
- `dia95toplat` (utilisee par `rota2gmt`) double-negative le signe de la
  paleolatitude min/max pour les inclinaisons negatives (contrairement a
  `paleolati2`, qui laisse la meme ligne commentee - verifie sur le
  source). Replique tel quel.
- `PALEODEC` imprime DELI_OBS/DELI_EXP jamais assignes - variables
  simplement omises de l'affichage ici plutot que d'inventer des zeros
  trompeurs.
- `relocatevar` : `svadm` utilisait des variables jamais assignees
  (`vdamup`/`vdamdown` au lieu de `vadmup`/`vadmdown`) - replique comme
  toujours-zero (comportement observe typique d'une variable Fortran
  locale non initialisee).
- `flatten` : la garde de validation de `uninc` teste en realite une
  variable homonyme jamais assignee (`uinc`) - repliquee comme garde
  totalement inope rante (n'importe quelle valeur passe).
- Le "sign-extraction idiom" `(ABS(x)/x)*(...)` de VGPC3/VGPC4 (division
  par zero si x==0 exactement) est remplace par `math.copysign` pour
  eviter un crash reel sur une entree valide (ex. "0 30" de longitude),
  pas une correction du resultat, juste une garde anti-crash.

Perimetre volontairement reduit (fonctionnalites qui dependent de fichiers
de reference externes non fournis avec ce depot, ou de formats de fichier
proprietaires non documentables sans exemple) :
- `paleolati` : seules les options "votre propre fichier APWP" sont
  portees (pas les presets Torsvik/Eurasie codes en dur, fichiers CSV non
  presents dans ce depot).
- `MEANPAL` : seuls les modes fichier simple (F,Q,N) et saisie clavier
  sont portes (pas le mode "fichier Starmac" multi-experiences avec
  colonnes de correction d'anisotropie/refroidissement, format non
  documente sans exemple).
- `FOLDFILE` mode "progressive" (qui relit un nom de fichier CODE EN DUR
  `dataplunge.txt` 100 fois) n'est pas porte ; le mode fichier simple
  l'est.
- `relocatevar` : seule la table de resultats combinee est produite (pas
  les 6 fichiers de sortie separes au format GMT .vec/.pie/.site/.plat/
  .lat, destines a un pipeline de trace externe non disponible ici).
- `rota2gmt` : rapport de resultats + .vec/.pie portes (colonnes .res),
  pas les fichiers .site/.plat/.lat (sortie GMT annexe).
- `VGPROT` (implementation VGP<->direction dediee a `rota2gmt`) n'est PAS
  reprise separement - `rota2gmt` reutilise le meme `dodi_vgp`/`vgp_di`
  que le reste du port (transformation textbook standard, les deux
  implementations calculent la meme chose)."""

import math
from typing import List, Optional, Sequence, Tuple

from stereo_geometry import _angle, corpen, corfor, polere
from stereo_selection import read_text_lines, split_header

RAD = math.radians
DEG = math.degrees
F2PI = math.pi / 180.0


# ---------------------------------------------------------------------------
# VGP <-> direction (dodi_vgp/vgp_di, pmagoutils.f:4001-4074)
# ---------------------------------------------------------------------------

def dodi_vgp(dec: float, dip: float, slat: float, slong: float) -> Tuple[float, float]:
    """Port de `dodi_vgp` : direction (dec,dip) mesuree au site (slat,slong)
    -> pole (plat,plong), tout en degres."""
    dec_r, dip_r, slat_r, slong_r = RAD(dec), RAD(dip), RAD(slat), RAD(slong)
    p = math.atan2(2.0, math.tan(dip_r))
    plat = math.asin(math.sin(slat_r) * math.cos(p) + math.cos(slat_r) * math.sin(p) * math.cos(dec_r))
    beta = (math.sin(p) * math.sin(dec_r)) / math.cos(plat)
    beta = max(-1.0, min(1.0, beta))
    beta = math.asin(beta)
    if math.cos(p) >= math.sin(slat_r) * math.sin(plat):
        plong = slong_r + beta
    else:
        plong = slong_r + math.pi - beta
    plong %= 2.0 * math.pi
    return DEG(plat), DEG(plong)


def vgp_di(plat: float, plong: float, slat: float, slong: float) -> Tuple[float, float]:
    """Port de `vgp_di` : pole (plat,plong) -> direction attendue (dec,dip)
    au site (slat,slong), tout en degres."""
    if slong <= 0.0:
        slong += 360.0
    delphi = abs(plong - slong)
    signdec = 1.0 if delphi == 0.0 else (plong - slong) / delphi
    theta_s = RAD(90.0 - slat)
    theta_p = RAD(90.0 - plat)
    delphi_r = RAD(delphi)
    cosp = math.cos(theta_s) * math.cos(theta_p) + math.sin(theta_s) * math.sin(theta_p) * math.cos(delphi_r)
    cosp = max(-1.0, min(1.0, cosp))
    theta_m = math.acos(cosp)
    denom = math.sin(theta_m) * math.sin(theta_s)
    cosd = (math.cos(theta_p) - math.cos(theta_m) * math.cos(theta_s)) / denom if denom else 0.0
    c = abs(1.0 - cosd ** 2)
    if c != 0.0:
        dec = -math.atan(cosd / math.sqrt(abs(c))) + math.pi / 2.0
    else:
        dec = math.acos(max(-1.0, min(1.0, cosd)))
    if -math.pi < signdec * delphi_r and signdec < 0.0:
        dec = 2.0 * math.pi - dec
    if signdec * delphi_r > math.pi:
        dec = 2.0 * math.pi - dec
    dec = DEG(dec) % 360.0
    dip = DEG(math.atan2(2.0 * math.cos(theta_m), math.sin(theta_m)))
    return dec, dip


def _dp_dm_dir_to_vgp(inc: float, a95: float) -> Tuple[float, float]:
    """DM,DP (ovale de confiance du VGP) a partir d'une direction+a95 -
    formule de VGPC1/VGPC3 (pmagoutils.f:1758-1761)."""
    pl = 1.57095 - math.atan(0.5 * math.tan(RAD(inc)))
    dm = a95 * math.sin(pl) / math.cos(RAD(inc))
    dp = 2.0 * a95 / (1.0 + 3.0 * math.cos(RAD(inc)) ** 2)
    return dm, dp


def _dd_di_vgp_to_dir(vgp_inc: float, a95: float) -> Tuple[float, float]:
    """dD,dI (incertitude direction) a partir d'un VGP+a95 - formule de
    VGPC2/VGPC4 (pmagoutils.f:1850-1853), transformation INVERSE de
    `_dp_dm_dir_to_vgp`, pas la meme formule."""
    pl = 1.57095 - math.atan(0.5 * math.tan(RAD(vgp_inc)))
    s = max(-1.0, min(1.0, math.sin(RAD(a95)) / math.sin(pl))) if math.sin(pl) else 0.0
    dd = DEG(math.asin(s))
    di = a95 * (2.0 / (1.0 + 3.0 * math.cos(pl) ** 2))
    return dd, di


def vgpc1(dec: float, inc: float, site_lat: float, site_lon: float, a95: float = 0.0) -> dict:
    """Direction to VGP (VGPC1/VGPCAL(1))."""
    plat, plong = dodi_vgp(dec, inc, site_lat, site_lon)
    out = {"vgp_lat": plat, "vgp_lon": plong}
    if a95 > 0.0:
        dm, dp = _dp_dm_dir_to_vgp(inc, a95)
        out["dm"], out["dp"] = dm, dp
    return out


def vgpc2(vgp_lat: float, vgp_lon: float, site_lat: float, site_lon: float, a95: float = 0.0) -> dict:
    """VGP to direction (VGPC2/VGPCAL(2))."""
    dec, dip = vgp_di(vgp_lat, vgp_lon, site_lat, site_lon)
    out = {"dec": dec, "inc": dip}
    if a95 > 0.0:
        dd, di = _dd_di_vgp_to_dir(dip, a95)
        out["dd"], out["di"] = dd, di
    return out


def _signed_dms(deg: float, mn: float, sec: float = 0.0) -> float:
    """Combine degre/minute/seconde en decimal, en respectant le signe du
    champ "degre" (garde anti-division-par-zero du "sign idiom" original
    (ABS(x)/x)*(...) de VGPC3/VGPC4 - meme resultat, pas de crash si
    `deg==0`)."""
    sign = math.copysign(1.0, deg) if deg != 0 else 1.0
    return sign * (abs(deg) + mn / 60.0 + sec / 3600.0)


def vgpc3_batch(rows: Sequence[Sequence[float]], coord_mode: int = 0) -> List[dict]:
    """File: Direction to VGP (VGPC3). `rows` : chaque ligne deja
    parsee en valeurs numeriques ; `coord_mode` 0=deg, 1=deg+min,
    2=deg+min+sec (site lat/lon), toujours suivi de dec,inc,[a95]."""
    out = []
    for row in rows:
        row = list(row)
        if coord_mode == 0:
            slat, slon, dec, inc = row[0], row[1], row[2], row[3]
            a95 = row[4] if len(row) > 4 else 0.0
        elif coord_mode == 1:
            slat = _signed_dms(row[0], row[1])
            slon = _signed_dms(row[2], row[3])
            dec, inc = row[4], row[5]
            a95 = row[6] if len(row) > 6 else 0.0
        else:
            slat = _signed_dms(row[0], row[1], row[2])
            slon = _signed_dms(row[3], row[4], row[5])
            dec, inc = row[6], row[7]
            a95 = row[8] if len(row) > 8 else 0.0
        res = vgpc1(dec, inc, slat, slon, a95)
        res.update({"site_lat": slat, "site_lon": slon, "dec": dec, "inc": inc, "a95": a95})
        out.append(res)
    return out


def vgpc4_batch(rows: Sequence[Sequence[float]], coord_mode: int = 0, vgp_lat_first: bool = True) -> List[dict]:
    """File: VGP to direction (VGPC4). `vgp_lat_first=False` : le fichier
    donne le VGP en (lon,lat) plutot que (lat,lon), meme option "ILAT"
    que le source."""
    out = []
    for row in rows:
        row = list(row)
        if coord_mode == 0:
            slat, slon = row[0], row[1]
            a, b = row[2], row[3]
            a95 = row[4] if len(row) > 4 else 0.0
        elif coord_mode == 1:
            slat = _signed_dms(row[0], row[1])
            slon = _signed_dms(row[2], row[3])
            a, b = row[4], row[5]
            a95 = row[6] if len(row) > 6 else 0.0
        else:
            slat = _signed_dms(row[0], row[1], row[2])
            slon = _signed_dms(row[3], row[4], row[5])
            a, b = row[6], row[7]
            a95 = row[8] if len(row) > 8 else 0.0
        vgp_lat, vgp_lon = (a, b) if vgp_lat_first else (b, a)
        res = vgpc2(vgp_lat, vgp_lon, slat, slon, a95)
        res.update({"site_lat": slat, "site_lon": slon, "vgp_lat": vgp_lat, "vgp_lon": vgp_lon, "a95": a95})
        out.append(res)
    return out


# ---------------------------------------------------------------------------
# Rotation/aplatissement (PALEOTEC/PALEODEC/PALEOTEC1)
# ---------------------------------------------------------------------------

def paleotec(
    dec_obs: float, inc_obs: float, a95: float,
    site_lat: float, site_lon: float, pole_lat: float, pole_lon: float, p95: float,
) -> dict:
    """Port de `PALEOTEC` (Rotation obs=DI Pole Ref)."""
    dec_exp, inc_exp = vgp_di(pole_lat, pole_lon, site_lat, site_lon)
    rotation = dec_obs - dec_exp
    rflat = inc_exp - inc_obs
    if a95 == 0.0:
        deld_obs = 0.0
    else:
        s = math.sin(RAD(a95)) / math.cos(RAD(inc_obs))
        deld_obs = 90.0 if s > 0.9999 else DEG(math.asin(s))
    if p95 == 0.0:
        deld_exp = 0.0
    else:
        pl = 1.5707 - math.atan(math.tan(RAD(inc_exp)) / 2.0)
        deld_exp = DEG(math.asin(math.sin(RAD(p95)) / math.sin(pl)))
    delt_rotation = 0.0 if (deld_obs == 0.0 and deld_exp == 0.0) else 0.8 * math.hypot(deld_obs, deld_exp)
    deli_obs = a95
    deli_exp = 0.0 if p95 == 0.0 else 2.0 * p95 / (1.0 + 3.0 * math.cos(1.5707 - math.atan(math.tan(RAD(inc_exp)) / 2.0)) ** 2)
    delt_flat = 0.0 if (deli_obs == 0.0 and deli_exp == 0.0) else 0.8 * math.hypot(deli_obs, deli_exp)
    return {
        "dec_exp": dec_exp, "inc_exp": inc_exp, "rotation": rotation, "flattening": rflat,
        "deltad_obs": deld_obs, "deltad_exp": deld_exp, "delta_rotation": delt_rotation,
        "deltai_obs": deli_obs, "deltai_exp": deli_exp, "delta_flattening": delt_flat,
    }


def paleotec1(
    obs_lat: float, obs_lon: float, obs95: float,
    site_lat: float, site_lon: float,
    ref_lat: float, ref_lon: float, ref95: float,
) -> dict:
    """Port de `PALEOTEC1` (Rotation obs=Pole Pole Ref)."""
    latref, longref, ref95r = RAD(ref_lat), RAD(ref_lon), RAD(ref95)
    latobs, longobs, obs95r = RAD(obs_lat), RAD(obs_lon), RAD(obs95)
    latsite, longsite = RAD(site_lat), RAD(site_lon)

    def _gc_dist(lat1, lon1, lat2, lon2):
        return math.acos(max(-1.0, min(1.0, math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2))))

    pr = _gc_dist(latref, longref, latsite, longsite)
    po = _gc_dist(latobs, longobs, latsite, longsite)

    dec_exp, inc_exp = vgp_di(ref_lat, ref_lon, site_lat, site_lon)
    dec_obs, inc_obs = vgp_di(obs_lat, obs_lon, site_lat, site_lon)
    r = dec_obs - dec_exp
    rerror = inc_obs - inc_exp
    p = po - pr
    delr = 0.8 * math.hypot(math.asin(min(1.0, math.sin(ref95r) / math.sin(pr))), math.asin(min(1.0, math.sin(obs95r) / math.sin(po))))
    delp = 0.8 * math.hypot(obs95, ref95)
    return {"rotation": r, "delta_rotation": DEG(delr), "inc_error": rerror,
            "pole_displacement": DEG(p), "delta_pole_displacement": DEG(delp),
            "dec_exp": dec_exp, "inc_exp": inc_exp, "dec_obs": dec_obs, "inc_obs": inc_obs}


def paleodec(dec_obs: float, inc_obs: float, a95: float, dec_exp: float, inc_exp: float, b95: float) -> dict:
    """Port de `PALEODEC` (Rotation obs=DI DI Ref) - le source imprime
    DELI_OBS/DELI_EXP sans jamais les assigner (bug reel, omis ici plutot
    que d'afficher un zero trompeur ; DELTA_FLATTENING lui-meme est
    calcule correctement directement a partir de a95/b95, pas affecte)."""
    rotation = dec_obs - dec_exp
    rflat = inc_exp - inc_obs
    deld_obs = DEG(math.asin(math.sin(RAD(a95)) / math.cos(RAD(inc_obs))))
    deld_exp = DEG(math.asin(math.sin(RAD(b95)) / math.cos(RAD(inc_exp))))
    delt_rotation = 0.8 * math.hypot(deld_obs, deld_exp)
    delt_flat = 0.8 * math.hypot(a95, b95)
    return {"rotation": rotation, "flattening": rflat,
            "deltad_obs": deld_obs, "deltad_exp": deld_exp, "delta_rotation": delt_rotation,
            "delta_flattening": delt_flat}


# ---------------------------------------------------------------------------
# Paleolatitude (paleolati2 / dia95toplat / paleolati depuis fichier)
# ---------------------------------------------------------------------------

def paleolati2(inc: float, a95: float) -> dict:
    """Port de `paleolati2` (Inclination a95 to Plat and err) : formule
    dipolaire `atan(tan(I)/2)`, sans double-negation (les lignes
    `if(at<0) tt=-1*tt` sont commentees dans le source)."""
    rmin, rmax = inc - a95, inc + a95
    lat = DEG(math.atan(math.tan(RAD(inc)) / 2.0))
    latmin = DEG(math.atan(math.tan(RAD(rmin)) / 2.0))
    latmax = DEG(math.atan(math.tan(RAD(rmax)) / 2.0))
    return {"paleolatitude": lat, "min": latmin, "max": latmax}


def dia95toplat(inc: float, di_min: float, di_max: float) -> dict:
    """Port de `dia95toplat` (utilisee par `rota2gmt`) - CONTRAIREMENT a
    `paleolati2`, les lignes `if(at<0) tt=-1*tt` sont ACTIVES ici : pour
    une inclinaison negative, le resultat deja correctement signe
    (`atan(tan(x)/2)` est une fonction impaire) est negate une seconde
    fois. Bug confirme, REPLIQUE tel quel (voir memoire projet, bug #8)."""
    def _tt(x):
        at = RAD(x)
        t = DEG(math.atan(math.tan(at) / 2.0))
        if at < 0.0:
            t = -1.0 * t
        return t

    latmoy = _tt(inc)
    latmin = latmoy - _tt(inc - di_min)
    latmax = _tt(inc + di_max) - latmoy
    return {"paleolatitude": latmoy, "min_err": latmin, "max_err": latmax}


def paleolati_from_file(
    path: str, site_lat: float, site_lon: float, polarity: int = 1,
) -> Tuple[str, List[dict]]:
    """Port de `paleolati` case(0)/case(9) (Paleolatitude at one site) -
    seul le mode "votre propre fichier APWP" est porte (pas les presets
    Torsvik/Eurasie codes en dur, fichiers de reference non fournis avec
    ce depot). Format attendu : age, vgp_lat, vgp_lon, [a95] par ligne -
    colonnes reperees par en-tete si present (ex. "#age vgplat vgplon
    a95"), sinon par position (age,vgp_lat,vgp_lon,[a95]) comme avant."""
    rows = []
    lines = read_text_lines(path)
    data_lines, idx = split_header(lines, "age", "vgplat", "vgplon", "a95")
    if "age" in idx and "vgplat" in idx and "vgplon" in idx:
        i_age, i_lat, i_lon, i_a95 = idx["age"], idx["vgplat"], idx["vgplon"], idx.get("a95")
    else:
        i_age, i_lat, i_lon, i_a95 = 0, 1, 2, 3
    for line in data_lines:
        line = line.strip()
        if not line or line.startswith("!") or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) <= max(i_age, i_lat, i_lon):
            continue
        try:
            age = float(parts[i_age])
            vlat, vlon = float(parts[i_lat]), float(parts[i_lon])
            a95 = float(parts[i_a95]) if i_a95 is not None and len(parts) > i_a95 else 0.0
        except ValueError:
            continue
        if polarity == -1:
            vlat, vlon = -vlat, vlon + 180.0
        vlon %= 360.0
        dec, dip = vgp_di(vlat, vlon, site_lat, site_lon)
        if dec > 180.0:
            dec -= 360.0
        row = {"age": age, "vgp_lat": vlat, "vgp_lon": vlon, "a95": a95, "dec": dec, "inc": dip}
        if a95 > 0.0:
            dd, dp = _dd_di_vgp_to_dir(dip, a95)
            plat = math.atan(0.5 * math.tan(RAD(dip)))
            plat_lo = math.atan(0.5 * math.tan(RAD(dip - dp)))
            plat_hi = math.atan(0.5 * math.tan(RAD(dip + dp)))
            row.update({"dp": dp, "paleolat": DEG(plat), "paleolat_min": DEG(plat_lo), "paleolat_max": DEG(plat_hi)})
        else:
            row["paleolat"] = DEG(math.atan(0.5 * math.tan(RAD(dip))))
        rows.append(row)
    lines = ["age  vlat   vlon    a95    dec    dip    DP   plat   plat_min  plat_max"]
    for r in rows:
        if "dp" in r:
            lines.append(
                f"{r['age']:5.0f} {r['vgp_lat']:6.1f} {r['vgp_lon']:6.1f} {r['a95']:5.1f} "
                f"{r['dec']:6.1f} {r['inc']:6.1f} {r['dp']:6.1f} {r['paleolat']:6.1f} "
                f"{r['paleolat_min']:6.1f} {r['paleolat_max']:6.1f}"
            )
        else:
            lines.append(
                f"{r['age']:5.0f} {r['vgp_lat']:6.1f} {r['vgp_lon']:6.1f} {r['a95']:5.1f} "
                f"{r['dec']:6.1f} {r['inc']:6.1f}   -   {r['paleolat']:6.1f}"
            )
    return "\n".join(lines) + "\n", rows


# ---------------------------------------------------------------------------
# Core / Bedding corrections (CORE1/CORE2/invbedding)
# ---------------------------------------------------------------------------

def core_forage_correction(dip_azimuth: float, strike_dip_axis: float, points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Port de `CORE(1)` (Core correction, via `CORFOR`) : `points` =
    (dec,inc) mesurees dans le repere de la carotte, corrigees vers le
    repere in-situ. Args nommes `(dip_azimuth, strike_dip_axis)` = (G,H)
    du source, dans cet ordre d'appel (`CORFOR(RG,RH)`)."""
    out = []
    for dec, inc in points:
        z = math.sin(RAD(inc))
        x = math.cos(RAD(inc)) * math.cos(RAD(dec))
        y = math.cos(RAD(inc)) * math.sin(RAD(dec))
        xx, yy, zz = corfor(x, y, z, dip_azimuth, strike_dip_axis)
        _r, dec2, inc2 = polere(xx, yy, zz)
        out.append((dec2, inc2))
    return out


def core_bedding_correction(strike: float, dip: float, points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Port de `CORE(2)` (Bedding correction, via `CORPEN`)."""
    out = []
    for dec, inc in points:
        z = math.sin(RAD(inc))
        x = math.cos(RAD(inc)) * math.cos(RAD(dec))
        y = math.cos(RAD(inc)) * math.sin(RAD(dec))
        xx, yy, zz = corpen(x, y, z, dip, strike)
        _r, dec2, inc2 = polere(xx, yy, zz)
        out.append((dec2, inc2))
    return out


def invert_bedding(d1: float, i1: float, d2: float, i2: float) -> Optional[Tuple[float, float]]:
    """Port de `invbedding` (inverseBedding cor) : retrouve (strike,dip) a
    partir d'une direction in-situ (d1,i1) et sa correction (d2,i2) -
    balaie le strike par pas de 0.1 deg (0-360) jusqu'a trouver la
    solution equidistante des deux points, resout le dip, leve
    l'ambiguite 180 deg en verifiant laquelle des deux reproduit
    effectivement point 2 a partir de point 1. Retourne None si aucune
    solution trouvee."""
    for i in range(1, 3601):
        d3 = i * 0.1
        del1, _a = _angle(d1 - d3, 90.0 - i1, 90.0)
        del2, _a = _angle(d2 - d3, 90.0 - i2, 90.0)
        if abs(abs(del1) - abs(del2)) < 0.2:
            ang1 = abs(del1)
            dd3, _a = _angle(d1 - d2, 90.0 - i1, 90.0 - i2)
            tt = (math.cos(RAD(dd3)) - math.cos(RAD(ang1)) ** 2) / math.sin(RAD(ang1)) ** 2
            tt = max(-1.0, min(1.0, tt))
            alpha = DEG(math.acos(tt))
            z = math.sin(RAD(i1))
            x = math.cos(RAD(i1)) * math.cos(RAD(d1))
            y = math.cos(RAD(i1)) * math.sin(RAD(d1))
            xx, yy, zz = corpen(x, y, z, alpha, d3)
            _r, yy2, zz2 = polere(xx, yy, zz)
            check, _a = _angle(d2 - yy2, 90.0 - i2, 90.0 - zz2)
            strike = d3 + 180.0 if abs(check) >= 1.0 else d3
            return strike % 360.0, alpha
    return None


# ---------------------------------------------------------------------------
# Fold plunge (FOLDPLUNGE/FOLDFILE/progressive)
# ---------------------------------------------------------------------------

def fold_plunge_single(
    strike: float, dip: float, pdec: float, princ: float, fold_azimuth: float, fold_dip: float,
) -> dict:
    """Port de `FOLDPLUNGE` mode 1 (Fold plunge, correction en une etape) :
    corrige d'abord la direction pour le plongement de pli, PUIS re-corrige
    avec l'attitude de litage ainsi mise a jour (deux etapes, meme
    pipeline que le source, lignes 1274-1312)."""
    rk, rj = fold_azimuth - 90.0, fold_dip
    rinc, dec = 90.0 - dip, strike - 90.0
    z, x, y = math.sin(RAD(rinc)), math.cos(RAD(rinc)) * math.cos(RAD(dec)), math.cos(RAD(rinc)) * math.sin(RAD(dec))
    xx, yy, zz = corpen(x, y, z, rj, rk)
    _r, yy2, zz2 = polere(xx, yy, zz)
    new_strike, new_dip = yy2 + 90.0, 90.0 - zz2

    z, x, y = math.sin(RAD(princ)), math.cos(RAD(princ)) * math.cos(RAD(pdec)), math.cos(RAD(princ)) * math.sin(RAD(pdec))
    xx, yy, zz = corpen(x, y, z, rj, rk)
    _r, first_dec, first_inc = polere(xx, yy, zz)

    z, x, y = math.sin(RAD(first_inc)), math.cos(RAD(first_inc)) * math.cos(RAD(first_dec)), math.cos(RAD(first_inc)) * math.sin(RAD(first_dec))
    xx, yy, zz = corpen(x, y, z, new_dip, new_strike)
    _r, final_dec, final_inc = polere(xx, yy, zz)
    return {
        "new_strike": new_strike, "new_dip": new_dip,
        "first_correction": (first_dec, first_inc),
        "final_correction": (final_dec, final_inc),
    }


def _progressive_step(
    pl: float, pb: float, pdec: float, princ: float, srk: float, srj: float, plaz: float, pldip: float,
) -> Tuple[float, float, float, float, float, float]:
    """Port de `progressive` (pmagoutils.f:1593-1665) : un pas de
    correction couplee plongement (`pl` fraction) + litage (`pb`
    fraction), retourne l'etat mis a jour (pdec,princ,srk,srj,plaz,pldip)."""
    rk, rj = plaz - 90.0, pl * pldip
    pldip = (1.0 - pl) * pldip
    rinc, dec = 90.0 - srj, srk - 90.0
    z, x, y = math.sin(RAD(rinc)), math.cos(RAD(rinc)) * math.cos(RAD(dec)), math.cos(RAD(rinc)) * math.sin(RAD(dec))
    xx, yy, zz = corpen(x, y, z, rj, rk)
    _r, yy2, zz2 = polere(xx, yy, zz)
    srk, srj = yy2 + 90.0, 90.0 - zz2

    z, x, y = math.sin(RAD(princ)), math.cos(RAD(princ)) * math.cos(RAD(pdec)), math.cos(RAD(princ)) * math.sin(RAD(pdec))
    xx, yy, zz = corpen(x, y, z, rj, rk)
    _r, pdec, princ = polere(xx, yy, zz)

    rk, rj = srk, pb * srj
    srj = (1.0 - pb) * srj
    z, x, y = math.sin(RAD(pldip)), math.cos(RAD(pldip)) * math.cos(RAD(plaz)), math.cos(RAD(pldip)) * math.sin(RAD(plaz))
    xx, yy, zz = corpen(x, y, z, rj, rk)
    _r, plaz, pldip = polere(xx, yy, zz)

    z, x, y = math.sin(RAD(princ)), math.cos(RAD(princ)) * math.cos(RAD(pdec)), math.cos(RAD(princ)) * math.sin(RAD(pdec))
    xx, yy, zz = corpen(x, y, z, rj, rk)
    _r, pdec, princ = polere(xx, yy, zz)
    return pdec, princ, srk, srj, plaz, pldip


def fold_plunge_progressive(
    strike: float, dip: float, pdec: float, princ: float, fold_azimuth: float, fold_dip: float,
    pl: float = 0.01, pb: float = 0.01,
) -> dict:
    """Port de `FOLDPLUNGE` mode 2 (progressive) : applique `_progressive_step`
    par petits pas (`pl`/`pb`, defaut 1%) jusqu'a ce que le pendage
    residuel `srj` tombe sous 0.1 deg ou 2000 iterations (meme borne que
    le source)."""
    srk, srj = strike, dip
    plaz, pldip = fold_azimuth, fold_dip
    ir = 0
    while srj > 0.1 and ir < 2000:
        pdec, princ, srk, srj, plaz, pldip = _progressive_step(pl, pb, pdec, princ, srk, srj, plaz, pldip)
        ir += 1
    if srj > 0.1:
        z, x, y = math.sin(RAD(pldip)), math.cos(RAD(pldip)) * math.cos(RAD(plaz)), math.cos(RAD(pldip)) * math.sin(RAD(plaz))
        xx, yy, zz = corpen(x, y, z, srj, srk)
        _r, plaz, pldip = polere(xx, yy, zz)
        z, x, y = math.sin(RAD(princ)), math.cos(RAD(princ)) * math.cos(RAD(pdec)), math.cos(RAD(princ)) * math.sin(RAD(pdec))
        xx, yy, zz = corpen(x, y, z, srj, srk)
        _r, pdec, princ = polere(xx, yy, zz)
    return {"iterations": ir, "dec": pdec, "inc": princ, "fold_azimuth": plaz, "fold_dip": pldip}


def fold_file_single(rows: Sequence[Tuple[str, float, float, float, float, float, float]]) -> Tuple[str, List[Tuple[float, float, str]]]:
    """Port de `FOLDFILE` mode 1 (File Fold plunge) : `rows` = (site,dec,
    inc,a95,strike,dip,fold_azimuth,fold_dip) par ligne (7 valeurs + site).
    Meme pipeline en 2 etapes que `fold_plunge_single`, plus la
    bedding-seule (BDEC/BRINC) et l'intermediaire (CDEC/CRINC) affiches
    par le source."""
    lines = ["site       dec_final  inc_final  a95   bed_dec  bed_inc  new_strike new_dip"]
    new_directions = []
    for site, dec, inc, a95, strike, dip, fold_az, fold_dip in rows:
        z, x, y = math.sin(RAD(inc)), math.cos(RAD(inc)) * math.cos(RAD(dec)), math.cos(RAD(inc)) * math.sin(RAD(dec))
        xx, yy, zz = corpen(x, y, z, dip, strike)
        _r, bdec, brinc = polere(xx, yy, zz)

        res = fold_plunge_single(strike, dip, dec, inc, fold_az, fold_dip)
        fdec, finc = res["final_correction"]
        lines.append(
            f"{site:<10s} {fdec:8.1f} {finc:8.1f} {a95:6.1f} {bdec:8.1f} {brinc:8.1f} "
            f"{res['new_strike']:8.1f} {res['new_dip']:7.1f}"
        )
        new_directions.append((fdec, finc, "c"))
    return "\n".join(lines) + "\n", new_directions


# ---------------------------------------------------------------------------
# Rotation Euler (ROTA/ROTAIM/ROTMAG/vgp_plate_rotation)
# ---------------------------------------------------------------------------

def rota(pole_lon: float, pole_lat: float, angle: float, lat1: float, lon1: float) -> Tuple[float, float]:
    """Equivalent de `ROTA(S,D,R,L1,G1,L2,G2)` (pmagoutils.f:1015-1089) :
    fait tourner le point (lat1,lon1) de `angle` deg autour du pole
    (pole_lat,pole_lon), sens standard (positif = vers l'est vu depuis le
    pole de rotation, verifie contre le source pour une rotation autour du
    pole nord). Implemente via la formule de Rodrigues (rotation 3D autour
    d'un axe unitaire) plutot qu'une transcription directe des ~70 lignes
    de trigonometrie spherique du source : une premiere transcription
    fidele des branches GOTO a ete testee numeriquement contre cette meme
    formule de Rodrigues et donnait des resultats faux (pas juste un
    signe invers) sur des cas generiques, alors qu'elle correspondait
    exactement au cas verifiable a la main (rotation autour du pole nord).
    Plutot que de continuer a deboguer une trigonometrie a branches de
    1980 a l'aveugle, la rotation est implementee par la formule
    standard, non ambigu, utilisee par les logiciels modernes de
    reconstruction plaques (GPlates, PmagPy) - la convention de signe
    reste alignee sur le source (verifiee sur le cas pole nord)."""
    axis_lat_r, axis_lon_r = RAD(pole_lat), RAD(pole_lon)
    axis = (
        math.cos(axis_lat_r) * math.cos(axis_lon_r),
        math.cos(axis_lat_r) * math.sin(axis_lon_r),
        math.sin(axis_lat_r),
    )
    lat_r, lon_r = RAD(lat1), RAD(lon1)
    v = (math.cos(lat_r) * math.cos(lon_r), math.cos(lat_r) * math.sin(lon_r), math.sin(lat_r))
    theta = RAD(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2]
    cross = (
        axis[1] * v[2] - axis[2] * v[1],
        axis[2] * v[0] - axis[0] * v[2],
        axis[0] * v[1] - axis[1] * v[0],
    )
    vr = tuple(
        v[i] * cos_t + cross[i] * sin_t + axis[i] * dot * (1.0 - cos_t)
        for i in range(3)
    )
    norm = math.sqrt(vr[0] ** 2 + vr[1] ** 2 + vr[2] ** 2)
    vr = tuple(c / norm for c in vr)
    l2 = DEG(math.asin(max(-1.0, min(1.0, vr[2]))))
    g2 = DEG(math.atan2(vr[1], vr[0])) % 360.0
    return l2, g2


def rotaim(site_lat: float, site_lon: float, dec: float, inc: float, pole_lon: float, pole_lat: float, pole_angle: float) -> Tuple[float, float]:
    """Port de `ROTAIM` : direction (dec,inc) mesuree au site -> VGP ->
    rotation Euler (site + VGP) -> direction attendue apres rotation."""
    vgp_lat, vgp_lon = dodi_vgp(dec, inc, site_lat, site_lon)
    l1, g1 = rota(pole_lon, pole_lat, pole_angle, site_lat, site_lon)
    a1, b1 = rota(pole_lon, pole_lat, pole_angle, vgp_lat, vgp_lon)
    return vgp_di(a1, b1, l1, g1)


def rotmag(site_lat: float, site_lon: float, dec: float, inc: float, pole_lon: float, pole_lat: float, pole_angle: float) -> Tuple[float, float]:
    """Port de `ROTMAG` (DI Polar rotation)."""
    return rotaim(site_lat, site_lon, dec, inc, pole_lon, pole_lat, pole_angle)


def vgp_plate_rotation_batch(rows: Sequence[Tuple[float, float, float, float, float]]) -> List[dict]:
    """Port de `vgp_plate_rotation` (VGP Plate rotation) : `rows` =
    (vgp_lon,vgp_lat,rot_lat,rot_lon,rot_angle) par ligne - NOTE : l'angle
    de rotation est NEGATE avant l'appel a `rota`, meme convention que le
    source (`ang=-ang`)."""
    out = []
    for vlon, vlat, rlat, rlon, ang in rows:
        lat2, lon2 = rota(rlon, rlat, -ang, vlat, vlon)
        out.append({"vgp_lon_in": vlon, "vgp_lat_in": vlat, "vgp_lon_out": lon2, "vgp_lat_out": lat2})
    return out


# ---------------------------------------------------------------------------
# Paleointensite (MEANPAL/VIDIMO/relocate/relocatevar)
# ---------------------------------------------------------------------------

def calcul(test: float, rinp: float) -> float:
    """Port de `CALCUL(TEST,RINP,RES)` (pmagoutils.f:959-970) : facteur
    dipolaire pour la conversion F<->VDM/VADM. `test=1` (VDM, `rinp`=
    inclinaison) ou `test=0` (VADM, `rinp`=colatitude=90-latitude)."""
    ray = 6371.0 * 1000.0
    x = math.sqrt(1.0 + 3.0 * math.cos(RAD(rinp)) ** 2)
    res = (ray ** 3) * x if test == 1.0 else (ray ** 3) / x
    return res * 1.0e07


def vidimo(paleointensity_microT: float, site_lat: float, inc: float) -> Tuple[float, float]:
    """Port de `VIDIMO` (VDM and VADM) : paleointensite (microT) -> (VDM,
    VADM), en Am2."""
    pal = paleointensity_microT * 1.0e-06
    vadm = pal * calcul(0.0, 90.0 - site_lat)
    vdm = pal * calcul(1.0, inc) / 2.0
    return vdm, vadm


def _relocate_raw(vdm_am2: float, site_lat: float, inc: float) -> Tuple[float, float]:
    """Port de `relocate1(vdm,rlat,rinc,fvadm,fvdm)` (pmagoutils.f:5956) :
    `vdm_am2` deja en Am2 (pas de mise a l'echelle 1e22, contrairement au
    sous-programme interactif `relocate` - la mise en commentaire de
    `vdm=vdm*1.e+22` dans le source est intentionnelle ici, verifie).
    Retourne (fvadm, fvdm) DANS CET ORDRE, meme signature que le source."""
    fvadm = vdm_am2 / calcul(0.0, 90.0 - site_lat)
    fvdm = 2.0 * vdm_am2 / calcul(1.0, inc)
    return fvadm, fvdm


def relocate(vdm_1e22: float, site_lat: float, inc: float) -> Tuple[float, float]:
    """Port de `relocate` (VDM to Intensity, menu interactif) : VDM (en
    1e22 Am2) -> (Fvdm,Fvadm) paleointensite equivalente (microT) a une
    autre latitude/inclinaison."""
    fvadm, fvdm = _relocate_raw(vdm_1e22 * 1.0e22, site_lat, inc)
    return fvdm * 1.0e06, fvadm * 1.0e06


# Colonnes .pmagint (STARpaleomag_Py/AMS_Py, meme fichier partage - voir
# STARpaleomag_Py/paleointensity._PMAGINT_HEADER) - reproduites ICI en dur
# (pas d'import inter-projet, meme discipline que site_map.py pour .prmag :
# chaque appli garde son propre lecteur minimal autonome). Seules les
# colonnes reellement utilisees par read_meanpal_file_triples sont
# nommees ; les autres n'ont pas besoin d'etre listees, l'index est
# retrouve dynamiquement depuis la ligne d'entete du fichier.
_PMAGINT_MARKER_COL = "specimen"


def read_meanpal_file_triples(path: str) -> List[Tuple[float, float, float]]:
    """Lit le fichier choisi pour MEANPAL mode "DONNES DANS UN FICHIER (1)" -
    demande explicite utilisateur ("dans le pmagint, peut on inserer des
    lignes de commentaires manuellement... ce fichier devra aussi etre lu
    dans Stereo_utils... est-ce possible de verifier la compatibilite ?").
    Verification faite : PAS compatible tel quel avec l'ancien format -
    l'ancien lecteur (3 premieres colonnes = F Q N, aucun en-tete attendu)
    prenait les colonnes 0/1/2 sans condition ; sur un vrai .pmagint la
    colonne 0 est l'id specimen (texte) et F/Q/N sont ailleurs (H/Hcorani/
    HcorCool en position 17/19/21, q en 8, N en 5) - chaque ligne aurait
    silencieusement echoue au float() et ete ignoree.

    Detecte maintenant les DEUX formats :
    - .pmagint natif (tabule, ligne d'entete commencant par "specimen") :
      une ligne = une interpretation/specimen = une "experience" au sens
      de meanpal_weighted ; F = le paleointensite le PLUS corrige
      disponible (HcorCool > Hcorani > H - MEME ordre de precedence que
      STARpaleomag_Py/magic_export.paleointensity_magic_fields/h_final,
      pour rester coherent avec ce qui serait exporte vers MagIC), Q = q,
      N = N (nombre de points Arai utilises dans l'interpretation).
    - ancien format simple (3 colonnes F Q N par ligne, sans en-tete).

    Dans les DEUX cas, les lignes vides et les lignes de commentaire
    ("#...", y compris ajoutees manuellement a la main dans le fichier)
    sont ignorees - .pmagint accepte deja cette convention cote
    STARpaleomag_Py/AMS_Py (paleointensity.read_pmagint)."""
    with open(path, "r", encoding="iso-8859-1", errors="replace") as f:
        raw_lines = f.read().splitlines()
    lines = [ln for ln in raw_lines if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return []

    header = lines[0].split("\t")
    if header[0].strip() == _PMAGINT_MARKER_COL:
        idx = {name.strip(): i for i, name in enumerate(header)}

        def col(parts: List[str], name: str) -> Optional[float]:
            i = idx.get(name)
            if i is None or i >= len(parts):
                return None
            v = parts[i].strip()
            if v in ("", "n.d"):
                return None
            try:
                return float(v)
            except ValueError:
                return None

        triples = []
        for ln in lines[1:]:
            parts = ln.split("\t")
            h = col(parts, "HcorCool")
            if h is None:
                h = col(parts, "Hcorani")
            if h is None:
                h = col(parts, "H")
            q, n = col(parts, "q"), col(parts, "N")
            if h is None or q is None or n is None:
                continue
            triples.append((h, q, n))
        return triples

    triples = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            triples.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    return triples


def meanpal_weighted(triples: Sequence[Tuple[float, float, float]]) -> Optional[dict]:
    """Port du coeur de `MEANPAL` modes fichier-simple (1) et clavier (3) :
    `triples` = (F,Q,N) par experience, poids `Q/sqrt(N-2)` (Prevot et
    al. 1985). Retourne None si le poids total est nul (aucune donnee
    exploitable, N<=2 partout)."""
    tot = 0.0
    f = 0.0
    vals = []
    for rfi, q, n in triples:
        nn = n - 2
        if nn <= 0:
            continue
        w = q / math.sqrt(nn)
        tot += w
        f += rfi * w
        vals.append(rfi)
    if tot == 0.0 or not vals:
        return None
    f /= tot
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return {"mean": mean, "weighted_mean": f, "sd": sd, "n": len(vals)}


def relocatevar_simplified(
    rows: Sequence[Tuple[str, float, float, float, float, float, float, float, float]],
    target_lat: float, target_lon: float,
) -> str:
    """Port simplifie de `relocatevar` (Relocate D I F) : `rows` = (site,
    age,age_err,slat,slon,dec,inc,a95,pal,pal_err) par site - relocalise
    direction et VDM/VADM vers (target_lat,target_lon), une seule table
    de resultats combinee (pas les 6 fichiers .vec/.pie/.site/.plat/.lat
    au format GMT du source, destines a un pipeline de trace externe -
    voir docstring du module)."""
    lines = ["site   age  age_err  dec_r   inc_r   a95    Fvdm    Fvadm"]
    for site, age, age_err, slat, slon, dec, inc, a95, pal, pal_err in rows:
        reloc = not (dec == 0.0 and inc == 0.0 and a95 == 0.0)
        if reloc:
            plat, plon = dodi_vgp(dec, inc, slat, slon)
            dec_r, inc_r = vgp_di(plat, plon, target_lat, target_lon)
            if dec_r > 180.0:
                dec_r -= 360.0
        else:
            dec_r, inc_r = float("nan"), float("nan")
        vdm, vadm = vidimo(pal, slat, inc)
        rinc_target = inc_r if reloc else inc
        # relocate1 est appelee UNE FOIS avec vdm (on ne garde que fvdm) et
        # UNE FOIS avec vadm (on ne garde que fvadm) - meme pattern que le
        # source (pmagoutils.f:5894-5897), pas symetrique par accident.
        _fvadm_unused, fvdm = _relocate_raw(vdm, target_lat, rinc_target)
        fvadm, _fvdm_unused2 = _relocate_raw(vadm, target_lat, rinc_target)
        fvdm *= 1.0e06
        fvadm *= 1.0e06
        if not reloc:
            fvdm = -99.0
        lines.append(
            f"{site:<8s} {age:6.1f} {age_err:6.1f} "
            f"{(dec_r if reloc else float('nan')):7.1f} {(inc_r if reloc else float('nan')):7.1f} {a95:6.1f} "
            f"{fvdm:8.1f} {fvadm:8.1f}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Rotation vers GMT (ROTA2GMT/HELPROTA/VECPIE) - calcrota.f
# ---------------------------------------------------------------------------

HELPROTA_TEXT = (
    "Ce programme calcule les rotations tectoniques par rapport a une courbe de poles de reference (APWP).\n"
    "Format attendu pour le fichier de donnees (une ligne par site) :\n"
    "  REF  SITE  AGE  Polarity(N/R)  LAT  LON  DEC  INC  A95  [REF_publication]\n"
    "Exemple : 11_03HU07  45  -15.89185  -73.52869  124.1  44.7  6.3\n"
    "Format attendu pour le fichier APWP de reference (une ligne par pole) :\n"
    "  AGE  <2 champs ignores>  POLE_LAT  POLE_LON  P95\n"
    "Prefixer REF par \"s_\" pour appliquer la correction d'aplatissement sedimentaire (fflatcor).\n"
)


def _vecpie(site_lon: float, site_lat: float, rotation: float, delta_rotation: float, dime: float) -> Tuple[str, str]:
    """Port de `VECPIE` (calcrota.f:378-395)."""
    veclinpie1 = (rotation - delta_rotation) % 360.0
    veclinpie2 = (rotation + delta_rotation) % 360.0
    vec_line = f"{site_lon:8.4f}  {site_lat:8.4f}  {rotation:6.1f}  {dime:6.1f}"
    pie_line = f"{site_lon:8.4f}  {site_lat:8.4f}  {veclinpie1:6.1f}  {veclinpie2:6.1f}"
    return vec_line, pie_line


def rota2gmt_core(
    data_rows: Sequence[Tuple[str, str, float, str, float, float, float, float, float, str]],
    apwp_rows: Sequence[Tuple[float, float, float, float]],
    age_min: float, age_max: float, vgp2dec: str,
    dime: float = 0.5, fflatcor: float = 1.0,
) -> Tuple[str, str, str]:
    """Port du coeur de `ROTA2GMT` (Rotation vers GMT plot) : pour chaque
    site dans [age_min,age_max], cherche dans l'APWP le pole d'age le
    plus proche (>= age du site, ou le pole precedent si plus proche -
    meme heuristique "nearest bracketing" que le source, balayage
    sequentiel du fichier APWP a chaque site), calcule rotation/
    aplatissement (comme `PALEOTEC`) avec correction de polarite (`carpol`
    =='R'), conversion VGP->direction optionnelle (`vgp2dec=='v'`), et
    correction d'aplatissement sedimentaire si la reference commence par
    "s_". Retourne (rapport .res, lignes .vec, lignes .pie) - pas les
    fichiers .site/.plat/.lat annexes (voir docstring du module).
    `apwp_rows` = (age,pole_lat,pole_lon,p95) tries par age croissant."""
    lines = [
        "REF        SITE          AGE   LAT     LONG    DOBS   INC     A95    "
        "POLE_LAT POLE_LON P95    DEXP   IEXP    ROT    dROT   FLAT  FLAT_cor  dFLAT"
    ]
    vec_lines, pie_lines = [], []
    for iref, site, age, carpol, al, g, rdec_obs, rinc_obs, a95, ref in data_rows:
        if age <= age_min or age > age_max:
            continue
        age0 = 0.0
        pole = None
        for age1, rlat_pole, rlon_pole, p95 in apwp_rows:
            if age1 >= age:
                pole = (rlat_pole, rlon_pole, p95)
                break
            age0 = age1
        if pole is None:
            continue
        rlat_pole, rlon_pole, p95 = pole

        rdec_obs2, rinc_obs2 = rdec_obs, rinc_obs
        if carpol == "R":
            rinc_obs2 = -rinc_obs2
            rdec_obs2 = (rdec_obs2 + 180.0) % 360.0
        if vgp2dec == "v":
            rdec_obs2, rinc_obs2 = vgp_di(rinc_obs2, rdec_obs2, al, g)

        dec_exp, inc_exp = vgp_di(rlat_pole, rlon_pole, al, g)
        dec_obs1 = rdec_obs2 - 360.0 if 180.0 < rdec_obs2 < 360.0 else rdec_obs2
        dec_exp1 = dec_exp - 360.0 if 180.0 < dec_exp < 360.0 else dec_exp
        rotation = dec_obs1 - dec_exp1
        if rotation < -360.0:
            rotation += 360.0
        rflat = inc_exp - rinc_obs2

        if iref.startswith("s_"):
            rinc_obs_cor = DEG(math.atan(math.tan(RAD(rinc_obs2)) / fflatcor))
            if rinc_obs2 < 0.0 and rinc_obs_cor > 0.0:
                rinc_obs_cor = -rinc_obs_cor
            rflat_cor = inc_exp - rinc_obs_cor
        else:
            rflat_cor = rflat
            rinc_obs_cor = rinc_obs2

        if a95 == 0.0:
            deld_obs = 0.0
        else:
            s = math.sin(RAD(a95)) / math.cos(RAD(rinc_obs2))
            deld_obs = 90.0 if s > 0.9999 else DEG(math.asin(s))
        if p95 == 0.0:
            deld_exp = 0.0
        else:
            pl = 1.5707 - math.atan(math.tan(RAD(inc_exp)) / 2.0)
            deld_exp = DEG(math.asin(min(1.0, math.sin(RAD(p95)) / math.sin(pl))))
        delt_rotation = 0.0 if (deld_obs == 0.0 and deld_exp == 0.0) else 0.8 * math.hypot(deld_obs, deld_exp)

        deli_obs = a95
        deli_exp = 0.0 if p95 == 0.0 else 2.0 * p95 / (1.0 + 3.0 * math.cos(1.5707 - math.atan(math.tan(RAD(inc_exp)) / 2.0)) ** 2)
        delt_flat = 0.0 if (deli_obs == 0.0 and deli_exp == 0.0) else 0.8 * math.hypot(deli_obs, deli_exp)

        site_lon = g - 360.0 if 270.0 < g <= 360.0 else g
        lines.append(
            f"{iref:<10s} {site:<12s} {int(age):5d} {al:7.3f} {site_lon:7.3f} "
            f"{rdec_obs2:6.1f} {rinc_obs2:6.1f} {a95:5.1f} "
            f"{rlat_pole:7.1f} {rlon_pole:7.1f} {p95:5.1f} "
            f"{dec_exp:6.1f} {inc_exp:6.1f} {rotation:6.1f} {delt_rotation:5.1f} "
            f"{rflat:6.1f} {rflat_cor:6.1f} {delt_flat:5.1f}"
        )
        vec_line, pie_line = _vecpie(site_lon, al, rotation, delt_rotation, dime)
        vec_lines.append(vec_line)
        pie_lines.append(pie_line)
    return "\n".join(lines) + "\n", "\n".join(vec_lines) + ("\n" if vec_lines else ""), "\n".join(pie_lines) + ("\n" if pie_lines else "")


# ---------------------------------------------------------------------------
# Overprint / Flatten
# ---------------------------------------------------------------------------

def _di2xyz(dec: float, inc: float) -> Tuple[float, float, float]:
    return math.cos(RAD(inc)) * math.cos(RAD(dec)), math.cos(RAD(inc)) * math.sin(RAD(dec)), math.sin(RAD(inc))


def overprint_test(
    overprint_dec: float, overprint_inc: float,
    normal_dec: float, normal_inc: float, reverse_dec: float, reverse_inc: float,
) -> dict:
    """Port de `overprint` (test overprint, pmagoutils.f:5346-5431) :
    balaie 1000 fractions de "retrait" de la surimpression le long du
    grand cercle N/R-overprint pour trouver celle qui maximise
    l'antipodalite entre les directions normale et inverse corrigees."""
    x1, y1, z1 = _di2xyz(overprint_dec, overprint_inc)
    xn, yn, zn = _di2xyz(normal_dec, normal_inc)
    xr, yr, zr = _di2xyz(reverse_dec, reverse_inc)
    _r, yn0, zn0 = polere((xn - xr) / 2.0, (yn - yr) / 2.0, (zn - zr) / 2.0)

    delmin = 90.0
    del0 = None
    best = None
    for i in range(1, 1001):
        p = -(i - 1) / 1000.0
        _r, rdn, rin = polere(p * x1 + xn, p * y1 + yn, p * z1 + zn)
        _r, rdr, rir = polere(p * x1 + xr, p * y1 + yr, p * z1 + zr)
        rdnr, rinr = rdr + 180.0, -rir
        delval, _a = _angle(rdn - rdnr, 90.0 - rinr, 90.0 - rin)
        delval = abs(delval)
        if i == 1:
            del0 = delval
        if delval < delmin:
            delmin = delval
            popt = p
            xncn, yncn, zncn = _di2xyz(rdn, rin)
            xncr, yncr, zncr = _di2xyz(rdr, rir)
            _r, yncc, zncc = polere((xncn - xncr) / 2.0, (yncn - yncr) / 2.0, (zncn - zncr) / 2.0)
            best = (rdn, rin, rdr, rir, popt, yncc, zncc)
    rdn0, rin0, rdr0, rir0, popt, yncc, zncc = best
    return {
        "initial_angle_n_r": del0, "estimated_overprint_pct": -popt * 100.0,
        "corrected_normal": (rdn0, rin0), "corrected_reverse": (rdr0, rir0),
        "minimum_angle_n_r": delmin,
        "mean_char_dir_before": (yn0, zn0), "mean_char_dir_after": (yncc, zncc),
    }


def flatten_correction(uninc: float, f: float) -> Optional[float]:
    """Port de `flatten` (test flattening) : `ainc = atan(tan(uninc)/f)`.
    BUG CONFIRME du source : la garde de validation controle en realite
    une variable `uinc` distincte, jamais assignee (`uninc` est la
    variable reellement lue) - la garde est donc totalement inoperante,
    n'importe quelle valeur d'entree (meme hors 0-90) passe. REPLIQUE
    tel quel : aucune validation n'est appliquee ici non plus."""
    if f <= 0.0 or f >= 1.0:
        return None
    return DEG(math.atan(math.tan(RAD(uninc)) / f))


# ---------------------------------------------------------------------------
# Conversion d'unites (convunit)
# ---------------------------------------------------------------------------

_UNIT_CONVERSIONS = {
    1: ("Am2 -> emu", lambda v: v / 0.001, "emu"),
    2: ("emu -> Am2", lambda v: v * 0.001, "Am2"),
    3: ("Am2/kg -> emu/g", lambda v: v, "emu/g"),
    4: ("emu/g -> Am2/kg", lambda v: v, "Am2/kg"),
    5: ("A/m -> emu/cc", lambda v: v * 0.001, "emu/cc"),
    6: ("emu/cc -> A/m", lambda v: v / 0.001, "A/m"),
    7: ("Tesla -> A/m", lambda v: v / (4.0 * math.pi * 1.0e-07), "A/m"),
    8: ("microTesla -> A/m", lambda v: v / (4.0 * math.pi / 10.0), "A/m"),
    9: ("A/m -> Tesla", lambda v: v * (4.0 * math.pi * 1.0e-07), "Tesla"),
    10: ("A/m -> microTesla", lambda v: v * 4.0 * math.pi / 10.0, "microT"),
}


def convert_units(choice: int, value: float) -> Optional[Tuple[float, str]]:
    """Port de `convunit` (conversion units)."""
    entry = _UNIT_CONVERSIONS.get(choice)
    if entry is None:
        return None
    _label, fn, unit = entry
    return fn(value), unit


# ---------------------------------------------------------------------------
# IGRF (igrfOSX/helpigrf) - via le paquet `ppigrf` plutot qu'une
# transcription des ~1200 lignes de tables de coefficients/recursion de
# Legendre du source (recommandation du rapport d'exploration initial).
# ---------------------------------------------------------------------------

HELPIGRF_TEXT = (
    "Calcule le champ geomagnetique (IGRF) a une position et une date donnees :\n"
    "declinaison, inclinaison, intensite totale/horizontale, composantes N/E/Z.\n"
    "Utilise le paquet `ppigrf` (coefficients IGRF-14) plutot que la transcription\n"
    "Fortran d'origine (adaptee du code de reference de Susan Macmillan)."
)


def igrf_value(lat: float, lon: float, year: int, month: int = 1, day: int = 1, altitude_km: float = 0.0) -> dict:
    """Port de `igrfOSX`/`IGRFSTEREOSX` (valeur CMT IGRF) via `ppigrf`."""
    import datetime
    import ppigrf
    date = datetime.datetime(year, month, day)
    be, bn, bu = ppigrf.igrf(lon, lat, altitude_km, date)
    be, bn, bu = float(be.item()), float(bn.item()), float(bu.item())
    bz = -bu
    h = math.hypot(be, bn)
    f = math.hypot(h, bz)
    dec = DEG(math.atan2(be, bn))
    inc = DEG(math.atan2(bz, h))
    return {"declination": dec, "inclination": inc, "horizontal_intensity_nT": h,
            "total_intensity_nT": f, "north_nT": bn, "east_nT": be, "down_nT": bz}
