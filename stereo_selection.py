"""
Modele de donnees minimal pour StereoUtils_Py (port de `StereoOSX_x.f95`,
reference/Stereo_V19) : listes equivalentes aux tableaux partages
`common /data/ adec/ainc` (D-I), `dic/inc/alph` (D-I-a95, "means"),
`glon/glat/ad1/ai1/ad2/ai2` (grands cercles) - le menu Data input du
Fortran d'origine (`stereograph.f`).

Convention d'en-tete (demande explicite utilisateur) : chaque lecteur de
fichier texte (ci-dessous, et dans `stereo_pmagutils.py`/`app.py`)
detecte automatiquement une eventuelle premiere ligne d'en-tete
auto-descriptive (commencant par "#", ou simplement composee de noms de
colonnes plutot que de nombres - ex. "#site dec inc strike dip") et
l'utilise pour repartir les colonnes par NOM plutot que par position/
"nombre de colonnes a sauter". Aucun en-tete detecte -> repli silencieux
sur l'ancien comportement positionnel (retrocompatible avec les fichiers
existants sans en-tete)."""

import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from stereo_geometry import corpen, polere

# Alias reconnus (en minuscules) pour chaque champ canonique - voir
# `_find_column`. Un meme fichier peut nommer "dec" declination/d/dec.
FIELD_ALIASES: Dict[str, set] = {
    # "rgb" reste la cle canonique en interne (ProjectEntry.rgb/
    # VgpProjectEntry.rgb, decode_color...) mais le fichier ecrit
    # desormais l'en-tete "color" (demande explicite utilisateur : "can
    # you change rgb by color") - accepter les deux noms de colonne
    # transparemment plutot que de renommer tous les champs internes qui
    # en dependent (draw_project, plot_vgp_project...).
    "rgb": {"rgb", "color"},
    "site": {"site", "id", "name", "sample", "specimen"},
    "dec": {"dec", "declination", "d"},
    "inc": {"inc", "inclination", "i"},
    "a95": {"a95", "alpha95", "alph"},
    "strike": {"strike", "str"},
    "dipdir": {"dipdir", "dipdirection", "dip_direction", "diredip", "dd"},
    "dip": {"dip", "pendage"},
    "glon": {"glon", "lon", "long", "longitude"},
    "glat": {"glat", "lat", "latitude"},
    "ad1": {"ad1"}, "ai1": {"ai1"}, "ad2": {"ad2"}, "ai2": {"ai2"},
    "age": {"age"},
    "vgplat": {"vgplat", "vgp_lat", "polelat", "pole_lat", "vlat"},
    "vgplon": {"vgplon", "vgp_lon", "polelon", "pole_lon", "vlon"},
    "p95": {"p95"},
    "azimuth": {"azimuth", "az", "foldaz", "foldazimuth"},
    "folddip": {"folddip", "fold_dip"},
    "d0": {"d0"}, "i0": {"i0"}, "d1": {"d1"}, "i1": {"i1"}, "d2": {"d2"}, "i2": {"i2"},
    "slat": {"slat", "sitelat", "site_lat", "lat", "latitude"},
    "slon": {"slon", "sitelon", "site_lon", "lon", "long", "longitude"},
    "rotlat": {"rotlat", "rot_lat"},
    "rotlon": {"rotlon", "rot_lon"},
    "rotangle": {"rotangle", "rot_angle", "angle"},
    "ageerr": {"ageerr", "age_err"},
    "pal": {"pal", "f", "intensity"},
    "palerr": {"palerr", "pal_err", "f_err", "ferr"},
    "ref": {"ref"},
    "iref": {"iref", "ref"},
    "refpub": {"refpub", "ref_pub", "publication", "pub"},
    "polarity": {"polarity", "carpol", "pol"},
    "litho": {"litho", "lithology"},
}


def read_text_lines(path: str) -> List[str]:
    """Essaie UTF-8 (fichiers modernes, accents/symboles) puis retombe sur
    iso-8859-1 si le decodage UTF-8 echoue."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
    except UnicodeDecodeError:
        with open(path, "r", encoding="iso-8859-1", errors="replace") as f:
            return f.readlines()


def detect_header(line: str) -> Optional[List[str]]:
    """Retourne les noms de colonnes (minuscules) si `line` ressemble a
    un en-tete auto-descriptif ("#" en tete, ou uniquement des tokens non
    numeriques) ; None sinon (ligne de donnees ou vide)."""
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("#"):
        stripped = stripped[1:].strip()
        if not stripped:
            return None
    elif stripped.startswith("!"):
        return None
    tokens = stripped.split()
    if not tokens:
        return None
    for t in tokens:
        try:
            float(t)
            return None
        except ValueError:
            continue
    return [t.lower() for t in tokens]


def find_column(header: Sequence[str], *field_keys: str) -> Optional[int]:
    """Index de la premiere colonne de `header` correspondant a l'un des
    noms canoniques `field_keys` (via `FIELD_ALIASES`, sinon le nom lui
    meme comme unique alias)."""
    aliases = set()
    for key in field_keys:
        aliases |= FIELD_ALIASES.get(key, {key})
    for i, name in enumerate(header):
        if name in aliases:
            return i
    return None


def split_header(lines: Sequence[str], *field_keys: str) -> Tuple[List[str], Dict[str, int]]:
    """Si `lines[0]` est un en-tete, retourne (lignes de donnees restantes,
    {champ_canonique: index_colonne}) pour chaque champ de `field_keys`
    trouve. Sinon (lines inchangees, {}) - l'appelant retombe alors sur
    son ancien decoupage positionnel/`skip_columns`."""
    if not lines:
        return list(lines), {}
    header = detect_header(lines[0])
    if header is None:
        return list(lines), {}
    index_map = {}
    for key in field_keys:
        idx = find_column(header, key)
        if idx is not None:
            index_map[key] = idx
    return list(lines[1:]), index_map


def _is_comment_or_blank(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("!") or stripped.startswith("#")


def read_di_file(path: str, skip_columns: int = 0) -> List[Tuple[float, float]]:
    """Equivalent de `openfilemag` (PmagPy_connect.f95:405-444) : lit un
    fichier texte, colonnes dec/inc reperees par en-tete si present, sinon
    `skip_columns` champs ignores avant dec/inc (meme convention que le
    prompt Fortran "number of variables to skip before D and I")."""
    out = []
    lines = read_text_lines(path)
    data_lines, idx = split_header(lines, "dec", "inc")
    i_dec, i_inc = (idx["dec"], idx["inc"]) if "dec" in idx and "inc" in idx else (skip_columns, skip_columns + 1)
    for line in data_lines:
        if _is_comment_or_blank(line):
            continue
        parts = line.split()
        if len(parts) <= max(i_dec, i_inc):
            continue
        try:
            dec, inc = float(parts[i_dec]), float(parts[i_inc])
        except ValueError:
            continue
        out.append((dec, inc))
    return out


def read_fold_file(
    path: str, skip_columns: int = 0, strike: bool = True,
) -> List[Tuple[float, float, float, float]]:
    """Equivalent du chargement de `foldtestpmagpy` (PmagPy_connect.f95:99-190) :
    (dec, inc, dip_direction, dip) par ligne. Colonnes reperees par
    en-tete si present (`strike` ou `dipdir` - determine alors
    AUTOMATIQUEMENT si la 3e colonne est un strike ou une direction de
    pendage, sans le prompt "Strike (1) or dip direction (2)?") ; sinon
    `skip_columns` champs ignores avant dec, et le parametre `strike`
    explicite le sens de la 3e colonne comme avant."""
    out = []
    lines = read_text_lines(path)
    data_lines, idx = split_header(lines, "dec", "inc", "strike", "dip", "dipdir")
    if "dec" in idx and "inc" in idx and "dip" in idx and ("strike" in idx or "dipdir" in idx):
        i_dec, i_inc, i_dip = idx["dec"], idx["inc"], idx["dip"]
        if "strike" in idx:
            i_sd, is_strike = idx["strike"], True
        else:
            i_sd, is_strike = idx["dipdir"], False
    else:
        i_dec, i_inc, i_sd, i_dip = skip_columns, skip_columns + 1, skip_columns + 2, skip_columns + 3
        is_strike = strike
    for line in data_lines:
        if _is_comment_or_blank(line):
            continue
        parts = line.split()
        if len(parts) <= max(i_dec, i_inc, i_sd, i_dip):
            continue
        try:
            dec = float(parts[i_dec])
            inc = float(parts[i_inc])
            str_or_dipdir = float(parts[i_sd])
            dip = float(parts[i_dip])
        except ValueError:
            continue
        dip_direction = str_or_dipdir + 90.0 if is_strike else str_or_dipdir
        out.append((dec, inc, dip_direction, dip))
    return out


def _tilt_correct(dec: float, inc: float, strike: float, dip: float) -> Tuple[float, float]:
    """dec,inc -> cartesien -> `corpen(dip,strike)` -> `polere` : meme
    pipeline que `opfil4`/`opfil5` (stereograph.f)."""
    rad = math.radians
    z = math.sin(rad(inc))
    x = math.cos(rad(inc)) * math.cos(rad(dec))
    y = math.cos(rad(inc)) * math.sin(rad(dec))
    xx, yy, zz = corpen(x, y, z, dip, strike)
    _mag, yy2, zz2 = polere(xx, yy, zz)
    return yy2, zz2


def read_di_tc_file(path: str, skip_columns: int = 0) -> List[Tuple[float, float]]:
    """Equivalent de `opfil4` (File [D-I_TC]) : (dec,inc,strike,dip) par
    ligne, colonnes reperees par en-tete si present, sinon `skip_columns`
    champs ignores avant dec. Correction de pendage (`corpen`) appliquee
    avant stockage."""
    out = []
    lines = read_text_lines(path)
    data_lines, idx = split_header(lines, "dec", "inc", "strike", "dip")
    if "dec" in idx and "inc" in idx and "strike" in idx and "dip" in idx:
        i_dec, i_inc, i_str, i_dip = idx["dec"], idx["inc"], idx["strike"], idx["dip"]
    else:
        i_dec, i_inc, i_str, i_dip = skip_columns, skip_columns + 1, skip_columns + 2, skip_columns + 3
    for line in data_lines:
        if _is_comment_or_blank(line):
            continue
        parts = line.split()
        if len(parts) <= max(i_dec, i_inc, i_str, i_dip):
            continue
        try:
            dec = float(parts[i_dec])
            inc = float(parts[i_inc])
            strike = float(parts[i_str])
            dip = float(parts[i_dip])
        except ValueError:
            continue
        out.append(_tilt_correct(dec, inc, strike, dip))
    return out


def read_di_a95_file(path: str, skip_columns: int = 0) -> List[Tuple[float, float, float]]:
    """Equivalent de `opfil2`/`mean` (File [D-I-a95]) : (dec,inc,alpha95)
    par ligne, colonnes reperees par en-tete si present, sinon
    `skip_columns` champs ignores avant dec."""
    out = []
    lines = read_text_lines(path)
    data_lines, idx = split_header(lines, "dec", "inc", "a95")
    if "dec" in idx and "inc" in idx and "a95" in idx:
        i_dec, i_inc, i_a95 = idx["dec"], idx["inc"], idx["a95"]
    else:
        i_dec, i_inc, i_a95 = skip_columns, skip_columns + 1, skip_columns + 2
    for line in data_lines:
        if _is_comment_or_blank(line):
            continue
        parts = line.split()
        if len(parts) <= max(i_dec, i_inc, i_a95):
            continue
        try:
            dec = float(parts[i_dec])
            inc = float(parts[i_inc])
            alph = float(parts[i_a95])
        except ValueError:
            continue
        out.append((dec, inc, alph))
    return out


def read_di_a95_tc_file(path: str, skip_columns: int = 0) -> List[Tuple[float, float, float]]:
    """Equivalent de `opfil5` (File [D-I-a95_TC]) : (dec,inc,alpha95,
    strike,dip) par ligne, colonnes reperees par en-tete si present, sinon
    `skip_columns`. Correction de pendage appliquee a dec/inc, alpha95
    recopie tel quel."""
    out = []
    lines = read_text_lines(path)
    data_lines, idx = split_header(lines, "dec", "inc", "a95", "strike", "dip")
    if all(k in idx for k in ("dec", "inc", "a95", "strike", "dip")):
        i_dec, i_inc, i_a95, i_str, i_dip = idx["dec"], idx["inc"], idx["a95"], idx["strike"], idx["dip"]
    else:
        i_dec, i_inc, i_a95, i_str, i_dip = (
            skip_columns, skip_columns + 1, skip_columns + 2, skip_columns + 3, skip_columns + 4)
    for line in data_lines:
        if _is_comment_or_blank(line):
            continue
        parts = line.split()
        if len(parts) <= max(i_dec, i_inc, i_a95, i_str, i_dip):
            continue
        try:
            dec = float(parts[i_dec])
            inc = float(parts[i_inc])
            alph = float(parts[i_a95])
            strike = float(parts[i_str])
            dip = float(parts[i_dip])
        except ValueError:
            continue
        dec_tc, inc_tc = _tilt_correct(dec, inc, strike, dip)
        out.append((dec_tc, inc_tc, alph))
    return out


def read_great_circle_file(path: str) -> List[Tuple[float, float, float, float, float, float]]:
    """Equivalent de `opfil3` (File [great circle]) : (glon,glat,ad1,ai1,
    ad2,ai2) par ligne - glon/glat = pole du grand cercle, ad1/ai1-ad2/ai2 =
    bornes optionnelles du secteur (0 si absentes, meme fallback que le
    source: `err=783` -> ad1=ai1=ad2=ai2=0 si seuls glon,glat sont lisibles).
    Colonnes reperees par en-tete si present, sinon glon,glat=colonnes
    0,1 et ad1..ai2=colonnes 2-5."""
    out = []
    lines = read_text_lines(path)
    data_lines, idx = split_header(lines, "glon", "glat", "ad1", "ai1", "ad2", "ai2")
    if "glon" in idx and "glat" in idx:
        i_glon, i_glat = idx["glon"], idx["glat"]
        sector_idx = (idx.get("ad1"), idx.get("ai1"), idx.get("ad2"), idx.get("ai2"))
        has_sector_header = all(v is not None for v in sector_idx)
    else:
        i_glon, i_glat = 0, 1
        sector_idx = (2, 3, 4, 5)
        has_sector_header = False
    for line in data_lines:
        if _is_comment_or_blank(line):
            continue
        parts = line.split()
        if len(parts) <= max(i_glon, i_glat):
            continue
        try:
            glon = float(parts[i_glon])
            glat = float(parts[i_glat])
        except ValueError:
            continue
        ad1 = ai1 = ad2 = ai2 = 0.0
        if has_sector_header:
            try:
                ad1, ai1, ad2, ai2 = (float(parts[i]) for i in sector_idx)
            except (ValueError, IndexError):
                ad1 = ai1 = ad2 = ai2 = 0.0
        elif not idx and len(parts) >= 6:
            try:
                ad1, ai1, ad2, ai2 = (float(parts[i]) for i in range(2, 6))
            except ValueError:
                ad1 = ai1 = ad2 = ai2 = 0.0
        out.append((glon, glat, ad1, ai1, ad2, ai2))
    return out


class VgpProjectEntry(NamedTuple):
    """Un VGP a tracer sur une carte (voir stereo_pmagpy.plot_vgp_project) -
    equivalent "carte" de stereo_project.ProjectEntry (reseau stereo) :
    meme esprit (symbole/couleur/taille par entree, editable a la main
    dans le fichier), mais pas le meme fichier ni le meme trace (un VGP
    est un point GEOGRAPHIQUE paleolat/paleolon, pas une direction
    dec/inc de stereonet - le format "Project" n'a pas de type d'entree
    pour lui, voir STARpaleomag_Py/export_stereo.py) - demande explicite
    utilisateur ("we will need to update the project in Stereo to manage
    the VGP plot")."""
    site: str
    paleolon: float
    paleolat: float
    dp: float
    dm: float
    p95: float
    symbol: str
    rgb: str
    size: float
    nb: int
    a95: float


def _parse_vgp_project_lines(lines: Sequence[str]) -> List[VgpProjectEntry]:
    """Coeur partage de read_vgp_project_file (fichier VGP autonome,
    colonnes dec/inc = VGP_lon/VGP_lat, meme convention que read_di_file/
    plot_vgps_on_map) ET du bloc "# VGP" d'un fichier "Stereo Project" a 3
    blocs (colonnes paleolon/paleolat - voir STARpaleomag_Py/export_stereo.
    export_stereo_project) - les DEUX noms de colonnes sont acceptes,
    n'importe lequel des deux suffit. symbol/rgb/size/dp/dm/p95/nb/a95
    restent optionnels - repli sur des valeurs par defaut (cercle noir
    taille 0.30, dp=dm=p95=0) pour un fichier plus ancien qui ne les
    ecrivait pas encore."""
    out: List[VgpProjectEntry] = []
    data_lines, idx = split_header(
        lines, "dec", "inc", "paleolon", "paleolat", "site", "id",
        "nb", "n", "a95", "dp", "dm", "p95", "symbol", "rgb", "size")
    i_lon = idx.get("paleolon", idx.get("dec"))
    i_lat = idx.get("paleolat", idx.get("inc"))
    if i_lon is None or i_lat is None:
        return out
    for i, line in enumerate(data_lines):
        if _is_comment_or_blank(line):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) <= max(i_lon, i_lat):
            continue

        def field(name: str, default: str = "", j=None) -> str:
            j = idx.get(name) if j is None else j
            return parts[j].strip() if j is not None and j < len(parts) else default

        try:
            paleolon, paleolat = float(field("", j=i_lon)), float(field("", j=i_lat))
        except ValueError:
            continue
        try:
            dp, dm = float(field("dp", "0")), float(field("dm", "0"))
        except ValueError:
            dp = dm = 0.0
        # p95 (rayon du cercle de confiance approximatif, voir
        # stereo_pmagpy.plot_vgp_project) : lu si la colonne existe (fichier
        # VGP autonome), sinon calcule depuis dp/dm (bloc "# VGP" d'un
        # fichier "Stereo Project" a 3 blocs, qui n'ecrit QUE dp/dm - voir
        # STARpaleomag_Py/export_stereo.export_stereo_project) plutot que
        # de retomber silencieusement sur 0.0 (aucun cercle trace).
        if "p95" in idx:
            try:
                p95 = float(field("p95", "0"))
            except ValueError:
                p95 = (dp + dm) / 2.0
        else:
            p95 = (dp + dm) / 2.0
        try:
            nb = int(float(field("nb") or field("n", "0")))
        except ValueError:
            nb = 0
        try:
            a95 = float(field("a95", "0"))
        except ValueError:
            a95 = 0.0
        try:
            size = float(field("size", "0.30"))
        except ValueError:
            size = 0.30
        site = field("site") or field("id") or f"vgp{i + 1}"
        symbol = field("symbol", "c") or "c"
        rgb = field("rgb", "0_0_0") or "0_0_0"
        out.append(VgpProjectEntry(site, paleolon, paleolat, dp, dm, p95, symbol, rgb, size, nb, a95))
    return out


def read_vgp_project_file(path: str) -> List[VgpProjectEntry]:
    """Lit un fichier VGP autonome ecrit par STARpaleomag_Py/export_stereo
    (voir _parse_vgp_project_lines pour le detail des colonnes/alias)."""
    return _parse_vgp_project_lines(read_text_lines(path))
