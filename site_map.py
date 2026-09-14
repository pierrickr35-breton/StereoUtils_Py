"""
Cartographie des sites d'un fichier .prmag (format STARpaleomag_Py) - menu
Pmag Utilities > Site Map - demande explicite utilisateur ("dans pmag_
utilities, ajouter un sous-menu pour tracer les sites sur Google Earth,
prmag to kml. Un autre GMT map pour tracer les sites sur une carte
topographique. Eventuellement une copie avec le simple matplotlib").

Lecture MINIMALE du .prmag (voir read_prmag_sites) : seule la PREMIERE
ligne d'entete de chaque bloc specimen (specimen/sample/site/lat/lon) est
lue, le reste (orientation/stratigraphie/mesures) est ignore - meme
philosophie et meme format que AMS_Py/ams_prmag.py.read_prmag_specimens
(StereoUtils_Py reste un projet independant, pas d'import cross-projet -
chaque app maintient son propre lecteur .prmag minimal, adapte a ses
propres besoins).

Trois sorties, chacune independante des deux autres :
- write_kml : un fichier .kml (Google Earth), une Placemark/Point par
  site.
- write_gmt_map_script : un script shell GMT6 pret a l'emploi - carte DE
  BASE (PAS de relief/topographie - demande explicite utilisateur
  "forget the topography, and use a basic map"), continent gris, lacs/
  rivieres en bleu (`gmt coast`), sites en symboles + etiquettes -
  ecartees automatiquement quand des sites sont proches (voir
  _declutter_label_positions : GMT lui-meme n'a PAS de placement anti-
  collision d'etiquettes, demande explicite utilisateur "I do not know
  if GMT can adjust the position of the labels when sites are close").
  Emplacement documente pour des ROUTES (GMT n'a PAS de base de donnees
  routiere integree - voir le commentaire dans le script genere). 3
  fichiers de donnees compagnons - genere pour etre execute par
  l'utilisateur (GMT n'est pas suppose etre installe/appelable depuis
  cette application, meme convention que pu_rota2gmt/"GMT rotation
  export" qui produit deja des fichiers texte pour un pipeline GMT
  externe, jamais d'appel direct a `gmt`).
- build_site_map_figure : equivalent matplotlib SIMPLE (pas de fond
  topographique/cotes - demande explicite "eventuellement une copie avec
  le simple matplotlib", volontairement minimal), juste les sites en
  lon/lat avec etiquettes."""

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from xml.sax.saxutils import escape as _xml_escape

from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector


@dataclass
class PrmagSite:
    """Un site, coordonnees moyennees sur tous ses specimens (devrait
    etre identique pour tous - moyenne par robustesse plutot que par
    necessite). `strike`/`dip` (pendage de la stratification, ligne C du
    bloc .prmag - "bed_dip_strike"/"bed_dip") None si non mesure -
    `geologic_type`/`lithology`/`age` (ligne D - "geologic_types"/
    "lithologies"/"age") chaine vide si absent - demande explicite
    utilisateur ("in the kml, can you add strike dip; Geologic type: and
    lithology" puis "and age"). `date` (ligne B - "date", format ISO8601
    tronque a YYYY-MM-DD) chaine vide si absent - demande explicite
    utilisateur (reprendre le style de l'ancien KML Fortran, qui
    affichait une date)."""
    site: str
    lat: float
    lon: float
    n_specimens: int = 0
    strike: Optional[float] = None
    dip: Optional[float] = None
    geologic_type: str = ""
    lithology: str = ""
    age: str = ""
    date: str = ""


def _prmag_kv_line(line: str) -> dict:
    """Meme format que AMS_Py/ams_prmag._prmag_kv_line : "cle:valeur"
    separes par des tabulations."""
    result = {}
    for chunk in line.split("\t"):
        if ":" not in chunk:
            continue
        k, _sep, v = chunk.partition(":")
        result[k.strip()] = v.strip()
    return result


def _text(v: Optional[str]) -> str:
    v = (v or "").strip()
    return "" if v == "n.d" else v


def _num(v: Optional[str], default: float = 0.0) -> float:
    v = (v or "").strip()
    if not v or v == "n.d":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _num_opt(v: Optional[str]) -> Optional[float]:
    """Comme `_num`, mais retourne None (pas 0.0) quand absent/"n.d" - 0
    est une valeur de pendage/strike valide, a distinguer de "non
    mesure"."""
    v = (v or "").strip()
    if not v or v == "n.d":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_prmag_sites(filepath: str, encoding: str = "utf-8") -> List[PrmagSite]:
    """Lit un .prmag (testlect.read_prmag_file cote STARpaleomag_Py pour
    le format complet) et retourne un site par nom de site rencontre,
    coordonnees = moyenne de `lat`/`lon` sur tous les specimens de ce
    site. Site sans nom (`site` vide dans le fichier) retombe sur les 6
    premiers caracteres du specimen id (annee+site, meme convention que
    AMS_Py.mean_result_site).

    strike/dip/geologic_type/lithology/age (lignes C et D du bloc,
    identiques a lat/lon dans leur principe - meme convention/donnee
    repetee sur chaque specimen d'un site) sont pris sur le PREMIER
    specimen rencontre pour chaque site plutot que moyennes : strike est
    un angle circulaire (moyenner betement 359 et 1 degre donnerait 180,
    faux) et geologic_type/lithology/age sont du texte, non moyennable -
    demande explicite utilisateur ("in the kml, can you add strike dip;
    Geologic type: and lithology" puis "and age")."""
    with open(filepath, "r", encoding=encoding) as f:
        lines = [raw.rstrip("\n") for raw in f]
    n = len(lines)
    i = 0

    def _skip_blank_and_comments():
        nonlocal i
        while i < n and (lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
            i += 1

    coords: Dict[str, List[tuple]] = {}
    extra: Dict[str, dict] = {}
    _skip_blank_and_comments()
    while i < n:
        if i + 3 >= n:
            break  # bloc incomplet en fin de fichier (meme garde que testlect.read_prmag_file)
        line_a = _prmag_kv_line(lines[i])
        line_b = _prmag_kv_line(lines[i + 1])
        line_c = _prmag_kv_line(lines[i + 2])
        line_d = _prmag_kv_line(lines[i + 3])
        specimen_id = _text(line_a.get("specimen"))
        if specimen_id:
            site = _text(line_a.get("site")) or specimen_id[:6]
            lat = _num(line_a.get("lat"))
            lon = _num(line_a.get("lon"))
            coords.setdefault(site, []).append((lat, lon))
            if site not in extra:
                date_m = re.match(r"(\d{4}-\d{2}-\d{2})", line_b.get("date") or "")
                extra[site] = {
                    "strike": _num_opt(line_c.get("bed_dip_strike")),
                    "dip": _num_opt(line_c.get("bed_dip")),
                    "age": _text(line_d.get("age")),
                    "geologic_type": _text(line_d.get("geologic_types")),
                    "lithology": _text(line_d.get("lithologies")),
                    "date": date_m.group(1) if date_m else "",
                }
        while i < n and lines[i].strip() != "":
            i += 1
        _skip_blank_and_comments()

    out = []
    for site, pts in coords.items():
        lat = sum(p[0] for p in pts) / len(pts)
        lon = sum(p[1] for p in pts) / len(pts)
        e = extra.get(site, {})
        out.append(PrmagSite(
            site=site, lat=lat, lon=lon, n_specimens=len(pts),
            strike=e.get("strike"), dip=e.get("dip"),
            geologic_type=e.get("geologic_type", ""),
            lithology=e.get("lithology", ""), age=e.get("age", ""),
            date=e.get("date", ""),
        ))
    out.sort(key=lambda s: s.site)
    return out


# ----------------------------------------------------------------------
# Google Earth (.kml)
# ----------------------------------------------------------------------

# Meme reference de style que l'ancien KML Fortran (icone punaise Google
# Earth "par defaut", jamais recalculee - demande explicite utilisateur
# "is it possible to have it more nicely like what I was doing in
# Fortran", reprend litteralement son <styleUrl>).
_DEFAULT_STYLE_URL = "root://styleMaps#default+nicon=0x304+hicon=0x314"


def _site_description_html(s: PrmagSite) -> str:
    """Corps HTML (dans un CDATA, voir write_kml) de la bulle d'info
    Google Earth d'un site - MEME STYLE que l'ancien KML Fortran
    (etiquettes en gras `<B>...</B>`, sauts de ligne `<BR>`, strike/dip
    sur la meme ligne) - demande explicite utilisateur ("is it possible
    to have it more nicely like what I was doing in Fortran"). Chaque
    champ absent (strike/dip None, date/geologic_type/lithology/age
    vides) est simplement omis plutot qu'affiche a 0/vide, pour ne pas
    laisser croire a une valeur mesuree qui ne l'est pas - la valeur 0.0
    elle-meme (strike ou dip reellement mesure a zero) reste affichee,
    comme dans l'exemple Fortran."""
    parts = [f" <B> Nb pmag cores : </B> {s.n_specimens} <BR>"]
    if s.date:
        parts.append(f"<B> date : </B> {_xml_escape(s.date)}<BR>")
    if s.strike is not None and s.dip is not None:
        parts.append(f"<B>  Strike : </B>  {s.strike:.1f} <B>dip : </B>  {s.dip:.1f}<BR>")
    elif s.strike is not None:
        parts.append(f"<B>  Strike : </B>  {s.strike:.1f}<BR>")
    elif s.dip is not None:
        parts.append(f"<B>dip : </B>  {s.dip:.1f}<BR>")
    if s.geologic_type:
        parts.append(f"<B> Geologic type : </B> {_xml_escape(s.geologic_type)}<BR>")
    if s.lithology:
        parts.append(f"<B> Lithology : </B> {_xml_escape(s.lithology)}<BR>")
    if s.age:
        parts.append(f"<B> Age : </B> {_xml_escape(s.age)}<BR>")
    return "".join(parts)


def write_kml(
    sites: List[PrmagSite], filepath: str, doc_name: str = "Sites",
    style_url: str = _DEFAULT_STYLE_URL,
) -> None:
    """Un fichier .kml (Google Earth) - une Placemark/Point par site,
    nom = id de site, description = nombre de specimens + date + strike/
    dip + geologic type + lithology + age quand disponibles (voir
    _site_description_html), `<open>1</open>` (bulle d'info ouverte par
    defaut) et `<styleUrl>` (icone) - meme structure que l'ancien KML
    Fortran (demande explicite utilisateur)."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        f"<name>{_xml_escape(doc_name)}</name>",
    ]
    for s in sites:
        lines.append("<Placemark>")
        lines.append(f"<name>{_xml_escape(s.site)}</name>")
        lines.append(f"<description><![CDATA[{_site_description_html(s)}]]></description>")
        lines.append("<open>1</open>")
        if style_url:
            lines.append(f"<styleUrl>{_xml_escape(style_url)}</styleUrl>")
        lines.append("<Point>")
        lines.append(f"<coordinates>{s.lon:.6f},{s.lat:.6f},0</coordinates>")
        lines.append("</Point>")
        lines.append("</Placemark>")
    lines.append("</Document>")
    lines.append("</kml>")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# GMT (script + fichier de donnees, a executer par l'utilisateur)
# ----------------------------------------------------------------------

def _declutter_label_positions(
    sites: List[PrmagSite], min_sep_deg: float,
) -> List[Tuple[float, float]]:
    """Position d'etiquette par site, ecartee des etiquettes DEJA
    placees d'au moins `min_sep_deg` - demande explicite utilisateur ("I
    do not know if GMT can adjust the position of the labels when sites
    are close") : reponse breve, GMT lui-meme n'a PAS de placement
    automatique anti-collision pour des etiquettes de points (contraste
    avec matplotlib.adjustText ou un moteur d'etiquetage SIG) - ce n'est
    donc PAS GMT qui decale les etiquettes ici, mais ce calcul Python en
    amont, qui ecrit une position d'etiquette DEJA ecartee dans le
    fichier de donnees compagnon. Heuristique simple (glouton, spirale
    en angle d'or) - PAS un algorithme sophistique d'optimisation
    globale, suffisant pour quelques dizaines de sites groupes."""
    placed: List[Tuple[float, float]] = []
    out: List[Tuple[float, float]] = []
    for s in sites:
        lon, lat = s.lon, s.lat
        radius = 0.0
        angle = 0.0
        while any(math.hypot(lon - p[0], lat - p[1]) < min_sep_deg for p in placed):
            radius += min_sep_deg * 0.35
            angle += 137.5  # angle d'or : repartition uniforme en spirale
            lon = s.lon + radius * math.cos(math.radians(angle))
            lat = s.lat + radius * math.sin(math.radians(angle))
        placed.append((lon, lat))
        out.append((lon, lat))
    return out


def write_gmt_map_script(
    sites: List[PrmagSite], script_path: str, region_pad_deg: float = 0.3,
    label_min_sep_deg: Optional[float] = None,
) -> str:
    """Ecrit un script shell GMT6 pret a l'emploi - carte DE BASE (PAS de
    relief topographique - demande explicite utilisateur "forget the
    topography, and use a basic map"), continent en gris, lacs/rivieres
    en bleu, plus (voir commentaire ROUTES dans le script genere) un
    emplacement pret pour des routes, quand l'utilisateur dispose d'un
    fichier vectoriel de routes (GMT/`coast` n'a PAS de base de donnees
    routiere integree - voir cette meme docstring plus bas).

    3 fichiers de donnees compagnons ("<script>_sites.txt"/"_labels.txt"/
    "_leaders.txt", lon/lat[/site]) : positions REELLES des sites (`gmt
    plot`, symboles), positions D'ETIQUETTE ecartees quand des sites sont
    proches (`gmt text` - voir _declutter_label_positions, `label_min_
    sep_deg` par defaut ~4% de l'etendue de la carte) et segments de
    rappel (petit trait reliant marqueur et etiquette quand celle-ci a du
    etre decalee, `gmt plot -M`, invisibles/nuls quand aucun decalage
    n'etait necessaire). Meme convention "genere pour un pipeline GMT
    externe" que pu_rota2gmt ("GMT rotation export"), aucun appel direct
    a `gmt` ici. Retourne le chemin du fichier de sites."""
    base, _ext = os.path.splitext(script_path)
    sites_path = base + "_sites.txt"
    labels_path = base + "_labels.txt"
    leaders_path = base + "_leaders.txt"

    lats = [s.lat for s in sites]
    lons = [s.lon for s in sites]
    lon_min, lon_max = min(lons) - region_pad_deg, max(lons) + region_pad_deg
    lat_min, lat_max = min(lats) - region_pad_deg, max(lats) + region_pad_deg

    if label_min_sep_deg is None:
        label_min_sep_deg = 0.04 * max(lon_max - lon_min, lat_max - lat_min)
    label_positions = _declutter_label_positions(sites, label_min_sep_deg)

    with open(sites_path, "w", encoding="utf-8") as f:
        for s in sites:
            f.write(f"{s.lon:.6f}\t{s.lat:.6f}\t{s.site}\n")
    with open(labels_path, "w", encoding="utf-8") as f:
        for s, (llon, llat) in zip(sites, label_positions):
            f.write(f"{llon:.6f}\t{llat:.6f}\t{s.site}\n")
    with open(leaders_path, "w", encoding="utf-8") as f:
        for s, (llon, llat) in zip(sites, label_positions):
            f.write(f"> {s.site}\n{s.lon:.6f}\t{s.lat:.6f}\n{llon:.6f}\t{llat:.6f}\n")

    ps_name = os.path.splitext(os.path.basename(script_path))[0]

    script = f"""#!/bin/bash
# Genere par StereoUtils_Py (Pmag Utilities > Site Map > GMT map) -
# necessite GMT >= 6 (https://www.generic-mapping-tools.org) ; a executer
# soi-meme (jamais appele automatiquement).
set -e
cd "$(dirname "$0")"

gmt begin {ps_name} pdf,png
    # Carte de base (pas de relief/topographie) : continent en gris
    # (-G), lacs/rivieres-lacs en bleu (-C), rivieres perennes majeures
    # en bleu (-I1 - remplacer par -Ia pour TOUTES les rivieres/canaux,
    # plus dense), frontieres administratives fines (-N1), trait de
    # cote fin (-W).
    gmt coast -R{lon_min:.3f}/{lon_max:.3f}/{lat_min:.3f}/{lat_max:.3f} -JM15c -Df \\
        -G200 -C135/206/250 -I1/0.5p,blue -N1/0.4p,gray40 -Wthin,black
    gmt basemap -Baf -BWSne

    # ROUTES : ni `coast` ni GMT en general n'incluent de base de
    # donnees routiere (contrairement au trait de cote/lacs/rivieres,
    # issus de GSHHG/Natural Earth, deja integres). Pour les ajouter,
    # telecharger un fichier vectoriel de routes (ex. Natural Earth
    # "ne_10m_roads", https://www.naturalearthdata.com/downloads/10m-cultural-vectors/roads/),
    # le convertir si besoin en table lon/lat multi-segments (`ogr2ogr
    # -f OGR_GMT roads.gmt ne_10m_roads.shp`), puis decommenter :
    # gmt plot roads.gmt -R -J -Wthin,gray50

    gmt plot "{os.path.basename(sites_path)}" -Sc0.25c -Gred -Wblack
    # Segments de rappel marqueur -> etiquette (visibles seulement si
    # l'etiquette a ete decalee pour eviter un site voisin - voir
    # _declutter_label_positions).
    gmt plot "{os.path.basename(leaders_path)}" -Wthinnest,gray40
    gmt text "{os.path.basename(labels_path)}" -F+f8p,Helvetica,black+jTL -D0.1c/0.1c
gmt end show
"""
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    try:
        os.chmod(script_path, 0o755)
    except OSError:
        pass
    return sites_path


# ----------------------------------------------------------------------
# matplotlib (simple, sans fond topographique/cotes)
# ----------------------------------------------------------------------

def _load_populated_places(
    extent: Tuple[float, float, float, float], max_places: int = 20,
) -> List[Tuple[str, float, float, float]]:
    """Villes/villages Natural Earth (couche cultural/populated_places,
    meme mecanisme de telechargement/cache que les cotes - voir docstring
    de build_site_map_figure) DANS l'etendue de la carte, triees par
    population decroissante et limitees a `max_places` (une etendue,
    meme petite, peut en contenir des dizaines - la garder lisible plutot
    que tout afficher) - demande explicite utilisateur ("can we have
    roads and cities"). Toujours '10m' (seule resolution ou Natural Earth
    publie ces points, contrairement au trait de cote qui existe aussi en
    50m/110m). Retourne (nom, lon, lat, population)."""
    import cartopy.io.shapereader as shpreader

    path = shpreader.natural_earth(resolution="10m", category="cultural", name="populated_places")
    lon_min, lon_max, lat_min, lat_max = extent
    found = []
    for rec in shpreader.Reader(path).records():
        lon, lat = rec.geometry.x, rec.geometry.y
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            name = rec.attributes.get("NAME") or rec.attributes.get("NAME_EN") or ""
            pop = float(rec.attributes.get("POP_MAX") or 0)
            found.append((name, lon, lat, pop))
    found.sort(key=lambda r: r[3], reverse=True)
    return found[:max_places]


def _coastline_scale(lon_span: float, lat_span: float) -> str:
    """Resolution Natural Earth/cartopy adaptee a l'etendue des sites -
    inutile de charger le detail '10m' pour une carte couvrant tout un
    continent, et '110m' serait grossier pour un groupe de sites serres
    sur quelques km."""
    span = max(lon_span, lat_span)
    if span > 20.0:
        return "110m"
    if span > 5.0:
        return "50m"
    return "10m"


def _attach_zoom_selector(ax, is_geoaxes: bool) -> RectangleSelector:
    """Ajoute un rectangle de selection (clic-glisser, bouton gauche) qui
    zoome la carte sur la zone tracee - demande explicite utilisateur
    ("is it possible to select an area on the map for a zoom"). Double-
    clic pour revenir a l'etendue initiale (memorisee au moment de cet
    appel). Retourne le RectangleSelector - l'APPELANT doit en garder
    une reference forte (ici : attache directement sur `ax`, qui reste
    lui-meme reference par la Figure/le canvas Tk) : matplotlib ne
    retient pas les widgets lui-meme, un RectangleSelector sans
    reference externe est silencieusement garbage-collecte et cesse de
    repondre aux evenements - piege classique documente dans
    matplotlib.widgets."""
    full_xlim = ax.get_xlim()
    full_ylim = ax.get_ylim()

    def on_select(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        if x0 == x1 or y0 == y1:
            return
        if is_geoaxes:
            ax.set_extent([x0, x1, y0, y1], crs=ax.projection)
        else:
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
        ax.figure.canvas.draw_idle()

    def on_button_press(event):
        if event.dblclick and event.inaxes is ax:
            if is_geoaxes:
                ax.set_extent([*full_xlim, *full_ylim], crs=ax.projection)
            else:
                ax.set_xlim(full_xlim)
                ax.set_ylim(full_ylim)
            ax.figure.canvas.draw_idle()

    selector = RectangleSelector(
        ax, on_select, useblit=False, button=[1], interactive=False,
        props=dict(facecolor="none", edgecolor="tab:blue", linewidth=1.0, linestyle="--"))
    ax.figure.canvas.mpl_connect("button_press_event", on_button_press)
    return selector


def build_site_map_figure(
    sites: List[PrmagSite], fig: Optional[Figure] = None, title: str = "Sites",
    basemap: bool = True,
) -> Tuple[Figure, bool, Optional[str]]:
    """Carte des sites (lon/lat) - demande explicite utilisateur ("le
    simple matplotlib" puis "is there a possibility to have a basic map
    too"). `basemap=True` (defaut) essaie un fond de carte reel via
    cartopy (cotes/terres/ocean/frontieres, meme bibliotheque deja
    utilisee ailleurs dans cette appli pour "Plot VGPs on Map" -
    ipmag.make_orthographic_map) - resolution Natural Earth choisie selon
    l'etendue des sites (voir _coastline_scale). Cartopy telecharge ces
    donnees au premier usage (mises en cache ensuite) : SANS reseau (ou
    si cartopy n'est pas installe), retombe sur la version simple
    (grille, pas de fond de carte) plutot que de planter - retourne
    (figure, fond_de_carte_reellement_trace, message_erreur) - le 3e
    element est le texte EXACT de l'exception rencontree (type + message),
    PAS un message generique, pour que l'utilisateur
    (et nous) puissions diagnostiquer la VRAIE cause (certificat SSL,
    pas de reseau, cartopy absent, etc.) plutot que de deviner - demande
    implicite : le premier message generique ("cartopy not installed, or
    no network/cached coastline data") s'est revele inutilisable pour
    diagnostiquer un echec reel rencontre par l'utilisateur.

    Dans les deux cas : un point par site + etiquette, aspect corrige par
    la latitude moyenne (1 degre de longitude vaut moins qu'1 degre de
    latitude en distance reelle, sauf a l'equateur) - cartopy le fait
    nativement via sa projection, la version de repli le fait a la
    main (`ax.set_aspect`) - et un rectangle de selection (clic-glisser)
    pour zoomer, double-clic pour revenir a l'etendue initiale (voir
    _attach_zoom_selector - demande explicite utilisateur "is it
    possible to select an area on the map for a zoom")."""
    # Conserve le VRAI canvas (ex. le FigureCanvasTkAgg persistant de
    # l'appli, deja attache a `fig` avant cet appel) - le forçage de
    # rendu interne (`FigureCanvasAgg(fig).draw()`, voir plus bas) a pour
    # effet de bord de REASSIGNER `fig.canvas` a ce canvas Agg jetable
    # (matplotlib le fait automatiquement a la construction d'un
    # FigureCanvasBase) : sans cette sauvegarde/restauration, le
    # RectangleSelector construit ensuite se brancherait sur le mauvais
    # canvas (celui, mort, de la validation) et ne recevrait alors JAMAIS
    # les clics de l'utilisateur sur le canvas Tk reellement affiche.
    if fig is None:
        fig = Figure(figsize=(6.5, 6.5), dpi=100)
    else:
        fig.clear()
    original_canvas = getattr(fig, "canvas", None)

    lons = [s.lon for s in sites]
    lats = [s.lat for s in sites]
    pad_lon = max(0.05, (max(lons) - min(lons)) * 0.15) if lons else 1.0
    pad_lat = max(0.05, (max(lats) - min(lats)) * 0.15) if lats else 1.0
    extent = (min(lons) - pad_lon, max(lons) + pad_lon, min(lats) - pad_lat, max(lats) + pad_lat)

    error_text: Optional[str] = None
    if basemap:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            scale = _coastline_scale(extent[1] - extent[0], extent[3] - extent[2])
            ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="#cfe3f0", zorder=0)
            ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="#f0ece0", zorder=0)
            ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.7, zorder=1)
            ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.4, linestyle=":", zorder=1)
            # Routes (Natural Earth "roads", seulement en 10m - demande
            # explicite utilisateur "can we have roads and cities") :
            # contrairement au script GMT (qui n'a PAS acces a une base
            # de donnees routiere et doit s'appuyer sur un fichier fourni
            # par l'utilisateur), cartopy telecharge/met en cache cette
            # couche exactement comme le trait de cote - meme mecanisme,
            # meme repli en cas d'echec (voir plus bas).
            roads = cfeature.NaturalEarthFeature(
                "cultural", "roads", "10m", edgecolor="dimgray", facecolor="none")
            ax.add_feature(roads, linewidth=0.5, zorder=1)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
            gl.top_labels = gl.right_labels = False

            # Villes/villages (Natural Earth "populated_places", meme
            # demande utilisateur) - marqueur distinct des sites pmag
            # (carre noir vs cercle rouge) pour ne pas les confondre ;
            # pas de decluttering ici (contrairement aux sites, voir
            # _declutter_label_positions cote GMT) - le zoom interactif
            # deja en place (_attach_zoom_selector) suffit a demeler une
            # zone dense au besoin.
            places = _load_populated_places(extent)
            if places:
                ax.scatter(
                    [p[1] for p in places], [p[2] for p in places], transform=ccrs.PlateCarree(),
                    s=18, marker="s", color="black", zorder=2)
                for name, lon, lat, _pop in places:
                    ax.annotate(
                        name, (lon, lat), xycoords=ax.transData, textcoords="offset points",
                        xytext=(4, -3), fontsize=6, style="italic", color="dimgray")

            ax.scatter(
                lons, lats, transform=ccrs.PlateCarree(), s=40, color="tab:red",
                edgecolor="black", linewidth=0.6, zorder=3)
            for s in sites:
                ax.annotate(
                    s.site, (s.lon, s.lat), xycoords=ax.transData, textcoords="offset points",
                    xytext=(4, 3), fontsize=7)
            ax.set_title(title)
            # PAS `fig.tight_layout()` ici (bug reel reproduit et isole -
            # voir rapport d'exploration : `tight_layout()` + gridliner
            # AVEC etiquettes sur une GeoAxes declenche `_update_title_
            # position` -> `gl._draw_gridliner`, ou cartopy/shapely
            # construisent parfois un polygone de bordure de carte NON
            # ferme selon l'extent -> `GEOSException: Points of
            # LinearRing do not form a closed linestring` - confirme
            # INDEPENDANT du fond de carte lui-meme (memes plantage sans
            # aucune feature ajoutee, uniquement gridlines+labels+
            # tight_layout) ; marges fixes via `subplots_adjust`
            # suffisent et evitent ce chemin de code entierement) -
            # demande explicite utilisateur, message d'erreur exact
            # rapporte apres le premier correctif de diagnostic.
            fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.08)
            # cartopy differe le chargement/tracé reel des features (cotes,
            # gridliner...) jusqu'au premier DRAW (network fetch des
            # donnees Natural Earth inclus) - un `add_feature()` qui
            # reussit ne garantit donc rien : on force ce rendu ICI, DANS
            # le try/except, pour que l'echec (pas de reseau/donnees en
            # cache) declenche le repli ci-dessous plutot que de planter
            # plus tard, hors de portee de ce garde-fou (constate via
            # `fig.savefig()` levant une GEOSException cartopy/shapely a
            # ce stade, alors qu'`add_feature()` lui-meme n'avait rien
            # signale).
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            FigureCanvasAgg(fig).draw()
            if original_canvas is not None:
                fig.canvas = original_canvas
            ax._zoom_selector = _attach_zoom_selector(ax, is_geoaxes=True)
            return fig, True, None
        except Exception as e:
            error_text = f"{type(e).__name__}: {e}"
            fig.clear()
            if original_canvas is not None:
                fig.canvas = original_canvas

    ax = fig.add_subplot(111)
    ax.scatter(lons, lats, s=40, color="tab:red", edgecolor="black", linewidth=0.6, zorder=3)
    for s in sites:
        ax.annotate(
            s.site, (s.lon, s.lat), textcoords="offset points", xytext=(4, 3), fontsize=7)

    if lats:
        mean_lat = sum(lats) / len(lats)
        ax.set_aspect(1.0 / max(0.05, math.cos(math.radians(mean_lat))), adjustable="box")
    ax.grid(True, linewidth=0.4, color="0.85")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    fig.tight_layout()
    ax._zoom_selector = _attach_zoom_selector(ax, is_geoaxes=False)
    return fig, False, error_text
