"""
Menu "Project" (StereoOSX_x.f95:360-515 + plotstereo.f95) : systeme de
trace multi-couches ("Illustrator layers") a partir d'un fichier texte
listant des directions/moyennes/grands cercles groupes par "layer" avec
couleur RGB et symbole individuels - independant du modele /data/ (D-I,
D-I-a95, grands cercles) utilise par les autres menus, meme si Export to
Project peut construire un projet a partir de ce modele.

Format de ligne (voir `Exemple_Stereo_dataFile.txt`/`project_stereo.txt`,
verifie sur `loadproject`, StereoOSX_x.f95:365-404) :
    layer  id  type(d/m/g)  dec  inc  alpha95  iplot(2/3)  symbol  rgb  size
`type='d'` : direction individuelle (`iplot=3` demarre un nouveau trace,
`iplot=2` relie au point precedent du meme calque par un arc de grand
cercle). `type='m'` : moyenne avec cercle de confiance a95. `type='g'` :
grand cercle (dec/inc = pole). Quirk reel du source, replique ici : une
ligne dont `alpha95==0.0` voit son `type` force a "d" au chargement, quel
que soit le type ecrit dans le fichier."""

import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from plotlib import PlotContext
from stereo_geometry import superc, gdcerc, circle
from stereo_net import _SYMBOL_TYPES, draw_stereo_net
import stereo_stats as ss
from stereo_selection import VgpProjectEntry, _parse_vgp_project_lines, split_header


class ProjectEntry(NamedTuple):
    layer: str
    id: str
    type: str  # 'd', 'm', 'g'
    dec: float
    inc: float
    alph: float
    iplot: int  # 2 (relie au precedent) ou 3 (nouveau trace)
    symb: str
    rgb: str  # "R_G_B" ou un nom de couleur (voir NAMED_COLORS)
    size: float


# Palette de ~20 couleurs nommees (extension hors source Fortran, demande
# explicite utilisateur : pouvoir ecrire un nom ("red") a la place de
# "R_G_B" dans un fichier projet, plutot que de devoir deviner un triple
# RGB a la main). `decode_color` accepte les deux formats.
NAMED_COLORS: dict = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 140, 0),
    "purple": (128, 0, 128),
    "pink": (255, 105, 180),
    "brown": (139, 69, 19),
    "gray": (128, 128, 128),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "lime": (50, 205, 50),
    "navy": (0, 0, 128),
    "teal": (0, 128, 128),
    "gold": (255, 215, 0),
    "violet": (148, 0, 211),
    "turquoise": (64, 224, 208),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "salmon": (250, 128, 114),
    "skyblue": (135, 206, 235),
}

# Symboles reconnus par `testcarsymb`/`_SYMBOL_TYPES` (plotstereo.f95:236-276).
SYMBOL_NAMES = {
    "c": "circle", "t": "triangle", "e": "star", "l": "diamond (losange)", "s": "square",
}


def decode_color(rgb: str) -> Tuple[int, int, int]:
    """Port de `decodcolor` (plotstereo.f95:277-289) : "R_G_B" -> (r,g,b) -
    ETENDU pour accepter aussi un nom de `NAMED_COLORS` (insensible a la
    casse). (0,0,0) si ni l'un ni l'autre format n'est reconnu (meme repli
    que le source)."""
    name = rgb.strip().lower()
    if name in NAMED_COLORS:
        return NAMED_COLORS[name]
    try:
        r, g, b = (int(v) for v in rgb.replace("_", " ").split())
        return r, g, b
    except (ValueError, TypeError):
        return 0, 0, 0


def _read_text_lines(path: str):
    """Essaie UTF-8 (fichiers projet modernes, ex. avec "±") puis retombe
    sur iso-8859-1 (convention des autres lecteurs de ce port, fichiers
    plus anciens) si le decodage UTF-8 echoue."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
    except UnicodeDecodeError:
        with open(path, "r", encoding="iso-8859-1", errors="replace") as f:
            return f.readlines()


def load_project(path: str) -> List[ProjectEntry]:
    """Port de `loadproject` (StereoOSX_x.f95:365-404) : lit un fichier
    projet (une ligne par entree, "!" en debut de ligne = commentaire,
    lecture list-directed = champs separes par des espaces)."""
    entries = []
    for line in _read_text_lines(path):
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        parts = stripped.split()
        if len(parts) < 10:
            continue
        try:
            layer, eid, etype = parts[0], parts[1], parts[2]
            dec, inc, alph = float(parts[3]), float(parts[4]), float(parts[5])
            iplot = int(float(parts[6]))
            symb, rgb = parts[7], parts[8]
            size = float(parts[9])
        except ValueError:
            continue
        if alph == 0.0:
            etype = "d"
        entries.append(ProjectEntry(layer, eid, etype, dec, inc, alph, iplot, symb, rgb, size))
    return entries


# Marqueurs de bloc (voir STARpaleomag_Py/export_stereo.export_stereo_project)
# -> cle de retour de load_project_blocks - reperes par sous-chaine,
# insensible a la casse, sur la ligne "# ..." qui precede l'en-tete de
# colonnes de chaque bloc.
_BLOCK_MARKERS = {
    "individual direction": "directions",
    "mean direction": "means",
    "vgp": "vgp",
}


def _split_project_blocks(lines: Sequence[str]) -> Dict[str, List[str]]:
    """Decoupe un fichier "Stereo Project" a 3 blocs (voir _BLOCK_MARKERS)
    en {cle_bloc: [lignes du bloc, EN-TETE DE COLONNES INCLUS]} - la ligne
    marqueur elle-meme ("# individual directions") n'est PAS incluse, la
    ligne suivante ("#layer\tid\t...", l'en-tete de colonnes attendu par
    split_header) l'est. {} si aucun marqueur reconnu (fichier "Project"
    classique a plat, sans bloc - voir load_project_blocks)."""
    blocks: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            marker = stripped.lstrip("#").strip().lower()
            matched = next((key for pat, key in _BLOCK_MARKERS.items() if pat in marker), None)
            if matched is not None:
                current = matched
                blocks.setdefault(current, [])
                continue
        if current is not None:
            blocks[current].append(line)
    return blocks


def _parse_direction_block(lines: Sequence[str], mean_type: Optional[str] = None) -> List[ProjectEntry]:
    """Lignes d'un bloc "individual directions" (mean_type=None, `type`
    lu colonne par colonne, 'd' ou 'g') ou "mean directions" (mean_type=
    'm', bloc entierement moyennes - pas de colonne `type`) d'un fichier
    "Stereo Project" a 3 blocs -> ProjectEntry (memes objets que
    load_project, directement compatibles avec draw_project/
    fisher_project/format_project_lines - `tilt_correction`/`info` sont
    lus par _split_project_blocks/split_header mais n'ont pas de champ
    correspondant dans ProjectEntry, volontairement ignores ici : ce sont
    des colonnes d'ARCHIVAGE/notes manuelles, pas des parametres de
    trace)."""
    field_keys = ["layer", "id", "dec", "inc", "alpha95", "symbol", "rgb", "size"]
    if mean_type is None:
        field_keys.append("type")
    data_lines, idx = split_header(lines, *field_keys)
    if "dec" not in idx or "inc" not in idx:
        return []
    out = []
    for line in data_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            continue
        parts = line.split("\t") if "\t" in line else line.split()

        def field(name: str, default: str = "") -> str:
            j = idx.get(name)
            return parts[j].strip() if j is not None and j < len(parts) else default

        try:
            dec, inc = float(field("dec")), float(field("inc"))
        except ValueError:
            continue
        try:
            alph = float(field("alpha95", "0"))
        except ValueError:
            alph = 0.0
        try:
            size = float(field("size", "0.30"))
        except ValueError:
            size = 0.30
        etype = mean_type or (field("type", "d") or "d")
        if alph == 0.0 and etype != "g":
            etype = "d"  # meme quirk que load_project (voir sa docstring)
        out.append(ProjectEntry(
            layer=field("layer", "Layer"), id=field("id", "?"), type=etype,
            dec=dec, inc=inc, alph=alph, iplot=3,
            symb=field("symbol", "c") or "c", rgb=field("rgb", "0_0_0") or "0_0_0", size=size,
        ))
    return out


def load_project_blocks(path: str) -> Tuple[List[ProjectEntry], List[VgpProjectEntry]]:
    """Lit un fichier "Stereo Project" a 3 blocs (individual directions/
    mean directions/VGP, chacun avec son propre en-tete de colonnes, une
    colonne `tilt_correction` 0-100 et une colonne `info` libre - voir
    STARpaleomag_Py/export_stereo.export_stereo_project) - demande
    explicite utilisateur ("can you manage a stereo project with three
    blocks (individual directions; mean directions, and VGP, each block
    with its own labels"). Retourne (entries, vgp_entries) :
    - `entries` : List[ProjectEntry] - directions + moyennes reunies,
      DIRECTEMENT utilisables par tout le reste du menu Project existant
      (Plot/List/Fisher Project) sans aucun changement la-bas.
    - `vgp_entries` : List[VgpProjectEntry] - voir
      stereo_pmagpy.plot_vgp_project (trace sur une carte, PAS le reseau
      stereo - un VGP est un point geographique, pas une direction).

    Fichier SANS aucun des 3 marqueurs de bloc reconnus (ancien format
    "Project" a plat, une seule liste sans en-tete) -> repli TRANSPARENT
    sur load_project (retro-compatible, `vgp_entries` vide)."""
    lines = _read_text_lines(path)
    blocks = _split_project_blocks(lines)
    if not blocks:
        return load_project(path), []
    entries = (
        _parse_direction_block(blocks.get("directions", []))
        + _parse_direction_block(blocks.get("means", []), mean_type="m")
    )
    vgp_entries = _parse_vgp_project_lines(blocks.get("vgp", []))
    return entries, vgp_entries


def load_vgp_entries(path: str) -> List[VgpProjectEntry]:
    """Resout les VGP d'un fichier, quel que soit son format - utilise par
    "Plot VGP Project..." (app.py) plutot que load_project_blocks
    directement : celui-ci retombe sur load_project (parseur STEREONET a
    plat, colonnes POSITIONNELLES) des qu'aucun marqueur de bloc n'est
    trouve - sur (dec/inc/site/nb/a95/dp/dm/p95/symbol/rgb/size), ce
    repli produirait un mauvais decoupage silencieux (des colonnes VGP
    interpretees a tort comme layer/id/type/dec/inc..., sans erreur
    visible) plutot qu'un simple fichier vide. Ici : bloc "# VGP" si
    present, sinon tente le fichier ENTIER comme VGP autonome (ancien
    format, dec/inc = VGP_lon/VGP_lat, voir
    stereo_selection.read_vgp_project_file) - jamais le repli
    load_project, non pertinent pour des VGP."""
    lines = _read_text_lines(path)
    blocks = _split_project_blocks(lines)
    if "vgp" in blocks:
        return _parse_vgp_project_lines(blocks["vgp"])
    return _parse_vgp_project_lines(lines)


def export_project(
    directions: Sequence[Tuple[float, float, str]],
    means: Sequence[Tuple[float, float, float, str]],
    great_circles: Sequence[Tuple[float, float, float, float, float, float]],
) -> List[ProjectEntry]:
    """Port de `export2project` (StereoOSX_x.f95:406-481) : construit des
    entrees de projet a partir de self.directions/self.means/
    self.great_circles - NOTE: le source original ecrit `dic(j)/inc(j)`
    (les tableaux de MOYENNES) pour les grands cercles au lieu de
    `glon(j)/glat(j)` (les vrais poles de grand cercle) - copier-coller
    manifestement fautif (les deux tableaux sont bien distincts dans
    `common /data/`), CORRIGE ici (utilise glon/glat, les vraies
    coordonnees du pole)."""
    entries = []
    for dec, inc, sym in directions:
        entries.append(ProjectEntry("Directions", "sample", "d", dec, inc, 0.0, 2, sym, "0_0_0", 0.3))
    for dec, inc, alph, sym in means:
        entries.append(ProjectEntry("Mean_Direction", "sample", "m", dec, inc, alph, 2, sym, "0_0_0", 0.5))
    for glon, glat, _ad1, _ai1, _ad2, _ai2 in great_circles:
        entries.append(ProjectEntry("GreatCircle", "sample", "g", glon, glat, 0.0, 2, "c", "0_0_0", 0.1))
    return entries


def format_project_lines(entries: Sequence[ProjectEntry]) -> str:
    """Port du format d'affichage de `listproject`/`export2project`
    (StereoOSX_x.f95:200 FORMAT) - une ligne par entree, directement
    recopiable dans un fichier projet."""
    lines = []
    for e in entries:
        lines.append(
            f"{e.layer:<14s} {e.id:<12s} {e.type}  {e.dec:6.1f}  {e.inc:6.1f}  {e.alph:6.1f}  "
            f"{e.iplot:2d}  {e.symb}  {e.rgb:<11s} {e.size:6.2f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _consecutive_layers(entries: Sequence[ProjectEntry]):
    """Groupe les entrees par calque CONSECUTIF (pas un regroupement
    global par nom - si un calque reapparait plus loin dans le fichier
    apres un autre calque, c'est un groupe separe), meme logique de
    balayage sequentiel que `fisherproject`/`plotproject`."""
    if not entries:
        return
    cur_layer = entries[0].layer
    cur = [entries[0]]
    for e in entries[1:]:
        if e.layer != cur_layer:
            yield cur_layer, cur
            cur_layer, cur = e.layer, []
        cur.append(e)
    yield cur_layer, cur


def fisher_project(
    entries: Sequence[ProjectEntry], rfilt: float = 0.0,
) -> Tuple[str, List[Tuple[str, float, float, float, str]]]:
    """Port de `fisherproject` (plotstereo.f95:353-411, menu Project >
    Fisher Project) : pour chaque groupe de lignes consecutives de meme
    calque (TOUS types confondus - le source ne filtre pas sur `.type`,
    replique tel quel), calcule la moyenne de Fisher (`ANGU`, avec
    separation en modes) et imprime une ligne pret-a-copier au format
    projet ("layer   mean  m  dec inc a95   3   e  0_0_0   0.55  N=n k=k").
    `rfilt` : un seul filtre partage pour tous les calques (le source
    reprompte `rfilt` a chaque calque via `ANGU`, simplifie ici a une
    valeur unique - n'affecte aucun resultat numerique, juste l'UX)."""
    if len(entries) < 2:
        return "not enough data\n", []
    lines = []
    layer_means: List[Tuple[str, float, float, float, str]] = []
    for layer, group in _consecutive_layers(entries):
        if len(group) < 2:
            continue
        pts = [(e.dec, e.inc) for e in group]
        text, means, last_stats = ss.angu(pts, rfilt=rfilt, return_last_stats=True)
        if not means:
            continue
        dec, inc, a95, _sym = means[-1]
        n, k = last_stats["n"], last_stats["k"]
        lines.append(
            f"{layer}   mean  m  {dec:6.1f}  {inc:6.1f}  {a95:6.1f}"
            f"    3   e  0_0_0   0.55     N= {n:3d} k= {k:6.1f}"
        )
        layer_means.append((layer, dec, inc, a95, "e"))
    return "\n".join(lines) + ("\n" if lines else ""), layer_means


# ---------------------------------------------------------------------------
# Trace (plotproject)
# ---------------------------------------------------------------------------

def draw_project(
    ctx: PlotContext,
    entries: Sequence[ProjectEntry],
    r: float,
    la: float = -90.0,
    phi: float = 0.0,
    iproj: int = 0,
) -> None:
    """Port de la boucle de trace de `plotproject` (plotstereo.f95:107-231)
    pour les 3 types d'entree ('m'/'g'/'d'), couleur RGB individuelle par
    entree (`decodcolor`+`newpencol`). Le trace de chemin des entrees 'd'
    (`iplot`) redemarre a chaque changement de calque (le source ne reset
    jamais son point "precedent" entre calques - un artefact de variable
    Fortran non reinitialisee plutot qu'un comportement voulu, non
    reproduit ici puisque les fichiers projet reels commencent toujours
    un calque par une ligne `iplot=3`)."""
    layer0 = None
    prev_point = None  # (dec, dip) du dernier point 'd' trace, reinitialise par calque
    for e in entries:
        if e.layer != layer0:
            layer0 = e.layer
            prev_point = None
            # Equivalent de `newlayer(plotdir(ikk).layer)` (plotstereo.f95) -
            # demande explicite utilisateur ("mon exportation svg anterieure
            # qui gardait les calques") : voir PlotContext.set_gid pour le
            # detail du mecanisme (Artist.set_gid plutot que reecrire
            # </g>/<g id=...> a la main).
            ctx.set_gid(layer0)
        ir, ig, ib = decode_color(e.rgb)
        ctx.newpencol(ir, ig, ib, ir, ig, ib)
        if e.type == "m":
            ctx.thickn(0.55)
            u, v, ifl = superc(la, phi, e.dec, e.inc, iproj)
            open_t, filled_t = _SYMBOL_TYPES.get(e.symb, _SYMBOL_TYPES["c"])
            ityp = open_t if ifl == 5 else filled_t
            ctx.symbol(v * r, u * r, e.size, ityp, -1)
            ctx.thickn(0.45)
            ei0, ed0 = circle(e.alph, e.inc, e.dec, 0.0)
            uu, vv, _ifl = superc(la, phi, ed0, ei0, iproj)
            ctx.plot(vv * r, uu * r, 3)
            ph = 0.0
            for _i in range(120):
                ph += 3.0
                ei, ed = circle(e.alph, e.inc, e.dec, ph)
                uu, vv, _ifl = superc(la, phi, ed, ei, iproj)
                ctx.plot(vv * r, uu * r, 2)
        elif e.type == "g":
            ctx.thickn(0.65)
            glat = e.inc
            if glat < 0.0:
                dipp, decc = glat + 90.0, e.dec
            else:
                dipp, decc = -glat + 90.0, e.dec + 180.0
            dec, dip = decc, dipp
            u, v, _ifl = superc(la, phi, dec, dip, iproj)
            ctx.plot(v * r, u * r, 3)
            for i in range(1, 5):
                dec2 = decc + 90.0
                dip2 = 0.0
                if i == 2:
                    dip2 = -dipp
                if i == 4:
                    dip2 = dipp
                gdcerc(ctx, la, phi, iproj, decc, dip, dec2, dip2, r)
                decc += 90.0
                dip = dip2
        elif e.type == "d":
            if e.iplot == 2 and prev_point is not None:
                gdcerc(ctx, la, phi, iproj, prev_point[0], prev_point[1], e.dec, e.inc, r)
            prev_point = (e.dec, e.inc)
            u, v, ifl = superc(la, phi, e.dec, e.inc, iproj)
            open_t, filled_t = _SYMBOL_TYPES.get(e.symb, _SYMBOL_TYPES["c"])
            ityp = open_t if ifl == 5 else filled_t
            ctx.symbol(v * r, u * r, e.size, ityp, -1)
    ctx.newpen(1)
    ctx.set_gid(None)


def build_project_figure(
    entries: Sequence[ProjectEntry],
    la: float = -90.0, phi: float = 0.0, iproj: int = 0, dim: float = 21.0,
    fig=None,
):
    """Equivalent de `plotproject` (menu Project > Plot Project) : cadre du
    reseau + toutes les entrees du projet."""
    from matplotlib.figure import Figure
    if fig is None:
        fig = Figure(figsize=(5.5, 5.5), dpi=100)
    else:
        fig.clear()
    ax = fig.add_subplot(111)
    ctx = PlotContext(ax)
    ctx.clear()
    ctx.plot(0.0, 0.0, -3)
    r = draw_stereo_net(ctx, la, phi, iproj, dim)
    draw_project(ctx, entries, r, la, phi, iproj)
    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    return fig
