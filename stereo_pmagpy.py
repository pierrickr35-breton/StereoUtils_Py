"""
Menu "Pmag_Python" : port de `PmagPy_connect.f95` (Stereo_V19). Le Fortran
d'origine ne reimplemente RIEN lui-meme ici - chaque routine ecrit les
directions en memoire dans un fichier texte puis SHELLE OUT (Terminal.app
via AppleScript, ou EXECUTE_COMMAND_LINE) vers un script PmagPy en ligne de
commande (`plot_map_pts.py`, `incfish.py`, `foldtest.py`, `common_mean.py`,
`find_EI.py`, `revtest.py`). Le port ici saute l'etape fichier+sous-processus
et appelle DIRECTEMENT la fonction Python equivalente du paquet PmagPy
installe (`pip install pmagpy` - meme paquet reutilise pour AMS_Py/
ams_bootstrap.py) :

  plotvgp          (plot_map_pts.py)  -> ipmag.make_orthographic_map + ipmag.plot_vgp
  meanincpmagpy    (incfish.py)       -> pmag.doincfish
  foldtestpmagpy   (foldtest.py)      -> ipmag.bootstrap_fold_test
  common_mean      (common_mean.py)   -> ipmag.common_mean_bootstrap
  find_EI          (find_EI.py)       -> ipmag.find_ei
  test_antipodal   (revtest.py)       -> ipmag.reversal_test_bootstrap

Ajout hors source Fortran (demande explicite utilisateur, pas un item du
menu Pmag_Python d'origine dans PmagPy_connect.f95) :
  fisher_mean                        -> pmag.fisher_mean (Fisher 1953,
      calcul direct via PmagPy plutot que le portage natif `ANGU` du menu
      Statistics - meme resultat numerique attendu sur un mode unimodal,
      mais SANS la separation en modes normal/reverse d'ANGU/MODES).

Les fonctions `ipmag.*` de haut niveau tracent elles-memes via pyplot (pas
de parametre `fig=` compatible avec un canvas Tkinter integre) : ce module
les appelle avec `save=True`, recupere le(s) fichier(s) image produits, et
laisse l'appelant (app.py) les recharger dans son propre panneau - plutot
que d'essayer de transferer une Figure pyplot vers un canvas different."""

import glob
import os
import tempfile
from typing import List, Optional, Tuple

from stereo_selection import read_text_lines, split_header
import field_uncertainty as fu

# cartopy (utilise par ipmag.make_orthographic_map pour "Plot VGPs on Map")
# telecharge les traits de cote Natural Earth au premier usage - la version
# Python.framework de python.org sur macOS n'a pas de certificats CA lies au
# trousseau systeme, ce qui fait echouer ce telechargement (SSL: CERTIFICATE_
# VERIFY_FAILED). Meme correctif que `pip install certify && Install
# Certificates.command` : pointer SSL_CERT_FILE vers le paquet certifi deja
# installe (dependance de pandas/matplotlib), avant tout import cartopy.
if "SSL_CERT_FILE" not in os.environ:
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from pmagpy import ipmag, pmag

from stereo_project import decode_color


def _run_and_capture(func, *args, fmt: str = "png", **kwargs) -> Tuple[object, List[str]]:
    """Appelle une fonction PmagPy qui trace via pyplot (`save`/`save_folder`
    /`fmt`), dans un dossier temporaire dedie, et retourne (resultat,
    fichiers images produits).

    Certaines fonctions PmagPy (ex. `common_mean_bootstrap`, appelee aussi
    par `reversal_test_bootstrap`) appellent `plt.show()` en interne, qui
    BLOQUE indefiniment avec le backend GUI par defaut (attend la fermeture
    d'une fenetre qui n'existe pas dans ce contexte non-interactif) - verifie
    en pratique (appel reste bloque >100s). On neutralise localement
    `plt.show` le temps de l'appel plutot que de forcer le backend global
    matplotlib a "Agg" (ce qui casserait le canvas Tkinter deja actif du
    reste de l'app)."""
    save_folder = tempfile.mkdtemp(prefix="stereoutils_")
    plt.close("all")
    original_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        result = func(*args, save=True, save_folder=save_folder, fmt=fmt, **kwargs)
    finally:
        plt.show = original_show
    plt.close("all")
    files = sorted(glob.glob(os.path.join(save_folder, f"*.{fmt}")))
    return result, files


def mean_inclination(directions: List[Tuple[float, float]], method: str = "mcfadden_reid") -> dict:
    """Equivalent de `meanincpmagpy`/`incfish.py` : `pmag.doincfish` sur
    les seules inclinaisons (McFadden & Reid 1982 par defaut)."""
    incs = [inc for _dec, inc in directions]
    return pmag.doincfish(incs, method=method)


def fisher_mean(directions: List[Tuple[float, float]]) -> dict:
    """Menu Pmag_Python > Fisher : `pmag.fisher_mean` (Fisher 1953)
    directement sur `directions`, sans separation en modes (contrairement
    a `stereo_stats.angu`, qui separe via `MODES` avant de calculer)."""
    di_block = [[dec, inc] for dec, inc in directions]
    fpars = pmag.fisher_mean(di_block)
    return {k: (int(v) if k == "n" else float(v)) for k, v in fpars.items()}


def generate_fisher_population(
    k: float, n: int, dec: float = 0.0, inc: float = 90.0, random_seed=None,
) -> List[Tuple[float, float]]:
    """Population synthetique de `n` directions fisheriennes (kappa `k`,
    direction moyenne `dec`/`inc`) - ajout hors source Fortran, demande
    explicite utilisateur ("trouver dans PmagPy la generation de
    population de directions avec une distribution fisherienne
    caracterisee par le parametre k (fishgen)") : aucun equivalent
    "fishgen" natif dans le Fortran de ce projet, direct `ipmag.fishrot`
    (genere `pmag.fshdev` dans le repere polaire standard - dec/inc
    fisherien autour du pole - puis fait pivoter le nuage vers dec/inc
    demandes via `pmag.dodirot_V`, meme technique que `pmag.fisher_mean`
    utilise en sens inverse pour ses propres tests)."""
    dec_arr, inc_arr = ipmag.fishrot(k=k, n=n, dec=dec, inc=inc, di_block=False, random_seed=random_seed)
    return [(float(d), float(i)) for d, i in zip(dec_arr, inc_arr)]


def bootstrap_ellipse(
    directions: List[Tuple[float, float]], num_sims: int = 1000, alpha: float = 0.05, random_seed=None,
) -> Tuple[dict, List[str]]:
    """Menu Pmag_Python > Bootstrap ellipse (ajout hors source Fortran,
    demande explicite) : `ipmag.mean_bootstrap_confidence` (Heslop et al.
    2023) + `ipmag.plot_bootstrap_confidence` - equivalent PmagPy du menu
    Statistics > Bootstrap ellipse, dont le portage natif Fortran a ete
    explicitement ecarte (RNG desactive dans le source, cf. memoire projet
    `project_stereoutils_pmagoutils_bugs.md` bug #1) - celui-ci est un
    vrai bootstrap, avec un generateur alcatoire fonctionnel.

    Contrairement a `mean_inclination`/`fisher_mean`/`find_elongation`
    (qui acceptent `save=True, save_folder=...`), `plot_bootstrap_confidence`
    dessine directement sur les axes courants : la figure est construite
    et sauvegardee manuellement ici plutot que via `_run_and_capture`."""
    di_block = [[dec, inc] for dec, inc in directions]
    params, confidence_di = ipmag.mean_bootstrap_confidence(
        di_block=di_block, num_sims=num_sims, alpha=alpha, random_seed=random_seed)
    params = {k: float(v) for k, v in params.items()}

    save_folder = tempfile.mkdtemp(prefix="stereoutils_")
    plt.close("all")
    fig = plt.figure(figsize=(6, 6), dpi=120)
    ipmag.plot_net(fig.number)
    ipmag.plot_bootstrap_confidence(params["dec"], params["inc"], confidence_di)
    path = os.path.join(save_folder, "bootstrap_ellipse.png")
    fig.savefig(path)
    plt.close("all")
    return params, [path]


def find_elongation(directions: List[Tuple[float, float]], nb: int = 1000, random_seed=None):
    """Equivalent de `find_EI`/`find_EI.py` : `ipmag.find_ei`, bootstrap de
    correction d'aplatissement (Tauxe & Kent)."""
    data = [[dec, inc] for dec, inc in directions]
    return _run_and_capture(ipmag.find_ei, data, nb=nb, random_seed=random_seed)


def test_reversal_antipodal(directions: List[Tuple[float, float]], random_seed=None):
    """Equivalent de `test_antipodal`/`revtest.py` : `ipmag.reversal_test_bootstrap`."""
    dec = [d for d, _i in directions]
    inc = [i for _d, i in directions]
    return _run_and_capture(ipmag.reversal_test_bootstrap, dec, inc, plot_stereo=True, random_seed=random_seed)


def test_common_mean(
    directions1: List[Tuple[float, float]], directions2: List[Tuple[float, float]],
    nb: int = 1000, random_seed=None,
):
    """Equivalent de `common_mean`/`common_mean.py` : `ipmag.common_mean_bootstrap`."""
    d1 = [[dec, inc] for dec, inc in directions1]
    d2 = [[dec, inc] for dec, inc in directions2]
    return _run_and_capture(ipmag.common_mean_bootstrap, d1, d2, NumSims=nb, random_seed=random_seed)


def fold_test(
    data: List[Tuple[float, float, float, float]],
    nb: int = 1000, bedding_error: float = 2.0, random_seed=None,
):
    """Equivalent de `foldtestpmagpy`/`foldtest.py` : `ipmag.bootstrap_fold_test`
    (Tauxe & Watson 1994). `data` : (dec,inc,dip_direction,dip) - convertie
    en tableau numpy, `pmag.dotilt_V` (appelee en interne) exige `.transpose()`
    et rejette une simple liste Python."""
    arr = np.array(data, dtype=float)
    return _run_and_capture(
        ipmag.bootstrap_fold_test, arr, num_sims=nb, bedding_error=bedding_error, random_seed=random_seed)


def tilt_corrected_directions(data: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float]]:
    """Directions tilt-corrigees (repere "Tilt-corrected", le 2e des 3
    graphiques de `ipmag.bootstrap_fold_test` - "eq_tc") : `data` =
    (dec,inc,dip_direction,dip), memes valeurs que celles passees a
    `fold_test`. Reproduit exactement `D,I=pmag.dotilt_V(Data)` tel
    qu'appele en interne par `bootstrap_fold_test` pour produire ce
    2e stereo (correction complete a 100%, PAS le pourcentage optimal du
    bootstrap - celui-ci varie par simulation et n'est pas trace tel
    quel)."""
    arr = np.array(data, dtype=float)
    dec, inc = pmag.dotilt_V(arr)
    return list(zip((float(d) for d in dec), (float(i) for i in inc)))


def plot_vgps_on_map(directions: List[Tuple[float, float]], view_lat: float = 0.0, view_lon: float = 0.0):
    """Equivalent de `plotvgp`/`plot_map_pts.py` : `ipmag.make_orthographic_map`
    + `ipmag.plot_vgp` (les dec/inc en memoire representent directement les
    longitude/latitude du VGP - meme convention que le Fortran, qui les
    ecrit tels quels dans le fichier passe a `plot_map_pts.py`)."""
    plt.close("all")
    ax = ipmag.make_orthographic_map(central_longitude=view_lon, central_latitude=view_lat)
    di_block = [[dec, inc] for dec, inc in directions]
    ipmag.plot_vgp(ax, di_block=di_block, color="k", marker="o", markersize=30)
    save_folder = tempfile.mkdtemp(prefix="stereoutils_")
    path = os.path.join(save_folder, "vgp_map.png")
    plt.savefig(path, dpi=120)
    plt.close("all")
    return None, [path]


# symbole (voir stereo_project.SYMBOL_NAMES, meme convention texte c/t/e/l/s
# que le reseau stereo) -> marqueur matplotlib pour ipmag.plot_vgp (qui
# attend un marqueur matplotlib reel, pas le code entier Fortran de
# stereo_net._SYMBOL_TYPES - deux systemes de symboles differents, celui-ci
# est le SEUL a s'appliquer ici puisque plot_vgp trace via pyplot/cartopy,
# pas via PlotContext).
_VGP_SYMBOL_TO_MARKER = {"c": "o", "t": "^", "e": "*", "l": "D", "s": "s"}


def plot_vgp_project(entries, view_lat: float = 0.0, view_lon: float = 0.0):
    """Equivalent-carte de stereo_project.draw_project (menu Project, reseau
    stereo) pour des VGP - demande explicite utilisateur ("we will need to
    update the project in Stereo to manage the VGP plot") : `entries`
    (stereo_selection.VgpProjectEntry, via read_vgp_project_file) porte
    DEJA symbole/couleur/taille par VGP (ecrits par STARpaleomag_Py/
    export_stereo.export_stereo_project) - contrairement a
    `plot_vgps_on_map` (une seule couleur/un seul marqueur pour TOUS les
    points, `self.directions` sans cette info), chaque VGP est ici trace
    INDIVIDUELLEMENT avec ses propres reglages.

    Ovale de confiance - demande explicite utilisateur ("is it possible to
    plot the VGP with their dp,dm ellipse") : la VRAIE ellipse dp/dm
    (asymetrique, orientee le long du meridien site->pole - voir Butler
    1992 fig. A.2) via `ipmag.plot_pole_dp_dm` DES QUE la position du SITE
    est connue (`e.site_lat`/`e.site_lon` non nuls - voir
    STARpaleomag_Py/export_stereo.export_stereo_project, colonnes
    site_lat/site_lon du bloc "# VGP") : c'est cette meme fonction PmagPy
    qui calcule l'angle de rotation correct (loi des cosinus spherique sur
    le triangle site/pole nord/paleopole), aucune nouvelle geometrie
    introduite ici. `plot_pole_dp_dm` trace AUSSI le site lui-meme (marqueur
    carre, meme couleur que le pole) - c'est la geometrie meme qui explique
    l'orientation de l'ellipse, pas une donnee en trop.

    Repli en simple CERCLE de rayon `p95` (moyenne dp/dm, voir
    calcul.dp_dm_from_a95) via `pmag.circ` quand le site n'est PAS connu
    (fichier plus ancien sans site_lat/site_lon, ou VGP sans site unique) -
    convention deja utilisee ailleurs dans ce meme projet pour les VGP
    d'un APWP (voir app.py, colonne "p95" de `read_apwp_file`).

    TROIS CORRECTIONS reelles apportees ici (demande explicite utilisateur,
    sur un vrai projet de 48 sites - Paleomag_2026/_Pmag_Data_VarSec/
    proj_VarsecChili.txt) :
    - `transform="Geodetic"` (au lieu du defaut "PlateCarree" de
      `ipmag.plot_pole_dp_dm`) : BUG REEL confirme sur ce fichier - des
      qu'un VGP est plus proche du pole geographique que son propre
      dp/dm (6 sites ici, ex. 10CL35 a 89.2°N pour un dm de 2.5°),
      `ipmag.ellipse` echantillonne des points de part et d'autre du
      pole ; relies par des droites en coordonnees lon/lat non projetees
      (PlateCarree) plutot que par le vrai grand cercle, ces points
      produisent une ellipse "polygonale" (aretes droites visibles,
      au lieu d'une courbe lisse) - confirme disparu (courbe a nouveau
      lisse sur les 6 sites concernes) avec ce transform, verifie par
      rendu compare des deux versions sur ce meme fichier.
    - `site_label="_nolegend_"` (convention matplotlib : un label
      commencant par "_" est exclu de la legende automatique) plutot que
      `f"{e.site} (site)"` : la legende comptait donc AUPARAVANT 2
      entrees par site (pole + site), la moitie d'entre elles ("(site)")
      ne correspondant meme pas a un marqueur VISIBLE des que le site
      (ici au Chili) est a l'oppose du point de vue choisi pour centrer
      les poles (ici pres du pole Nord) - une projection orthographique
      ne montre jamais qu'un seul hemisphere a la fois. Le marqueur carre
      du site reste trace sur la carte quand il EST visible (ce fut
      seulement la ligne de legende dediee qui disparait) - la geometrie
      de l'ellipse (voir docstring ci-dessus) continue d'en avoir besoin.
    - Legende sortie du cadre de la carte (`bbox_to_anchor`) et sur 2
      colonnes, petite police : `loc=2` (dans les axes) recouvrait
      litteralement la carte des qu'il y a plus qu'une poignee de sites
      (48 ici -> 48 lignes de legende, avant meme la duplication
      "(site)" ci-dessus)."""
    plt.close("all")
    ax = ipmag.make_orthographic_map(central_longitude=view_lon, central_latitude=view_lat)
    for e in entries:
        marker = _VGP_SYMBOL_TO_MARKER.get(e.symbol, "o")
        # decode_color (stereo_project.py) accepte deja "R_G_B" ET les noms
        # de NAMED_COLORS (~22, dont certains - "skyblue", "lime"... - ne
        # sont pas garantis reconnus tels quels par matplotlib) : convertir
        # systematiquement en triplet 0-1 plutot que de transmettre `e.rgb`
        # tel quel, pour rester coherent avec le rendu du reseau stereo
        # (Project) sur EXACTEMENT les memes chaines de couleur.
        r, g, b = decode_color(e.rgb)
        color = (r / 255.0, g / 255.0, b / 255.0)
        markersize = max(e.size, 0.05) * 60
        has_site = e.site_lat != 0.0 or e.site_lon != 0.0
        if has_site and (e.dp > 0.0 or e.dm > 0.0):
            ipmag.plot_pole_dp_dm(
                ax, e.paleolon, e.paleolat, e.site_lon, e.site_lat, e.dp, e.dm,
                pole_label=e.site, site_label="_nolegend_",
                pole_color=color, pole_edgecolor=color, pole_marker=marker,
                site_color=color, site_edgecolor=color, site_marker="s",
                markersize=markersize, legend=False, transform="Geodetic",
            )
        else:
            ipmag.plot_vgp(
                ax, di_block=[[e.paleolon, e.paleolat]],
                color=color, marker=marker, markersize=markersize, label=e.site,
            )
            if e.p95 > 0.0:
                lons, lats = pmag.circ(e.paleolon, e.paleolat, e.p95)
                ax.plot(lons, lats, color=color, linewidth=1, transform=ccrs.Geodetic())
    if entries:
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7,
                   ncol=2 if len(entries) > 20 else 1, borderaxespad=0.0)
    save_folder = tempfile.mkdtemp(prefix="stereoutils_")
    path = os.path.join(save_folder, "vgp_project_map.png")
    # bbox_inches="tight" : sans ca, la legende SORTIE du cadre de la carte
    # (voir ci-dessus, bbox_to_anchor) serait coupee a la marge par defaut
    # de la figure plutot que d'agrandir le PNG pour l'inclure entierement.
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return None, [path]


# ---------------------------------------------------------------------------
# Modeles de champ paleomagnetique/paleosecular-variation (series temporelles
# D/I/F predites a un site) - demande explicite utilisateur ("est-ce
# possible d'ajouter la creation des series temporelles de champ magnetique
# a partir des modeles GUFM et les modeles Cals3K, Cals10k, Schadiff etc;
# voir ce qui est deja disponible dans PmagPy").
#
# `pmagpy/coefficients.py` embarque deja les coefficients de Gauss de tous
# ces modeles ; `pmag.doigrf`/`ipmag.igrf` (deja utilise ailleurs dans ce
# module, voir plot_vgps_on_map) les selectionne via son parametre `mod`.
# AUCUNE nouvelle geometrie/formule introduite ici - seulement une boucle
# sur la date, plus l'affichage/export du resultat.
#
# Bornes de date NON recopiees d'un docstring pmagpy (voir ci-dessous, `mod`
# reellement teste contre de vraies requetes plutot que suppose) : lues en
# direct sur chaque `coefficients.get_*()` (min/max de sa propre liste
# d'epoques) - un changement de version de pmagpy les garde a jour tout
# seul, et permet a `predicted_field_curve` de sauter proprement (warning,
# pas d'exception) toute date que le modele ne couvre pas reellement.
#
# DEUX ECARTS REELS decouverts en testant chaque modele avec de vraies
# requetes (lat=-39/lon=-72, plusieurs dates) avant d'ecrire ce module :
#   - Le docstring de `pmag.doigrf`/`ipmag.igrf` liste 'cals10k.1b' comme
#     valeur attendue pour `mod` - mais le code de `doigrf` (branche
#     date<1900) ne teste QUE la chaine exacte 'cals10k' pour choisir son
#     increment temporel (50 ans) ; passer 'cals10k.1b' tombe dans un autre
#     cas et leve `ValueError: x not in list` des que date<1900. `mod=
#     "cals10k"` (sans le suffixe) fonctionne correctement - c'est cette
#     chaine qui est utilisee ci-dessous.
#   - `mod='ggf100k'` (Panovska et al., 2018) : BUG REEL dans
#     `coefficients.get_ggf100k()` - sa liste d'epoques (`models`, 606
#     entrees) et son tableau de coefficients (`coeffs`, 602 lignes) n'ont
#     PAS la meme longueur, ce qui leve `IndexError` des que la date
#     tombe sur l'une des 4 epoques manquantes (constate y compris sur
#     LA SEULE date "ronde" documentee, 1850). Retire de la liste ci-
#     dessous plutot que d'exposer un choix qui echoue presque a coup sur.
#
# GUFM1 (Jackson et al., 2000) - CABLE ICI A LA MAIN (pas via pmag.doigrf/
# ipmag.igrf, qui ne le connaissent pas du tout) : un module dedie
# (`pmagpy/gufm.py`) existe bien dans cette copie locale de pmagpy, avec
# ses coefficients de Gauss par pas de 5 ans, mais s'arrete a l'echelon
# 1940-1945 SANS aucun `else`/`return` au-dela (`gufm.coeffs(date)` pour
# date>=1945 renvoie silencieusement `None`). Remarque utilisateur qui
# rend ce trou sans consequence pratique : le menu IGRF (mod="" ci-dessus)
# couvre DEJA 1900 a aujourd'hui, donc la seule plage ou GUFM1 apporterait
# quelque chose d'unique (avant le debut d'IGRF) est 1600-1900, entierement
# couverte par les donnees presentes dans le fichier - inutile de completer
# 1945-1990 pour rendre GUFM1 utile ici. Voir `_gufm1_dif` pour le calcul
# (meme technique que pmag.doigrf pour ses autres modeles paleo : secular
# variation entre deux echelons de 5 ans, puis `pmag.magsyn`) et pour les
# deux garde-fous necessaires (sys.exit() interne pour date<1600 ;
# troncature reelle a 120 coefficients - degre <=10 - avant magsyn, les
# 224 coefficients de GUFM1, degre 14, n'etant pas tous exploites par
# cette implementation de magsyn, verifie identique tronque/non tronque).
_FIELD_MODEL_SPECS = [
    ("IGRF14 (Alken et al., 2021)", ""),
    ("GUFM1 (Jackson et al., 2000)", "gufm1"),
    ("ARCH3k (Korte et al., 2009)", "arch3k"),
    ("CALS3k.4b (Korte & Constable, 2011)", "cals3k"),
    ("CALS10k.1b (Korte et al., 2011)", "cals10k"),
    ("CALS10k.2 (Constable et al., 2016)", "cals10k.2"),
    ("PFM9k (Nilsson et al., 2014)", "pfm9k"),
    ("HFM.OL1.A1 (Constable et al., 2016)", "hfm10k"),
    ("SHA.DIF.14k (Pavon-Carrasco et al., 2014)", "shadif14k"),
    ("SHAWQ2k (Campuzano et al., 2019)", "shawq2k"),
    ("SHAWQ-Iberia (Osete et al., 2020)", "shawqIA"),
]


def field_model_choices() -> List[Tuple[str, str]]:
    """(label, mod) pour chaque modele de `_FIELD_MODEL_SPECS`, le label
    portant sa plage de dates REELLE (lue sur `coefficients.py`, voir
    docstring de section ci-dessus) - IGRF14 seul n'a pas de `coefficients.
    get_*()` dedie (voir pmag.doigrf, `models, igrf14coeffs =
    cf.get_igrf14()` inconditionnel), affiche avec sa plage documentee."""
    from pmagpy import coefficients as cf
    getters = {
        "arch3k": cf.get_arch3k, "cals3k": cf.get_cals3k, "cals10k": cf.get_cals10k,
        "cals10k.2": cf.get_cals10k_2, "pfm9k": cf.get_pfm9k, "hfm10k": cf.get_hfm10k,
        "shadif14k": cf.get_shadif14k, "shawq2k": cf.get_shawq2k, "shawqIA": cf.get_shawqIA,
    }
    choices = []
    for label, mod in _FIELD_MODEL_SPECS:
        if mod == "gufm1":
            # Voir docstring de section : plage codee en dur (pas de
            # coefficients.get_*() pour ce modele) - 1600 borne basse
            # verifiee (gufm.coeffs leve plus bas), 1940 borne haute
            # verifiee empiriquement (le dernier echelon du fichier,
            # 1940-1945, ne peut pas fournir la SV puisque l'echelon
            # suivant, 1945, n'existe pas - voir _gufm1_dif).
            choices.append((f"{label} [1600 to 1940]", mod))
        elif mod in getters:
            models, _ = getters[mod]()
            choices.append((f"{label} [{int(min(models))} to {int(max(models))}]", mod))
        else:
            choices.append((f"{label} [1900 to present]", mod))
    return choices


def _gufm1_dif(lat: float, lon: float, alt_km: float, date: float) -> Tuple[float, float, float]:
    """D/I/F pour GUFM1 (Jackson et al., 2000) - voir la docstring de
    section pour le contexte complet (pourquoi ce modele n'est pas
    accessible via pmag.doigrf/ipmag.igrf, et pourquoi sa plage 1945-1990
    manquante n'a pas besoin d'etre completee ici).

    Meme technique que `pmag.doigrf` pour ses propres modeles paleo
    (arch3k, cals10k...) : secular variation calculee entre l'echelon de
    5 ans courant et le suivant, puis `pmag.magsyn` (la meme routine
    Malin & Barraclough qu'utilise doigrf) pour interpoler a la date
    exacte demandee - aucune formule nouvelle, seulement le branchement
    manquant vers `gufm.coeffs`.

    Deux garde-fous, tous deux verifies empiriquement avant d'ecrire cette
    fonction (voir conversation) plutot que de faire confiance au fichier :
    - date<1600 : `gufm.coeffs` appelle `sys.exit()` dans ce cas - PAS une
      Exception normale (`SystemExit` derive de BaseException) qui
      tuerait l'application entiere si elle remontait jusqu'a la boucle
      Tkinter, sans etre interceptee par le `except Exception` de
      `predicted_field_curve`. Verifie ICI en amont pour lever un
      ValueError normal a la place.
    - date>=1940 (dernier echelon du fichier sans suite) : `gufm.coeffs`
      pour l'echelon suivant (+5 ans) retourne silencieusement `None`
      (aucune exception) - detecte explicitement pour lever un message
      clair plutot que de laisser le `zip` suivant echouer avec un
      TypeError opaque.

    Les tableaux de coefficients de GUFM1 font 224 elements (degre 14) ;
    `pmag.magsyn` (implementation Malin & Barraclough, meme routine que
    pour tous les autres modeles de ce fichier) n'en exploite que les 120
    premiers (degre <=10) - verifie : resultat rigoureusement identique en
    tronquant explicitement a 120 avant l'appel. Les degres 11-14 de
    GUFM1 sont donc silencieusement ignores, comme ils le seraient de
    toute facon pour rester a la meme resolution que les autres modeles."""
    if date < 1600:
        raise ValueError("GUFM1 has no data before 1600")
    from pmagpy import gufm
    model = date - (date % 5.0)
    gh = gufm.coeffs(model)
    gh_next = gufm.coeffs(model + 5.0)
    if gh_next is None:
        raise ValueError("GUFM1 has no data at/after 1945 (see field_model_choices: use IGRF14 from 1900)")
    sv = [(b - a) / 5.0 for a, b in zip(gh, gh_next)]
    colat = 90.0 - lat
    x, y, z, f = pmag.magsyn(gh, sv, model, date, 1, alt_km, colat, lon % 360)
    dec, inc, _ = pmag.cart2dir((x, y, z))
    return float(dec), float(inc), float(f)


def predicted_field_curve(
    lat: float, lon: float, alt_km: float, date_start: float, date_end: float,
    step: float, mod: str,
) -> Tuple[List[Tuple[float, float, float, float, Optional[float], Optional[float], Optional[float]]], List[str]]:
    """Serie temporelle D/I/F predite au site (lat,lon,alt_km) par le
    modele `mod` (voir `_FIELD_MODEL_SPECS`, "" = IGRF14, "gufm1" = voir
    `_gufm1_dif`, tout le reste = `ipmag.igrf`), du plus vieux au plus
    recent quel que soit l'ordre de `date_start`/`date_end`.

    Retourne (points, warnings) - points = liste de (date, dec, inc,
    f_uT, ddec, dinc, df_uT) une entree PAR DATE OU le modele a repondu ;
    warnings = une entree par date en echec (hors de la plage reellement
    couverte par ce modele, ou toute autre erreur numerique) plutot que
    de faire echouer tout le calcul - meme esprit "un point illisible est
    saute, pas fatal" que ams_asc.parse_asc_file. Intensite convertie de
    nT (retour natif de `ipmag.igrf`/`pmag.magsyn`) en microTesla,
    l'unite utilisee partout ailleurs dans ce projet pour la
    paleointensite.

    ddec/dinc/df_uT (demande explicite utilisateur : "dans le dossier
    Field_models, il y a des vieilles sources en Fortran pour calculer
    des incertitudes... est-ce possible de voir si une partie peut etre
    integree") : None sauf pour `mod` in ('cals3k','cals10k'), les deux
    SEULS modeles pour lesquels une incertitude par coefficient est
    disponible localement (voir field_uncertainty.py pour le format, la
    verification contre le Fortran d'origine recompile, et pourquoi
    CALS10k.2 n'en beneficie PAS malgre la meme famille de modele).
    Pour ces deux `mod`, dec/inc/f_uT eux-memes sont AUSSI recalcules via
    field_uncertainty (meme synthese B-spline que l'incertitude, pas
    `ipmag.igrf`) plutot que d'accoler une incertitude "neuve" a un point
    moyen calcule par un chemin different (interpolation lineaire par
    morceaux de doigrf entre epoques tabulees) - verifie IDENTIQUE a
    ipmag.igrf aux epoques exactement tabulees, mais potentiellement
    legerement different ENTRE deux epoques (vraie spline cubique ici,
    pas une simple droite) ; garder le MEME chemin de calcul pour le
    point moyen et son enveloppe evite une incoherence visuelle entre
    les deux sur le graphique."""
    date_start, date_end = min(date_start, date_end), max(date_start, date_end)
    step = abs(step) or 1.0
    kwargs = {} if mod == "" else {"mod": mod}
    uncertain = fu.has_uncertainty_model(mod)
    points: List[Tuple[float, float, float, float, Optional[float], Optional[float], Optional[float]]] = []
    warnings: List[str] = []
    date = date_start
    while date <= date_end + 1e-9:
        try:
            if uncertain:
                r = fu.field_uncertainty_at(lat, lon, alt_km, date, mod)
                if r is None:
                    raise ValueError("date outside the uncertainty model's spline range")
                dec, inc, f_nt, ddec, dinc, df_nt = r
                points.append((date, dec, inc, f_nt / 1000.0, ddec, dinc, df_nt / 1000.0))
            elif mod == "gufm1":
                dec, inc, f_nt = _gufm1_dif(lat, lon, alt_km, date)
                points.append((date, float(dec), float(inc), float(f_nt) / 1000.0, None, None, None))
            else:
                dec, inc, f_nt = ipmag.igrf([date, alt_km, lat, lon], **kwargs)
                points.append((date, float(dec), float(inc), float(f_nt) / 1000.0, None, None, None))
        except Exception as e:
            warnings.append(f"{date:g}: {type(e).__name__}: {e}")
        date += step
    return points, warnings


def read_field_curve_data(path: str) -> List[Tuple[float, float, float, float, Optional[float], Optional[float]]]:
    """Points reels (age, dec, inc, a95, intensity_uT_ou_None,
    dintensity_ou_None) a superposer a une courbe predite - demande
    explicite utilisateur ("est-ce possible de plotter des donnees en
    comparaison des modeles de champ, menu Predicted field curve").
    Colonnes reperees par en-tete si present (age/dec/inc/a95, +
    intensity/dintensity optionnelles), sinon position 0-3(-5) - meme
    convention que le reste de ce projet (voir stereo_selection.
    split_header). Une ligne sans intensity/dintensity lisible laisse
    simplement ces deux valeurs a None (superposee aux 2 premiers
    sous-graphiques seulement, pas au 3e)."""
    lines = read_text_lines(path)
    data_lines, idx = split_header(lines, "age", "dec", "inc", "a95", "intensity", "dintensity")
    if all(k in idx for k in ("age", "dec", "inc", "a95")):
        i_age, i_dec, i_inc, i_a95 = idx["age"], idx["dec"], idx["inc"], idx["a95"]
        i_f, i_df = idx.get("intensity"), idx.get("dintensity")
    else:
        i_age, i_dec, i_inc, i_a95, i_f, i_df = 0, 1, 2, 3, 4, 5

    out = []
    for line in data_lines:
        line = line.strip()
        if not line or line.startswith("!") or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) <= max(i_age, i_dec, i_inc, i_a95):
            continue
        try:
            age = float(parts[i_age])
            dec, inc, a95 = float(parts[i_dec]), float(parts[i_inc]), float(parts[i_a95])
        except ValueError:
            continue
        f_ut = dfut = None
        if i_f is not None and len(parts) > i_f:
            try:
                f_ut = float(parts[i_f])
                if i_df is not None and len(parts) > i_df:
                    dfut = float(parts[i_df])
            except ValueError:
                f_ut = dfut = None
        out.append((age, dec, inc, a95, f_ut, dfut))
    return out


def plot_field_curve(
    points: List[Tuple[float, float, float, float, Optional[float], Optional[float], Optional[float]]],
    title: str = "",
    data_points: Optional[List[Tuple[float, float, float, float, Optional[float], Optional[float]]]] = None,
):
    """3 sous-graphiques empiles (Dec/Inc/Intensity vs date), meme
    convention de sauvegarde (PNG dans un dossier temporaire, chemin
    retourne pour `app._show_images`) que les autres fonctions de trace de
    ce module.

    `points[i]` = (date, dec, inc, f_uT, ddec, dinc, df_uT) - les 3
    derniers (voir stereo_pmagpy.predicted_field_curve/
    field_uncertainty.py) sont None pour la plupart des modeles (pas
    d'incertitude disponible localement) : une enveloppe ombree
    (mean +/- incertitude) est alors tracee en plus du trait moyen,
    SEULEMENT sur les points qui en portent une - demande explicite
    utilisateur ("il y a des vieilles sources en Fortran pour calculer
    des incertitudes... est-ce possible d'integrer une partie").

    `data_points` (optionnel - voir read_field_curve_data) : donnees REELLES
    datees (age, dec, inc, a95, intensity_ou_None, dintensity_ou_None),
    superposees a la courbe predite en barres d'erreur - demande explicite
    utilisateur ("plotter des donnees en comparaison des modeles de
    champ"). L'intensite (3e sous-graphique) n'est marquee que pour les
    points qui en portent une - un site purement directionnel reste donc
    superpose sur Dec/Inc seulement, sans laisser croire a une intensite
    nulle sur le 3e sous-graphique."""
    plt.close("all")
    dates = [p[0] for p in points]
    # Declinaison ramenee a [-180, 180] (300 -> -60) plutot que le [0,360)
    # natif d'ipmag.igrf/pmag.cart2dir - demande explicite utilisateur
    # ("pour les graphiques de declinaison, comme il s'agit de champ
    # recent, faire l'echelle entre -60 et +60 (300 =-60)") : evite le saut
    # 350->5 a chaque passage par 0/360, et rend l'echelle -60/+60 ci-
    # dessous lisible pour une declinaison qui oscille autour de 0 (champ
    # recent - une excursion paleosecular-variation de plusieurs dizaines
    # de degres resterait visible, juste coupee au bord du cadre si elle
    # depasse cette echelle plutot que de la re-elargir automatiquement).
    dec = [p[1] - 360.0 if p[1] > 180.0 else p[1] for p in points]
    inc = [p[2] for p in points]
    f_ut = [p[3] for p in points]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    axes[0].plot(dates, dec, "b.-", label="predicted" if data_points else None)
    axes[0].set_ylabel("Declination (°)")
    axes[0].set_ylim(-60.0, 60.0)
    axes[1].plot(dates, inc, "r.-")
    axes[1].set_ylabel("Inclination (°)")
    axes[2].plot(dates, f_ut, "g.-")
    axes[2].set_ylabel("Intensity (µT)")
    axes[2].set_xlabel("Date (years CE)")
    # Enveloppe d'incertitude (mean +/- 1 sigma) - matplotlib casse
    # naturellement le remplissage la ou les valeurs sont NaN, donc les
    # points SANS incertitude (ddec/dinc/df_uT=None) laissent simplement
    # un trou dans la bande plutot que de fausser l'echelle ou planter.
    if any(p[4] is not None for p in points):
        ddec = np.array([p[4] if p[4] is not None else np.nan for p in points])
        dinc = np.array([p[5] if p[5] is not None else np.nan for p in points])
        df_ut = np.array([p[6] if p[6] is not None else np.nan for p in points])
        dec_a, inc_a, f_a = np.array(dec), np.array(inc), np.array(f_ut)
        axes[0].fill_between(dates, dec_a - ddec, dec_a + ddec, color="b", alpha=0.15, linewidth=0)
        axes[1].fill_between(dates, inc_a - dinc, inc_a + dinc, color="r", alpha=0.15, linewidth=0)
        axes[2].fill_between(dates, f_a - df_ut, f_a + df_ut, color="g", alpha=0.15, linewidth=0)
    if data_points:
        d_ages = [p[0] for p in data_points]
        d_dec = [p[1] - 360.0 if p[1] > 180.0 else p[1] for p in data_points]
        d_inc = [p[2] for p in data_points]
        d_a95 = [p[3] for p in data_points]
        axes[0].errorbar(d_ages, d_dec, yerr=d_a95, fmt="ko", ms=4, capsize=3, label="data")
        axes[1].errorbar(d_ages, d_inc, yerr=d_a95, fmt="ko", ms=4, capsize=3)
        f_ages = [p[0] for p in data_points if p[4] is not None]
        f_vals = [p[4] for p in data_points if p[4] is not None]
        f_errs = [p[5] if p[5] is not None else 0.0 for p in data_points if p[4] is not None]
        if f_vals:
            axes[2].errorbar(f_ages, f_vals, yerr=f_errs, fmt="ko", ms=4, capsize=3)
        axes[0].legend(loc="best", fontsize=8)
    if title:
        axes[0].set_title(title)
    for ax in axes:
        ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    save_folder = tempfile.mkdtemp(prefix="stereoutils_")
    path = os.path.join(save_folder, "field_curve.png")
    fig.savefig(path, dpi=120)
    plt.close("all")
    return path


def write_field_curve_file(
    points: List[Tuple[float, float, float, float, Optional[float], Optional[float], Optional[float]]],
    path: str,
    lat: float, lon: float, alt_km: float, model_label: str,
) -> None:
    """Fichier texte tabule (date, dec, inc, intensite en uT, +
    incertitudes ddec/dinc/dintensite si disponibles - voir
    field_uncertainty.py), en-tete commente (#...) - meme convention que
    les fichiers "Project" (STARpaleomag_Py/export_stereo), directement
    relisable par un tableur ou un autre script sans plus de traitement.
    Les colonnes d'incertitude valent "n.d" (convention de ce projet pour
    une donnee manquante) quand `mod` n'a pas d'incertitude disponible
    localement."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# predicted field curve - model: {model_label}\n")
        f.write(f"# site: lat={lat:.4f}  lon={lon:.4f}  altitude={alt_km:.2f} km\n")
        f.write("#date\tdec\tinc\tintensity_uT\tddec\tdinc\tdintensity_uT\n")
        for date, dec, inc, f_ut, ddec, dinc, df_ut in points:
            ddec_s = f"{ddec:.2f}" if ddec is not None else "n.d"
            dinc_s = f"{dinc:.2f}" if dinc is not None else "n.d"
            df_ut_s = f"{df_ut:.3f}" if df_ut is not None else "n.d"
            f.write(
                f"{date:g}\t{dec:.2f}\t{inc:.2f}\t{f_ut:.3f}\t{ddec_s}\t{dinc_s}\t{df_ut_s}\n"
            )
