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
    export_stereo.export_poles_to_stereo) - contrairement a
    `plot_vgps_on_map` (une seule couleur/un seul marqueur pour TOUS les
    points, `self.directions` sans cette info), chaque VGP est ici trace
    INDIVIDUELLEMENT avec ses propres reglages (une boucle + un appel
    `ipmag.plot_vgp` par entree, plutot qu'un seul appel groupe sur
    `di_block`).

    Ovale de confiance : APPROXIME par un CERCLE de rayon `p95` (moyenne
    dp/dm, deja calculee cote STARpaleomag_Py - voir
    calcul.dp_dm_from_a95), PAS la vraie ellipse dp/dm (asymetrique
    nord-sud/est-ouest) - simplification deliberee, PAS un oubli : c'est
    deja la convention utilisee ailleurs dans ce meme projet pour les VGP
    d'un APWP (voir app.py, colonne "p95" de `read_apwp_file` / le menu
    correspondant), et `pmag.circ` (deja utilise/verifie par PmagPy,
    aucune nouvelle formule geometrique introduite ici) ne sait tracer
    qu'un cercle, pas une ellipse orientee - une vraie ellipse dp/dm
    demanderait de connaitre l'azimut site->VGP (angle de declinaison
    paleomagnetique), non reconstruit ici. `dp`/`dm` individuels restent
    neanmoins dans le fichier/l'entree pour un usage futur plus precis."""
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
        ipmag.plot_vgp(
            ax, di_block=[[e.paleolon, e.paleolat]],
            color=color, marker=marker, markersize=max(e.size, 0.05) * 60,
            label=e.site,
        )
        if e.p95 > 0.0:
            lons, lats = pmag.circ(e.paleolon, e.paleolat, e.p95)
            ax.plot(lons, lats, color=color, linewidth=1, transform=ccrs.Geodetic())
    save_folder = tempfile.mkdtemp(prefix="stereoutils_")
    path = os.path.join(save_folder, "vgp_project_map.png")
    plt.savefig(path, dpi=120)
    plt.close("all")
    return None, [path]
