"""
Hysteresis : port des routines principales de Curie_OSX pour les cycles
d'hysteresis (reference/../Curie_Hyst_AWE - hysteresis.f, menu
"Hysteresis") - demande explicite utilisateur ("integrer les deux
fonctions principales de Curie_OSX ... et des donnees d'Hysteresis").
L'AUTRE fonction principale (K-T, susceptibilite vs temperature) est
portee dans curie_kt.py, deja fait.

DEUX formats reels supportes, chacun avec sa propre liste d'echantillons -
demande explicite utilisateur ("dans ce dossier il y a des donnees de VSM
et un exemple de fichier liste ... et ici des vieilles donnees d'AGM") :

1) AGM (Princeton Measurements MicroMag "Model 2900 ASCII Data File",
   /Users/pierrickroperch/Paleomag_data/Hyste_Chili_2, annees 2013-2014) -
   port de `selectfichhystAGM` (hysteresis.f:717-945) : fichier <id> (boucle
   complete, sature->sature en passant par le champ negatif) + fichier
   <id>-r (courbe de desaimantation/remanence DCD, memes 2 colonnes) - 2
   lignes d'entete (parametres instrument, puis ligne vide) ignorees, puis
   des lignes "champ(T),moment(Am2)" - liste d'echantillons Liste_Chile2.txt
   : "<id> <masse_mg>". Verifie CHAMP PAR CHAMP contre Resu_Chile2.txt
   (sortie DEJA calculee par le Fortran d'origine, meme fichier .txt que
   `CURIE_output.txt`) pour l'echantillon CL2603 : JsMax/Jrs/Hcr EXACTS,
   Js_Ferro/Hc/suscepPara exacts a <0.1% pres pour `valsat_frac~0.665-0.67`
   (le seuil "haut champ" du fit paramagnetique EST un reglage utilisateur,
   `prefhyste`/`valsat` - PAS une constante figee : la valeur precise
   utilisee pour ce lot historique de 2014 n'est pas recuperable sans le
   fichier de preference d'origine, d'ou `valsat_frac` expose ici comme
   parametre, defaut 0.7 = le defaut committe du Fortran).

2bis) VFTB (Petersen Instruments Variable Field Translation Balance,
   export .hys texte) - demande explicite utilisateur ("dans Stereo, est
   ce possible d'ajouter a la lecture des fichiers hysteresis, un format
   supplementaire (VFTB)"), verifie sur un vrai fichier
   (24WH0205_277mg_RGV.hys) : 1ere ligne "name: <nom>\tweight: <masse>
   mg" (cle:valeur, meme convention que .prmag), puis un ou plusieurs
   blocs "Set N:" (une ligne d'entete de colonnes par bloc, colonnes
   identifiees par NOM - "field / Oe", "mag / emu / g" - PAS une
   position fixe, les colonnes temp/time/std dev/suscep ne sont pas
   utilisees ici). Contrairement a AGM/VSM, masse et nom sont DEJA dans
   le fichier lui-meme - pas de liste d'echantillons separee necessaire.
   Champ en Oe (converti en Tesla, x1e-4) ; "mag" est DEJA normalise par
   la masse (emu/g, magnetisation specifique CGS) - reconverti ici en
   moment BRUT (Am2) via la masse du fichier, pour rester dans la MEME
   convention (HystLoop.moment non-normalise) que les 2 autres formats
   et reutiliser exactement le meme compute_hysteresis. Un fichier VFTB
   "simple" (une seule boucle, PAS de courbe de remanence/DCD separee)
   est courant : voir compute_hysteresis(backfield=None) - Hcr/Jrs
   deviennent alors indisponibles (nan), le reste (JsMax/Js_Ferro/Hc/
   suscepPara) reste calculable normalement a partir de la boucle seule.

2) VSM moderne (LakeShore/MicroMag, export CSV natif,
   /Users/pierrickroperch/Paleomag_data/VSM_Nov2022, 2022) - PAS un format
   du Fortran d'origine (logiciel different/plus recent) : bloc d'entete
   texte (#SAMPLE SETTINGS, #HYSTERESIS MEASUREMENT/#REMANENCE CURVES
   MEASUREMENT...) jusqu'a la ligne colonnes "Step,Iteration,Segment,Field
   (...) [T],Moment (m) [A.m2],...", puis les donnees CSV - boucle dans
   "<id> - 1.csv", courbe de remanence/DCD dans "<id> - 2.csv" (memes
   fichiers pour un meme specimen) - liste d'echantillons listeVSM_nov22.txt
   : "<indice> <id> <masse_mg>" (indice ignore, meme convention que les
   listes K-T - voir curie_kt.read_kt_sample_list). Le fichier "<id> - 1
   Corrections.csv" (deja slope-corrige par le logiciel VSM lui-meme,
   fenetre haut-champ fixe 300-500mT) n'est PAS utilise ici : ce module
   relit systematiquement le brut et applique SA PROPRE correction
   parametrable, pour un pipeline identique AGM/VSM.

Pipeline de calcul (compute_hysteresis) commun aux deux formats : une fois
`field`/`moment` (Tesla, Am2) et `mass_mg` obtenus, le reste est
STRICTEMENT le meme algorithme quelle que soit la source - port de
`suppara`/`calculhyste`/`LINREG` (hysteresis.f:258-470), deja utilise/
verifie ci-dessus. `LINREG` (regression lineaire par moindres carres,
methode des sommes) est reimplementee via `numpy.polyfit` (meme resultat
mathematique, pas un portage bit-a-bit des sommes manuelles).

Graphiques reconstruits DIRECTEMENT avec matplotlib, forme CARREE (meme
choix que curie_kt.py - demande explicite utilisateur "is it possible to
have a plot with a square shape" puis "the same logic applies to the
hysteresis plots") : PAS de port pixel-pres via plotlib.PlotContext."""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.figure import Figure


@dataclass
class HystLoop:
    """Une courbe brute (boucle d'hysteresis OU courbe de remanence/DCD -
    meme structure pour les deux, seule la SIGNIFICATION differe) : champ
    (Tesla) et moment (Am2, PAS encore normalise par la masse - voir
    compute_hysteresis)."""
    sample: str
    path: str
    field: np.ndarray
    moment: np.ndarray


def _read_field_moment_csv(path: str, skip_lines: int, delimiter: str = ",") -> Tuple[np.ndarray, np.ndarray]:
    """Lit un fichier `<champ><delim><moment>` a 2 colonnes, en sautant
    les `skip_lines` premieres lignes (entete) - ignore silencieusement
    toute ligne qui ne se reduit pas a exactement 2 valeurs numeriques
    (lignes vides, trailer "Model 2900 Data File ends", etc. - meme
    tolerance que le filtre de longueur du Fortran, mais base sur le
    CONTENU plutot que sur un nombre de caracteres fixe, plus robuste)."""
    field: List[float] = []
    moment: List[float] = []
    with open(path, "r", encoding="iso-8859-1", errors="replace") as f:
        for _ in range(skip_lines):
            next(f, None)
        for line in f:
            parts = line.strip().split(delimiter)
            if len(parts) != 2:
                continue
            try:
                field.append(float(parts[0]))
                moment.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(field), np.array(moment)


# ----------------------------------------------------------------------
# AGM (Princeton MicroMag "Model 2900 ASCII Data File") - voir docstring
# de module, verifie contre Resu_Chile2.txt.
# ----------------------------------------------------------------------

def read_agm_file(path: str) -> HystLoop:
    """Equivalent de la lecture fichier de `selectfichhystAGM`
    (hysteresis.f:762-798, boucle ET fichier -r partagent le meme format -
    voir docstring de module) : 2 lignes d'entete, puis "champ,moment"."""
    field, moment = _read_field_moment_csv(path, skip_lines=2)
    sample = os.path.splitext(os.path.basename(path))[0]
    return HystLoop(sample=sample, path=path, field=field, moment=moment)


@dataclass
class AgmListEntry:
    """Une ligne de Liste_Chile2.txt (voir read_agm_sample_list) :
    <id> <masse_mg>."""
    filename: str
    masse: float


def read_agm_sample_list(path: str) -> Dict[str, AgmListEntry]:
    """Equivalent de la liste lue par `selectfichhystAGM`
    (`fich(i).sample, fich(i).masse`) : format reel observe (Liste_
    Chile2.txt) "<id> <masse_mg>", sans indice de ligne (contrairement
    aux listes K-T) - cle de jointure : id en MAJUSCULES."""
    entries: Dict[str, AgmListEntry] = {}
    with open(path, "r", encoding="iso-8859-1", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                masse = float(parts[1])
            except ValueError:
                continue
            entries[parts[0].upper()] = AgmListEntry(parts[0], masse)
    return entries


# ----------------------------------------------------------------------
# VSM moderne (export CSV natif) - voir docstring de module.
# ----------------------------------------------------------------------

def read_vsm_csv(path: str) -> HystLoop:
    """Lit un export CSV VSM moderne (Nov2022) : saute le bloc d'entete
    texte jusqu'a la ligne de colonnes "Step,Iteration,Segment,Field
    (...) [T],Moment (m) [A.m2],..." (position variable d'un fichier a
    l'autre selon le type de mesure - #HYSTERESIS vs #REMANENCE - d'ou
    la recherche par CONTENU plutot qu'un skip_lines fixe comme pour
    l'AGM), puis Field=colonne 4 (index 3), Moment=colonne 5 (index 4)
    de chaque ligne de donnees - memes colonnes pour la boucle d'hysteresis
    ET la courbe de remanence/DCD (memes en-tetes de table dans les
    deux types de fichier)."""
    with open(path, "r", encoding="iso-8859-1", errors="replace") as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Step,Iteration,Segment,Field"):
            header_idx = i
            break
    field: List[float] = []
    moment: List[float] = []
    if header_idx is not None:
        for line in lines[header_idx + 1:]:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            try:
                field.append(float(parts[3]))
                moment.append(float(parts[4]))
            except ValueError:
                continue
    sample = os.path.splitext(os.path.basename(path))[0]
    return HystLoop(sample=sample, path=path, field=np.array(field), moment=np.array(moment))


@dataclass
class VsmListEntry:
    """Une ligne de listeVSM_nov22.txt (voir read_vsm_sample_list) :
    <id> <masse_mg>."""
    filename: str
    masse: float


def read_vsm_sample_list(path: str) -> Dict[str, VsmListEntry]:
    """Format reel observe (listeVSM_nov22.txt) : "<indice> <id>
    <masse_mg>" (indice ignore, meme convention que curie_kt.
    read_kt_sample_list - probablement ajoute par un tableur, jamais lu
    par le Fortran d'origine puisque ce format VSM n'a PAS d'equivalent
    Fortran). Cle de jointure : id en MAJUSCULES."""
    entries: Dict[str, VsmListEntry] = {}
    with open(path, "r", encoding="iso-8859-1", errors="replace") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0].isdigit():
                parts = parts[1:]
            if len(parts) < 2:
                continue
            try:
                masse = float(parts[1])
            except ValueError:
                continue
            entries[parts[0].upper()] = VsmListEntry(parts[0], masse)
    return entries


def vsm_paths_for(base_dir: str, sample_id: str) -> Tuple[str, str]:
    """Chemins des 2 fichiers CSV d'un specimen VSM ("<id> - 1.csv" =
    boucle d'hysteresis, "<id> - 2.csv" = courbe de remanence/DCD - meme
    convention de nommage que /Users/pierrickroperch/Paleomag_data/
    VSM_Nov2022, verifiee sur plusieurs specimens reels)."""
    return (
        os.path.join(base_dir, f"{sample_id} - 1.csv"),
        os.path.join(base_dir, f"{sample_id} - 2.csv"),
    )


# ----------------------------------------------------------------------
# VFTB (Petersen Instruments, export .hys) - voir docstring de module,
# verifie sur 24WH0205_277mg_RGV.hys.
# ----------------------------------------------------------------------

def _vftb_kv_line(line: str) -> Dict[str, str]:
    """"name: <nom>\tweight: <masse> mg" -> {"name": "<nom>", "weight":
    "<masse> mg"} - meme principe cle:valeur tabule que .prmag, colonnes
    dans un ORDRE quelconque."""
    result: Dict[str, str] = {}
    for chunk in line.split("\t"):
        if ":" not in chunk:
            continue
        k, _sep, v = chunk.partition(":")
        result[k.strip().lower()] = v.strip()
    return result


def read_vftb_file(path: str) -> Tuple[str, Optional[float], List[HystLoop]]:
    """Lit un fichier VFTB .hys (voir docstring de module) : nom+masse
    depuis la 1ere ligne, puis une `HystLoop` par bloc "Set N:" rencontre
    (colonnes "field"/"mag" retrouvees PAR NOM sur la ligne d'entete de
    chaque bloc, insensible a la casse - les autres colonnes eventuelles,
    temp/time/std dev/suscep, sont ignorees). "mag" (emu/g, DEJA
    normalise par la masse) est reconverti en moment BRUT (Am2) via la
    masse lue en 1ere ligne, pour rester dans la meme convention
    (HystLoop.moment non-normalise) que AGM/VSM - `emu/g * masse(mg) *
    1e-6 = Am2` (1 emu = 1e-3 Am2, masse(mg)/1000 = masse(g)). Si la
    masse n'est pas trouvable/valide dans l'entete, "mag" est laisse tel
    quel (emu/g) - l'appelant doit alors fournir la masse autrement et
    ne PAS reutiliser compute_hysteresis sans reconversion.

    Retourne (nom, masse_mg ou None, liste de HystLoop - une par "Set",
    dans l'ordre du fichier ; typiquement 1 seule (boucle complete, pas
    de courbe de remanence/DCD separee) ou 2 (boucle + backfield)."""
    with open(path, "r", encoding="iso-8859-1", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]
    if not lines:
        return os.path.splitext(os.path.basename(path))[0], None, []

    header = _vftb_kv_line(lines[0])
    name = header.get("name") or os.path.splitext(os.path.basename(path))[0]
    mass_mg: Optional[float] = None
    m = re.match(r"[-+]?[\d.]+", header.get("weight", "").strip())
    if m:
        try:
            mass_mg = float(m.group())
        except ValueError:
            mass_mg = None

    loops: List[HystLoop] = []
    i, n = 1, len(lines)
    while i < n:
        if lines[i].strip().lower().startswith("set"):
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            if i >= n:
                break
            cols = [c.strip().lower() for c in lines[i].split("\t")]
            field_idx = next((j for j, c in enumerate(cols) if c.startswith("field")), None)
            mag_idx = next((j for j, c in enumerate(cols) if c.startswith("mag")), None)
            i += 1
            field: List[float] = []
            moment: List[float] = []
            while i < n and lines[i].strip():
                parts = lines[i].split("\t")
                if field_idx is not None and mag_idx is not None and len(parts) > max(field_idx, mag_idx):
                    try:
                        h_oe = float(parts[field_idx])
                        mag_emu_g = float(parts[mag_idx])
                    except ValueError:
                        i += 1
                        continue
                    field.append(h_oe * 1.0e-4)  # Oe -> Tesla
                    moment.append(mag_emu_g * mass_mg * 1.0e-6 if mass_mg else mag_emu_g)
                i += 1
            field_arr, moment_arr = np.array(field), np.array(moment)
            # VFTB mesure souvent une courbe de premiere aimantation
            # (0 -> +Hmax) AVANT la boucle proprement dite - demande
            # explicite utilisateur ("le calcul du paramagnetisme est
            # incorrect") : verifie sur un vrai fichier
            # (24WH0205_277mg_RGV.hys) que field[0] n'etait PAS pres du
            # champ maximal (0.0027 T alors que le maximum, 0.982 T,
            # n'arrive qu'a l'indice 36/181) - compute_hysteresis suppose
            # justement field[0]~+Hmax (convention AGM/VSM, ou la boucle
            # demarre deja a saturation) pour placer son seuil "haut
            # champ" (`valsat_frac * field[0]`) : sans ce retrait, ce
            # seuil tombe pres de 0 et le fit paramagnetique utilise des
            # points de la courbe de premiere aimantation, pas seulement
            # les branches haute-saturation de la boucle. On retire tout
            # ce qui precede le maximum GLOBAL du champ (si differe de
            # l'indice 0) - verifie que le minimum tombe alors bien au
            # milieu du tableau restant, signe d'une boucle bien formee
            # (branche descendante puis remontante, symetrique).
            if len(field_arr):
                imax = int(np.argmax(field_arr))
                if imax > 0:
                    field_arr, moment_arr = field_arr[imax:], moment_arr[imax:]
            loops.append(HystLoop(sample=name, path=path, field=field_arr, moment=moment_arr))
        else:
            i += 1
    return name, mass_mg, loops


# ----------------------------------------------------------------------
# Calcul (commun AGM/VSM/VFTB) - port de suppara/calculhyste/LINREG
# (hysteresis.f:258-470), verifie contre Resu_Chile2.txt (CL2603).
# ----------------------------------------------------------------------

@dataclass
class HysteresisResult:
    """Resultat complet pour UN specimen - memes champs que la ligne de
    `Resu_Chile2.txt`/`CURIE_output.txt` (voir calculhyste) : masse(mg)
    JsMax Js_Ferro Jrs Hc Hcr Jrs/Js_ferro Hcr/Hc suscepPara. Les tableaux
    de courbe (`field`/`moment_norm`/`moment_norm_corrected`/
    `backfield_*`) sont conserves pour le trace (build_hysteresis_figure),
    deja normalises par la masse (Am2/kg)."""
    sample: str
    mass_mg: float
    js_max: float
    js_ferro: float
    jrs: float
    hc_mt: float
    hcr_mt: float
    jrs_over_jsferro: float
    hcr_over_hc: float
    susc_para_si: float
    field: np.ndarray
    moment_norm: np.ndarray
    moment_norm_corrected: np.ndarray
    backfield_field: np.ndarray
    backfield_norm: np.ndarray


def _ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Regression lineaire Y=slope*X+intercept par moindres carres -
    equivalent de `LINREG` (hysteresis.f:382-470, meme resultat
    mathematique que ses sommes manuelles, via `numpy.polyfit`)."""
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _last_sign_change(arr: np.ndarray, lo: int, hi: int) -> Optional[int]:
    """Dernier indice i dans [lo,hi) ou sign(arr[i]) != sign(arr[i+1]) -
    meme logique que les boucles `id`/`im` de `suppara`."""
    idx = None
    for i in range(lo, hi):
        if np.sign(arr[i]) != np.sign(arr[i + 1]):
            idx = i
    return idx


def compute_hysteresis(
    loop: HystLoop, backfield: Optional[HystLoop], mass_mg: float, valsat_frac: float = 0.7,
) -> Optional[HysteresisResult]:
    """Equivalent du pipeline `selectfichhystAGM` (calcul de JsMax/Hc brut)
    + `suppara` (correction paramagnetique + Hc post-correction) +
    `calculhyste` (assemblage du resultat) - voir docstring de module
    pour l'etat de verification. `valsat_frac` : fraction de `field[0]`
    (premier point de la boucle, pres de la saturation) au-dela de
    laquelle un point est considere "haut champ" pour le fit
    paramagnetique - equivalent du reglage `prefhyste`/`valsat`, PAS une
    constante figee. Retourne None si la boucle est trop courte ou si
    aucun point ne depasse le seuil haut-champ des deux cotes.

    `backfield` : None (ou une courbe vide) quand aucune courbe de
    remanence/DCD separee n'est disponible - demande explicite
    utilisateur (ajout du format VFTB, dont un fichier "simple" ne
    contient souvent qu'UNE boucle) : Hcr/Jrs deviennent alors "nan"
    (indisponibles), le reste (JsMax/Js_Ferro/Hc/suscepPara, qui ne
    dependent que de la boucle elle-meme) reste calcule normalement."""
    x = loop.field
    n = len(x)
    if n < 6:
        return None
    mass_kg = mass_mg * 1.0e-6
    jsnorm = loop.moment / mass_kg
    js_max = float(np.max(loop.moment)) / mass_kg

    imin = int(np.argmin(x))
    valsat = valsat_frac * x[0]
    b1 = np.arange(0, imin + 1)
    b1 = b1[x[b1] >= valsat]
    b2 = np.arange(imin, n)
    b2 = b2[x[b2] <= -valsat]
    if len(b1) + len(b2) < 2:
        return None
    xfit = np.concatenate([x[b1], -x[b2]])
    yfit = np.concatenate([jsnorm[b1], -jsnorm[b2]])
    slope, js_ferro = _ols(xfit, yfit)
    susc_para_si = slope * 4.0 * np.pi * 1.0e-7

    jsnorm_corr = jsnorm - slope * x
    mil = n // 2
    idd = _last_sign_change(jsnorm_corr, 0, mil)
    imm = _last_sign_change(jsnorm_corr, mil, n - 1)
    if idd is None or imm is None:
        hc_mt = float("nan")
    else:
        x2 = np.array([-jsnorm_corr[idd], -jsnorm_corr[idd + 1], jsnorm_corr[imm], jsnorm_corr[imm + 1]])
        y2 = np.array([-x[idd], -x[idd + 1], x[imm], x[imm + 1]])
        _slope2, hc = _ols(x2, y2)
        hc_mt = hc * 1000.0

    if backfield is not None and len(backfield.field):
        bfield = backfield.field
        jsrem = backfield.moment / mass_kg
        jrs = float(jsrem[0])
        icr = None
        for i in range(len(jsrem)):
            if jsrem[i] < 0.0:
                icr = i
                break
        if icr is None or icr == 0:
            hcr_mt = float("nan")
        else:
            xr = -bfield[icr] + bfield[icr - 1]
            yr = jsrem[icr - 1] / (jsrem[icr - 1] - jsrem[icr])
            hcr_mt = (-bfield[icr - 1] + xr * yr) * 1000.0
    else:
        bfield = np.array([])
        jsrem = np.array([])
        jrs = float("nan")
        hcr_mt = float("nan")

    return HysteresisResult(
        sample=loop.sample, mass_mg=mass_mg,
        js_max=js_max, js_ferro=js_ferro, jrs=jrs,
        hc_mt=hc_mt, hcr_mt=hcr_mt,
        jrs_over_jsferro=(jrs / js_ferro) if js_ferro else float("nan"),
        hcr_over_hc=(hcr_mt / hc_mt) if hc_mt else float("nan"),
        susc_para_si=susc_para_si,
        field=x, moment_norm=jsnorm, moment_norm_corrected=jsnorm_corr,
        backfield_field=bfield, backfield_norm=jsrem,
    )


def format_hysteresis_result(res: HysteresisResult) -> str:
    """Equivalent des lignes ecrites par `calculhyste` (console +
    CURIE_output.txt)."""
    return (
        f" Echantillon: {res.sample}\n"
        f" Masse      : {res.mass_mg:g} mg\n"
        f" Js         : {res.js_max:.5g} Am2/kg\n"
        f" Js corrige : {res.js_ferro:.5g} Am2/kg\n"
        f" Remanence  : {res.jrs:.5g} Am2/kg\n"
        f" Hc         : {res.hc_mt:.4g} mT\n"
        f" Hcr        : {res.hcr_mt:.4g} mT\n"
        f" Jrs/Js_ferro: {res.jrs_over_jsferro:.4g}   Hcr/Hc: {res.hcr_over_hc:.4g}\n"
        f" susceptibilite paramagnetique+diamagnetique: {res.susc_para_si:.4g} m3/kg\n"
    )


def _register_xaxis_pick(ax) -> None:
    """Permet de cliquer sur l'axe X (ligne/graduations/etiquette) pour
    changer sa borne (champ max affiche) - meme mecanisme que AMS_Py
    (ams_xy._register_xaxis_pick/app._on_plot_pick, `set_picker`
    declenche un pick_event des qu'un clic tombe pres de l'axe, pas
    seulement sur les donnees) - demande explicite utilisateur ("to
    click on the Xaxis to change the scale (Max field value)")."""
    ax.xaxis.set_picker(True)
    ax.xaxis._su_pick_kind = "hyst_xaxis"


def build_hysteresis_figure(
    res: HysteresisResult,
    fig: Optional[Figure] = None,
    max_field: Optional[float] = None,
    normalize: bool = False,
) -> Figure:
    """Graphique hysteresis (equivalent NON pixel-pres, matplotlib direct
    - voir docstring de module) de `plothyste` (hysteresis.f:471-589) :
    boucle brute (fine, grise) + boucle corrigee de la pente paramagnetique
    (couleur), et courbe de remanence/DCD (verte) sur le MEME panneau -
    Hc/Hcr marques par des lignes verticales pointillees. Forme CARREE
    (`set_box_aspect(1)`, demande explicite utilisateur "the same logic
    applies to the hysteresis plots").

    `normalize` (demande explicite utilisateur "is it possible to
    normalize each plot", precisee ensuite "best to normalize each one
    by its own max") : quand True, CHAQUE courbe est divisee par SON
    PROPRE maximum absolu (boucle brute, boucle corrigee, remanence
    chacune independamment) - memes references que `selectfichhystAGM`
    utilise pour son export GMT (hysteresis.f:857-864 : `js/rmax`,
    `jsnorm/rnormplot`, `jsrem/jsrem(1)` - un maximum PROPRE a chaque
    courbe, pas une reference partagee). Chaque courbe sature ainsi pres
    de +/-1 independamment des deux autres. Sans normalisation (defaut),
    les courbes restent en Am2/kg (comparaison directe entre
    echantillons).

    `max_field` (defaut None = auto sur les donnees) : borne explicite de
    l'axe X (Tesla), symetrique (-max_field, +max_field) - peut aussi
    etre change en cliquant sur l'axe X (voir _register_xaxis_pick /
    app._on_plot_pick, demande explicite utilisateur "to click on the
    Xaxis to change the scale (Max field value)")."""
    if fig is None:
        fig = Figure(figsize=(6.0, 6.0), dpi=100)
    else:
        fig.clear()
    ax = fig.add_subplot(111)
    ax.set_box_aspect(1)

    if normalize:
        def _by_own_max(arr: np.ndarray) -> np.ndarray:
            ref = float(np.max(np.abs(arr))) if len(arr) else 0.0
            return arr / ref if ref else arr
        raw_y = _by_own_max(res.moment_norm)
        corr_y = _by_own_max(res.moment_norm_corrected)
        rem_y = _by_own_max(res.backfield_norm)
        ylabel = "Moment (normalized)"
    else:
        raw_y, corr_y, rem_y = res.moment_norm, res.moment_norm_corrected, res.backfield_norm
        ylabel = "Moment (Am²/kg)"

    ax.plot(res.field, raw_y, "-", color="0.75", linewidth=0.8, label="raw loop")
    ax.plot(res.field, corr_y, "-", color="tab:red", linewidth=1.2, label="corrected loop")
    ax.plot(res.backfield_field, rem_y, "-", color="tab:green", linewidth=1.2,
            label="remanence (DCD)")

    if np.isfinite(res.hc_mt):
        ax.axvline(res.hc_mt / 1000.0, color="tab:red", linestyle=":", linewidth=0.9)
        ax.axvline(-res.hc_mt / 1000.0, color="tab:red", linestyle=":", linewidth=0.9)
    if np.isfinite(res.hcr_mt):
        ax.axvline(-res.hcr_mt / 1000.0, color="tab:green", linestyle=":", linewidth=0.9)

    ax.axhline(0.0, color="0.6", linewidth=0.6)
    ax.axvline(0.0, color="0.6", linewidth=0.6)
    if max_field is not None and max_field > 0:
        ax.set_xlim(-max_field, max_field)
    ax.set_xlabel("Field (T)")
    ax.set_ylabel(ylabel)
    ax.set_title(res.sample)
    ax.legend(fontsize=8, loc="best")
    _register_xaxis_pick(ax)
    fig.tight_layout()
    return fig
