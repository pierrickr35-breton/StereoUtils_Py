"""
K-T (susceptibilite en fonction de la temperature) : port des routines
principales de Curie_OSX (reference/../Curie_Hyst_AWE - CurieOSX_x.f95,
menu "KLY3_CS3", et divers.f pour saisie/tauxe/minmax) - demande explicite
utilisateur ("integrer les deux fonctions principales de Curie_OSX, le
traitement des donnees de susceptibilite (courbes K-T)... ajouter un menu
K-T"). L'AUTRE fonction principale (Hysteresis, hysteresis.f) n'est PAS
portee ici - demandee separement, pas encore traitee.

Format reel VERIFIE sur les fichiers .CUR/.CLW de /Users/pierrickroperch/
Paleomag_data/Curie_Chili_95_10_11_12 (Chili-10-11, Chili_Curie_2012,
Villarica_95) : une ligne d'entete "TEMP TSUSC CSUSC NSUSC BULKS FERRT
FERRB TIME EMPTY/NONAME" puis, par ligne, 9 colonnes dont seules les 2
premieres (TEMP, TSUSC) sont utilisees - CSUSC/NSUSC valent 0.00 dans TOUS
les fichiers reels inspectes (jamais la branche "susceptibilite deja
corrigee" de `saisie`, divers.f:88-98, qui lirait 3 colonnes) : ce module
ne porte donc que la branche a 2 colonnes, la seule rencontree en
pratique. `.CLW` (cycle basse temperature, ex. -193.5 a +25 degC) partage
EXACTEMENT le meme format de colonnes que `.CUR` (chauffe four) - seule la
plage de temperature differe.

Listes d'echantillons (liste_Chili_2010.txt/liste_Chili_2011.txt/
Liste_Curie_Chili_2012, meme repertoire, PLUS un exemple utilisateur avec
la variante a 999/vase vide - voir docstring KTListEntry) : port de
`openlistekt` (CurieOSX_x.f95:431-473) - format reel observe : UN INDICE
en tete de chaque ligne (absent du READ Fortran d'origine, probablement
ajoute par un tableur/editeur lors d'une resauvegarde ulterieure - jamais
lu par le programme, ignore ici) puis <nom_fichier> <suscep> <masse>
<zerodia> [<vase_vide>] separes par des espaces/tabulations
(list-directed).

Deux corrections four possibles (voir apply_furnace_correction) : une
valeur CONSTANTE (`zerodia`, ex. -142.8 - la seule presente dans les
listes Chili 95/10/11/12) soustraite telle quelle a toute la courbe, ou
(zerodia==999, demande explicite utilisateur "999 indicate that a whole
empty vessel should be used ... when it is slightly contaminated and the
magnetic susceptibility of the sample is also very low") la courbe COMPLETE
(fonction de la temperature, chauffe/refroidissement separes) d'un vase
vide mesure a part et lui-meme liste dans le meme fichier - voir
correctdiamag_tables/apply_vessel_correction, port de `correctdiamag`
(CurieOSX_x.f95:581-654).

Graphiques reconstruits DIRECTEMENT avec matplotlib (meme choix que
ams_xy.py cote AMS_Py / xygraph.py cote STARpaleomag_Py) - demande
explicite utilisateur ("Faire les graphics sans passer par mes anciennes
fonctions (plot, symbol etc)") : PAS de port pixel-pres via
plotlib.PlotContext/symbo1 ici, contrairement au stereonet natif de ce
projet (stereo_net.py)."""

import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.figure import Figure


@dataclass
class KTCurve:
    """Une courbe K-T brute ou corrigee : temperature/susceptibilite
    APPARIEES, meme ordre chronologique que le fichier .CUR/.CLW (voir
    read_cur_file) - `sample` : nom de fichier sans extension, 12
    premiers caracteres (meme convention que `fich(i).sample` cote
    Fortran, openlistekt: `fil2(1:nlen-4)`). `corrected`/`normalized`
    tracent le pipeline deja applique (voir apply_furnace_correction/
    normalize_by_mass/normalize_by_volume), pour eviter de re-appliquer
    une normalisation par erreur (meme garde que le Fortran, "susceptibi-
    lite deja normalisee")."""
    sample: str
    path: str
    temp: np.ndarray
    tsusc: np.ndarray
    corrected: bool = False
    normalized: Optional[str] = None  # None / "mass" / "volume"


def read_cur_file(path: str) -> KTCurve:
    """Equivalent de `saisie` (divers.f:3-182), branche NON corrigee
    (reponse 'N' a "susceptibilite corrigee o/N?" - la SEULE rencontree
    dans les fichiers reels Chili 95/10/11/12, voir docstring de module) :
    une ligne d'entete ignoree, puis TEMP/TSUSC en 1ere/2eme colonne de
    chaque ligne suivante - les 7 autres colonnes sont ignorees, meme
    comportement que le READ list-directed Fortran `read(2,*) xit(i),
    yy(i)` sur une ligne a plus de 2 valeurs (valeurs en trop simplement
    non lues, pas une erreur)."""
    temps: List[float] = []
    tsusc: List[float] = []
    with open(path, "r", encoding="iso-8859-1", errors="replace") as f:
        next(f, None)
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                t, s = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            temps.append(t)
            tsusc.append(s)
    sample = os.path.splitext(os.path.basename(path))[0][:12]
    return KTCurve(sample=sample, path=path, temp=np.array(temps), tsusc=np.array(tsusc))


@dataclass
class KTListEntry:
    """Une ligne de liste_*.txt (voir openlistekt) : parametres de
    correction/normalisation d'UN specimen. `suscep` : susceptibilite
    bulk INDEPENDANTE (SI, ex. 1.695E-02 = 0.01695 SI) utilisee par
    normalize_by_volume pour ancrer la courbe - 0 = non fournie, utiliser
    `masse`/normalize_by_mass a la place (demande explicite utilisateur -
    "when susceptibility is 0, the normalization is by mass, otherwise
    it assume that the sample has this magnetic susceptibility"). `masse`
    : masse en mg. `zerodia` : soit une valeur CONSTANTE a soustraire
    (correction four, voir apply_furnace_correction), soit le SENTINEL
    999 signifiant "utiliser la courbe complete d'un vase vide" -
    `empty_vessel` porte alors son nom de fichier (5eme colonne,
    None sinon) - demande explicite utilisateur ("999 indicate that a
    whole empty vessel should be used ... when it is slightly
    contaminated and the magnetic susceptibility of the sample is also
    very low")."""
    filename: str
    suscep: float
    masse: float
    zerodia: float
    empty_vessel: Optional[str] = None


def read_kt_sample_list(path: str) -> Dict[str, KTListEntry]:
    """Equivalent de `openlistekt` (CurieOSX_x.f95:431-473) - voir
    docstring de module pour l'ecart de format reel (indice de ligne en
    tete, ignore) et pour la 5eme colonne optionnelle (nom du vase vide,
    seulement quand zerodia==999). Cle de jointure avec un fichier .CUR/
    .CLW : nom de fichier en MAJUSCULES (casse variable observee dans les
    listes reelles)."""
    entries: Dict[str, KTListEntry] = {}
    with open(path, "r", encoding="iso-8859-1", errors="replace") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0].isdigit():
                parts = parts[1:]
            if len(parts) < 4:
                continue
            filename = parts[0]
            try:
                suscep, masse, zerodia = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            empty_vessel = parts[4] if zerodia == 999.0 and len(parts) >= 5 else None
            entries[filename.upper()] = KTListEntry(filename, suscep, masse, zerodia, empty_vessel)
    return entries


def apply_furnace_correction(curve: KTCurve, zerodia: float) -> KTCurve:
    """Equivalent de `corrfour` (CurieOSX_x.f95:240-291), branche
    constante (`zerodia != 999` - voir apply_vessel_correction pour
    l'autre branche) : soustrait la meme valeur a TOUTE la courbe.
    Fonction PURE (retourne une nouvelle KTCurve, ne modifie pas
    `curve`)."""
    return replace(curve, tsusc=curve.tsusc - zerodia, corrected=True)


def _smooft(y: np.ndarray, pts: float) -> np.ndarray:
    """Filtre passe-bas equivalent a `SMOOFT`/`REALFT`/`FOUR1`
    (filterCurie.f95, Numerical Recipes) : detrend lineaire entre le
    premier et le dernier point, complete de zeros jusqu'a la prochaine
    puissance de 2 >= n+2*pts, filtre chaque frequence k par
    max(0, 1-(pts*k/m)^2) (MEME formule que le Fortran, verifiee
    algebriquement identique - y compris son cas special "Nyquist",
    (1-0.25*pts^2), qui est cette meme formule evaluee a k=m/2), transformee
    inverse, puis re-ajoute la tendance lineaire retiree au debut.
    Reimplementee via `numpy.fft.rfft`/`irfft` (normalisation automatique
    correcte) plutot qu'un portage litteral de REALFT/FOUR1 (qui gerent
    eux-memes une normalisation manuelle, `/MO2`, propre a leur FFT non
    normalisee - non necessaire ici, numpy s'en charge) : MEME filtre,
    implementation differente. `pts` : equivalent du parametre "fenetre"
    demande par l'interface (`filtered window ... pts`)."""
    n = len(y)
    if n < 4:
        return y.copy()
    m = 2
    nmin = n + int(2.0 * pts)
    while m < nmin:
        m *= 2
    y1, yn = y[0], y[-1]
    j = np.arange(n)
    trend = (y1 * (n - 1 - j) + yn * j) / (n - 1)
    padded = np.zeros(m)
    padded[:n] = y - trend
    spec = np.fft.rfft(padded)
    k = np.arange(len(spec))
    const = (pts / m) ** 2
    fac = np.maximum(0.0, 1.0 - const * k * k)
    filtered = np.fft.irfft(spec * fac, n=m)
    return filtered[:n] + trend


def correctdiamag_tables(vessel_curve: KTCurve) -> Tuple[np.ndarray, np.ndarray]:
    """Equivalent de `correctdiamag` (CurieOSX_x.f95:581-654) : a partir
    de la courbe COMPLETE d'un vase vide (`vessel_curve`), construit deux
    tables de susceptibilite indexees par TEMPERATURE ENTIERE (0..800
    degC, index = degres) - heating[.]/cooling[.] - a soustraire point
    par point de l'echantillon (voir apply_vessel_correction). La courbe
    du vase est d'abord lissee (meme filtre que `smooth`, pts=20 fixe
    - valeur EXACTE du Fortran, "npt=20"), puis:
      - en-dessous de la temperature de depart du vase / au-dessus de sa
        temperature max : valeur constante (moyenne des 3 points de
        bord correspondants), memes bornes/moyennes que le Fortran ;
      - entre les deux : valeur du point du vase a cette temperature
        entiere (branche chauffe/refroidissement distinguee par le sens
        de variation de la temperature, comme le Fortran) ;
      - les degres non couverts par un point exact du vase (ecart de
        temperature entre 2 mesures voisines > 1 degre) sont combles par
        report de la valeur du degre precedent (meme "forward fill" que
        le Fortran, `suscepfour(1,i)=suscepfour(1,i-1)` quand toujours a
        la sentinelle -9999)."""
    tf = vessel_curve.temp
    nf = len(tf)
    valf = _smooft(vessel_curve.tsusc, pts=20.0)

    rising = tf[:-1] < tf[1:]
    rising_idx = np.nonzero(rising)[0]
    imax0 = int(rising_idx[-1]) if len(rising_idx) else 0  # 0-based, dernier point encore en hausse

    suscepatmax = float(np.mean(valf[max(0, imax0 - 2): imax0 + 1]))
    susceptminfin = float(np.mean(valf[-3:]))
    susceptmindeb = float(np.mean(valf[:3]))
    tmax = float(tf[imax0])
    tmindeb = float(tf[0])
    tminfin = float(tf[-1])

    heating = np.full(801, np.nan)
    cooling = np.full(801, np.nan)
    for deg in range(801):
        if deg < int(tmindeb):
            heating[deg] = susceptmindeb
        if deg < int(tminfin):
            cooling[deg] = susceptminfin
        if deg > int(tmax):
            heating[deg] = suscepatmax
            cooling[deg] = suscepatmax

    for i in range(nf - 1):
        deg = int(tf[i])
        if not (0 <= deg <= 800):
            continue
        if tf[i] < tf[i + 1]:
            heating[deg] = valf[i]
        else:
            cooling[deg] = valf[i]

    for deg in range(1, 801):
        if np.isnan(heating[deg]):
            heating[deg] = heating[deg - 1]
        if np.isnan(cooling[deg]):
            cooling[deg] = cooling[deg - 1]
    if np.isnan(heating[0]):
        heating[0] = 0.0
    if np.isnan(cooling[0]):
        cooling[0] = 0.0
    return heating, cooling


def apply_vessel_correction(curve: KTCurve, vessel_curve: KTCurve) -> KTCurve:
    """Equivalent de la branche `zerodia==999` de `corrfour`
    (CurieOSX_x.f95:264-289) : soustrait, POINT PAR POINT, la valeur de
    la table heating/cooling du vase vide (voir correctdiamag_tables) a
    la temperature entiere de CE point - branche chauffe/refroidissement
    du point choisie par le sens de variation local de sa propre
    temperature (le dernier point de la courbe est toujours traite comme
    refroidissement, meme convention que le Fortran). Fonction PURE."""
    heating, cooling = correctdiamag_tables(vessel_curve)
    t = curve.temp
    y = curve.tsusc.copy()
    n = len(y)
    for i in range(n - 1):
        deg = min(800, max(0, int(t[i])))
        y[i] -= heating[deg] if t[i] < t[i + 1] else cooling[deg]
    if n:
        deg = min(800, max(0, int(t[-1])))
        y[-1] -= cooling[deg]
    return replace(curve, tsusc=y, corrected=True)


def normalize_by_mass(curve: KTCurve, mass_mg: float) -> KTCurve:
    """Equivalent de `normas` (CurieOSX_x.f95:293-330) : `yy = 10*yy/
    masse` (mg), resultat en 1e-6 m3/kg (commentaire Fortran, transcrit
    tel quel). Refuse silencieusement de re-normaliser une courbe deja
    normalisee (meme garde que le Fortran, "susceptibilite deja
    normalisee") - retourne `curve` inchangee dans ce cas."""
    if curve.normalized is not None:
        return curve
    return replace(curve, tsusc=10.0 * curve.tsusc / mass_mg, normalized="mass")


def normalize_by_volume(curve: KTCurve, suscep_bulk_si: float, is_cur: bool = True) -> KTCurve:
    """Equivalent de `norvol` (CurieOSX_x.f95:332-372) : ancre la courbe
    sur une susceptibilite bulk INDEPENDANTE (`suscep_bulk_si`, DEJA en
    SI, ex. 0.01695 - demande explicite utilisateur "the magnetic
    susceptibility in the list is given in SI (no need for the 10^-5
    factor)" : le Fortran convertit sa propre saisie interactive
    "valeur ... en 1e-5 SI" via `/1.0e-05` avant de re-multiplier par
    `1.0e-5` plus loin - un aller-retour qui s'annule algebriquement et
    ne fait que decrire la convention du PROMPT interactif d'origine, pas
    une unite reellement appliquee a `suscep_bulk_si`/au resultat ; ce
    port prend directement la valeur SI, sans ce detour), au PREMIER
    point pour un .CUR (avant chauffe) ou au DERNIER pour tout autre type
    (ex. .CLW, apres le cycle froid - `is_cur` reproduit le test sur
    l'extension de fichier cote Fortran). Meme garde "deja normalisee"
    que normalize_by_mass. Resultat en SI (meme unite que
    `suscep_bulk_si`)."""
    if curve.normalized is not None:
        return curve
    val0 = curve.tsusc[0] if is_cur else curve.tsusc[-1]
    if val0 == 0.0:
        return curve
    return replace(curve, tsusc=suscep_bulk_si * curve.tsusc / val0, normalized="volume")


def minmax_temperature(curve: KTCurve) -> Tuple[int, int]:
    """Equivalent de `minmax` (divers.f:688-706) : indices (0-based) du
    minimum et du maximum de temperature sur TOUT le tableau (pas
    forcement les extremites - un cycle four heating/cooling standard a
    son maximum quelque part au milieu)."""
    return int(np.argmin(curve.temp)), int(np.argmax(curve.temp))


def curie_point_second_derivative(curve: KTCurve) -> Tuple[Optional[float], Optional[float]]:
    """Equivalent de `tauxe` (divers.f:740-816). MALGRE son nom, PAS la
    methode des deux tangentes de Tauxe et al. (1996) - la methode
    REELLEMENT implementee est celle du MAXIMUM DE LA DERIVEE SECONDE
    (derivees premiere/seconde par differences centrees sur 3 points, /3
    - transcrit tel quel), calculee separement sur la branche de CHAUFFE
    (jusqu'a l'index du maximum de temperature) et de REFROIDISSEMENT
    (apres) - voir minmax_temperature pour cet index de separation.

    Simplification assumee (bornes exactes des boucles Fortran NON
    reproduites au point pres - `dm`/`dmm` sont ici calcules sur toute la
    plage interieure valide via des tranches numpy vectorisees, au lieu
    des indices de boucle `jstart`/`jend` exacts du Fortran qui excluent
    1-2 points de bord en plus) : une recherche de MAXIMUM sur 300-400
    points n'est pas sensible a 1-2 points de bord exclus ou non - la
    valeur de temperature de Curie retournee est inchangee en pratique.

    Retourne (temp_curie_chauffe, temp_curie_refroidissement) - None pour
    une branche trop courte (<4 points utiles) pour calculer une derivee
    seconde."""
    y = curve.tsusc
    x = curve.temp
    n = len(y)
    if n < 6:
        return None, None
    _itmin, itmax = minmax_temperature(curve)

    dm = np.full(n, np.nan)
    dm[1:-1] = (y[2:] - y[:-2]) / 3.0

    dmm = np.full(n, np.nan)
    dmm[2:-2] = (dm[3:-1] - dm[1:-3]) / 3.0

    def _argmax_in(lo: int, hi: int) -> Optional[int]:
        # [lo, hi) sur dmm, en ignorant les NaN de bord.
        if hi - lo < 1:
            return None
        seg = dmm[lo:hi]
        if np.all(np.isnan(seg)):
            return None
        return lo + int(np.nanargmax(seg))

    idx_heat = _argmax_in(2, max(itmax, 2))
    temp_heat = float(x[idx_heat]) if idx_heat is not None else None

    idx_cool = _argmax_in(itmax + 1, n - 2)
    temp_cool = float(x[idx_cool]) if idx_cool is not None else None

    return temp_heat, temp_cool


def build_kt_figure(
    curve: KTCurve,
    curie_heating: Optional[float] = None,
    curie_cooling: Optional[float] = None,
    fig: Optional[Figure] = None,
) -> Figure:
    """Graphique K-T (susceptibilite vs temperature) - equivalent NON
    pixel-pres (matplotlib direct - voir docstring de module) de
    `tracerklyhot`/`tracerklycold` (divers.f:1084-1416) : branche de
    chauffe (jusqu'a l'index du maximum de temperature) en trait plein,
    refroidissement en tirets - meme distinction visuelle que le Fortran
    (deux sous-routines de trace separees), reunies ici sur un seul axe.
    Les temperatures de Curie (curie_point_second_derivative), si
    fournies, sont marquees par une ligne verticale pointillee de la
    couleur de leur branche."""
    if fig is None:
        fig = Figure(figsize=(6.0, 6.0), dpi=100)
    else:
        fig.clear()
    ax = fig.add_subplot(111)
    # forme carree du panneau de trace (X=temperature, Y=susceptibilite,
    # unites differentes - set_box_aspect force un CADRE carre a
    # l'affichage sans imposer une echelle "1 unite X = 1 unite Y" comme
    # le ferait set_aspect('equal'), qui n'aurait pas de sens ici) -
    # demande explicite utilisateur ("is it possible to have a plot with
    # a square shape").
    ax.set_box_aspect(1)

    _itmin, itmax = minmax_temperature(curve)
    heat_t, heat_y = curve.temp[: itmax + 1], curve.tsusc[: itmax + 1]
    cool_t, cool_y = curve.temp[itmax:], curve.tsusc[itmax:]

    ax.plot(heat_t, heat_y, "-", color="tab:red", linewidth=1.3, label="heating")
    if len(cool_t) > 1:
        ax.plot(cool_t, cool_y, "--", color="tab:blue", linewidth=1.3, label="cooling")

    if curie_heating is not None:
        ax.axvline(curie_heating, color="tab:red", linestyle=":", linewidth=0.9)
    if curie_cooling is not None:
        ax.axvline(curie_cooling, color="tab:blue", linestyle=":", linewidth=0.9)

    ax.axhline(0.0, color="0.75", linewidth=0.6)
    ax.set_xlabel("Temperature (°C)")
    unit = {
        "mass": "susceptibility (1e-6 m³/kg)",
        "volume": "susceptibility (SI)",
    }.get(curve.normalized, "susceptibility (raw, instrument units)")
    ax.set_ylabel(unit)
    ax.set_title(curve.sample)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig
