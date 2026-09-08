"""
Trace du reseau stereographique/equiaire et des donnees : port de
`plotdata` (stereograph.f:131-528). Cadre du reseau (cercle, graduations,
N/E/S/W) - formules verifiees identiques a `draw_stereo_net` deja porte
cote Starmac_Py (memes constantes magiques `dimster/68.76` etc.).

Symboles des directions (`adecsym`, un caractere par point) : c=cercle,
t=triangle, e=etoile, l=losange, s=carre - plein si l'hemisphere est
"visible" (ifl!=5 cote `superc`), ouvert sinon (memes types que
`plotlib._label2`, table deja partagee avec Starmac/AMS)."""

import math
from typing import List, Optional, Tuple

from matplotlib.figure import Figure

from plotlib import PlotContext
from stereo_geometry import superc, gdcerc, circle

_SYMBOL_TYPES = {
    "c": (8, 14), "t": (10, 16), "e": (13, 19), "l": (12, 18), "s": (9, 15),
}


def draw_stereo_net(
    ctx: PlotContext, la: float = -90.0, phi: float = 0.0, iproj: int = 0,
    dim: float = 21.0, show_orient_label: bool = False,
) -> float:
    """Port du cadre du reseau dans `plotdata` (stereograph.f:200-251).
    Retourne le rayon `r` (cm)."""
    r = dim / 3.0
    dimster = dim

    ctx.newpen(1)
    ctx.thickn(1.0)
    ctx.circl2(0.0, 0.0, r, 1, 0)

    # petits traits vers l'interieur a N/E/S/W (stereograph.f:201-230),
    # oublies dans une premiere passe - coordonnees brutes (pas de swap
    # `plott`, ces `plot()` du source ne passent pas par ce wrapper).
    ctx.plot(r, 0.0, 3)
    ctx.plot(r - dimster / 60.0, 0.0, 2)
    ctx.plottxt(r + dimster / 190.9, -dimster / 65.6233, dimster / 38.158692, "E")

    ctx.plot(0.0, r, 3)
    ctx.plot(0.0, r - dimster / 60.0, 2)
    ctx.plottxt(-dimster / 68.76, (r + dimster / 21.6) - dimster / 27.63, dimster / 38.158692, "N")

    ctx.plot(-r + dimster / 60.0, 0.0, 3)
    ctx.plot(-r, 0.0, 2)
    ctx.plottxt(-r - 4 * dimster / 84, -dimster / 65.6233, dimster / 38.158692, "W")

    ctx.plot(0.0, -r, 3)
    ctx.plot(0.0, -r + dimster / 60.0, 2)
    ctx.plottxt(-dimster / 68.76, (-r - dimster / 1050.0) - dimster / 27.63, dimster / 38.158692, "S")

    ctx.newpen(1)
    ctx.thickn(0.25)
    teta = 0.0
    for _j in range(4):
        dipp = 10.0
        for _k in range(17):
            u, v, ifl = superc(la, phi, teta, 90.0 - dipp, iproj)
            if ifl != 5 and (u * u + v * v) <= 0.98:
                ctx.symbol(v * r, u * r, dim / 72.0, 5, -1)
            dipp += 10.0
        teta += 90.0
    # poles (dip=+-90) : symbo2 remappe ifl-8 (nord, =3->5 "plus") et
    # ifl-1 (sud, =10->8 "cercle ouvert") - PAS le meme type pour les 2
    # poles, verifie sur le source (stereograph.f:246-251).
    u, v, ifl = superc(la, phi, 0.0, 90.0, iproj)
    if ifl != 5:
        ctx.symbol(v * r, u * r, dim / 35.0, 5, -1)
    u, v, ifl = superc(la, phi, 0.0, -90.0, iproj)
    if ifl != 5:
        ctx.symbol(v * r, u * r, dim / 35.0, 8, -1)

    ctx.newpen(1)
    return r


def draw_stereo_data(
    ctx: PlotContext,
    directions: List[Tuple[float, float, str]],
    r: float,
    la: float = -90.0,
    phi: float = 0.0,
    iproj: int = 0,
    path_between_points: bool = False,
    point_size: float = 0.3,
) -> None:
    """Port de la boucle de trace des donnees dans `plotdata`
    (stereograph.f:419-527) : `directions` = liste de (dec, inc, symbole
    'c'/'t'/'e'/'l'/'s'). Si `path_between_points`, les points consecutifs
    sont relies par un arc de grand cercle (`gdcerc`, equivalent de
    `test=.false.` cote Fortran - variable nommee a l'envers de son
    intitule, verifie sur le source)."""
    if not directions:
        return
    ctx.thickn(0.55)
    dec1, dip1 = directions[0][0], directions[0][1]
    for i, (dec, dip, sym) in enumerate(directions):
        if i > 0 and path_between_points:
            gdcerc(ctx, la, phi, iproj, dec1, dip1, dec, dip, r)
        u, v, ifl = superc(la, phi, dec, dip, iproj)
        open_t, filled_t = _SYMBOL_TYPES.get(sym, _SYMBOL_TYPES["c"])
        ityp = open_t if ifl == 5 else filled_t
        ctx.symbol(v * r, u * r, point_size, ityp, -1)
        dec1, dip1 = dec, dip
    ctx.newpen(1)


def draw_stereo_means(
    ctx: PlotContext,
    means: List[Tuple[float, float, float, str]],
    r: float,
    la: float = -90.0,
    phi: float = 0.0,
    iproj: int = 0,
) -> None:
    """Port de la boucle des directions moyennes dans `plotdata`
    (stereograph.f:302-389) : `means` = liste de (dec,inc,alpha95,symbole).
    Chaque moyenne est tracee avec son symbole PUIS son cercle de confiance
    a95 (`circle`, 121 points par pas de 3 deg - un alpha95=0 degenere
    simplement en un point au centre, sans cas particulier dans le
    source). Le trace du cercle change de "couleur" (pen 5/bleu si le
    point est sur l'hemisphere visible, pen 3/rouge sinon) a chaque
    changement de cote, meme logique que le source (`icol`/`ippen`)."""
    if not means:
        return
    for dec, dip, alph, sym in means:
        # icolm(j)=5 (bleu) pour toute moyenne creee via Data input
        # (dataman/mean/opfil2/opfil5) - pas de champ couleur distinct
        # dans ce modele simplifie, valeur fixe verifiee sur le source.
        ctx.newpen(5)
        ctx.thickn(0.55)
        u, v, ifl = superc(la, phi, dec, dip, iproj)
        open_t, filled_t = _SYMBOL_TYPES.get(sym, _SYMBOL_TYPES["c"])
        ityp = open_t if ifl == 5 else filled_t
        ctx.symbol(v * r, u * r, 0.5, ityp, -1)

        ctx.thickn(0.45)
        ei0, ed0 = circle(alph, dip, dec, 0.0)
        icol1 = -1 if ei0 < 0 else 1
        ctx.newpen(3 if icol1 == -1 else 5)
        uu, vv, _ifl = superc(la, phi, ed0, ei0, iproj)
        ctx.plot(vv * r, uu * r, 3)
        ph = 0.0
        for _i in range(120):
            ph += 3.0
            ei, ed = circle(alph, dip, dec, ph)
            icol = -1 if ei < 0 else 1
            if icol != icol1:
                ctx.newpen(3 if icol == -1 else 5)
                icol1 = icol
            uu, vv, _ifl = superc(la, phi, ed, ei, iproj)
            ctx.plot(vv * r, uu * r, 2)
    ctx.newpen(1)


def draw_stereo_great_circles(
    ctx: PlotContext,
    great_circles: List[Tuple[float, float, float, float, float, float]],
    r: float,
    la: float = -90.0,
    phi: float = 0.0,
    iproj: int = 0,
) -> None:
    """Port de la boucle des grands cercles dans `plotdata`
    (stereograph.f:391-417) : `great_circles` = liste de (glon,glat,ad1,
    ai1,ad2,ai2) - glon/glat = pole du grand cercle (ad1..ai2, bornes de
    secteur optionnelles, pas encore utilisees ici - reservees au menu
    Statistics/FISHGC). Chaque grand cercle est trace en 4 arcs de 90 deg
    (`gdcerc`) entre des points equidistants du pole, exactement comme le
    source (pas via `circle(90,...)` directement)."""
    if not great_circles:
        return
    ctx.newpen(1)
    ctx.thickn(0.65)
    for glon, glat, _ad1, _ai1, _ad2, _ai2 in great_circles:
        if glat < 0.0:
            dipp = glat + 90.0
            decc = glon
        else:
            dipp = -glat + 90.0
            decc = glon + 180.0
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
    ctx.newpen(1)


def build_stereo_figure(
    directions: List[Tuple[float, float, str]],
    la: float = -90.0,
    phi: float = 0.0,
    iproj: int = 0,
    dim: float = 21.0,
    path_between_points: bool = False,
    point_size: float = 0.3,
    means: Optional[List[Tuple[float, float, float, str]]] = None,
    great_circles: Optional[List[Tuple[float, float, float, float, float, float]]] = None,
    fig: Optional[Figure] = None,
) -> Figure:
    """Equivalent de `plotdata` (menu Command > Plot-screen) : cadre du
    reseau + directions/moyennes/grands cercles charges/saisis."""
    if fig is None:
        fig = Figure(figsize=(5.5, 5.5), dpi=100)
    else:
        fig.clear()
    ax = fig.add_subplot(111)
    ctx = PlotContext(ax)
    ctx.clear()
    ctx.plot(0.0, 0.0, -3)

    r = draw_stereo_net(ctx, la, phi, iproj, dim)
    draw_stereo_means(ctx, means or [], r, la, phi, iproj)
    draw_stereo_great_circles(ctx, great_circles or [], r, la, phi, iproj)
    draw_stereo_data(ctx, directions, r, la, phi, iproj, path_between_points, point_size)

    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    return fig
