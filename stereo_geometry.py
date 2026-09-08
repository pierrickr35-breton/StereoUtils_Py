"""
Geometrie de projection stereographique/equiaire : port de `superc`/`gdcerc`/
`angle` (stereograph.f:1010-1163) - VERIFIE formule a formule identique a la
version deja portee et validee cote Starmac_Py (`stereo.py`), meme
bibliotheque de calcul reprise a l'identique entre les deux applications
(memes noms de variables, meme structure). Repris ici tel quel plutot que
retranscrit independamment.

`superc` gere un pole de projection OBLIQUE (la != +-90, heritage Supermap
- voir memoire projet "Supermap heritage") - PAS simplifie ici,
contrairement a AMS_Py (dont l'appli d'origine, plus recente, n'utilise que
`stecor`, la version simple sans pole oblique)."""

import math
from typing import Tuple

from plotlib import PlotContext


def _angle(p: float, a: float, b: float) -> Tuple[float, float]:
    """Port de `angle` (stereograph.f:1078-1096) : distance angulaire (deg)."""
    pr, ar, br = math.radians(p), math.radians(a), math.radians(b)
    cs = math.cos(ar) * math.cos(br) + math.sin(ar) * math.sin(br) * math.cos(pr)
    cs = min(cs, 1.0)
    y = math.sqrt(max(0.0, 1.0 - cs * cs))
    delr = math.atan2(y, cs)
    dell = math.sin(delr) * math.sin(br)
    if dell == 0.0:
        dell = 0.000001
    cs2 = (math.cos(ar) - math.cos(delr) * math.cos(br)) / dell
    cs2 = min(cs2, 1.0)
    y2 = math.sqrt(max(0.0, 1.0 - cs2 * cs2))
    ang = math.atan2(y2, cs2)
    if p < 0.0:
        ang = -ang
    return math.degrees(delr), math.degrees(ang)


def superc(la: float, phi: float, dec: float, dip: float, iproj: int) -> Tuple[float, float, int]:
    """Port de `superc` (stereograph.f:1097-1163) : projette (dec,dip)
    autour du pole de projection (la,phi) - iproj=1: equiaire/Schmidt,
    sinon stereographique/Wulff. Retourne (u,v,ifl) - ifl=5 si la
    direction tombe sur l'hemisphere "cache" (symbole ouvert), 11 sinon
    (symbole plein)."""
    eps = 1e-6
    ifl = 11

    if abs(la) != 90.0:
        teta = phi - dec
        a = 90.0 - dip
        b = 90.0 - la
        if abs(teta) >= 180.0:
            teta = teta + 360.0 if teta < 0 else teta - 360.0
        tet = abs(teta)
        delta, ang1 = _angle(tet, a, b)
        if abs(delta) < 90.0:
            ifl = 5
            delt = 180.0 - delta if delta >= 0 else -180.0 - delta
            del_, ang = _angle(ang1, delt, b)
            del_ = abs(del_)
            dip = 90.0 - del_
            dec = phi + ang if teta <= 0 else phi - ang
    else:
        if dip < 0.0:
            dip = -dip
            ifl = 5

    if iproj == 1:
        la2, phi2 = -la, phi + 180.0
    else:
        la2, phi2 = la, phi

    cos0, sin0 = math.cos(math.radians(la2)), math.sin(math.radians(la2))
    sinph = math.sin(math.radians(dec - phi2))
    cosph = math.cos(math.radians(dec - phi2))
    sinla = math.sin(math.radians(dip))
    cosla = math.sqrt(max(0.0, 1.0 - sinla * sinla))
    cosa = sinla * sin0 + cosla * cos0 * cosph
    sina = math.sqrt(max(0.0, 1.0 + eps - cosa * cosa))
    sinb = cosla * sinph / sina if sina else 0.0
    cosb = (sinla * cos0 - cosla * sin0 * cosph) / sina if sina else 0.0

    if iproj != 1:
        r = (1.0 + cosa) / sina
        return r * cosb, r * sinb, ifl

    if cosa + 1.0 <= 0.0:
        return 1.0, 1.0, ifl
    r = sina / math.sqrt(0.5 + 0.5 * cosa)
    u = r * cosb / math.sqrt(2.0)
    v = -r * sinb / math.sqrt(2.0)
    return u, v, ifl


def circle(al: float, ai: float, ad: float, ph: float) -> Tuple[float, float]:
    """Port de `circle` (stereograph.f:1054-1077) : un point (dec,inc) sur le
    petit/grand cercle de rayon angulaire `al` autour du pole (ad,ai), pour un
    angle parametrique `ph` (0-360, balaie le cercle). `al=90` trace un grand
    cercle. Utilise pour les cercles de confiance a95 des directions
    moyennes (menu Data > mean/opfil2/opfil5)."""
    sal, cal = math.sin(math.radians(al)), math.cos(math.radians(al))
    si, ci = math.sin(math.radians(ai)), math.cos(math.radians(ai))
    sd, cd = math.sin(math.radians(ad)), math.cos(math.radians(ad))
    zpx, zpy, zpz = cal * cd * ci, cal * sd * ci, cal * si
    sp, cp = math.sin(math.radians(ph)), math.cos(math.radians(ph))
    xp, yp = sal * cp, sal * sp
    x = xp * (sd * sd + cd * cd * si) + yp * sd * cd * (si - 1.0) + zpx
    y = xp * (cd * sd * si - cd * sd) + yp * (cd * cd + sd * sd * si) + zpy
    z = xp * (-cd * ci) + yp * (-sd * ci) + zpz
    s = math.hypot(x, y)
    ed = math.degrees(math.atan2(y / s, x / s)) if s else 0.0
    if ed < 0.0:
        ed += 360.0
    ei = math.degrees(math.atan2(z, s)) if s else (90.0 if z > 0 else -90.0)
    return ei, ed


def polere(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Port de `POLERE` (pmagoutils.f:2398) : cartesien -> (magnitude,
    declinaison, inclinaison en degres). Formule-identique a la version deja
    portee/verifiee cote Starmac_Py (`selection.py:polere`, meme atan2)."""
    horiz = math.hypot(x, y)
    mag = math.hypot(horiz, z)
    if mag == 0.0:
        return 0.0, 0.0, 0.0
    dec = math.degrees(math.atan2(y, x))
    if dec < 0.0:
        dec += 360.0
    inc = math.degrees(math.atan2(z, horiz))
    return mag, dec, inc


def corpen(x: float, y: float, z: float, dip: float, strike: float) -> Tuple[float, float, float]:
    """Port de `CORPEN(J,K)` (pmagoutils.f:1680) : correction de pendage
    (rotation de `dip` autour de la direction de `strike`), sur le
    cartesien (x,y,z) commun /CART/. Formule-identique a Starmac_Py
    (`selection.py:corpen`, memes noms J=dip,K=strike)."""
    jr, kr = math.radians(dip), math.radians(strike)
    cj, sj = math.cos(jr), math.sin(jr)
    ck, sk = math.cos(kr), math.sin(kr)
    xx = x * (ck ** 2 + sk ** 2 * cj) + y * (ck * sk * (1.0 - cj)) - z * sk * sj
    yy = x * ck * sk * (1.0 - cj) + y * (sk ** 2 + cj * ck ** 2) + z * sj * ck
    zz = x * sj * sk - y * sj * ck + z * cj
    return xx, yy, zz


def corfor(x: float, y: float, z: float, dip_azimuth: float, axis: float) -> Tuple[float, float, float]:
    """Port de `CORFOR(G,H)` (pmagoutils.f:1670) : correction de forage
    (repere echantillon -> repere in-situ), rotation autour de l'axe de
    la carotte. Formule-identique a Starmac_Py (`selection.py:corfor`,
    memes noms N/E/VV=x/y/z, G=dip_azimuth, H=axis)."""
    gr, hr = math.radians(dip_azimuth), math.radians(axis)
    cg, sg = math.cos(gr), math.sin(gr)
    ch, sh = math.cos(hr), math.sin(hr)
    xx = x * cg * sh + y * ch + z * sh * sg
    yy = -x * cg * ch + y * sh - z * ch * sg
    zz = -x * sg + z * cg
    return xx, yy, zz


def gdcerc(
    ctx: PlotContext, la: float, phi: float, iproj: int,
    dec1: float, dip1: float, dec2: float, dip2: float, r: float,
) -> None:
    """Port de `gdcerc` (stereograph.f:1010-1053) : trace l'arc de grand
    cercle entre deux directions projetees, par petits pas angulaires
    (~3 deg) - identique a la version Starmac."""
    decc = dec1
    teta = dec1 - dec2
    a = 90.0 - dip2
    if a == 0.0:
        a = 0.1
    if a == 180.0:
        a = 179.9
    delti = 0.0
    icont = 1
    b = 90.0 - dip1
    if b == 180.0:
        b = 179.9
    if b == 0.0:
        b = 0.1
    if abs(teta) >= 180.0:
        teta = teta + 360.0 if teta < 0 else teta - 360.0
    tet = abs(teta)
    pas = 3.0
    delta, ang = _angle(tet, a, b)
    a = 0.0
    ifll = 0

    while True:
        u, v, ifl = superc(la, phi, dec1, dip1, iproj)
        if ifl == 11 and ifll == 11:
            icont = 0
        if ifl == 11 and ifll == 5:
            icont = 1
        if ifl == 5 and ifll == 11:
            icont = 1
        # Fortran : call plott(u*r,-v*r,...) -> plott fait plot(-y,x,...) ->
        # plot(-(-v*r), u*r,...) = plot(v*r, u*r,...) - simplifie ici direct
        ctx.plot(v * r, u * r, 2 + (icont % 2))
        if a == -1.0:
            return
        icont += 1
        if abs(delti - delta) < 3.0:
            dec1, dip1, a = dec2, dip2, -1.0
            continue
        delti += pas
        ai, pp = _angle(ang, delti, b)
        dec1 = decc - pp if teta > 0 else decc + pp
        if dec1 > 360.0:
            dec1 -= 360.0
        dip1 = 90.0 - ai
        ifll = ifl
