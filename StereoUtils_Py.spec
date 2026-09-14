# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files

# Meme raison que AMS_Py.spec/STARpaleomag_Py.spec : pmagpy expedie des
# fichiers de donnees hors .py (field_models/, data_model/) que
# PyInstaller ne detecte pas automatiquement.
datas = collect_data_files('pmagpy')
# Guide utilisateur statique (Help > User Guide) - doit etre EXTRAIT sous
# le meme nom de dossier ('help/') pour que _resource_path le retrouve
# une fois empaquete.
datas += [('help', 'help')]

# matplotlib charge le backend SVG dynamiquement au moment de
# fig.savefig(path, format="svg") (Export SVG...), PAS via un `import`
# statique visible dans le source - PyInstaller ne le detecte donc pas
# tout seul et l'app packagee echoue a l'export avec "No module named
# matplotlib.backends.backend_svg" (repere sur un vrai .app construit,
# fonctionnait depuis les sources). backend_pdf ajoute par precaution/
# coherence avec les autres .spec de ce projet, meme si non utilise ici.
hiddenimports = ['matplotlib.backends.backend_svg', 'matplotlib.backends.backend_pdf']

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='StereoUtils_Py',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='StereoUtils_Py',
)
app = BUNDLE(
    coll,
    name='StereoUtils_Py.app',
    icon=None,
    bundle_identifier=None,
)
