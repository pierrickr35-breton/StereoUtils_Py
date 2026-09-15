"""
StereoUtils_Py : port Python de Stereo_V19 (reference/Stereo_V19),
application Fortran/Qt5 d'utilitaires paleomagnetiques (stereonets,
statistiques directionnelles, corrections structurales, connecteur
PmagPy), meme heritage/architecture que Starmac_Py/AMS_Py (console texte +
panneau graphique).

ETAT : tous les menus sont portes - "PmagPy-tools" (stereo_pmagpy.py, ne
reimplemente rien, appelle directement le paquet pmagpy installe plutot
que shell-out vers un script PmagPy standalone comme le faisait le
Fortran), "Graphics" (ex-"Projection", fusionne avec l'ancien menu
"Command" - Plot stereo/Clear Screen/Graph title/Export SVG y vivent
desormais aussi, ainsi que Plot stereo project/Plot VGPs on Map/Plot VGP
Project - demande explicite utilisateur "change the menu projection by
Graphics, add plot stereo in this menu, export svg" puis "move plot
stereo project; VGP on map VGP project; within the graphic menu"), "Data"
(reseau stereographique/equiaire natif + chargement/saisie/liste/
suppression D-I, D-I-a95 ("means"), grands cercles - stereo_net.py/
stereo_geometry.py/stereo_selection.py), "Statistics" (hors Bootstrap
ellipse, portee sous PmagPy-tools - stereo_stats.py), "Project" (systeme
de calques Illustrator multi-couches - stereo_project.py) et "Pmag
Utilities" (pmagoutils.f - VGP/rotation/correction structurale/
paleointensite/IGRF - stereo_pmagutils.py) - demande explicite
utilisateur ("is it possible to write a guide for stereo") : voir
help/StereoUtils_Py_Guide.html (menu Help) pour le detail complet."""

import glob
import os
import subprocess
import sys
import tempfile
import tkinter as tk
import webbrowser
from tkinter import ttk, filedialog, messagebox

from matplotlib import image as mpimg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from stereo_selection import (
    read_di_file, read_fold_file, read_di_tc_file,
    read_di_a95_file, read_di_a95_tc_file,
    read_great_circle_file, read_text_lines, split_header,
)
from stereo_net import build_stereo_figure
import stereo_pmagpy as sp
import stereo_stats as ss
import stereo_project as sproj
import stereo_pmagutils as pu
from curie_kt import (
    read_cur_file, read_kt_sample_list, apply_furnace_correction,
    apply_vessel_correction, normalize_by_mass, normalize_by_volume,
    curie_point_second_derivative, build_kt_figure,
)
from curie_hyst import (
    read_agm_file, read_agm_sample_list, read_vsm_csv, read_vsm_sample_list,
    vsm_paths_for, compute_hysteresis, format_hysteresis_result, build_hysteresis_figure,
)
from site_map import read_prmag_sites, write_kml, write_gmt_map_script, build_site_map_figure

_SYMBOL_CHARS = {"c", "t", "e", "l", "s"}

# Raccourcis clavier - premiere entree de ce type dans StereoUtils_Py,
# meme convention que AMS_Py/STARpaleomag_Py app._SHORTCUTS_MAC/_WIN/
# SHORTCUTS/_labeled/_setup_shortcuts (demande explicite utilisateur
# "can you add a shortcut to the menu normalize") : PAS d'`accelerator=`
# Tk (sur Aqua, `accelerator=` fait intercepter la combinaison par le
# menu natif sans jamais invoquer la commande Tcl, court-circuitant
# `bind_all` - bug reel deja rencontre et documente cote AMS_Py/
# STARpaleomag_Py) - le raccourci est affiche dans le LIBELLE (_labeled)
# et lie separement via `bind_all` (_setup_shortcuts).
_SHORTCUTS_MAC = {
    "hystnorm": ("Cmd+N", "<Command-n>"),
}
_SHORTCUTS_WIN = {
    "hystnorm": ("Ctrl+Shift+N", "<Control-Shift-N>"),
}
SHORTCUTS = _SHORTCUTS_WIN if sys.platform.startswith("win") else _SHORTCUTS_MAC


def _resource_path(*parts: str) -> str:
    """Chemin absolu d'une ressource livree AVEC l'appli (le guide
    utilisateur HTML, voir ouvrir_user_guide) - equivalent de
    STARpaleomag_Py/AMS_Py app._resource_path : fonctionne aussi bien
    lancee depuis les sources (repertoire de ce fichier) qu'empaquetee par
    PyInstaller (sys._MEIPASS) - demande explicite utilisateur ("is it
    possible to write a guide for stereo")."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


class StereoUtilsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("StereoUtils_Py - Paleomagnetic Stereo Utilities")
        self.root.geometry("1100x700")

        self.directions = []     # List[(dec, inc, symbol)] - common /data/ adec/ainc (il)
        self.means = []          # List[(dec, inc, alpha95, symbol)] - common /data/ dic/inc/alph (im)
        self.great_circles = []  # List[(glon, glat, ad1, ai1, ad2, ai2)] - common /data/ glon/glat/... (imm)
        self.project_entries = []  # List[ProjectEntry] - common /dataplot/ plotdir (nbdata)
        self.project_vgp_entries = []  # List[VgpProjectEntry] - bloc "# VGP" du dernier Load Project
        self._current_graphic = None
        self._last_image_files = []

        # Projection state (equivalent /proj/ la,phi,iproj,dim - stereograph.f)
        self.iproj = 1          # 0=stereographique (Wulff), 1=equiaire (Schmidt) - equiaire par defaut
        self.la = -90.0         # pole de projection : latitude
        self.phi = 0.0          # pole de projection : longitude
        self.dim = 21.0         # diametre du reseau (cm) - defaut Fortran
        self.point_size = 0.3   # taille des symboles (cm)
        self.path_between_points = False
        self.graph_title = ""

        # K-T (susceptibilite vs temperature, curie_kt.py) - demande
        # explicite utilisateur ("integrer les deux fonctions principales
        # de Curie_OSX ... ajouter un menu K-T").
        self.kt_curve = None          # curie_kt.KTCurve courante
        self.kt_sample_list = {}      # Dict[str, KTListEntry], voir read_kt_sample_list
        self.kt_list_dir = None       # repertoire de la liste chargee (resout les noms de fichier)
        self.kt_curie_heating = None
        self.kt_curie_cooling = None

        # Hysteresis (curie_hyst.py) - demande explicite utilisateur
        # ("integrer les deux fonctions principales de Curie_OSX ... les
        # donnees d'Hysteresis").
        self.hyst_sample_list = {}    # Dict[str, AgmListEntry|VsmListEntry]
        self.hyst_list_dir = None
        self.hyst_format = None       # "AGM" ou "VSM"
        self.hyst_result = None       # curie_hyst.HysteresisResult courant
        self.hyst_valsat_frac = 0.7   # seuil haut-champ du fit paramagnetique (prefhyste/valsat)
        self.hyst_normalize = tk.BooleanVar(value=False)  # demande explicite "normalize each plot"
        self.hyst_max_field = None    # None = auto ; sinon borne X explicite (Tesla)

        # Site Map (site_map.py) - demande explicite utilisateur ("is
        # there a possibility to have a basic map too" dans la version
        # matplotlib du menu Site Map).
        self.sitemap_basemap = tk.BooleanVar(value=True)

        self._setup_menu()
        self._setup_shortcuts()

        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        self.graph_frame = ttk.Frame(self.paned_window, width=550)
        self.fig = Figure(figsize=(5.2, 5.2), dpi=100)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.paned_window.add(self.graph_frame, weight=1)
        # Clic-sur-axe-X pour changer l'echelle (champ max) du graphique
        # Hysteresis - voir curie_hyst._register_xaxis_pick/_on_plot_pick
        # - demande explicite utilisateur ("to click on the Xaxis to
        # change the scale (Max field value)").
        self.canvas_fig.mpl_connect("pick_event", self._on_plot_pick)

        self.text_frame = ttk.Frame(self.paned_window, width=550)
        self.text_area = tk.Text(
            self.text_frame, bg="#ffffff", fg="#000000", insertbackground="black",
            font=("Courier", 14), wrap="none",
        )
        text_yscroll = ttk.Scrollbar(self.text_frame, orient=tk.VERTICAL, command=self.text_area.yview)
        text_xscroll = ttk.Scrollbar(self.text_frame, orient=tk.HORIZONTAL, command=self.text_area.xview)
        self.text_area.configure(yscrollcommand=text_yscroll.set, xscrollcommand=text_xscroll.set)
        self.text_area.tag_configure("prompt", foreground="#c0392b")
        self.text_area.grid(row=0, column=0, sticky="nsew")
        text_yscroll.grid(row=0, column=1, sticky="ns")
        text_xscroll.grid(row=1, column=0, sticky="ew")
        self.text_frame.rowconfigure(0, weight=1)
        self.text_frame.columnconfigure(0, weight=1)
        self.paned_window.add(self.text_frame, weight=1)

        self.root.after(200, self._activate_window)

    def _activate_window(self):
        self.root.attributes("-topmost", True)
        self.root.after(50, lambda: self.root.attributes("-topmost", False))
        try:
            subprocess.run(
                ["osascript", "-e",
                 f'tell application "System Events" to set frontmost of '
                 f'(first process whose unix id is {os.getpid()}) to true'],
                check=False, capture_output=True, timeout=2,
            )
        except (subprocess.SubprocessError, OSError):
            pass

    # ------------------------------------------------------------------
    # Console texte (meme motif que Starmac_Py/AMS_Py)
    # ------------------------------------------------------------------

    def _afficher(self, text):
        if self.text_area.get("1.0", "end-1c").strip():
            self.text_area.insert(tk.END, "\n" + "-" * 60 + "\n")
        self.text_area.insert(tk.END, text)
        self.text_area.see(tk.END)

    def _console_input(self, prompt, default=""):
        self.text_area.insert(tk.END, prompt, "prompt")
        start_index = self.text_area.index("end-1c")
        self.text_area.mark_set(tk.INSERT, tk.END)
        self.text_area.see(tk.END)
        self.text_area.focus_set()

        outcome = {"value": None}
        done = tk.BooleanVar(value=False)

        def on_return(event):
            typed = self.text_area.get(start_index, "end-1c")
            outcome["value"] = typed if typed.strip() else default
            self.text_area.insert(tk.END, "\n")
            done.set(True)
            return "break"

        def on_escape(event):
            self.text_area.delete(start_index, "end-1c")
            outcome["value"] = None
            self.text_area.insert(tk.END, "\n")
            done.set(True)
            return "break"

        ret_id = self.text_area.bind("<Return>", on_return)
        esc_id = self.text_area.bind("<Escape>", on_escape)
        self.text_area.wait_variable(done)
        self.text_area.unbind("<Return>", ret_id)
        self.text_area.unbind("<Escape>", esc_id)
        self.text_area.see(tk.END)
        return outcome["value"]

    def _split_pasted_rows(self, text):
        """Coupe le texte capture par `_console_input` en lignes de
        donnees individuelles - demande explicite utilisateur : coller
        plusieurs lignes/colonnes de donnees d'un coup (ex. depuis BBEdit)
        dans une boucle de saisie manuelle doit traiter CHAQUE ligne
        collee comme une entree separee, pas seulement la premiere.
        `_console_input` capture tout le texte tape/colle avant l'appui
        final sur Retour, y compris les retours a la ligne internes d'un
        collage multi-lignes - sans ce decoupage, `line.split()` fusionne
        silencieusement toutes les lignes collees en une seule rangee de
        tokens et n'en garde que les premiers, perdant le reste sans
        avertissement. Retourne la liste des lignes non vides (chacune
        encore a decouper par `.split()` par l'appelant)."""
        return [ln.strip() for ln in text.split("\n") if ln.strip()]

    def _save_manual_entry_backup(self, kind, header, rows_text):
        """Sauvegarde une session de saisie manuelle multiple dans un
        fichier texte temporaire, avec un en-tete auto-descriptif
        (rechargeable directement via le menu "Load from file"
        correspondant - detection d'en-tete automatique, voir
        stereo_selection.py) - demande explicite utilisateur : garder une
        trace facilement rechargeable de chaque saisie manuelle, plutot
        que de perdre les valeurs si la session Python se termine."""
        fd, path = tempfile.mkstemp(prefix=f"stereoutils_{kind}_", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write(rows_text)
        self._afficher(f"(also saved to {path} - reload it anytime via the matching \"Load from file\" menu item)\n")
        return path

    def _not_implemented(self, name):
        messagebox.showinfo(
            "Not yet ported",
            f"'{name}' is not yet ported (only the Pmag_Python menu is, for now).",
        )

    def _redraw_canvas(self):
        """Force un vrai cycle Tk <Configure> puis aligne la Figure sur la
        taille REELLE du widget canvas - meme correctif que Starmac_Py
        (`app.py:_redraw_canvas`) : sans ce "secouement" de la fenetre,
        matplotlib/FigureCanvasTkAgg n'affichait parfois le contenu
        correctement qu'apres un redimensionnement MANUEL de la fenetre
        par l'utilisateur (le widget Tk est la source de verite pour la
        taille de la Figure, et ne se resynchronise que sur un vrai
        evenement <Configure>)."""
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        self.root.geometry(f"{w + 1}x{h + 1}")
        self.root.update_idletasks()
        self.root.geometry(f"{w}x{h}")
        self.root.update_idletasks()

        dpi = self.fig.dpi
        canvas_w = self.canvas_fig.get_tk_widget().winfo_width()
        canvas_h = self.canvas_fig.get_tk_widget().winfo_height()
        if canvas_w > 1 and canvas_h > 1:
            self.fig.set_size_inches(canvas_w / dpi, canvas_h / dpi, forward=False)

        self.canvas_fig.draw()

    def _show_images(self, files):
        """Charge et affiche le(s) fichier(s) image produits par une
        fonction PmagPy (qui trace elle-meme via pyplot, sans parametre
        `fig=` compatible avec ce canvas) - empiles verticalement si
        plusieurs, meme motif que xygraph.py cote Starmac."""
        if not files:
            self._afficher("(no plot produced)\n")
            return
        self._last_image_files = files
        self.fig.clear()
        n = len(files)
        self.fig.set_size_inches(6.0, 5.0 * n, forward=True)
        for i, path in enumerate(files, start=1):
            ax = self.fig.add_subplot(n, 1, i)
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.axis("off")
        self.fig.tight_layout()
        self._redraw_canvas()

    # ------------------------------------------------------------------
    # Graphics / Projection (reseau stereo natif - stereo_net.py)
    # ------------------------------------------------------------------

    def plot_screen(self):
        """Equivalent de `plotdata` (menu Graphics > Plot stereo, ex-menu
        "Command", fusionne dans "Graphics" - demande explicite
        utilisateur "change the menu projection by Graphics, add plot
        stereo in this menu") : trace le cadre du reseau + les directions
        chargees, dans le canvas integre (pas de fichier image
        intermediaire, contrairement au Pmag_Python)."""
        self._last_image_files = []
        build_stereo_figure(
            self.directions, la=self.la, phi=self.phi, iproj=self.iproj,
            dim=self.dim, path_between_points=self.path_between_points,
            point_size=self.point_size, means=self.means,
            great_circles=self.great_circles, fig=self.fig,
        )
        if self.graph_title:
            self.fig.axes[0].set_title(self.graph_title)
        self._redraw_canvas()

    def clear_screen(self):
        """Equivalent de `clrtty` (menu Graphics > Clear Screen, ex-menu
        "Command")."""
        self.fig.clear()
        self.fig.set_size_inches(5.5, 5.5, forward=True)
        self._redraw_canvas()

    def set_graph_title(self):
        """Equivalent de `title` (menu Graphics > Graph title, ex-menu
        "Command")."""
        title = self._console_input("graph title : ", self.graph_title)
        if title is None:
            return
        self.graph_title = title
        if self.fig.axes:
            self.plot_screen()

    def _set_iproj(self, iproj):
        self.iproj = iproj
        self._afficher(f"projection : {'equal area (Schmidt)' if iproj == 1 else 'stereographic (Wulff)'}\n")

    def set_pole_projection(self):
        """Equivalent de `projpole` (menu Projection > Pole of projection) :
        latitude/longitude du pole de projection (la=+-90 : projection
        polaire standard, sinon oblique via `superc`)."""
        la_s = self._console_input("latitude of the pole of projection (-90 to 90) : ", str(self.la))
        if la_s is None:
            return
        phi_s = self._console_input("longitude of the pole of projection : ", str(self.phi))
        if phi_s is None:
            return
        try:
            la = float(la_s)
        except ValueError:
            la = self.la
        try:
            phi = float(phi_s)
        except ValueError:
            phi = self.phi
        if la < -90.0 or la > 90.0:
            la = -90.0
        self.la, self.phi = la, phi
        self._afficher(f"pole of projection : la={self.la:.1f}  phi={self.phi:.1f}\n")

    def set_diameter(self):
        """Equivalent de `diam` (menu Projection > Diameter)."""
        dim_s = self._console_input("diameter of the net (cm) : ", str(self.dim))
        if dim_s is None:
            return
        try:
            dim = float(dim_s)
        except ValueError:
            dim = self.dim
        if dim <= 0.0:
            dim = 21.0
        self.dim = dim
        self._afficher(f"net diameter : {self.dim:.1f} cm\n")

    def set_symbol_size_color(self):
        """Equivalent de `sizepoint` (menu Graphics > Symbol size &
        color) : taille (cm) - la couleur des symboles individuels suit le
        caractere saisi au chargement (`c`,`t`,`e`,`l`,`s`), pas un
        parametre global cote source."""
        size_s = self._console_input("symbol size (cm) : ", str(self.point_size))
        if size_s is None:
            return
        try:
            size = float(size_s)
        except ValueError:
            size = self.point_size
        if size <= 0.0:
            size = 0.3
        self.point_size = size
        self._afficher(f"symbol size : {self.point_size:.2f} cm\n")

    def export_svg(self):
        """Equivalent fonctionnel de STARpaleomag_Py/app.exporter_svg (menu
        Graphics > Export SVG...) - demande explicite utilisateur ("export
        svg") : pas besoin d'un code d'export separe, matplotlib ecrit un
        SVG directement depuis la Figure actuellement affichee dans le
        canvas integre (`self.fig`, quel que soit le trace en cours -
        reseau stereo, projet, carte VGP...)."""
        if not self.fig.axes:
            messagebox.showwarning("No graphic", "Plot something first (e.g. Plot stereo).")
            return
        path = filedialog.asksaveasfilename(
            title="Export as SVG", defaultextension=".svg",
            initialfile="stereo_plot.svg",
            filetypes=[("SVG", "*.svg"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.fig.savefig(path, format="svg")
        except Exception as e:
            messagebox.showerror("Error", f"SVG export failed:\n{e}")
            return
        self._afficher(f"Graphic exported: {path}\n")

    def ouvrir_user_guide(self):
        """Ouvre le guide utilisateur (StereoUtils_Py Guide) dans le
        navigateur systeme - equivalent de STARpaleomag_Py/AMS_Py
        app.ouvrir_user_guide (meme principe : un fichier HTML STATIQUE
        livre EN LOCAL avec l'appli, voir _resource_path,
        help/StereoUtils_Py_Guide.html) - demande explicite utilisateur
        ("is it possible to write a guide for stereo")."""
        guide_path = _resource_path("help", "StereoUtils_Py_Guide.html")
        if not os.path.exists(guide_path):
            messagebox.showerror("Error", f"User guide not found:\n{guide_path}")
            return
        webbrowser.open(f"file://{guide_path}")

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    @staticmethod
    def _labeled(text, shortcut_name):
        """Libelle avec le raccourci entre parentheses, SANS `accelerator=`
        (voir AMS_Py/STARpaleomag_Py app._labeled - meme raison, voir
        commentaire pres de SHORTCUTS)."""
        return f"{text}    ({SHORTCUTS[shortcut_name][0]})"

    def _setup_shortcuts(self):
        """Lie reellement les raccourcis affiches dans les libelles de
        menu (voir SHORTCUTS/_labeled) - jusque-la purement decoratifs."""
        bindings = {
            "hystnorm": self.hyst_toggle_normalize,
        }
        for name, callback in bindings.items():
            self.root.bind_all(SHORTCUTS[name][1], lambda event, cb=callback: cb())

    def _setup_menu(self):
        menubar = tk.Menu(self.root)

        data_menu = tk.Menu(menubar, tearoff=0)

        load_menu = tk.Menu(data_menu, tearoff=0)
        load_menu.add_command(label="File [D-I]...", command=self.load_di_file)
        load_menu.add_command(label="File [D-I_TC]...", command=self.load_di_tc_file)
        load_menu.add_command(label="File [D-I-a95]...", command=self.load_di_a95_file)
        load_menu.add_command(label="File [D-I-a95_TC]...", command=self.load_di_a95_tc_file)
        load_menu.add_separator()
        load_menu.add_command(label="File [great circle]...", command=self.load_great_circle_file)
        data_menu.add_cascade(label="Load from file", menu=load_menu)

        manual_menu = tk.Menu(data_menu, tearoff=0)
        manual_menu.add_command(label="Manual [D-I]...", command=self.enter_di_manual)
        manual_menu.add_command(label="Manual [D-I-a95]...", command=self.enter_di_a95_manual)
        manual_menu.add_command(label="Manual [great circle]...", command=self.enter_great_circle_manual)
        data_menu.add_cascade(label="Manual entry", menu=manual_menu)

        data_menu.add_command(label="List-data", command=self.list_data)

        delete_menu = tk.Menu(data_menu, tearoff=0)
        delete_menu.add_command(label="Delete D-I line...", command=self.delete_di_line)
        delete_menu.add_command(label="Delete D-I-a95 line...", command=self.delete_mean_line)
        delete_menu.add_command(label="Delete GC line...", command=self.delete_gc_line)
        data_menu.add_cascade(label="Delete", menu=delete_menu)

        data_menu.add_command(label="Initialize...", command=self.initialize_data)
        data_menu.add_separator()
        data_menu.add_command(label="Help Data (file headers)", command=self.help_data)
        menubar.add_cascade(label="Data", menu=data_menu)

        self._iproj_var = tk.IntVar(value=self.iproj)
        proj_menu = tk.Menu(menubar, tearoff=0)
        proj_menu.add_command(label="Plot stereo", command=self.plot_screen)
        proj_menu.add_command(label="Plot stereo project", command=self.plot_project)
        proj_menu.add_command(label="Plot VGPs on Map", command=self.plot_vgps_on_map)
        proj_menu.add_command(label="Plot VGP Project...", command=self.plot_vgp_project)
        proj_menu.add_command(label="Clear Screen", command=self.clear_screen)
        proj_menu.add_command(label="Graph title...", command=self.set_graph_title)
        proj_menu.add_separator()
        proj_menu.add_radiobutton(
            label="Stereographic (Wulff)", variable=self._iproj_var, value=0,
            command=lambda: self._set_iproj(0))
        proj_menu.add_radiobutton(
            label="Equal area (Schmidt)", variable=self._iproj_var, value=1,
            command=lambda: self._set_iproj(1))
        proj_menu.add_separator()
        proj_menu.add_command(label="Pole of projection...", command=self.set_pole_projection)
        proj_menu.add_command(label="Diameter...", command=self.set_diameter)
        proj_menu.add_command(label="Symbol size & color...", command=self.set_symbol_size_color)
        proj_menu.add_separator()
        proj_menu.add_command(label="Export SVG...", command=self.export_svg)
        menubar.add_cascade(label="Graphics", menu=proj_menu)

        stats_menu = tk.Menu(menubar, tearoff=0)
        stats_menu.add_command(label="Fisher Statistics...", command=self.stat_fisher_statistics)
        stats_menu.add_command(label="Fisher Dir+GC...", command=self.stat_fisher_dir_gc)
        stats_menu.add_command(label="Fisher recursive...", command=self.stat_fisher_recursive)
        stats_menu.add_command(label="St Dev dipole...", command=self.stat_stddip)
        stats_menu.add_command(label="Fisher Strati", command=self.stat_strati)
        stats_menu.add_command(label="progressive Fold test...", command=self.stat_foldtest)
        stats_menu.add_command(label="Mean Inclination only", command=self.stat_meaninc)
        stats_menu.add_command(label="smallcircles Intersect", command=self.stat_intersect)
        stats_menu.add_command(label="Reversal angle...", command=self.stat_reversal_angle)
        stats_menu.add_command(label="difference angle...", command=self.stat_diff_angle)
        menubar.add_cascade(label="Statistics", menu=stats_menu)

        project_menu = tk.Menu(menubar, tearoff=0)
        project_menu.add_command(label="Load Project...", command=self.load_project)
        project_menu.add_command(label="Export to Project", command=self.export_to_project)
        project_menu.add_command(label="List Project", command=self.list_project)
        project_menu.add_command(label="Init Project", command=self.init_project)
        project_menu.add_separator()
        project_menu.add_command(label="Fisher Project...", command=self.fisher_project)
        project_menu.add_separator()
        project_menu.add_command(label="Help Project (colors & symbols)", command=self.help_project)
        menubar.add_cascade(label="Project", menu=project_menu)

        pu_menu = tk.Menu(menubar, tearoff=0)
        vgp_menu = tk.Menu(pu_menu, tearoff=0)
        vgp_menu.add_command(label="Direction to VGP...", command=self.pu_vgpc1)
        vgp_menu.add_command(label="File: Direction to VGP...", command=self.pu_vgpc3)
        vgp_menu.add_command(label="VGP to direction...", command=self.pu_vgpc2)
        vgp_menu.add_command(label="File: VGP to direction...", command=self.pu_vgpc4)
        pu_menu.add_cascade(label="VGP conversions", menu=vgp_menu)

        rot_menu = tk.Menu(pu_menu, tearoff=0)
        rot_menu.add_command(label="Rotation obs=DI  Pole Ref...", command=self.pu_paleotec)
        rot_menu.add_command(label="Rotation obs=DI  DI Ref...", command=self.pu_paleodec)
        rot_menu.add_command(label="Rotation obs=Pole Pole Ref...", command=self.pu_paleotec1)
        pu_menu.add_cascade(label="Rotation/flattening tests", menu=rot_menu)

        plat_menu = tk.Menu(pu_menu, tearoff=0)
        plat_menu.add_command(label="Paleolatitude at one site (file)...", command=self.pu_paleolati)
        plat_menu.add_command(label="Inclination a95 to Plat and err...", command=self.pu_paleolati2)
        pu_menu.add_cascade(label="Paleolatitude", menu=plat_menu)

        gmt_menu = tk.Menu(pu_menu, tearoff=0)
        gmt_menu.add_command(label="Rotation vers GMT plot...", command=self.pu_rota2gmt)
        gmt_menu.add_command(label="Aide Rotation", command=self.pu_helprota)
        pu_menu.add_cascade(label="GMT rotation export", menu=gmt_menu)

        sitemap_menu = tk.Menu(pu_menu, tearoff=0)
        sitemap_menu.add_command(label="prmag to KML (Google Earth)...", command=self.pu_sitemap_kml)
        sitemap_menu.add_command(label="prmag to GMT map script...", command=self.pu_sitemap_gmt)
        sitemap_menu.add_command(label="Plot sites (matplotlib)...", command=self.pu_sitemap_plot)
        sitemap_menu.add_checkbutton(
            label="Basic basemap (coastlines, roads, cities)", variable=self.sitemap_basemap)
        pu_menu.add_cascade(label="Site Map", menu=sitemap_menu)

        core_menu = tk.Menu(pu_menu, tearoff=0)
        core_menu.add_command(label="Core correction...", command=self.pu_core1)
        core_menu.add_command(label="Bedding correction...", command=self.pu_core2)
        core_menu.add_command(label="inverseBedding cor...", command=self.pu_invbedding)
        pu_menu.add_cascade(label="Core/bedding corrections", menu=core_menu)

        fold_menu = tk.Menu(pu_menu, tearoff=0)
        fold_menu.add_command(label="Fold plunge...", command=self.pu_foldplunge)
        fold_menu.add_command(label="File Fold plunge...", command=self.pu_foldfile)
        pu_menu.add_cascade(label="Fold plunge", menu=fold_menu)

        euler_menu = tk.Menu(pu_menu, tearoff=0)
        euler_menu.add_command(label="DI Polar rotation...", command=self.pu_rotmag)
        euler_menu.add_command(label="VGP Plate rotation...", command=self.pu_vgp_plate_rotation)
        pu_menu.add_cascade(label="Euler rotation", menu=euler_menu)

        pal_menu = tk.Menu(pu_menu, tearoff=0)
        pal_menu.add_command(label="mean paleointensity...", command=self.pu_meanpal)
        pal_menu.add_command(label="VDM and VADM...", command=self.pu_vidimo)
        pal_menu.add_command(label="VDM to Intensity...", command=self.pu_relocate)
        pal_menu.add_command(label="Relocate D I F...", command=self.pu_relocatevar)
        pu_menu.add_cascade(label="Paleointensity", menu=pal_menu)

        igrf_menu = tk.Menu(pu_menu, tearoff=0)
        igrf_menu.add_command(label="valeur CMT IGRF...", command=self.pu_igrf)
        igrf_menu.add_command(label="Help calcul IGRF", command=self.pu_helpigrf)
        pu_menu.add_cascade(label="IGRF", menu=igrf_menu)

        pu_menu.add_command(label="conversion units...", command=self.pu_convunit)
        pu_menu.add_separator()
        pu_menu.add_command(label="test overprint...", command=self.pu_overprint)
        pu_menu.add_command(label="test flattening...", command=self.pu_flatten)
        menubar.add_cascade(label="Pmag Utilities", menu=pu_menu)

        pmag_menu = tk.Menu(menubar, tearoff=0)
        pmag_menu.add_command(label="Fisher", command=self.pmagpy_fisher)
        pmag_menu.add_separator()
        pmag_menu.add_command(label="Bootstrap ellipse", command=self.pmagpy_bootstrap_ellipse)
        pmag_menu.add_separator()
        pmag_menu.add_command(label="Find Elongation", command=self.find_elongation)
        pmag_menu.add_separator()
        pmag_menu.add_command(label="Reversal antipodality", command=self.test_reversal_antipodal)
        pmag_menu.add_separator()
        pmag_menu.add_command(label="Test common mean", command=self.test_common_mean)
        pmag_menu.add_separator()
        pmag_menu.add_command(label="Fold Test", command=self.fold_test)
        pmag_menu.add_separator()
        pmag_menu.add_command(label="Mean Inclination", command=self.mean_inclination)
        menubar.add_cascade(label="PmagPy-tools", menu=pmag_menu)

        kt_menu = tk.Menu(menubar, tearoff=0)
        kt_menu.add_command(label="Open .CUR/.CLW File...", command=self.kt_open_file)
        kt_menu.add_command(label="Open Sample List...", command=self.kt_open_sample_list)
        kt_menu.add_command(label="Select File from List...", command=self.kt_select_from_list)
        kt_menu.add_separator()
        kt_menu.add_command(label="Furnace Correction", command=self.kt_apply_furnace_correction)
        kt_menu.add_command(label="Normalize by Mass", command=self.kt_normalize_by_mass)
        kt_menu.add_command(label="Normalize by Volume", command=self.kt_normalize_by_volume)
        kt_menu.add_separator()
        kt_menu.add_command(label="Curie Point (2nd derivative)", command=self.kt_curie_point)
        kt_menu.add_separator()
        kt_menu.add_command(label="Plot K-T", command=self.kt_plot)
        menubar.add_cascade(label="K-T", menu=kt_menu)

        hyst_menu = tk.Menu(menubar, tearoff=0)
        hyst_menu.add_command(label="Open AGM Sample List...", command=self.hyst_open_agm_list)
        hyst_menu.add_command(label="Open VSM Sample List...", command=self.hyst_open_vsm_list)
        hyst_menu.add_command(label="Select Sample to Plot...", command=self.hyst_select_from_list)
        hyst_menu.add_separator()
        hyst_menu.add_command(label="Paramagnetic Fit Threshold...", command=self.hyst_set_valsat)
        hyst_menu.add_checkbutton(
            label=self._labeled("Normalize Plot", "hystnorm"),
            variable=self.hyst_normalize, command=self.hyst_refresh_plot)
        menubar.add_cascade(label="Hysteresis", menu=hyst_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="User Guide", command=self.ouvrir_user_guide)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    # ------------------------------------------------------------------
    # Data input (stereograph.f) - chaque chargement/saisie AJOUTE aux
    # listes existantes (`il`/`im`/`imm` s'incrementent depuis leur valeur
    # courante dans le source, ne les remplacent pas).
    # ------------------------------------------------------------------

    def _prompt_path_and_symbol(self):
        """Les deux prompts partages par opfil1/opfil4/dataman ("do you
        want a path between points" + "symbole c,t,l,e,s") - retourne
        (path_between_points, symbol) ou None si annule."""
        path_s = self._console_input("do you want a path between points (y/n) : ", "n")
        if path_s is None:
            return None
        self.path_between_points = path_s.strip().lower().startswith("y")
        sym_s = self._console_input("symbol c, t, l, e, s for square : ", "c")
        if sym_s is None:
            return None
        sym = sym_s.strip().lower()
        if sym not in _SYMBOL_CHARS:
            sym = "c"
        return self.path_between_points, sym

    def _prompt_symbol_only(self):
        """Le prompt symbole seul (mean/opfil2 - pas de prompt "path")."""
        sym_s = self._console_input("symbol c, t, l, e, s for square : ", "c")
        if sym_s is None:
            return None
        sym = sym_s.strip().lower()
        if sym not in _SYMBOL_CHARS:
            sym = "c"
        return sym

    # -- File [D-I] / [D-I_TC] / [D-I-a95] / [D-I-a95_TC] -----------------

    def load_di_file(self):
        """Equivalent de `opfil1` (File [D-I])."""
        path = filedialog.askopenfilename(
            title="File [D-I]", filetypes=[("Text", "*.txt *.dat"), ("All files", "*.*")])
        if not path:
            return
        pairs = read_di_file(path)
        picked = self._prompt_path_and_symbol()
        if picked is None:
            return
        _path_between, sym = picked
        self.directions.extend((dec, inc, sym) for dec, inc in pairs)
        self._afficher(f"{len(pairs)} direction(s) loaded from {os.path.basename(path)} (D-I total: {len(self.directions)})\n")

    def load_di_tc_file(self):
        """Equivalent de `opfil4` (File [D-I_TC]) : (dec,inc,strike,dip),
        correction de pendage appliquee avant stockage."""
        path = filedialog.askopenfilename(
            title="File [D-I_TC]", filetypes=[("Text", "*.txt *.dat"), ("All files", "*.*")])
        if not path:
            return
        pairs = read_di_tc_file(path)
        picked = self._prompt_path_and_symbol()
        if picked is None:
            return
        _path_between, sym = picked
        self.directions.extend((dec, inc, sym) for dec, inc in pairs)
        self._afficher(f"{len(pairs)} tilt-corrected direction(s) loaded from {os.path.basename(path)} (D-I total: {len(self.directions)})\n")

    def load_di_a95_file(self):
        """Equivalent de `opfil2`/`mean` (File [D-I-a95])."""
        path = filedialog.askopenfilename(
            title="File [D-I-a95]", filetypes=[("Text", "*.txt *.dat"), ("All files", "*.*")])
        if not path:
            return
        triples = read_di_a95_file(path)
        sym = self._prompt_symbol_only()
        if sym is None:
            return
        self.means.extend((dec, inc, alph, sym) for dec, inc, alph in triples)
        self._afficher(f"{len(triples)} mean(s) loaded from {os.path.basename(path)} (D-I-a95 total: {len(self.means)})\n")

    def load_di_a95_tc_file(self):
        """Equivalent de `opfil5` (File [D-I-a95_TC]) : (dec,inc,alpha95,
        strike,dip), correction de pendage appliquee - PAS de prompt
        symbole cote source (omission reelle : les entrees opfil5
        n'obtiennent jamais leur propre symbole en Fortran), symbole 'c'
        par defaut ici."""
        path = filedialog.askopenfilename(
            title="File [D-I-a95_TC]", filetypes=[("Text", "*.txt *.dat"), ("All files", "*.*")])
        if not path:
            return
        triples = read_di_a95_tc_file(path)
        self.means.extend((dec, inc, alph, "c") for dec, inc, alph in triples)
        self._afficher(f"{len(triples)} tilt-corrected mean(s) loaded from {os.path.basename(path)} (D-I-a95 total: {len(self.means)})\n")

    # -- File [great circle] (opfil3) --------------------------------------

    def load_great_circle_file(self):
        path = filedialog.askopenfilename(title="File [great circle]")
        if not path:
            return
        gcs = read_great_circle_file(path)
        self.great_circles.extend(gcs)
        self._afficher(f"{len(gcs)} great circle(s) loaded from {os.path.basename(path)} (GC total: {len(self.great_circles)})\n")

    # -- Manual entry (dataman/mean/gdci) ----------------------------------

    def enter_di_manual(self):
        """Equivalent de `dataman` (manual [D-I]) : saisie ligne a ligne
        "dec inc" au clavier, ligne vide pour arreter - accepte aussi un
        collage multi-lignes (ex. deux colonnes copiees depuis BBEdit) :
        chaque ligne collee est traitee comme une entree separee."""
        self._afficher("declination inclination\n(blank line to stop, paste multiple lines OK)\n")
        n0 = len(self.directions)
        i = n0 + 1
        while True:
            line = self._console_input(f"{i:3d}: ")
            if line is None or not line.strip():
                break
            for subline in self._split_pasted_rows(line):
                parts = subline.split()
                if len(parts) < 2:
                    self._afficher(f"  (invalid, skipped: \"{subline}\")\n")
                    continue
                try:
                    dec, inc = float(parts[0]), float(parts[1])
                except ValueError:
                    self._afficher(f"  (invalid, skipped: \"{subline}\")\n")
                    continue
                self.directions.append((dec, inc, "c"))
                i += 1
        added = len(self.directions) - n0
        if added == 0:
            return
        picked = self._prompt_path_and_symbol()
        if picked is not None:
            _path_between, sym = picked
            for j in range(n0, len(self.directions)):
                dec, inc, _sym = self.directions[j]
                self.directions[j] = (dec, inc, sym)
        self._afficher(f"{added} direction(s) entered (D-I total: {len(self.directions)})\n")
        rows_text = "".join(f"{dec:.2f} {inc:.2f}\n" for dec, inc, _sym in self.directions[n0:])
        self._save_manual_entry_backup("DI", "#dec inc", rows_text)

    def enter_di_a95_manual(self):
        """Equivalent de `mean` (manual [D-I-a95]) : saisie "dec inc
        alpha95", ligne vide pour arreter - accepte aussi un collage
        multi-lignes (colonnes copiees depuis BBEdit par exemple)."""
        self._afficher("declination inclination and alpha 95\n(blank line to stop, paste multiple lines OK)\n")
        n0 = len(self.means)
        i = n0 + 1
        while True:
            line = self._console_input(f"{i:3d}: ")
            if line is None or not line.strip():
                break
            for subline in self._split_pasted_rows(line):
                parts = subline.split()
                if len(parts) < 3:
                    self._afficher(f"  (invalid, skipped: \"{subline}\")\n")
                    continue
                try:
                    dec, inc, alph = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    self._afficher(f"  (invalid, skipped: \"{subline}\")\n")
                    continue
                self.means.append((dec, inc, alph, "c"))
                i += 1
        added = len(self.means) - n0
        if added == 0:
            return
        sym = self._prompt_symbol_only()
        if sym is not None:
            for j in range(n0, len(self.means)):
                dec, inc, alph, _sym = self.means[j]
                self.means[j] = (dec, inc, alph, sym)
        self._afficher(f"{added} mean(s) entered (D-I-a95 total: {len(self.means)})\n")
        rows_text = "".join(f"{dec:.2f} {inc:.2f} {alph:.2f}\n" for dec, inc, alph, _sym in self.means[n0:])
        self._save_manual_entry_backup("DIa95", "#dec inc a95", rows_text)

    def enter_great_circle_manual(self):
        """Equivalent de `gdci` (manual [great circle]) : saisie "glon glat
        [ad1 ai1 ad2 ai2]", ligne vide pour arreter - accepte aussi un
        collage multi-lignes."""
        self._afficher("great circle (long,lat) [+ optional sector ad1 ai1 ad2 ai2]\n(blank line to stop, paste multiple lines OK)\n")
        n0 = len(self.great_circles)
        i = n0 + 1
        while True:
            line = self._console_input(f"{i:3d}: ")
            if line is None or not line.strip():
                break
            for subline in self._split_pasted_rows(line):
                parts = subline.split()
                if len(parts) < 2:
                    self._afficher(f"  (invalid, skipped: \"{subline}\")\n")
                    continue
                try:
                    glon, glat = float(parts[0]), float(parts[1])
                    if len(parts) >= 6:
                        ad1, ai1, ad2, ai2 = (float(p) for p in parts[2:6])
                    else:
                        ad1 = ai1 = ad2 = ai2 = 0.0
                except ValueError:
                    self._afficher(f"  (invalid, skipped: \"{subline}\")\n")
                    continue
                self.great_circles.append((glon, glat, ad1, ai1, ad2, ai2))
                i += 1
        added = len(self.great_circles) - n0
        self._afficher(f"{added} great circle(s) entered (GC total: {len(self.great_circles)})\n")
        if added > 0:
            rows_text = "".join(
                f"{glon:.2f} {glat:.2f} {ad1:.2f} {ai1:.2f} {ad2:.2f} {ai2:.2f}\n"
                for glon, glat, ad1, ai1, ad2, ai2 in self.great_circles[n0:]
            )
            self._save_manual_entry_backup("GC", "#glon glat ad1 ai1 ad2 ai2", rows_text)

    def help_data(self):
        """Aide "Load from file" (ajout hors source Fortran, demande
        explicite) : rappelle la ligne d'en-tete auto-descriptive
        reconnue pour chaque type de fichier du menu Data, et les noms de
        colonnes acceptes (voir `stereo_selection.FIELD_ALIASES`)."""
        text = (
            "Data > Load from file : optional header line\n"
            "\n"
            "Each file can start with a header line (leading \"#\", or simply\n"
            "the column names) giving the column order - otherwise it falls\n"
            "back to the classic positional order (dec, inc, ... starting\n"
            "from the 1st column, with no site/id column up front).\n"
            "\n"
            "  File [D-I]            : #dec inc\n"
            "  File [D-I_TC]         : #dec inc strike dip\n"
            "  File [D-I-a95]        : #dec inc a95\n"
            "  File [D-I-a95_TC]     : #dec inc a95 strike dip\n"
            "  File [great circle]   : #glon glat [ad1 ai1 ad2 ai2]\n"
            "\n"
            "A \"site\"/\"id\"/\"name\" column at the start of the line is detected\n"
            "and skipped automatically IF a header is present (otherwise, add\n"
            "a header to your file rather than relying on a fixed column\n"
            "offset).\n"
            "\n"
            "Recognized column names (case-insensitive):\n"
            "  dec      : dec, declination, d\n"
            "  inc      : inc, inclination, i\n"
            "  a95      : a95, alpha95, alph\n"
            "  strike   : strike, str\n"
            "  dipdir   : dipdir, dip_direction, dd\n"
            "  dip      : dip\n"
            "  glon     : glon, lon, long, longitude\n"
            "  glat     : glat, lat, latitude\n"
            "  ad1/ai1/ad2/ai2 : ad1, ai1, ad2, ai2 (great circle sector)\n"
            "  site     : site, id, name, sample, specimen\n"
        )
        self._afficher(text)

    # -- List / Delete / Initialize -----------------------------------------

    def list_data(self):
        """Equivalent de `listdata` (List-data)."""
        if not self.directions and not self.means and not self.great_circles:
            self._afficher("no data in memory ??\n")
            return
        lines = ["data list\n"]
        if self.means:
            lines.append("\n  mean data")
            for i, (dec, inc, alph, sym) in enumerate(self.means, start=1):
                lines.append(f"{i:4d}:{dec:7.1f}{inc:7.1f}{alph:7.1f}  sym={sym}")
        if self.great_circles:
            lines.append("\n  great circles (long,lat)")
            for i, (glon, glat, ad1, ai1, ad2, ai2) in enumerate(self.great_circles, start=1):
                if (ad1, ai1, ad2, ai2) == (0.0, 0.0, 0.0, 0.0):
                    lines.append(f"{i:4d}:{glon:7.1f}{glat:7.1f}")
                else:
                    lines.append(f"{i:4d}:{glon:7.1f}{glat:7.1f}    {ad1:7.1f}{ai1:7.1f}{ad2:7.1f}{ai2:7.1f}")
        if self.directions:
            lines.append("\n  declination-inclination")
            for i, (dec, inc, sym) in enumerate(self.directions, start=1):
                lines.append(f"{i:4d}:{dec:7.1f}{inc:7.1f}  sym={sym}")
        self._afficher("\n".join(lines) + "\n")

    def delete_di_line(self):
        """Equivalent de `delet` (Delete D-I line)."""
        if not self.directions:
            messagebox.showwarning("No data", "No D-I line to delete.")
            return
        lines = [f"{i:4d}:{d:7.1f}{inc:7.1f}" for i, (d, inc, _s) in enumerate(self.directions, start=1)]
        self._afficher("declination-inclination\n" + "\n".join(lines) + "\n")
        n_s = self._console_input("what number : ", "0")
        if n_s is None:
            return
        try:
            n = int(n_s)
        except ValueError:
            return
        if n <= 0 or n > len(self.directions):
            return
        del self.directions[n - 1]
        self._afficher(f"line {n} deleted (D-I total: {len(self.directions)})\n")

    def delete_mean_line(self):
        """Equivalent de `delmean` (Delete D-I-a95 line)."""
        if not self.means:
            messagebox.showwarning("No data", "No D-I-a95 line to delete.")
            return
        lines = [f"{i:4d}:{d:7.1f}{inc:7.1f}{a:7.1f}" for i, (d, inc, a, _s) in enumerate(self.means, start=1)]
        self._afficher("mean data\n" + "\n".join(lines) + "\n")
        n_s = self._console_input("what number : ", "0")
        if n_s is None:
            return
        try:
            n = int(n_s)
        except ValueError:
            return
        if n <= 0 or n > len(self.means):
            return
        del self.means[n - 1]
        self._afficher(f"line {n} deleted (D-I-a95 total: {len(self.means)})\n")

    def delete_gc_line(self):
        """Equivalent de `delgc` (Delete GC line)."""
        if not self.great_circles:
            messagebox.showwarning("No data", "No great circle line to delete.")
            return
        lines = [f"{i:4d}:{g:7.1f}{lat:7.1f}" for i, (g, lat, *_r) in enumerate(self.great_circles, start=1)]
        self._afficher("great circles (long,lat)\n" + "\n".join(lines) + "\n")
        n_s = self._console_input("what number : ", "0")
        if n_s is None:
            return
        try:
            n = int(n_s)
        except ValueError:
            return
        if n <= 0 or n > len(self.great_circles):
            return
        del self.great_circles[n - 1]
        self._afficher(f"line {n} deleted (GC total: {len(self.great_circles)})\n")

    def initialize_data(self):
        """Equivalent de `initia` (Initialize) : demande y/n pour chaque
        liste non vide, dans l'ordre means -> great circles -> D-I (meme
        ordre que le source ; le "clear ellipses" du Fortran n'a pas
        d'equivalent ici, aucune donnee Statistics/Bootstrap n'existe
        encore dans ce port)."""
        if self.means:
            ans = self._console_input("initialize the mean directions list (y/n) : ", "n")
            if ans is not None and ans.strip().lower().startswith("y"):
                self.means = []
                self._afficher("mean directions list cleared\n")
        if self.great_circles:
            ans = self._console_input("initialize the great circles list (y/n) : ", "n")
            if ans is not None and ans.strip().lower().startswith("y"):
                self.great_circles = []
                self._afficher("great circles list cleared\n")
        if self.directions:
            ans = self._console_input("initialize the list (y/n) ? : ", "n")
            if ans is not None and ans.strip().lower().startswith("y"):
                self.directions = []
                self._afficher("D-I list cleared\n")

    def _di_pairs(self):
        """(dec,inc) sans le symbole - format attendu par stereo_pmagpy.py."""
        return [(d, inc) for d, inc, _sym in self.directions]

    # ------------------------------------------------------------------
    # Statistics (stereo_stats.py - port de pmagoutils.f, hors Bootstrap
    # ellipse cf. memoire projet). Chaque routine qui produit une moyenne
    # l'ajoute a self.means (symbole "e", meme convention que le source),
    # visible ensuite via Command > Plot-screen.
    # ------------------------------------------------------------------

    def stat_fisher_statistics(self):
        """Equivalent de `ANGU` (Statistics > Fisher Statistics)."""
        if len(self.directions) < 2:
            messagebox.showwarning("Not enough data", "Load at least 2 directions first.")
            return
        rfilt_s = self._console_input(
            "Do you want to filter the data at more xx deg from the mean? (blank = no) : ", "")
        if rfilt_s is None:
            return
        try:
            rfilt = float(rfilt_s) if rfilt_s.strip() else 0.0
        except ValueError:
            rfilt = 0.0
        text, means = ss.angu(self._di_pairs(), rfilt=rfilt)
        self._afficher(text)
        self.means.extend(means)

    def stat_fisher_dir_gc(self):
        """Equivalent de `FISHGC` (Statistics > Fisher Dir+GC, McFadden &
        McElhinny 1988)."""
        if (len(self.directions) + len(self.great_circles)) < 3:
            messagebox.showwarning(
                "Not enough data", "Need directions + great circles >= 3 total.")
            return
        sect_s = self._console_input("DO YOU WANT TO TAKE SECTORS INTO ACCOUNT? Y/N : ", "n")
        if sect_s is None:
            return
        use_sectors = sect_s.strip().lower().startswith("y")
        start_dec = start_inc = 0.0
        if not self.directions:
            start_s = self._console_input("DIRECTION DE DEPART (0,0) PAR DEFAULT: ", "")
            if start_s is None:
                return
            parts = start_s.split()
            if len(parts) >= 2:
                try:
                    start_dec, start_inc = float(parts[0]), float(parts[1])
                except ValueError:
                    start_dec = start_inc = 0.0
        text, mean = ss.fishgc(
            self._di_pairs(), self.great_circles, use_sectors, start_dec, start_inc)
        self._afficher(text)
        if mean is not None:
            self.means.append(mean)

    def stat_fisher_recursive(self):
        """Equivalent de `angurecur` (Statistics > Fisher recursive)."""
        if len(self.directions) < 3:
            messagebox.showwarning("Not enough data", "Load at least 3 directions first.")
            return
        cutoff_s = self._console_input("angle cutoff (0-90) : ", "20")
        if cutoff_s is None:
            return
        try:
            cutoff = float(cutoff_s)
        except ValueError:
            cutoff = 20.0
        if cutoff < 0.0 or cutoff > 90.0:
            cutoff = 20.0
        text, means = ss.angurecur(self._di_pairs(), cutoff)
        self._afficher(text)
        self.means.extend(means)

    def stat_stddip(self):
        """Equivalent de `STDDIP` (Statistics > St Dev dipole)."""
        if not self.directions:
            messagebox.showwarning("No data", "Load directions first.")
            return
        dec_s = self._console_input("DIRECTION DE REFERENCE - dec : ", "0")
        if dec_s is None:
            return
        inc_s = self._console_input("DIRECTION DE REFERENCE - inc : ", "90")
        if inc_s is None:
            return
        try:
            ref_dec = float(dec_s)
        except ValueError:
            ref_dec = 0.0
        try:
            ref_inc = float(inc_s)
        except ValueError:
            ref_inc = 90.0
        self._afficher(ss.stddip(self._di_pairs(), ref_dec, ref_inc))

    def stat_strati(self):
        """Equivalent de `STRATI` (Statistics > Fisher Strati) : mute les
        directions (dec+90) et REMPLACE la liste des grands cercles par
        les poles calcules - meme effet de bord que le source (`IMM=M`)."""
        if len(self.directions) < 3:
            messagebox.showwarning("Not enough data", "Load at least 3 directions first.")
            return
        text, new_directions, new_gcs = ss.strati(self.directions)
        self._afficher(text)
        self.directions = new_directions
        self.great_circles = new_gcs

    def stat_foldtest(self):
        """Equivalent de `FOLDTEST` (Statistics > progressive Fold test) :
        charge son PROPRE fichier (site dec inc azimuth dip, ou sans site),
        independant de self.directions - REMPLACE self.directions par les
        directions optimalement depliees par site (meme effet de bord que
        le source, IL=NUMDIR)."""
        path = filedialog.askopenfilename(
            title="progressive Fold test - file (site dec inc azimuth dip)")
        if not path:
            return
        rows = []
        with open(path, "r", encoding="iso-8859-1", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        site = parts[0]
                        dec, inc, az, dip = (float(p) for p in parts[1:5])
                        rows.append((site, dec, inc, az, dip))
                        continue
                    except ValueError:
                        pass
                if len(parts) >= 4:
                    try:
                        dec, inc, az, dip = (float(p) for p in parts[:4])
                        rows.append((f"site{len(rows) + 1}", dec, inc, az, dip))
                    except ValueError:
                        continue
        if len(rows) < 2:
            messagebox.showerror("Error", "Need at least 2 sites (site dec inc azimuth dip).")
            return
        text, new_directions = ss.fold_test_native(rows)
        self._afficher(text)
        self.directions = new_directions

    def stat_meaninc(self):
        """Equivalent de `MEANINC` (Statistics > Mean Inclination only,
        McFadden & Reid 1982 natif - distinct du menu Pmag_Python > Mean
        Inclination qui appelle PmagPy)."""
        if len(self.directions) < 2:
            messagebox.showwarning("Not enough data", "Load at least 2 directions first.")
            return
        res = ss.mean_inclination_native([inc for _d, inc in self._di_pairs()])
        if res is None:
            return
        self._afficher(
            f" MOYENNE ARITHMETIQUE: INCLINAISON = {res['arithmetic_mean']:.2f}\n"
            f" INCLINAISON BIAISEE (ARASON) {res['biased_arason']:.2f}\n"
            f" INCLINAISON NON BIAISEE (MCFADDEN): {res['unbiased_mcfadden']:.2f}\n"
            f" PARAMETRE DE PRECISION: {res['precision']:.2f}\n"
            f" ESTIMATION DE ALP95: {res['alpha95']:.2f}\n"
            f" INCLINAISON MOYENNE: {res['unbiased_mcfadden']:.2f} +/-{res['alpha95']:.2f}\n"
            f"                      {res['biased_arason']:.2f} +{res['alpha95_pos']:.2f} -{res['alpha95_neg']:.2f}\n"
        )

    def stat_intersect(self):
        """Equivalent du coeur de `intersect2` (Statistics > smallcircles
        Intersect) : utilise TOUJOURS les 3 (ou 2) premieres moyennes en
        memoire, meme limitation que le source."""
        if len(self.means) < 2:
            messagebox.showwarning(
                "Not enough data", "Need at least 2 mean directions (D-I-a95) in memory.")
            return
        text, points = ss.intersect_smallcircles(self.means)
        self._afficher(text)
        self.directions.extend(points)

    def stat_reversal_angle(self):
        """Equivalent de `reversalangle` (Statistics > Reversal angle)."""
        if not self.directions:
            messagebox.showwarning("No data", "Load directions first.")
            return
        dec_s = self._console_input("Expected direction in normal polarity - dec : ", "0")
        if dec_s is None:
            return
        inc_s = self._console_input("Expected direction in normal polarity - inc : ", "0")
        if inc_s is None:
            return
        try:
            d_expect = float(dec_s)
        except ValueError:
            d_expect = 0.0
        try:
            r_expect = float(inc_s)
        except ValueError:
            r_expect = 0.0
        self._afficher(ss.reversal_angle(self._di_pairs(), d_expect, r_expect))

    def stat_diff_angle(self):
        """Equivalent de `difangle` (Statistics > difference angle) :
        fichier (D0,I0,D1,I1) par ligne."""
        path = filedialog.askopenfilename(title="difference angle - file (D0 I0 D1 I1)")
        if not path:
            return
        self._afficher(ss.diff_angle_file(path))

    # ------------------------------------------------------------------
    # Project (stereo_project.py - port de plotstereo.f95 +
    # StereoOSX_x.f95:360-515) : systeme de calques Illustrator
    # independant du modele D-I/D-I-a95/grands cercles.
    # ------------------------------------------------------------------

    def load_project(self):
        """Equivalent de `loadproject` (Project > Load Project) - lit
        AUSSI, silencieusement, un fichier "Stereo Project" a 3 blocs
        (individual directions/mean directions/VGP - voir
        stereo_project.load_project_blocks) quand il en trouve un ; sinon
        repli automatique sur l'ancien format a plat (retro-compatible).
        `self.project_vgp_entries` recoit le bloc VGP eventuel (vide pour
        un ancien fichier), consultable via "Plot VGP Project..."."""
        path = filedialog.askopenfilename(
            title="Load Project", filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        self.project_entries, self.project_vgp_entries = sproj.load_project_blocks(path)
        msg = f"{len(self.project_entries)} project entrie(s) loaded from {os.path.basename(path)}"
        if self.project_vgp_entries:
            msg += f" ({len(self.project_vgp_entries)} VGP entrie(s) also available - see Plot VGP Project...)"
        self._afficher(msg + "\n")

    def export_to_project(self):
        """Equivalent de `export2project` (Project > Export to Project) :
        construit le projet a partir de self.directions/self.means/
        self.great_circles (REMPLACE self.project_entries, meme effet que
        le source qui reinitialise `i=0`), affiche au format pret-a-copier
        et propose de l'enregistrer dans un fichier."""
        self.project_entries = sproj.export_project(self.directions, self.means, self.great_circles)
        text = sproj.format_project_lines(self.project_entries)
        self._afficher("-------\n" + text + "-------\ncopy these lines to a project file and edit with a text editor\n")
        if not self.project_entries:
            return
        out_path = filedialog.asksaveasfilename(
            title="Save project file", defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        if not out_path:
            return
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        self._afficher(f"saved to {os.path.basename(out_path)}\n")

    def list_project(self):
        """Equivalent de `listproject` (Project > List Project)."""
        if not self.project_entries:
            self._afficher("no data in project\n")
            return
        self._afficher(sproj.format_project_lines(self.project_entries))

    def init_project(self):
        """Equivalent de `initproject`/`initdataproject` (Project > Init
        Project)."""
        self.project_entries = []
        self._afficher("project cleared\n")

    def plot_project(self):
        """Equivalent de `plotproject` (menu Graphics > Plot stereo project,
        ex-menu "Project" - demande explicite utilisateur "move plot
        stereo project... within the graphic menu")."""
        if not self.project_entries:
            messagebox.showwarning("No data", "Load or export a project first.")
            return
        self._last_image_files = []
        sproj.build_project_figure(
            self.project_entries, la=self.la, phi=self.phi, iproj=self.iproj,
            dim=self.dim, fig=self.fig,
        )
        if self.graph_title:
            self.fig.axes[0].set_title(self.graph_title)
        self._redraw_canvas()

    def fisher_project(self):
        """Equivalent de `fisherproject` (Project > Fisher Project) :
        moyenne de Fisher par calque (lignes consecutives de meme nom de
        calque) - un seul filtre `rfilt` demande une fois pour tous les
        calques (le source reprompte a chaque calque via `ANGU`, simplifie
        ici pour l'ergonomie, sans effet sur les valeurs calculees)."""
        if len(self.project_entries) < 2:
            messagebox.showwarning("Not enough data", "Load or export a project first.")
            return
        rfilt_s = self._console_input(
            "Do you want to filter the data at more xx deg from the mean? (blank = no) : ", "")
        if rfilt_s is None:
            return
        try:
            rfilt = float(rfilt_s) if rfilt_s.strip() else 0.0
        except ValueError:
            rfilt = 0.0
        text, layer_means = sproj.fisher_project(self.project_entries, rfilt=rfilt)
        self._afficher(text)

    def help_project(self):
        """Aide visuelle Project (ajout hors source Fortran, demande
        explicite) : palette de couleurs nommees affichee avec un
        vrai echantillon colore (utilisables directement dans un fichier
        projet a la place de "R_G_B") + legende des symboles."""
        if self.text_area.get("1.0", "end-1c").strip():
            self.text_area.insert(tk.END, "\n" + "-" * 60 + "\n")
        self.text_area.insert(tk.END, "Project file colors - use the name directly instead of \"R_G_B\" :\n\n")
        for name, (r, g, b) in sproj.NAMED_COLORS.items():
            tag = f"swatch_{name}"
            if tag not in self.text_area.tag_names():
                self.text_area.tag_configure(tag, background=f"#{r:02x}{g:02x}{b:02x}")
            self.text_area.insert(tk.END, "    ", tag)
            self.text_area.insert(tk.END, f"  {name:<10s} ({r}_{g}_{b})\n")
        self.text_area.insert(tk.END, "\nProject file symbols (letter used in the \"symb\" column) :\n\n")
        for letter, desc in sproj.SYMBOL_NAMES.items():
            self.text_area.insert(tk.END, f"  {letter}   {desc}\n")
        self.text_area.see(tk.END)

    # ------------------------------------------------------------------
    # Pmag Utilities (stereo_pmagutils.py - port de pmagoutils.f +
    # calcrota.f, hors drillcore et Echelle polarite)
    # ------------------------------------------------------------------

    def _prompt_float(self, prompt, default=0.0):
        s = self._console_input(prompt, str(default))
        if s is None:
            return None
        try:
            return float(s)
        except ValueError:
            return default

    def _prompt_floats(self, prompt, n):
        """Lit UNE ligne et y cherche `n` valeurs separees par des
        espaces - meme convention que les `READ(*,*) A,B,C` Fortran a
        plusieurs valeurs sur une seule ligne (ex. "STRIKE AND DIP:",
        "DECLINATION, INCLINATION AND A95:"). Retourne None si annule
        (Echap), ligne vide, ou nombre/format de valeurs invalide."""
        line = self._console_input(prompt)
        if line is None or not line.strip():
            return None
        parts = line.split()
        if len(parts) < n:
            return None
        try:
            return [float(p) for p in parts[:n]]
        except ValueError:
            return None

    def _prompt_coord_mode(self):
        s = self._console_input("COORDONNES EN DEG (0), DEG MIN (1), DEG MIN SEC (2) : ", "0")
        if s is None:
            return None
        try:
            return max(0, min(2, int(s)))
        except ValueError:
            return 0

    # -- VGP conversions ----------------------------------------------------

    def pu_vgpc1(self):
        site = self._prompt_floats("LATITUDE AND LONGITUDE OF THE SITE, ex. \"-10 -70\" : ", 2)
        if site is None:
            return
        site_lat, site_lon = site
        while True:
            line = self._console_input("DECLINATION, INCLINATION +? A95\n(blank to stop, paste multiple lines OK)\n: ")
            if line is None or not line.strip():
                break
            for subline in self._split_pasted_rows(line):
                parts = subline.split()
                if len(parts) < 2:
                    continue
                try:
                    dec, inc = float(parts[0]), float(parts[1])
                    a95 = float(parts[2]) if len(parts) > 2 else 0.0
                except ValueError:
                    continue
                res = pu.vgpc1(dec, inc, site_lat, site_lon, a95)
                if a95 > 0:
                    self._afficher(f"VGP LAT: {res['vgp_lat']:6.1f}  LONG: {res['vgp_lon']:6.1f}  DM={res['dm']:.1f}  DP={res['dp']:.1f}\n")
                else:
                    self._afficher(f"VGP LAT: {res['vgp_lat']:6.1f}  LONG: {res['vgp_lon']:6.1f}\n")

    def pu_vgpc2(self):
        site = self._prompt_floats("LATITUDE AND LONGITUDE OF THE SITE, ex. \"-10 -70\" : ", 2)
        if site is None:
            return
        site_lat, site_lon = site
        while True:
            line = self._console_input("LATITUDE, LONGITUDE (VGP) +? A95\n(blank to stop, paste multiple lines OK)\n: ")
            if line is None or not line.strip():
                break
            for subline in self._split_pasted_rows(line):
                parts = subline.split()
                if len(parts) < 2:
                    continue
                try:
                    vlat, vlon = float(parts[0]), float(parts[1])
                    a95 = float(parts[2]) if len(parts) > 2 else 0.0
                except ValueError:
                    continue
                res = pu.vgpc2(vlat, vlon, site_lat, site_lon, a95)
                if a95 > 0:
                    self._afficher(f"DEC: {res['dec']:6.1f}  INC: {res['inc']:6.1f}  DD={res['dd']:.1f}  DI={res['di']:.1f}\n")
                else:
                    self._afficher(f"DECLI: {res['dec']:6.1f}  INCLI: {res['inc']:6.1f}\n")

    def pu_vgpc3(self):
        path = filedialog.askopenfilename(title="File: Direction to VGP (site lat lon, dec, inc, [a95])")
        if not path:
            return
        lines = read_text_lines(path)
        data_lines, idx = split_header(lines, "slat", "slon", "dec", "inc", "a95")
        col_order = None
        if "slat" in idx and "slon" in idx and "dec" in idx and "inc" in idx:
            mode = 0
            col_order = [idx["slat"], idx["slon"], idx["dec"], idx["inc"]]
            if "a95" in idx:
                col_order.append(idx["a95"])
        else:
            mode = self._prompt_coord_mode()
            if mode is None:
                return
        rows = []
        for line in data_lines:
            parts = line.split()
            if not parts:
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if col_order:
                if max(col_order) >= len(vals):
                    continue
                vals = [vals[i] for i in col_order]
            rows.append(vals)
        results = pu.vgpc3_batch(rows, mode)
        lines = ["Decli    Incli    a95       Longitude Latitude  a95 b95"]
        for r in results:
            if "dm" in r:
                lines.append(f"{r['dec']:6.1f}  {r['inc']:6.1f}  {r['a95']:5.1f}   {r['vgp_lon']:6.1f}  {r['vgp_lat']:6.1f}  {r['dm']:5.1f} {r['dp']:5.1f}")
            else:
                lines.append(f"{r['dec']:6.1f}  {r['inc']:6.1f}              {r['vgp_lon']:6.1f}  {r['vgp_lat']:6.1f}")
        self._afficher("\n".join(lines) + "\n")

    def pu_vgpc4(self):
        path = filedialog.askopenfilename(title="File: VGP to direction (site lat lon, VGP, [a95])")
        if not path:
            return
        lines = read_text_lines(path)
        data_lines, idx = split_header(lines, "slat", "slon", "vgplat", "vgplon", "a95")
        col_order = None
        lat_first = True
        if "slat" in idx and "slon" in idx and "vgplat" in idx and "vgplon" in idx:
            mode = 0
            col_order = [idx["slat"], idx["slon"], idx["vgplat"], idx["vgplon"]]
            if "a95" in idx:
                col_order.append(idx["a95"])
        else:
            order_s = self._console_input("VGP as (lat,lon) (1) or (lon,lat) (2) in the file : ", "1")
            if order_s is None:
                return
            lat_first = order_s.strip() != "2"
            mode = self._prompt_coord_mode()
            if mode is None:
                return
        rows = []
        for line in data_lines:
            parts = line.split()
            if not parts:
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if col_order:
                if max(col_order) >= len(vals):
                    continue
                vals = [vals[i] for i in col_order]
            rows.append(vals)
        results = pu.vgpc4_batch(rows, mode, lat_first)
        lines = ["Longitude  latitude  a95     Decli    Incli   a95  b95"]
        for r in results:
            if "dd" in r:
                lines.append(f"{r['vgp_lon']:6.1f}  {r['vgp_lat']:6.1f}  {r['a95']:5.1f}   {r['dec']:6.1f}  {r['inc']:6.1f}  {r['dd']:5.1f} {r['di']:5.1f}")
            else:
                lines.append(f"{r['vgp_lon']:6.1f}  {r['vgp_lat']:6.1f}            {r['dec']:6.1f}  {r['inc']:6.1f}")
        self._afficher("\n".join(lines) + "\n")

    # -- Rotation/flattening tests -------------------------------------------

    def pu_paleotec(self):
        dia = self._prompt_floats("DECLINATION, INCLINATION AND A95, ex. \"20 45 5\" : ", 3)
        if dia is None:
            return
        dec, inc, a95 = dia
        site = self._prompt_floats("LATITUDE AND LONGITUDE OF THE SITE, ex. \"-10 -70\" : ", 2)
        if site is None:
            return
        slat, slon = site
        pole = self._prompt_floats("LATITUDE, LONGITUDE, P95 (reference pole), ex. \"48.6 317.6 2\" : ", 3)
        if pole is None:
            return
        plat, plon, p95 = pole
        r = pu.paleotec(dec, inc, a95, slat, slon, plat, plon, p95)
        self._afficher(
            f"DECL_EXP: {r['dec_exp']:6.1f}  INCL_EXP: {r['inc_exp']:6.1f}  ROTATION: {r['rotation']:6.1f}  FLATTENING: {r['flattening']:6.1f}\n"
            f"DELTA_ROTATION: {r['delta_rotation']:6.1f}   DELTA_FLATTENING: {r['delta_flattening']:6.1f}\n"
        )

    def pu_paleodec(self):
        dir1 = self._prompt_floats("1ERE DIRECTION: DEC, INC AND A95, ex. \"10 40 3\" : ", 3)
        if dir1 is None:
            return
        d1, i1, a95 = dir1
        dir2 = self._prompt_floats("2EME DIRECTION: DEC, INC AND A95, ex. \"10 40 3\" : ", 3)
        if dir2 is None:
            return
        d2, i2, b95 = dir2
        r = pu.paleodec(d1, i1, a95, d2, i2, b95)
        self._afficher(f"ROTATION: {r['rotation']:6.1f}  FLATTENING: {r['flattening']:6.1f}  DELTA_ROTATION: {r['delta_rotation']:6.1f}  DELTA_FLATTENING: {r['delta_flattening']:6.1f}\n")

    def pu_paleotec1(self):
        obs = self._prompt_floats("DATA (obs pole): LATITUDE, LONGITUDE, A95, ex. \"48.6 317.6 2\" : ", 3)
        if obs is None:
            return
        olat, olon, o95 = obs
        site = self._prompt_floats("LATITUDE AND LONGITUDE OF THE SITE, ex. \"-10 -70\" : ", 2)
        if site is None:
            return
        slat, slon = site
        ref = self._prompt_floats("REFERENCE: LATITUDE, LONGITUDE, P95, ex. \"48.6 317.6 2\" : ", 3)
        if ref is None:
            return
        rlat, rlon, r95 = ref
        r = pu.paleotec1(olat, olon, o95, slat, slon, rlat, rlon, r95)
        self._afficher(
            f"R={r['rotation']:.1f}  dR={r['delta_rotation']:.1f}  Ierr={r['inc_error']:.1f}  "
            f"LAT_DEPLACEMENT={r['pole_displacement']:.1f}  dL={r['delta_pole_displacement']:.1f}\n"
        )

    # -- Paleolatitude --------------------------------------------------------

    def pu_paleolati(self):
        path = filedialog.askopenfilename(title="APWP file (age, vgp_lat, vgp_lon, [a95])")
        if not path:
            return
        site = self._prompt_floats("site lat, lon, ex. \"-10 -70\" : ", 2)
        if site is None:
            return
        slat, slon = site
        pol_s = self._console_input("Polarity Normal (1) or Reverse (-1) : ", "1")
        if pol_s is None:
            return
        try:
            polarity = int(pol_s)
        except ValueError:
            polarity = 1
        text, _rows = pu.paleolati_from_file(path, slat, slon, polarity)
        self._afficher(text)

    def pu_paleolati2(self):
        ia = self._prompt_floats("normal polarity Inclination and a95, ex. \"45 5\" : ", 2)
        if ia is None:
            return
        inc, a95 = ia
        r = pu.paleolati2(inc, a95)
        self._afficher(f"paleolatitude : {r['paleolatitude']:.1f}  Min: {r['min']:.1f}  Max: {r['max']:.1f}\n")

    # -- GMT rotation export --------------------------------------------------

    def pu_rota2gmt(self):
        data_path = filedialog.askopenfilename(
            title="ROTA2GMT - data file (REF SITE AGE Polarity LAT LON DEC INC A95 [REF_pub])")
        if not data_path:
            return
        apwp_path = filedialog.askopenfilename(title="ROTA2GMT - reference APWP file")
        if not apwp_path:
            return
        agerange = self._prompt_floats("INTERVALLE DE TEMPS CHOISI, ex. \"0 300\" : ", 2)
        if agerange is None:
            return
        agemin, agemax = agerange
        vgp2dec = self._console_input("data as VGP (v) or Direction (d) : ", "d")
        if vgp2dec is None:
            return
        dime = self._prompt_float("Length of the arrow (0.2-1.5) : ", 0.5)
        if dime is None:
            return
        fflatcor = self._prompt_float("correction for shallowing in sediments (0.4-1) : ", 1.0)
        if fflatcor is None:
            return
        if fflatcor < 0.4 or fflatcor > 1.0:
            fflatcor = 1.0

        data_rows = []
        data_lines = read_text_lines(data_path)
        data_lines, didx = split_header(
            data_lines, "iref", "site", "age", "polarity", "slat", "slon", "dec", "inc", "a95", "refpub")
        if all(k in didx for k in ("iref", "site", "age", "polarity", "slat", "slon", "dec", "inc", "a95")):
            d_i = didx
        else:
            d_i = {"iref": 0, "site": 1, "age": 2, "polarity": 3, "slat": 4, "slon": 5, "dec": 6, "inc": 7, "a95": 8}
            if "refpub" in didx:
                d_i["refpub"] = didx["refpub"]
        for line in data_lines:
            line = line.strip()
            if not line or line.startswith("!") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) <= max(d_i["iref"], d_i["site"], d_i["age"], d_i["polarity"], d_i["slat"], d_i["slon"], d_i["dec"], d_i["inc"], d_i["a95"]):
                continue
            try:
                iref, site = parts[d_i["iref"]], parts[d_i["site"]]
                age = float(parts[d_i["age"]])
                carpol = parts[d_i["polarity"]]
                al, g, rdec, rinc, a95 = (
                    float(parts[d_i["slat"]]), float(parts[d_i["slon"]]),
                    float(parts[d_i["dec"]]), float(parts[d_i["inc"]]), float(parts[d_i["a95"]]))
            except ValueError:
                continue
            if "refpub" in d_i and len(parts) > d_i["refpub"]:
                ref = parts[d_i["refpub"]]
            else:
                ref = "none"
            data_rows.append((iref, site, age, carpol, al, g, rdec, rinc, a95, ref))

        apwp_rows = []
        apwp_lines = read_text_lines(apwp_path)
        apwp_lines, aidx = split_header(apwp_lines, "age", "vgplat", "vgplon", "p95")
        if all(k in aidx for k in ("age", "vgplat", "vgplon", "p95")):
            i_age, i_lat, i_lon, i_p95 = aidx["age"], aidx["vgplat"], aidx["vgplon"], aidx["p95"]
        else:
            i_age, i_lat, i_lon, i_p95 = 0, 3, 4, 5
        for line in apwp_lines:
            line = line.strip()
            if not line or line.startswith("!") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) <= max(i_age, i_lat, i_lon, i_p95):
                continue
            try:
                age1 = float(parts[i_age])
                rlat_pole, rlon_pole, p95 = float(parts[i_lat]), float(parts[i_lon]), float(parts[i_p95])
            except ValueError:
                continue
            apwp_rows.append((age1, rlat_pole, rlon_pole, p95))
        apwp_rows.sort(key=lambda r: r[0])

        report, vec, pie = pu.rota2gmt_core(data_rows, apwp_rows, agemin, agemax, vgp2dec, dime, fflatcor)
        self._afficher(report)
        out_path = filedialog.asksaveasfilename(title="Save results (.res)", defaultextension=".res")
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report)
            base = out_path.rsplit(".", 1)[0]
            with open(base + ".vec", "w", encoding="utf-8") as f:
                f.write(vec)
            with open(base + ".pie", "w", encoding="utf-8") as f:
                f.write(pie)
            self._afficher(f"saved {os.path.basename(base)}.res/.vec/.pie\n")

    def pu_helprota(self):
        self._afficher(pu.HELPROTA_TEXT)

    # -- Site Map (site_map.py, port ad hoc - pas de source Fortran) ----------

    def _pu_sitemap_open_prmag(self):
        """Ouvre un .prmag et en extrait les sites (voir site_map.
        read_prmag_sites) - factorise entre les 3 entrees du sous-menu
        "Site Map". Retourne (path, sites) ou (None, None) si annule/
        aucun site trouve."""
        path = filedialog.askopenfilename(
            title="Open .prmag", filetypes=[("prmag", "*.prmag"), ("All files", "*.*")])
        if not path:
            return None, None
        sites = read_prmag_sites(path)
        if not sites:
            messagebox.showwarning("No sites", "No site (with a specimen block) found in this file.")
            return None, None
        return path, sites

    def pu_sitemap_kml(self):
        """Equivalent de "prmag to kml" (demande explicite utilisateur
        "ajouter un sous-menu pour tracer les sites sur Google Earth,
        prmag to kml") - voir site_map.write_kml."""
        path, sites = self._pu_sitemap_open_prmag()
        if sites is None:
            return
        default_name = os.path.splitext(os.path.basename(path))[0] + ".kml"
        out_path = filedialog.asksaveasfilename(
            title="Save KML", defaultextension=".kml", initialfile=default_name,
            filetypes=[("KML", "*.kml"), ("All files", "*.*")])
        if not out_path:
            return
        write_kml(sites, out_path, doc_name=os.path.splitext(os.path.basename(path))[0])
        self._afficher(f"{len(sites)} site(s) written to {out_path} (open in Google Earth).\n")

    def pu_sitemap_gmt(self):
        """Equivalent de "GMT map pour tracer les sites sur une carte
        topographique" (demande explicite utilisateur) - voir site_map.
        write_gmt_map_script. Genere un script shell GMT6 (relief distant
        @earth_relief_01m) + son fichier de donnees compagnon, A EXECUTER
        PAR L'UTILISATEUR (GMT n'est jamais appele depuis cette
        application, meme convention que "Rotation vers GMT plot...")."""
        path, sites = self._pu_sitemap_open_prmag()
        if sites is None:
            return
        default_name = os.path.splitext(os.path.basename(path))[0] + "_map.sh"
        out_path = filedialog.asksaveasfilename(
            title="Save GMT map script", defaultextension=".sh", initialfile=default_name,
            filetypes=[("Shell script", "*.sh"), ("All files", "*.*")])
        if not out_path:
            return
        data_path = write_gmt_map_script(sites, out_path)
        self._afficher(
            f"{len(sites)} site(s) - GMT script saved to {out_path}\n"
            f"(data file: {data_path})\n"
            f"Run it yourself (needs GMT >= 6 installed): bash \"{out_path}\"\n")

    def pu_sitemap_plot(self):
        """Equivalent de "une copie avec le simple matplotlib" (demande
        explicite utilisateur "eventuellement une copie avec le simple
        matplotlib", puis "is there a possibility to have a basic map
        too") - voir site_map.build_site_map_figure. Le fond de carte
        (case "Basic basemap (coastlines)", cartopy) est TENTE si coche,
        avec repli vers la version simple sans reseau/donnees Natural
        Earth en cache - affiche le texte EXACT de l'erreur rencontree
        (pas un message generique - demande explicite utilisateur, le
        premier message generique s'etant revele inutilisable pour
        diagnostiquer l'echec reellement rencontre)."""
        path, sites = self._pu_sitemap_open_prmag()
        if sites is None:
            return
        _fig, got_basemap, error_text = build_site_map_figure(
            sites, fig=self.fig, title=os.path.splitext(os.path.basename(path))[0],
            basemap=self.sitemap_basemap.get())
        self._redraw_canvas()
        if self.sitemap_basemap.get() and not got_basemap:
            self._afficher(
                f"note: basemap unavailable ({error_text}) - showing the simple version "
                "instead.\n")

    # -- Core/bedding corrections ---------------------------------------------

    def pu_core1(self):
        gh = self._prompt_floats("DIP AND STRIKE (forage), ex. \"10 20\" : ", 2)
        if gh is None:
            return
        g, h = gh
        results = []
        while True:
            line = self._console_input("DECLINATION, INCLINATION, ex. \"30 40\"\n(blank to stop, paste multiple lines OK)\n: ")
            if line is None or not line.strip():
                break
            for subline in self._split_pasted_rows(line):
                parts = subline.split()
                if len(parts) < 2:
                    continue
                try:
                    dec, inc = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                (dec2, inc2), = pu.core_forage_correction(g, h, [(dec, inc)])
                self._afficher(f"DEC: {dec2:6.1f}   INC: {inc2:6.1f}\n")
                results.append((dec2, inc2))
        if results:
            rows_text = "".join(f"{d:.2f} {i:.2f}\n" for d, i in results)
            self._save_manual_entry_backup("core1", "#dec inc", rows_text)

    def pu_core2(self):
        mode_s = self._console_input("Is the bedding attitude given as Strike (1) or Dip direction (2)? : ", "1")
        if mode_s is None:
            return
        use_dipdir = mode_s.strip() == "2"
        label = "DIP DIRECTION AND DIP" if use_dipdir else "STRIKE AND DIP"
        rkrj = self._prompt_floats(f"{label}, ex. \"10 20\" : ", 2)
        if rkrj is None:
            return
        rk, rj = rkrj
        if use_dipdir:
            # dip direction -> strike : -90 deg (correction utilisateur -
            # `read_fold_file` ajoute 90 pour strike->dipdir, donc
            # l'operation inverse dipdir->strike est bien -90, pas +90).
            rk = (rk - 90.0) % 360.0
            self._afficher(f"(dip direction {rkrj[0]:.1f} -> strike {rk:.1f} for the correction)\n")
        results = []
        while True:
            line = self._console_input("DECLINATION, INCLINATION, ex. \"30 40\"\n(blank to stop, paste multiple lines OK)\n: ")
            if line is None or not line.strip():
                break
            for subline in self._split_pasted_rows(line):
                parts = subline.split()
                if len(parts) < 2:
                    continue
                try:
                    dec, inc = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                (dec2, inc2), = pu.core_bedding_correction(rk, rj, [(dec, inc)])
                self._afficher(f"DEC: {dec2:6.1f}   INC: {inc2:6.1f}\n")
                results.append((dec2, inc2))
        if results:
            rows_text = "".join(f"{d:.2f} {i:.2f}\n" for d, i in results)
            self._save_manual_entry_backup("core2", "#dec inc", rows_text)

    def pu_invbedding(self):
        use_file = self._console_input("data in a file ? y/N : ", "n")
        if use_file is None:
            return
        if use_file.strip().lower() == "y":
            path = filedialog.askopenfilename(title="inverseBedding cor - file (d1 i1 d2 i2)")
            if not path:
                return
            raw_lines = read_text_lines(path)
            data_lines, idx = split_header(raw_lines, "d1", "i1", "d2", "i2")
            if all(k in idx for k in ("d1", "i1", "d2", "i2")):
                cols = [idx["d1"], idx["i1"], idx["d2"], idx["i2"]]
            else:
                cols = [0, 1, 2, 3]
                data_lines = raw_lines
            lines = []
            for line in data_lines:
                parts = line.split()
                if len(parts) <= max(cols):
                    continue
                try:
                    d1, i1, d2, i2 = (float(parts[c]) for c in cols)
                except ValueError:
                    continue
                res = pu.invert_bedding(d1, i1, d2, i2)
                if res:
                    lines.append(f"{d1:7.1f} {i1:7.1f} {res[0]:7.1f} {res[1]:7.1f}")
            self._afficher("\n".join(lines) + "\n")
        else:
            di1 = self._prompt_floats("in situ D1,I1, ex. \"30 40\" : ", 2)
            if di1 is None:
                return
            d1, i1 = di1
            di2 = self._prompt_floats("bed corrected D2,I2, ex. \"30 40\" : ", 2)
            if di2 is None:
                return
            d2, i2 = di2
            res = pu.invert_bedding(d1, i1, d2, i2)
            if res is None:
                self._afficher("no solution found\n")
            else:
                self._afficher(f"strike: {res[0]:6.1f}   dip: {res[1]:6.1f}\n")

    # -- Fold plunge -----------------------------------------------------------

    def pu_foldplunge(self):
        mode = self._console_input("CORRECTION OF PLUNGE FIRST (1) OR PROGRESSIVE (2) : ", "1")
        if mode is None:
            return
        sd_mode_s = self._console_input("Is the bedding attitude given as Strike (1) or Dip direction (2)? : ", "1")
        if sd_mode_s is None:
            return
        sd_use_dipdir = sd_mode_s.strip() == "2"
        sd_label = "DIP DIRECTION AND DIP" if sd_use_dipdir else "STRIKE AND DIP"
        sd = self._prompt_floats(f"{sd_label}, ex. \"50 20\" : ", 2)
        if sd is None:
            return
        strike, dip = sd
        if sd_use_dipdir:
            strike = (strike - 90.0) % 360.0
            self._afficher(f"(dip direction {sd[0]:.1f} -> strike {strike:.1f} for the correction)\n")
        di = self._prompt_floats("DECLINATION, INCLINATION, ex. \"10 40\" : ", 2)
        if di is None:
            return
        dec, inc = di
        fold = self._prompt_floats("FOLD AZIMUTH, FOLD DIP, ex. \"140 15\" : ", 2)
        if fold is None:
            return
        faz, fdip = fold
        if mode.strip() == "2":
            r = pu.fold_plunge_progressive(strike, dip, dec, inc, faz, fdip)
            self._afficher(f"iterations: {r['iterations']}  DEC: {r['dec']:6.1f}  INC: {r['inc']:6.1f}\n")
        else:
            r = pu.fold_plunge_single(strike, dip, dec, inc, faz, fdip)
            self._afficher(
                f"STRIKE: {r['new_strike']:6.1f}  DIP: {r['new_dip']:6.1f}\n"
                f"first correction  DEC: {r['first_correction'][0]:6.1f}  INC: {r['first_correction'][1]:6.1f}\n"
                f"final correction  DEC: {r['final_correction'][0]:6.1f}  INC: {r['final_correction'][1]:6.1f}\n"
            )

    def pu_foldfile(self):
        path = filedialog.askopenfilename(
            title="File Fold plunge (site dec inc a95 strike|dipdir dip fold_azimuth fold_dip)")
        if not path:
            return
        lines = read_text_lines(path)
        data_lines, idx = split_header(lines, "site", "dec", "inc", "a95", "strike", "dipdir", "dip", "azimuth", "folddip")
        use_dipdir = False
        if "dipdir" in idx and all(k in idx for k in ("site", "dec", "inc", "a95", "dip", "azimuth", "folddip")):
            cols = [idx["site"], idx["dec"], idx["inc"], idx["a95"], idx["dipdir"], idx["dip"], idx["azimuth"], idx["folddip"]]
            use_dipdir = True
        elif all(k in idx for k in ("site", "dec", "inc", "a95", "strike", "dip", "azimuth", "folddip")):
            cols = [idx["site"], idx["dec"], idx["inc"], idx["a95"], idx["strike"], idx["dip"], idx["azimuth"], idx["folddip"]]
        else:
            mode_s = self._console_input(
                "No recognized header - is column 5 Strike (1) or Dip direction (2)? : ", "1")
            if mode_s is None:
                return
            use_dipdir = mode_s.strip() == "2"
            cols = list(range(8))
            data_lines = lines
        rows = []
        for line in data_lines:
            parts = line.split()
            if len(parts) <= max(cols):
                continue
            try:
                site = parts[cols[0]]
                dec, inc, a95, strike_or_dipdir, dip, faz, fdip = (float(parts[c]) for c in cols[1:8])
            except ValueError:
                continue
            strike = (strike_or_dipdir - 90.0) % 360.0 if use_dipdir else strike_or_dipdir
            rows.append((site, dec, inc, a95, strike, dip, faz, fdip))
        if not rows:
            messagebox.showerror("Error", "No valid rows (need site dec inc a95 strike|dipdir dip fold_azimuth fold_dip).")
            return
        text, new_directions = pu.fold_file_single(rows)
        self._afficher(text)
        self.directions = new_directions

    # -- Euler rotation ----------------------------------------------------

    def pu_rotmag(self):
        while True:
            site = self._prompt_floats("LATITUDE AND LONGITUDE OF THE SITE, ex. \"-10 -70\" : ", 2)
            if site is None:
                return
            slat, slon = site
            di = self._prompt_floats("DECLINAISON AND INCLINAISON, ex. \"30 40\" : ", 2)
            if di is None:
                return
            dec, inc = di
            pole = self._prompt_floats("POLE DE ROTATION (LONG LAT ANGLE), ex. \"100 30 25\" : ", 3)
            if pole is None:
                return
            plon, plat, pang = pole
            dec2, inc2 = pu.rotmag(slat, slon, dec, inc, plon, plat, pang)
            self._afficher(f"DECLINATION= {dec2:6.1f}  INCLINATION= {inc2:5.1f}\n")
            again = self._console_input("DO YOU WANT A NEW CALCULATION (Y/N) : ", "n")
            if again is None or again.strip().lower() != "y":
                return

    def pu_vgp_plate_rotation(self):
        path = filedialog.askopenfilename(title="VGP Plate rotation - file (vgp_lon vgp_lat rot_lat rot_lon rot_angle)")
        if not path:
            return
        lines = read_text_lines(path)
        data_lines, idx = split_header(lines, "vgplon", "vgplat", "rotlat", "rotlon", "rotangle")
        if all(k in idx for k in ("vgplon", "vgplat", "rotlat", "rotlon", "rotangle")):
            cols = [idx["vgplon"], idx["vgplat"], idx["rotlat"], idx["rotlon"], idx["rotangle"]]
        else:
            cols = [0, 1, 2, 3, 4]
            data_lines = lines
        rows = []
        for line in data_lines:
            parts = line.split()
            if len(parts) <= max(cols):
                continue
            try:
                rows.append(tuple(float(parts[c]) for c in cols))
            except ValueError:
                continue
        results = pu.vgp_plate_rotation_batch(rows)
        lines = [f"in: {r['vgp_lon_in']:.1f} {r['vgp_lat_in']:.1f}  out: {r['vgp_lon_out']:.1f} {r['vgp_lat_out']:.1f}" for r in results]
        self._afficher("\n".join(lines) + "\n")

    # -- Paleointensity --------------------------------------------------------

    def pu_meanpal(self):
        mode = self._console_input("DATA FROM A FILE (1), KEYBOARD (3) : ", "3")
        if mode is None:
            return
        if mode.strip() == "1":
            path = filedialog.askopenfilename(
                title="mean paleointensity - file (F Q N, or a .pmagint)")
            if not path:
                return
            kind, data = pu.read_meanpal_file(path)
            if not data:
                self._afficher("no exploitable data in file\n")
                return
            if kind == "pmagint":
                self._pu_meanpal_pmagint_loop(data)
            else:
                res = pu.meanpal_weighted(data)
                if res is None:
                    self._afficher("no exploitable data (N<=2 everywhere)\n")
                    return
                self._afficher(
                    f"MEAN: {res['mean']:.1f}   WEIGHTED MEAN: {res['weighted_mean']:.1f}   "
                    f"SD: {res['sd']:.1f}   N: {res['n']}\n")
            return

        triples = []
        self._afficher("ENTER FIELD VALUE, Q, AND THE NUMBER OF STEPS\n(blank to stop, paste multiple lines OK)\n")
        while True:
            line = self._console_input("F Q N : ")
            if line is None or not line.strip():
                break
            for subline in self._split_pasted_rows(line):
                parts = subline.split()
                if len(parts) < 3:
                    continue
                try:
                    triples.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    continue
        res = pu.meanpal_weighted(triples)
        if res is None:
            self._afficher("no exploitable data (N<=2 everywhere)\n")
            return
        self._afficher(f"MEAN: {res['mean']:.1f}   WEIGHTED MEAN: {res['weighted_mean']:.1f}   SD: {res['sd']:.1f}   N: {res['n']}\n")

    def _pu_meanpal_pmagint_loop(self, rows):
        """CASE(2) "FICHIER STARMAC" du Fortran (pmagoutils.f:612-824,
        partie selection+calcul - la partie VDM/VADM en aval n'est pas
        portee ici, hors perimetre "mean paleointensity") : affiche les
        lignes numerotees, demande une plage i j, calcule/affiche la
        table 3 series (0/ANI/cool) sur cette plage, puis propose de
        recommencer - demande explicite utilisateur ("l'utilisateur doit
        choisir les donnees a moyenner... Voir source en Fortran")."""
        listing = "\n".join(f"{i:3d} : {r.label}" for i, r in enumerate(rows, start=1))
        self._afficher(listing + "\n")
        while True:
            rng = self._console_input(
                f"select lines i j for the mean calculation (1-{len(rows)}, blank to stop) : ", "")
            if rng is None or not rng.strip():
                return
            parts = rng.replace(",", " ").split()
            if len(parts) < 2:
                self._afficher("invalid range - expected two numbers \"i j\"\n")
                continue
            try:
                i, j = int(parts[0]), int(parts[1])
            except ValueError:
                self._afficher("invalid range - expected two numbers \"i j\"\n")
                continue
            subset = rows[max(1, i) - 1: j]
            if not subset:
                self._afficher("empty range\n")
                continue
            table = pu.meanpal_table(subset)
            if table is None:
                self._afficher("no exploitable data (N<=2 everywhere)\n")
            else:
                self._afficher(pu.format_meanpal_table(table))
            again = self._console_input("DO YOU WANT ANOTHER CALCULATION (Y/N) : ", "n")
            if again is None or again.strip().lower() != "y":
                return

    def pu_vidimo(self):
        while True:
            pal = self._prompt_float("PALEOINTENSITY (microT) : ")
            if pal is None:
                return
            slat = self._prompt_float("LATITUDE OF THE SITE : ")
            if slat is None:
                return
            inc = self._prompt_float("PALEOMAGNETIC INCLINATION : ")
            if inc is None:
                return
            vdm, vadm = pu.vidimo(pal, slat, inc)
            self._afficher(f"VDM={vdm:.3e}    VADM={vadm:.3e}\n")
            again = self._console_input("DO YOU WANT ANOTHER CALCULATION (Y/N) : ", "n")
            if again is None or again.strip().lower() != "y":
                return

    def pu_relocate(self):
        while True:
            vdm = self._prompt_float("VDM * 1.e22 Am2 : ")
            if vdm is None:
                return
            slat = self._prompt_float("LATITUDE OF THE SITE : ")
            if slat is None:
                return
            inc = self._prompt_float("PALEOMAGNETIC INCLINATION : ")
            if inc is None:
                return
            fvdm, fvadm = pu.relocate(vdm, slat, inc)
            self._afficher(f"Fvdm={fvdm:.1f}    Fvadm={fvadm:.1f}\n")
            again = self._console_input("DO YOU WANT ANOTHER CALCULATION (Y/N) : ", "n")
            if again is None or again.strip().lower() != "y":
                return

    def pu_relocatevar(self):
        path = filedialog.askopenfilename(
            title="Relocate D I F - file (site age age_err lat lon dec inc a95 F F_err)")
        if not path:
            return
        target = self._prompt_floats("relocation site - lat, lon, ex. \"-15 -72\" : ", 2)
        if target is None:
            return
        target_lat, target_lon = target
        lines = read_text_lines(path)
        data_lines, idx = split_header(
            lines, "site", "age", "ageerr", "slat", "slon", "dec", "inc", "a95", "pal", "palerr")
        if all(k in idx for k in ("site", "age", "ageerr", "slat", "slon", "dec", "inc", "a95", "pal", "palerr")):
            cols = [idx["site"], idx["age"], idx["ageerr"], idx["slat"], idx["slon"],
                    idx["dec"], idx["inc"], idx["a95"], idx["pal"], idx["palerr"]]
        else:
            cols = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # legacy: site,litho(skipped),age,age_err,slat,slon,dec,inc,a95,pal,pal_err
            data_lines = lines
        rows = []
        for line in data_lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            parts = line.split()
            if len(parts) <= max(cols):
                continue
            try:
                site = parts[cols[0]]
                age, age_err, slat, slon, dec, inc, a95, pal, pal_err = (float(parts[c]) for c in cols[1:10])
            except ValueError:
                continue
            rows.append((site, age, age_err, slat, slon, dec, inc, a95, pal, pal_err))
        self._afficher(pu.relocatevar_simplified(rows, target_lat, target_lon))

    # -- IGRF --------------------------------------------------------------

    def pu_igrf(self):
        lat = self._prompt_float("site latitude : ")
        if lat is None:
            return
        lon = self._prompt_float("site longitude : ")
        if lon is None:
            return
        year_s = self._console_input("year : ", "2020")
        if year_s is None:
            return
        try:
            year = int(float(year_s))
        except ValueError:
            year = 2020
        alt = self._prompt_float("altitude (km) : ", 0.0)
        if alt is None:
            return
        try:
            r = pu.igrf_value(lat, lon, year, altitude_km=alt)
        except ImportError:
            messagebox.showerror("ppigrf not available", "The 'ppigrf' package is required for IGRF calculations.")
            return
        self._afficher(
            f"D={r['declination']:6.1f}  I={r['inclination']:6.1f}  "
            f"F={r['total_intensity_nT']:8.1f} nT  H={r['horizontal_intensity_nT']:8.1f} nT  "
            f"N={r['north_nT']:8.1f}  E={r['east_nT']:8.1f}  Z={r['down_nT']:8.1f}\n"
        )

    def pu_helpigrf(self):
        self._afficher(pu.HELPIGRF_TEXT)

    # -- Unit conversion / overprint / flatten ---------------------------------

    def pu_convunit(self):
        menu_text = (
            "Am2 -> emu : 1\n emu -> Am2 : 2\n Am2/kg -> emu/g : 3\n emu/g -> Am2/kg : 4\n"
            "A/m -> emu/cc : 5\n emu/cc -> A/m : 6\n Tesla -> A/m : 7\n microTesla -> A/m : 8\n"
            "A/m -> Tesla : 9\n A/m -> microTesla : 10\n"
        )
        self._afficher(menu_text)
        choice_s = self._console_input("choice : ", "1")
        if choice_s is None:
            return
        try:
            choice = int(choice_s)
        except ValueError:
            return
        value = self._prompt_float("valeur : ")
        if value is None:
            return
        res = pu.convert_units(choice, value)
        if res is None:
            return
        val, unit = res
        self._afficher(f"{val:g} {unit}\n")

    def pu_overprint(self):
        ov = self._prompt_floats("direction of the overprint, ex. \"0 60\" : ", 2)
        if ov is None:
            return
        od, oi = ov
        nrm = self._prompt_floats("normal polarity direction in situ, D I, ex. \"10 40\" : ", 2)
        if nrm is None:
            return
        nd, ni = nrm
        rev = self._prompt_floats("reverse polarity direction in situ, D I, ex. \"195 -35\" : ", 2)
        if rev is None:
            return
        rd, ri = rev
        r = pu.overprint_test(od, oi, nd, ni, rd, ri)
        self._afficher(
            f"angle between initial N and R directions: {r['initial_angle_n_r']:.1f}\n"
            f"likely overprint: {r['estimated_overprint_pct']:.1f} %\n"
            f"char dir corrected: N={r['corrected_normal'][0]:.1f}/{r['corrected_normal'][1]:.1f}  "
            f"R={r['corrected_reverse'][0]:.1f}/{r['corrected_reverse'][1]:.1f}\n"
            f"estimated minimum angle between N and R: {r['minimum_angle_n_r']:.1f}\n"
            f"mean char dir before: {r['mean_char_dir_before'][0]:.1f}/{r['mean_char_dir_before'][1]:.1f}\n"
            f"mean char dir after correction: {r['mean_char_dir_after'][0]:.1f}/{r['mean_char_dir_after'][1]:.1f}\n"
        )

    def pu_flatten(self):
        while True:
            line = self._console_input("value of the uncorrected inclination within 0-90 (blank to stop) : ")
            if line is None or not line.strip():
                return
            try:
                uninc = float(line)
            except ValueError:
                return
            line2 = self._console_input("value of the f factor within 0-1 : ")
            if line2 is None or not line2.strip():
                return
            try:
                f = float(line2)
            except ValueError:
                return
            res = pu.flatten_correction(uninc, f)
            if res is None:
                return
            self._afficher(f"corrected value : {res:.1f}\n")

    # ------------------------------------------------------------------
    # Pmag_Python (stereo_pmagpy.py - port de PmagPy_connect.f95)
    # ------------------------------------------------------------------

    def pmagpy_fisher(self):
        """Menu Pmag_Python > Fisher (ajout hors source Fortran, demande
        explicite) : `pmag.fisher_mean` directement sur self.directions -
        pas de separation en modes normal/reverse (contrairement a
        Statistics > Fisher Statistics, qui appelle `MODES` avant Fisher)."""
        if len(self.directions) < 2:
            messagebox.showwarning("Not enough data", "Load at least 2 directions first.")
            return
        res = sp.fisher_mean(self._di_pairs())
        self._afficher(
            f" Fisher mean (PmagPy) : n={res['n']}  dec={res['dec']:.1f}  inc={res['inc']:.1f}\n"
            f" r={res['r']:.3f}  k={res['k']:.1f}  a95={res['alpha95']:.1f}  csd={res['csd']:.1f}\n"
        )

    def pmagpy_bootstrap_ellipse(self):
        """Menu Pmag_Python > Bootstrap ellipse (ajout hors source
        Fortran, demande explicite) : `ipmag.mean_bootstrap_confidence`
        (Heslop et al. 2023) - equivalent PmagPy, avec un vrai generateur
        aleatoire, du menu Statistics > Bootstrap ellipse dont le portage
        natif Fortran a ete explicitement ecarte (RNG desactive dans le
        source)."""
        if len(self.directions) < 3:
            messagebox.showwarning("Not enough data", "Load at least 3 directions first.")
            return
        nb_s = self._console_input("number of bootstrap simulations (default=1000) : ", "1000")
        if nb_s is None:
            return
        try:
            nb = int(nb_s)
        except ValueError:
            nb = 1000
        if nb <= 0:
            nb = 1000
        params, files = sp.bootstrap_ellipse(self._di_pairs(), num_sims=nb)
        self._afficher(
            f" bootstrap mean (Heslop et al. 2023) : dec={params['dec']:.1f}  inc={params['inc']:.1f}\n"
            f" T_critical={params['T_critical']:.3f}\n"
        )
        self._show_images(files)

    def mean_inclination(self):
        """Equivalent de `meanincpmagpy` (menu Pmag_Python > Mean
        Inclination) : `incfish.py` -> `pmag.doincfish`."""
        if len(self.directions) < 2:
            messagebox.showwarning("Not enough data", "Load at least 2 directions first.")
            return
        res = sp.mean_inclination(self._di_pairs())
        self._afficher(
            f" inclination (McFadden & Reid 1982) : n={res['n']}  inc={res['inc']:.1f}\n"
            f" k={res['k']:.1f}  a95={res['alpha95']:.1f}  csd={res['csd']:.1f}\n"
            f" confidence limits: [{res['lower_confidence_limit']:.1f}, {res['upper_confidence_limit']:.1f}]\n"
        )

    def find_elongation(self):
        """Equivalent de `find_EI` (menu Pmag_Python > Find Elongation) :
        `find_EI.py` -> `ipmag.find_ei`."""
        if not self.directions:
            messagebox.showwarning("No data", "Load directions first.")
            return
        nb_s = self._console_input("number of iteration (100 to 2000) : ", "2000")
        if nb_s is None:
            return
        try:
            nb = int(nb_s)
        except ValueError:
            nb = 2000
        if nb < 100 or nb > 2000:
            nb = 2000
        _res, files = sp.find_elongation(self._di_pairs(), nb=nb)
        self._show_images(files)
        lines = ["  declination-inclination (data used for this plot)"]
        for i, (dec, inc) in enumerate(self._di_pairs(), start=1):
            lines.append(f"{i:4d}:{dec:7.1f}{inc:7.1f}")
        self._afficher("\n".join(lines) + "\n")

    def test_reversal_antipodal(self):
        """Equivalent de `test_antipodal` (menu Pmag_Python > Reversal
        antipodality) : `revtest.py` -> `ipmag.reversal_test_bootstrap`."""
        if not self.directions:
            messagebox.showwarning(
                "No data", "Load a file with normal and reverse directions first.")
            return
        _res, files = sp.test_reversal_antipodal(self._di_pairs())
        self._show_images(files)

    def _load_or_enter_di(self, label):
        """Charge un jeu de directions (dec,inc) depuis un fichier OU par
        saisie manuelle ligne par ligne "dec inc" (blank pour arreter) -
        utilise par Test common mean, qui a besoin de 2 jeux de donnees
        DEDIES independants de self.directions (contrairement au reste
        de l'app), et n'offrait jusqu'ici que le chargement fichier."""
        mode = self._console_input(f"{label} - load from file (f) or enter manually (m) : ", "f")
        if mode is None:
            return None
        if mode.strip().lower().startswith("m"):
            dirs = []
            self._afficher(f"{label} : enter \"dec inc\" per line\n(blank to stop, paste multiple lines OK)\n")
            i = 1
            while True:
                line = self._console_input(f"{i:3d}: ")
                if line is None or not line.strip():
                    break
                for subline in self._split_pasted_rows(line):
                    parts = subline.split()
                    if len(parts) < 2:
                        continue
                    try:
                        dec, inc = float(parts[0]), float(parts[1])
                    except ValueError:
                        continue
                    dirs.append((dec, inc))
                    i += 1
            if dirs:
                rows_text = "".join(f"{d:.2f} {inc:.2f}\n" for d, inc in dirs)
                self._save_manual_entry_backup("DI", "#dec inc", rows_text)
            return dirs
        path = filedialog.askopenfilename(title=label)
        if not path:
            return None
        return read_di_file(path)

    def test_common_mean(self):
        """Equivalent de `common_mean` (menu Pmag_Python > Test common
        mean) : `common_mean.py` -> `ipmag.common_mean_bootstrap`. Demande
        2 jeux de donnees, comme le Fortran ("open the first/second data
        set") - fichier ou saisie manuelle au choix pour chacun."""
        self._afficher("comparison of two sets of direction to test if they have a common mean\n")
        dirs1 = self._load_or_enter_di("First data set (D-I)")
        if dirs1 is None:
            return
        dirs2 = self._load_or_enter_di("Second data set (D-I)")
        if dirs2 is None:
            return
        if len(dirs1) < 2 or len(dirs2) < 2:
            messagebox.showerror("Error", "Both data sets need at least 2 directions.")
            return
        _res, files = sp.test_common_mean(dirs1, dirs2)
        self._show_images(files)

    def fold_test(self):
        """Equivalent de `foldtestpmagpy` (menu Pmag_Python > Fold Test) :
        `foldtest.py` -> `ipmag.bootstrap_fold_test` (Tauxe & Watson 1994).
        Donnees (dec,inc,strike|dip_direction,dip) - fichier ou saisie
        manuelle au choix."""
        self._afficher(
            " Fold test with PmagPy\n"
            " You need D, I, Strike (or dip direction), dip for each sample\n"
        )
        mode = self._console_input("load from file (f) or enter manually (m) : ", "f")
        if mode is None:
            return
        if mode.strip().lower().startswith("m"):
            str_s = self._console_input("Will you enter Strike (1) or dip direction (2)? : ", "1")
            if str_s is None:
                return
            strike = str_s.strip() != "2"
            label = "strike" if strike else "dip direction"
            self._afficher(f"enter \"dec inc {label} dip\" per line\n(blank to stop, paste multiple lines OK)\n")
            data = []
            i = 1
            while True:
                line = self._console_input(f"{i:3d}: ")
                if line is None or not line.strip():
                    break
                for subline in self._split_pasted_rows(line):
                    parts = subline.split()
                    if len(parts) < 4:
                        continue
                    try:
                        dec, inc, sd, dip = (float(p) for p in parts[:4])
                    except ValueError:
                        continue
                    dip_direction = sd + 90.0 if strike else sd
                    data.append((dec, inc, dip_direction, dip))
                    i += 1
            if data:
                rows_text = "".join(f"{d:.2f} {inc:.2f} {dd:.2f} {dip:.2f}\n" for d, inc, dd, dip in data)
                self._save_manual_entry_backup("foldtest", "#dec inc dipdir dip", rows_text)
        else:
            path = filedialog.askopenfilename(title="Fold test data (D I Str/DipDir Dip)")
            if not path:
                return
            header = split_header(read_text_lines(path), "dec", "inc", "strike", "dip", "dipdir")[1]
            has_header = "dec" in header and "inc" in header and "dip" in header and ("strike" in header or "dipdir" in header)
            if has_header:
                skip, strike = 0, True  # ignored by read_fold_file when the header itself resolves the columns
            else:
                skip_s = self._console_input("number of variables to skip before D : ", "0")
                if skip_s is None:
                    return
                try:
                    skip = int(skip_s)
                except ValueError:
                    skip = 0
                str_s = self._console_input("In the file is it Strike (1) or dip direction (2)? : ", "1")
                if str_s is None:
                    return
                strike = str_s.strip() != "2"
            data = read_fold_file(path, skip_columns=skip, strike=strike)
        if len(data) < 3:
            messagebox.showerror("Error", "Need at least 3 dec/inc/strike/dip rows.")
            return
        angle_s = self._console_input(
            "ANGLE (circular standard deviation) for uncertainty on bedding poles (default=2.0) : ", "2.0")
        if angle_s is None:
            return
        try:
            angle = float(angle_s)
        except ValueError:
            angle = 2.0
        nb_s = self._console_input("number of bootstrap (default=1000) : ", "1000")
        if nb_s is None:
            return
        try:
            nb = int(nb_s)
        except ValueError:
            nb = 1000
        if nb <= 0:
            nb = 1000
        _res, files = sp.fold_test(data, nb=nb, bedding_error=angle)
        self._show_images(files)
        tc = sp.tilt_corrected_directions(data)
        lines = ["  declination-inclination (tilt-corrected, 2nd stereo \"Tilt-corrected\"/eq_tc)"]
        for i, (dec, inc) in enumerate(tc, start=1):
            lines.append(f"{i:4d}:{dec:7.1f}{inc:7.1f}")
        self._afficher("\n".join(lines) + "\n")

    def plot_vgps_on_map(self):
        """Equivalent de `plotvgp` (menu Graphics > Plot VGPs on Map,
        ex-menu "Pmag_Python"/"PmagPy-tools" - demande explicite
        utilisateur "move plot stereo project; VGP on map VGP project;
        within the graphic menu") : `plot_map_pts.py` ->
        `ipmag.make_orthographic_map` + `ipmag.plot_vgp` (les dec/inc en
        memoire representent directement longitude/latitude du VGP, meme
        convention que le Fortran)."""
        if not self.directions:
            messagebox.showwarning("No data", "Load directions (VGP lon/lat) first.")
            return
        lat_s = self._console_input("latitude of the view point : ", "0")
        if lat_s is None:
            return
        lon_s = self._console_input("longitude of the view point : ", "0")
        if lon_s is None:
            return
        try:
            lat = float(lat_s)
        except ValueError:
            lat = 0.0
        try:
            lon = float(lon_s)
        except ValueError:
            lon = 0.0
        if lat < -90 or lat > 90:
            lat = 0.0
        _res, files = sp.plot_vgps_on_map(self._di_pairs(), view_lat=lat, view_lon=lon)
        self._show_images(files)

    def plot_vgp_project(self):
        """Equivalent VGP de "Plot stereo project" - demande explicite
        utilisateur ("when we load a project, can we load the VGP too and
        keep them in memory for later plot on a map? as with the mean
        directions") : utilise DIRECTEMENT `self.project_vgp_entries`
        (deja rempli par "Load Project" - voir load_project/
        stereo_project.load_project_blocks), SANS reparcourir de fichier -
        exactement le meme fonctionnement que `plot_project` avec
        `self.project_entries`. Pour charger un AUTRE fichier VGP, passer
        par "Load Project" d'abord (qui remplace aussi
        self.project_vgp_entries), pas par un dialogue separe ici."""
        if not self.project_vgp_entries:
            messagebox.showwarning(
                "No data", "Load a project with a VGP block first (Graphics > Load Project...).")
            return
        lat_s = self._console_input("latitude of the view point : ", "0")
        if lat_s is None:
            return
        lon_s = self._console_input("longitude of the view point : ", "0")
        if lon_s is None:
            return
        try:
            lat = float(lat_s)
        except ValueError:
            lat = 0.0
        try:
            lon = float(lon_s)
        except ValueError:
            lon = 0.0
        if lat < -90 or lat > 90:
            lat = 0.0
        _res, files = sp.plot_vgp_project(self.project_vgp_entries, view_lat=lat, view_lon=lon)
        self._show_images(files)

    # ------------------------------------------------------------------
    # K-T (susceptibilite vs temperature - curie_kt.py, port de Curie_OSX
    # menu "KLY3_CS3") - demande explicite utilisateur ("integrer les
    # deux fonctions principales de Curie_OSX, le traitement des donnees
    # de susceptibilite (courbes K-T) ... ajouter un menu K-T").
    # ------------------------------------------------------------------

    def _kt_lookup_entry(self):
        """Retrouve l'entree de la liste d'echantillons (voir kt_open_
        sample_list) correspondant a self.kt_curve, par nom de fichier
        (MAJUSCULES) - None si aucune liste n'est chargee ou si ce
        fichier n'y figure pas."""
        if not self.kt_sample_list or self.kt_curve is None:
            return None
        return self.kt_sample_list.get(os.path.basename(self.kt_curve.path).upper())

    def _kt_correct_with_entry(self, curve, entry, base_dir):
        """Applique la correction four decrite par `entry` (voir
        curie_kt.apply_furnace_correction/apply_vessel_correction) -
        factorise entre kt_apply_furnace_correction (correction manuelle
        d'une courbe deja ouverte) et kt_select_from_list (pipeline
        automatique, meme comportement que `selectfich`). Retourne
        (nouvelle_courbe, message) ou (None, message_erreur) - n'affiche
        rien elle-meme, laisse l'appelant decider (_showerror vs
        _afficher)."""
        if entry.zerodia == 999.0:
            if not entry.empty_vessel:
                return None, "code 999 but no empty-vessel filename in the list"
            vessel_path = os.path.join(base_dir, entry.empty_vessel)
            if not os.path.exists(vessel_path):
                return None, f"empty-vessel file not found: {vessel_path}"
            vessel_curve = read_cur_file(vessel_path)
            corrected = apply_vessel_correction(curve, vessel_curve)
            return corrected, f"furnace correction: full empty-vessel curve {entry.empty_vessel}"
        corrected = apply_furnace_correction(curve, entry.zerodia)
        return corrected, f"furnace correction: subtracted {entry.zerodia:g}"

    def kt_open_file(self):
        """Equivalent de `openkt` (CurieOSX_x.f95:189-201) -> `saisie`
        (divers.f:3-182, branche a 2 colonnes - voir curie_kt.py)."""
        path = filedialog.askopenfilename(
            title="Open .CUR/.CLW File",
            filetypes=[("Curie K-T", "*.CUR *.cur *.CLW *.clw"), ("All files", "*.*")])
        if not path:
            return
        self.kt_curve = read_cur_file(path)
        self.kt_curie_heating = None
        self.kt_curie_cooling = None
        entry = self._kt_lookup_entry()
        msg = (
            f"{self.kt_curve.sample}: {len(self.kt_curve.temp)} point(s), "
            f"T = {self.kt_curve.temp.min():.1f} to {self.kt_curve.temp.max():.1f} °C\n"
        )
        if entry is not None:
            msg += (
                f"(matching sample-list entry found: suscep={entry.suscep:g} SI, "
                f"masse={entry.masse:g} mg, zerodia={entry.zerodia:g})\n"
            )
        self._afficher(msg)

    def kt_open_sample_list(self):
        """Equivalent de `openlistekt` (CurieOSX_x.f95:431-473, voir
        curie_kt.read_kt_sample_list pour l'ecart de format reel)."""
        path = filedialog.askopenfilename(
            title="Open Sample List", filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        self.kt_sample_list = read_kt_sample_list(path)
        self.kt_list_dir = os.path.dirname(path)
        self._afficher(f"{len(self.kt_sample_list)} sample-list entry/entries loaded from "
                        f"{os.path.basename(path)}\n")

    def kt_select_from_list(self):
        """Equivalent de `selectfich` (CurieOSX_x.f95:536-572, CTRL+L) :
        parcourt self.kt_sample_list - pour l'entree choisie, ouvre son
        fichier .CUR/.CLW, applique AUTOMATIQUEMENT la correction four
        (constante ou vase vide selon `zerodia`, voir _kt_correct_with_
        entry) PUIS la normalisation appropriee (volume si `suscep!=0`,
        sinon masse si `masse!=0`, sinon aucune - meme choix que le
        Fortran), calcule le point de Curie et trace la courbe - boucle
        jusqu'a "0"/Escape pour passer a l'echantillon suivant (demande
        explicite utilisateur "when we open the list, we should be able
        to go through and select the sample to plot")."""
        if not self.kt_sample_list:
            self._showwarning("No sample list", "Open a sample list first (Open Sample List...).")
            return
        entries = list(self.kt_sample_list.values())
        lines = []
        for i, e in enumerate(entries, start=1):
            extra = f"  vessel={e.empty_vessel}" if e.empty_vessel else ""
            lines.append(
                f"{i:3d} : {e.filename:<16} suscep={e.suscep:g}  masse={e.masse:g}  "
                f"zerodia={e.zerodia:g}{extra}")
        self._afficher("\n".join(lines) + "\n")

        while True:
            choice_s = self._console_input(
                f"Line number to open+plot (0 or Escape to stop, 1-{len(entries)}): ", "0")
            if choice_s is None:
                return
            choice_s = choice_s.strip()
            if not choice_s or choice_s == "0":
                return
            try:
                idx = int(choice_s)
            except ValueError:
                self._showerror("Error", "Must be an integer.")
                continue
            if not (1 <= idx <= len(entries)):
                self._showerror("Error", f"Out of range (1-{len(entries)}).")
                continue
            self._kt_process_list_entry(entries[idx - 1])

    def _kt_process_list_entry(self, entry):
        """Pipeline automatique d'UNE entree de liste - voir kt_select_
        from_list. Facteur commun avec les etapes manuelles (kt_open_
        file/kt_apply_furnace_correction/kt_normalize_by_.../kt_curie_
        point/kt_plot), mais enchainees sans intervention (meme
        comportement que `selectfich`)."""
        base_dir = self.kt_list_dir or "."
        path = os.path.join(base_dir, entry.filename)
        if not os.path.exists(path):
            self._showerror("Error", f"File not found: {path}")
            return
        curve = read_cur_file(path)

        corrected, msg = self._kt_correct_with_entry(curve, entry, base_dir)
        if corrected is None:
            self._showerror("Error", f"{entry.filename}: {msg}")
            return
        curve = corrected

        if entry.suscep:
            is_cur = os.path.splitext(path)[1].upper() == ".CUR"
            curve = normalize_by_volume(curve, entry.suscep, is_cur=is_cur)
        elif entry.masse:
            curve = normalize_by_mass(curve, entry.masse)

        self.kt_curve = curve
        self.kt_curie_heating, self.kt_curie_cooling = curie_point_second_derivative(curve)
        heat = f"{self.kt_curie_heating:.1f}" if self.kt_curie_heating is not None else "n/a"
        cool = f"{self.kt_curie_cooling:.1f}" if self.kt_curie_cooling is not None else "n/a"
        self._afficher(
            f"{entry.filename}: {len(curve.temp)} point(s), {msg}, "
            f"normalized={curve.normalized or 'no'} - "
            f"Curie heating: {heat} °C, cooling: {cool} °C\n"
        )
        build_kt_figure(
            curve, curie_heating=self.kt_curie_heating,
            curie_cooling=self.kt_curie_cooling, fig=self.fig)
        self._redraw_canvas()

    def kt_apply_furnace_correction(self):
        """Equivalent de `corrfour` (CurieOSX_x.f95:240-291) - deux
        branches (voir curie_kt.apply_furnace_correction/apply_vessel_
        correction) : `zerodia` pris dans la liste d'echantillons si ce
        fichier y figure (sinon demande directement, meme repli que le
        Fortran quand `iselect==999`, ouverture d'un fichier isole sans
        liste) - si `zerodia==999`, le fichier du vase vide associe
        (`entry.empty_vessel`) est ouvert et sa courbe COMPLETE soustraite
        point par point plutot qu'une simple constante (demande explicite
        utilisateur - "999 indicate that a whole empty vessel should be
        used")."""
        if self.kt_curve is None:
            self._showwarning("No K-T curve", "Open a .CUR/.CLW file first.")
            return
        entry = self._kt_lookup_entry()

        if entry is not None and entry.zerodia == 999.0:
            base_dir = self.kt_list_dir or os.path.dirname(self.kt_curve.path)
            corrected, msg = self._kt_correct_with_entry(self.kt_curve, entry, base_dir)
            if corrected is None:
                self._showerror("Error", msg)
                return
            self.kt_curve = corrected
            self._afficher(f"Furnace correction applied ({msg}).\n")
            return

        default = f"{entry.zerodia:g}" if entry is not None else "-142.8"
        zerodia_s = self._console_input("Furnace/holder background value to subtract: ", default)
        if zerodia_s is None:
            return
        try:
            zerodia = float(zerodia_s)
        except ValueError:
            self._showerror("Error", "Must be a number.")
            return
        self.kt_curve = apply_furnace_correction(self.kt_curve, zerodia)
        self._afficher(f"Furnace correction applied (subtracted {zerodia:g}).\n")

    def kt_normalize_by_mass(self):
        """Equivalent de `normas` (CurieOSX_x.f95:293-330, voir
        curie_kt.normalize_by_mass)."""
        if self.kt_curve is None:
            self._showwarning("No K-T curve", "Open a .CUR/.CLW file first.")
            return
        if self.kt_curve.normalized is not None:
            self._showwarning("Already normalized", "This curve is already normalized.")
            return
        entry = self._kt_lookup_entry()
        default = f"{entry.masse:g}" if entry is not None and entry.masse else ""
        mass_s = self._console_input("Sample mass (mg): ", default)
        if mass_s is None:
            return
        try:
            mass_mg = float(mass_s)
        except ValueError:
            self._showerror("Error", "Must be a number.")
            return
        if mass_mg <= 0.0:
            self._showerror("Error", "Mass must be > 0.")
            return
        self.kt_curve = normalize_by_mass(self.kt_curve, mass_mg)
        self._afficher(f"Normalized by mass ({mass_mg:g} mg) - values in 1e-6 m³/kg.\n")

    def kt_normalize_by_volume(self):
        """Equivalent de `norvol` (CurieOSX_x.f95:332-372, voir
        curie_kt.normalize_by_volume)."""
        if self.kt_curve is None:
            self._showwarning("No K-T curve", "Open a .CUR/.CLW file first.")
            return
        if self.kt_curve.normalized is not None:
            self._showwarning("Already normalized", "This curve is already normalized.")
            return
        entry = self._kt_lookup_entry()
        default = f"{entry.suscep:g}" if entry is not None and entry.suscep else ""
        suscep_s = self._console_input("Independent bulk susceptibility (SI): ", default)
        if suscep_s is None:
            return
        try:
            suscep = float(suscep_s)
        except ValueError:
            self._showerror("Error", "Must be a number.")
            return
        is_cur = os.path.splitext(self.kt_curve.path)[1].upper() == ".CUR"
        self.kt_curve = normalize_by_volume(self.kt_curve, suscep, is_cur=is_cur)
        self._afficher(f"Normalized by volume (bulk={suscep:g} SI) - values in SI.\n")

    def kt_curie_point(self):
        """Equivalent de `tauxe` (divers.f:740-816 - voir curie_kt.
        curie_point_second_derivative pour l'ecart avec le nom "Tauxe
        method")."""
        if self.kt_curve is None:
            self._showwarning("No K-T curve", "Open a .CUR/.CLW file first.")
            return
        self.kt_curie_heating, self.kt_curie_cooling = curie_point_second_derivative(self.kt_curve)
        heat = f"{self.kt_curie_heating:.1f}" if self.kt_curie_heating is not None else "n/a"
        cool = f"{self.kt_curie_cooling:.1f}" if self.kt_curie_cooling is not None else "n/a"
        self._afficher(
            f"Curie temperature (2nd-derivative maximum) - heating: {heat} °C, "
            f"cooling: {cool} °C\n"
        )

    def kt_plot(self):
        """Equivalent de `plotkt` (CurieOSX_x.f95:232-235) ->
        `tracerklyhot`/`tracerklycold` (voir curie_kt.build_kt_figure -
        matplotlib direct, pas plotlib.PlotContext, demande explicite
        utilisateur "Faire les graphics sans passer par mes anciennes
        fonctions (plot, symbol etc)")."""
        if self.kt_curve is None:
            self._showwarning("No K-T curve", "Open a .CUR/.CLW file first.")
            return
        build_kt_figure(
            self.kt_curve, curie_heating=self.kt_curie_heating,
            curie_cooling=self.kt_curie_cooling, fig=self.fig)
        self._redraw_canvas()

    # ------------------------------------------------------------------
    # Hysteresis (curie_hyst.py, port de Curie_OSX menu "Hysteresis") -
    # demande explicite utilisateur ("integrer les deux fonctions
    # principales de Curie_OSX ... et des donnees d'Hysteresis").
    # ------------------------------------------------------------------

    def hyst_open_agm_list(self):
        """Equivalent de la liste lue par `selectfichhystAGM`
        (hysteresis.f:717-945, voir curie_hyst.read_agm_sample_list) -
        format AGM historique (Princeton MicroMag "Model 2900 ASCII Data
        File", verifie sur /Users/pierrickroperch/Paleomag_data/
        Hyste_Chili_2)."""
        path = filedialog.askopenfilename(
            title="Open AGM Sample List", filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        self.hyst_sample_list = read_agm_sample_list(path)
        self.hyst_list_dir = os.path.dirname(path)
        self.hyst_format = "AGM"
        self._afficher(
            f"{len(self.hyst_sample_list)} AGM sample-list entry/entries loaded from "
            f"{os.path.basename(path)}\n")

    def hyst_open_vsm_list(self):
        """Format VSM moderne (voir curie_hyst.read_vsm_sample_list),
        verifie sur /Users/pierrickroperch/Paleomag_data/VSM_Nov2022 -
        PAS un format du Fortran d'origine."""
        path = filedialog.askopenfilename(
            title="Open VSM Sample List", filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        self.hyst_sample_list = read_vsm_sample_list(path)
        self.hyst_list_dir = os.path.dirname(path)
        self.hyst_format = "VSM"
        self._afficher(
            f"{len(self.hyst_sample_list)} VSM sample-list entry/entries loaded from "
            f"{os.path.basename(path)}\n")

    def hyst_set_valsat(self):
        """Equivalent de `prefhyste`/`valsat` (CurieOSX_x.f95 - fraction
        du champ maximal au-dela de laquelle un point est considere
        "haut champ" pour le fit paramagnetique, voir curie_hyst.
        compute_hysteresis) - PAS une constante figee, demande explicite
        (verifie ne pas reproduire exactement Resu_Chile2.txt avec la
        valeur par defaut 0.7, voir docstring de module)."""
        s = self._console_input(
            "Paramagnetic fit high-field threshold (fraction of max field, 0-1): ",
            f"{self.hyst_valsat_frac:g}")
        if s is None:
            return
        try:
            v = float(s)
        except ValueError:
            self._showerror("Error", "Must be a number.")
            return
        if not (0.0 < v < 1.0):
            self._showerror("Error", "Must be between 0 and 1.")
            return
        self.hyst_valsat_frac = v
        self._afficher(f"Paramagnetic fit threshold set to {v:g}.\n")

    def hyst_select_from_list(self):
        """Equivalent de la boucle de selection de `selectfichhystAGM`
        (hysteresis.f:745-751, meme principe que kt_select_from_list
        cote K-T - demande explicite utilisateur "when we open the list,
        we should be able to go through and select the sample to plot",
        appliquee ici aussi) : pour l'entree choisie, ouvre ses 2
        fichiers (boucle + remanence/DCD, format AGM ou VSM selon
        `self.hyst_format`), calcule Js/Jrs/Hc/Hcr et trace - boucle
        jusqu'a "0"/Escape."""
        if not self.hyst_sample_list:
            self._showwarning(
                "No sample list", "Open an AGM or VSM sample list first.")
            return
        entries = list(self.hyst_sample_list.values())
        lines = [
            f"{i:3d} : {e.filename:<14} masse={e.masse:g} mg"
            for i, e in enumerate(entries, start=1)
        ]
        self._afficher(f"({self.hyst_format})\n" + "\n".join(lines) + "\n")

        while True:
            choice_s = self._console_input(
                f"Line number to open+plot (0 or Escape to stop, 1-{len(entries)}): ", "0")
            if choice_s is None:
                return
            choice_s = choice_s.strip()
            if not choice_s or choice_s == "0":
                return
            try:
                idx = int(choice_s)
            except ValueError:
                self._showerror("Error", "Must be an integer.")
                continue
            if not (1 <= idx <= len(entries)):
                self._showerror("Error", f"Out of range (1-{len(entries)}).")
                continue
            self._hyst_process_entry(entries[idx - 1])

    def _hyst_process_entry(self, entry):
        """Pipeline automatique d'UNE entree de liste - voir hyst_select_
        from_list."""
        try:
            if self.hyst_format == "AGM":
                loop_path = os.path.join(self.hyst_list_dir, entry.filename)
                backfield_path = loop_path + "-r"
                if not (os.path.exists(loop_path) and os.path.exists(backfield_path)):
                    self._showerror(
                        "Error", f"File(s) not found for {entry.filename} "
                        f"(expected {entry.filename} and {entry.filename}-r).")
                    return
                loop = read_agm_file(loop_path)
                backfield = read_agm_file(backfield_path)
            else:
                hyst_path, backfield_path = vsm_paths_for(self.hyst_list_dir, entry.filename)
                if not (os.path.exists(hyst_path) and os.path.exists(backfield_path)):
                    self._showerror(
                        "Error", f"File(s) not found for {entry.filename} "
                        f"(expected \"{entry.filename} - 1.csv\" and \"{entry.filename} - 2.csv\").")
                    return
                loop = read_vsm_csv(hyst_path)
                backfield = read_vsm_csv(backfield_path)
        except OSError as e:
            self._showerror("Error", f"{entry.filename}: {e}")
            return

        res = compute_hysteresis(loop, backfield, entry.masse, valsat_frac=self.hyst_valsat_frac)
        if res is None:
            self._showerror(
                "Error",
                f"{entry.filename}: could not compute hysteresis parameters "
                "(loop too short, or no point above the high-field threshold).")
            return
        res.sample = entry.filename
        self.hyst_result = res
        self.hyst_max_field = None  # reinitialise l'echelle X pour un nouvel echantillon
        self._afficher(format_hysteresis_result(res))
        self.hyst_refresh_plot()

    def hyst_refresh_plot(self):
        """Redessine self.hyst_result avec les reglages courants
        (self.hyst_normalize/self.hyst_max_field) SANS relire les
        fichiers - appelee par la case a cocher "Normalize Plot" et par
        le clic-sur-axe-X (voir _on_plot_pick)."""
        if self.hyst_result is None:
            return
        build_hysteresis_figure(
            self.hyst_result, fig=self.fig,
            max_field=self.hyst_max_field, normalize=self.hyst_normalize.get())
        self._redraw_canvas()

    def hyst_toggle_normalize(self):
        """Bascule "Normalize Plot" - appelee par le raccourci clavier
        (voir SHORTCUTS["hystnorm"]/_setup_shortcuts), la case a cocher
        elle-meme bascule sa propre variable au clic et appelle
        directement hyst_refresh_plot."""
        self.hyst_normalize.set(not self.hyst_normalize.get())
        self.hyst_refresh_plot()

    def _on_plot_pick(self, event):
        """Gestionnaire unique de clic-sur-axe (pick_event) - pour
        l'instant uniquement l'axe X du graphique Hysteresis (voir
        curie_hyst._register_xaxis_pick) - demande explicite utilisateur
        ("to click on the Xaxis to change the scale (Max field
        value)")."""
        kind = getattr(event.artist, "_su_pick_kind", None)
        if kind == "hyst_xaxis":
            self._prompt_hyst_max_field()

    def _prompt_hyst_max_field(self):
        """Invite pour changer la borne X (champ max, Tesla) du
        graphique Hysteresis - symetrique (-max, +max)."""
        default = "" if self.hyst_max_field is None else f"{self.hyst_max_field:g}"
        s = self._console_input("Max field to display (T, empty = auto): ", default)
        if s is None:
            return
        s = s.strip()
        if not s:
            self.hyst_max_field = None
        else:
            try:
                v = float(s)
            except ValueError:
                self._showerror("Error", "Must be a number.")
                return
            if v <= 0:
                self._showerror("Error", "Must be > 0.")
                return
            self.hyst_max_field = v
        self.hyst_refresh_plot()


def main():
    root = tk.Tk()
    # Force l'encodage systeme de Tcl a utf-8 - meme correctif que
    # AMS_Py/STARpaleomag_Py (demande explicite utilisateur "l'appli
    # installee ne fonctionne pas tres bien, par exemple probleme de
    # texte Latin lors de l'importation. Pas de pb depuis le terminal") :
    # Tcl/Tk devine son "system encoding" depuis LANG/LC_ALL au
    # demarrage - absent pour un .app lance depuis le Finder/Dock
    # (contrairement a un Terminal, qui herite la locale du shell).
    root.tk.call("encoding", "system", "utf-8")
    StereoUtilsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
