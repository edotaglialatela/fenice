"""Simulation setup dialog (PyQt6).

Collects every simulation parameter in four tabs and validates them; the result
(:attr:`SetupDialog.sim_params`) is written as a config file for the C++ solver
by :mod:`gui.run`.

* **Configuration** - mesh, thermodynamics, BCs, numerical schemes, output, MPI.
* **Immersed Boundaries** - IBM cylinder / sphere geometry.
* **Solution Mapping** - hot start from a checkpoint ``.npz`` (grid metadata is
  read from the file; legacy files without it ask for the source Nx / Ny).
* **AMR** - block-structured adaptive mesh refinement settings.

The test-case list and each case's defaults come from the solver itself
(:func:`gui.solver.case_defaults`).
"""
import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QFrame,
                             QInputDialog,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
                             QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from gui.solver import ROOT_DIR, available_cases, case_defaults, save_case
from gui.theme import COLORS

_BOOL_KEYS = {"use_viscous", "use_bodyforce", "resume", "use_mpi", "use_ibm", "use_muscl", "use_hot_start", "use_amr"}


def _gui_values(params):
    """Solver config strings -> values the dialog setters expect (bools, BC labels)."""
    labels = {v: k for k, v in SetupDialog._BC_LABELS.items()}
    out = {}
    for key, value in params.items():
        if key in _BOOL_KEYS:
            out[key] = value.strip().lower() in ("1", "true", "yes", "on")
        elif key.startswith("bc_"):
            out[key] = labels.get(value, value)
        else:
            out[key] = value
    return out


class SetupDialog(QDialog):
    """PyQt6 dialog for configuring a simulation run.

    Presents a three-tab form:

    * **Configuration** — mesh dimensions, thermodynamics (ideal/stiffened gas),
      diffusive physics (Navier-Stokes), body forces, boundary conditions,
      numerical schemes (MUSCL, Riemann solver, time integration), output
      settings, and MPI parallelism.
    * **Immersed Boundaries** — enable/configure the IBM cylinder or sphere.
    * **Solution Mapping** — enable hot-start from a coarse ``.npz`` checkpoint;
      auto-detects source grid dimensions from file metadata or falls back to
      manual Nx/Ny entry for legacy files.

    On acceptance, :attr:`sim_params` is populated with validated, typed values
    ready to pass to :meth:`Run._setup_from_params`.

    Attributes
    ----------
    sim_params : dict
        Populated by :meth:`validate_and_accept`; empty until the dialog is
        accepted.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fenice — Simulation Setup")
        self.setMinimumWidth(960)
        self.setMinimumHeight(660)

        master_layout = QHBoxLayout(self)
        master_layout.setContentsMargins(20, 20, 20, 20)
        master_layout.setSpacing(20)

        self.tabs = QTabWidget()
        self.config_tab   = QWidget()
        self.ibm_tab      = QWidget()
        self.hotstart_tab = QWidget()
        self.amr_tab      = QWidget()
        self.tabs.addTab(self.config_tab,   "Configuration")
        self.tabs.addTab(self.ibm_tab,      "Immersed Boundaries")
        self.tabs.addTab(self.hotstart_tab, "Solution Mapping")
        self.tabs.addTab(self.amr_tab,      "AMR")

        self._setup_config_tab()
        self._setup_ibm_tab()
        self._setup_solution_mapping_tab()
        self._setup_amr_tab()

        main_right_layout = QVBoxLayout()
        main_right_layout.setSpacing(14)
        main_right_layout.addWidget(self._build_header())
        main_right_layout.addWidget(self.tabs, stretch=1)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        ok_btn = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText("Run Simulation")
        ok_btn.setObjectName("primaryButton")
        cancel_btn = self.buttons.button(QDialogButtonBox.StandardButton.Cancel)
        cancel_btn.setObjectName("secondaryButton")
        save_case_btn = self.buttons.addButton("Save as case…", QDialogButtonBox.ButtonRole.ActionRole)
        save_case_btn.setObjectName("secondaryButton")
        save_case_btn.setToolTip("Save this configuration as a new test case under cases/")
        save_case_btn.clicked.connect(self._save_as_case)
        self.buttons.accepted.connect(self.validate_and_accept)
        self.buttons.rejected.connect(self.reject)
        main_right_layout.addWidget(self.buttons)

        master_layout.addLayout(main_right_layout, stretch=1)
        master_layout.addWidget(self._build_logo_panel())

        self.sim_params = {}

        # Connect signals after all widgets have been created
        self.bc_left.currentTextChanged.connect(self._update_bc_inputs)
        self.bc_right.currentTextChanged.connect(self._update_bc_inputs)
        self.bc_bottom.currentTextChanged.connect(self._update_bc_inputs)
        self.bc_top.currentTextChanged.connect(self._update_bc_inputs)
        self.bc_front.currentTextChanged.connect(self._update_bc_inputs)
        self.bc_back.currentTextChanged.connect(self._update_bc_inputs)
        self.input_nz.textChanged.connect(self._update_dim_inputs)
        self.input_nz.textChanged.connect(self._update_hot_start_target_info)
        
        self.check_viscous.toggled.connect(self._toggle_viscous_inputs)
        self.check_bodyforce.toggled.connect(self._toggle_bodyforce_inputs)
        self.check_mpi.toggled.connect(self._toggle_mpi_inputs)
        self.check_ibm.toggled.connect(self._toggle_ibm_inputs)
        
        self.eos_combo.currentTextChanged.connect(self._update_eos_ui)
        self.combo_testcase.currentTextChanged.connect(self._apply_test_case_defaults)

        self.check_muscl.toggled.connect(self._toggle_muscl_inputs)
        self.combo_time.currentTextChanged.connect(self._toggle_bdf2_inputs)

        self.check_hot_start.toggled.connect(self._toggle_hot_start_inputs)
        self.btn_hot_start_browse.clicked.connect(self._browse_hot_start_file)
        self.input_nx.textChanged.connect(self._update_hot_start_target_info)
        self.input_ny.textChanged.connect(self._update_hot_start_target_info)

        self.check_amr.toggled.connect(self._toggle_amr_inputs)

        self._apply_initial_defaults()

    def _build_header(self):
        """Title block above the tab strip (app name + one-line purpose)."""
        header = QWidget()
        header.setObjectName("appHeader")
        h = QVBoxLayout(header)
        h.setContentsMargins(2, 0, 2, 2)
        h.setSpacing(2)
        title = QLabel("Fenice")
        title.setObjectName("appTitle")
        subtitle = QLabel("Compressible finite-volume simulation — run configuration")
        subtitle.setObjectName("appSubtitle")
        h.addWidget(title)
        h.addWidget(subtitle)
        return header

    def _build_logo_panel(self):
        """The brand painting, matted and captioned like a gallery card."""
        panel = QWidget()
        panel.setObjectName("logoPanel")
        v = QVBoxLayout(panel)
        v.setContentsMargins(16, 16, 16, 14)
        v.setSpacing(12)

        self.image_label = QLabel()
        self.image_label.setObjectName("logoImage")
        image_path = os.path.join(ROOT_DIR, "aesthetics", "logo.png")
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(320, 560, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText("Logo not found")
            self.image_label.setMinimumSize(320, 480)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.image_label, stretch=1)

        caption = QLabel("Fenice")
        caption.setObjectName("logoCaption")
        caption_sub = QLabel("Euler · Navier–Stokes · MUSCL · HLLC · IBM · AMR")
        caption_sub.setObjectName("logoCaptionSub")
        caption_sub.setWordWrap(True)
        v.addWidget(caption)
        v.addWidget(caption_sub)
        return panel

    def _setup_config_tab(self):
        layout = QVBoxLayout(self.config_tab)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # 0. Test case
        g0 = QGroupBox("0. Initial Problem")
        f0 = QFormLayout()
        self.combo_testcase = QComboBox()
        self.combo_testcase.addItems(available_cases())
        f0.addRow("Test case:", self.combo_testcase)
        g0.setLayout(f0)
        scroll_layout.addWidget(g0)

        # 1. Mesh
        g1 = QGroupBox("1. Spatial Domain")
        f1 = QFormLayout()
        self.input_Lx = QLineEdit("1.0");  self.input_Ly = QLineEdit("1.0")
        self.input_nx = QLineEdit("100");  self.input_ny = QLineEdit("100")
        self.input_Lz = QLineEdit("1.0");  self.input_nz = QLineEdit("1")
        f1.addRow("Lx [m]:", self.input_Lx);  f1.addRow("Ly [m]:", self.input_Ly)
        f1.addRow("Nx:",     self.input_nx);  f1.addRow("Ny:",     self.input_ny)
        f1.addRow("Nz (1 = 2-D):", self.input_nz)
        self.label_Lz = QLabel("Lz [m]:")
        f1.addRow(self.label_Lz, self.input_Lz)
        g1.setLayout(f1)
        scroll_layout.addWidget(g1)

        # 2. Thermodynamics
        g2 = QGroupBox("2. Thermodynamics")
        f2 = QFormLayout()
        self.eos_combo = QComboBox()
        self.eos_combo.addItems(["Ideal Gas", "Stiffened Gas"])
        self.input_gamma = QLineEdit("1.4")
        self.input_p_inf = QLineEdit("0.0")
        self.label_p_inf = QLabel("p_inf [Pa]:")
        f2.addRow("EOS model:", self.eos_combo)
        f2.addRow("Gamma:",     self.input_gamma)
        f2.addRow(self.label_p_inf, self.input_p_inf)
        g2.setLayout(f2)
        scroll_layout.addWidget(g2)

        # 3. Diffusive Physics
        g2b = QGroupBox("3. Diffusive Physics")
        f2b = QFormLayout()
        self.check_viscous = QCheckBox("Enable Viscous Fluxes (Navier-Stokes)")
        self.input_mu = QLineEdit("1e-3")
        self.input_pr = QLineEdit("0.72")
        self.label_mu = QLabel("Dynamic viscosity (mu):")
        self.label_pr = QLabel("Prandtl number (Pr):")
        f2b.addRow("", self.check_viscous)
        f2b.addRow(self.label_mu, self.input_mu)
        f2b.addRow(self.label_pr, self.input_pr)
        g2b.setLayout(f2b)
        scroll_layout.addWidget(g2b)

        # 4. Body Forces
        g2c = QGroupBox("4. Body Forces")
        f2c = QFormLayout()
        self.check_bodyforce = QCheckBox("Enable Body Forces (e.g. Gravity)")
        self.input_gx = QLineEdit("0.0")
        self.input_gy = QLineEdit("-9.81")
        self.input_gz = QLineEdit("0.0")
        self.label_gx = QLabel("Acceleration X (gx) [m/s²]:")
        self.label_gy = QLabel("Acceleration Y (gy) [m/s²]:")
        self.label_gz = QLabel("Acceleration Z (gz) [m/s²]:")
        f2c.addRow("", self.check_bodyforce)
        f2c.addRow(self.label_gx, self.input_gx)
        f2c.addRow(self.label_gy, self.input_gy)
        f2c.addRow(self.label_gz, self.input_gz)
        g2c.setLayout(f2c)
        scroll_layout.addWidget(g2c)

        # 5. Boundary Conditions
        g3 = QGroupBox("5. Boundary Conditions")
        f3 = QFormLayout()
        bc_opts = ["Inviscid Wall (Slip)", "Viscous Wall (No-Slip)", "Moving Wall",
                   "Velocity Inlet", "Mass Flow Inlet", "Pressure Outlet",
                   "Transmissive", "Non-Reflecting Outlet", "Periodic"]
        self.bc_left = QComboBox(); self.bc_left.addItems(bc_opts)
        self.bc_right = QComboBox(); self.bc_right.addItems(bc_opts)
        self.bc_bottom = QComboBox(); self.bc_bottom.addItems(bc_opts)
        self.bc_top = QComboBox(); self.bc_top.addItems(bc_opts)
        f3.addRow("Left:", self.bc_left); f3.addRow("Right:", self.bc_right)
        f3.addRow("Bottom:", self.bc_bottom); f3.addRow("Top:", self.bc_top)
        self.bc_front = QComboBox(); self.bc_front.addItems(bc_opts)
        self.bc_back = QComboBox(); self.bc_back.addItems(bc_opts)
        self.label_bc_front = QLabel("Front (z = 0):"); self.label_bc_back = QLabel("Back (z = Lz):")
        f3.addRow(self.label_bc_front, self.bc_front); f3.addRow(self.label_bc_back, self.bc_back)
        self.label_u_in = QLabel("Inlet Velocity (u_in) [m/s]:"); self.input_u_in = QLineEdit("0.0")
        self.label_rho_in = QLabel("Inlet Density (rho_in) [kg/m³]:"); self.input_rho_in = QLineEdit("1.0")
        self.label_p_out = QLabel("Outlet Pressure (p_out) [Pa]:"); self.input_p_out = QLineEdit("1.0")
        f3.addRow(self.label_u_in, self.input_u_in); f3.addRow(self.label_rho_in, self.input_rho_in)
        f3.addRow(self.label_p_out, self.input_p_out)
        g3.setLayout(f3)
        scroll_layout.addWidget(g3)

        # 6. Numerical Schemes
        g4 = QGroupBox("6. Numerical Schemes")
        f4 = QFormLayout()
        self.combo_riemann = QComboBox()
        self.combo_riemann.addItems(["HLLC", "HLLC (Low-Mach Thornber)", "HLLC (Preconditioned)"])
        self.combo_time = QComboBox()
        self.combo_time.addItems(["Explicit Euler", "SSP Runge-Kutta 2", "Runge-Kutta 4", "BDF2 Dual Time"])
        self.input_cfl = QLineEdit("0.5")
        self.check_muscl = QCheckBox("Enable 2nd-Order Spatial (MUSCL)")
        self.combo_limiter = QComboBox()
        self.combo_limiter.addItems(["Van Leer", "MC", "Koren", "Minmod"])
        self.label_limiter = QLabel("Slope Limiter:")
        f4.addRow("", self.check_muscl)
        f4.addRow(self.label_limiter, self.combo_limiter)
        f4.addRow("Riemann solver:", self.combo_riemann)
        f4.addRow("Time integration:", self.combo_time)
        f4.addRow("CFL number:", self.input_cfl)
        self.label_dt_phys   = QLabel("Physical dt [s]:")
        self.input_dt_phys   = QLineEdit("0.1")
        self.label_max_inner = QLabel("Max inner iterations:")
        self.input_max_inner = QLineEdit("50")
        self.label_inner_tol = QLabel("Inner tol (relative):")
        self.input_inner_tol = QLineEdit("1e-3")
        f4.addRow(self.label_dt_phys,   self.input_dt_phys)
        f4.addRow(self.label_max_inner, self.input_max_inner)
        f4.addRow(self.label_inner_tol, self.input_inner_tol)
        g4.setLayout(f4)
        scroll_layout.addWidget(g4)

        # 7. Simulation & Output
        g5 = QGroupBox("7. Simulation & Output")
        f5 = QFormLayout()
        self.input_t_end = QLineEdit("0.005"); self.input_save_interval = QLineEdit("1")
        self.check_resume = QCheckBox("Resume from Checkpoint (.npz)")
        f5.addRow("End time [s]:", self.input_t_end)
        f5.addRow("Save every N steps:", self.input_save_interval)
        f5.addRow("", self.check_resume)
        g5.setLayout(f5)
        scroll_layout.addWidget(g5)

        # 8. Parallel Computation (MPI)
        g6 = QGroupBox("8. Parallel Computation (MPI)")
        f6 = QFormLayout()
        self.check_mpi = QCheckBox("Enable MPI Domain Decomposition")
        self.label_ncpus = QLabel("Number of CPUs (Ncpus):")
        self.spin_ncpus = QSpinBox(); self.spin_ncpus.setRange(2, 512); self.spin_ncpus.setValue(2)
        f6.addRow("", self.check_mpi)
        f6.addRow(self.label_ncpus, self.spin_ncpus)
        g6.setLayout(f6)
        scroll_layout.addWidget(g6)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

    def _setup_ibm_tab(self):
        layout = QVBoxLayout(self.ibm_tab)
        g_ibm = QGroupBox("Immersed Boundary Parameters")
        f_ibm = QFormLayout()
        
        self.check_ibm = QCheckBox("Enable Immersed Boundary Method")
        self.combo_shape = QComboBox(); self.combo_shape.addItems(["Cylinder", "Sphere", "NACA 0012", "Airfoil"])
        self.combo_shape.setEditable(True)   # any "NACA dddd"; Airfoil reads the case's ibm_file
        self.input_radius = QLineEdit("0.1"); self.input_cx = QLineEdit("0.5"); self.input_cy = QLineEdit("0.5")
        self.input_cz = QLineEdit("0.5"); self.l_cz = QLabel("CZ (3-D):")
        
        self.l_shape = QLabel("Shape:"); self.l_rad = QLabel("Radius:"); self.l_cx = QLabel("CX:"); self.l_cy = QLabel("CY:")
        
        f_ibm.addRow("", self.check_ibm)
        f_ibm.addRow(self.l_shape, self.combo_shape)
        f_ibm.addRow(self.l_rad, self.input_radius)
        f_ibm.addRow(self.l_cx, self.input_cx)
        f_ibm.addRow(self.l_cy, self.input_cy)
        f_ibm.addRow(self.l_cz, self.input_cz)
        g_ibm.setLayout(f_ibm)
        layout.addWidget(g_ibm)
        layout.addStretch()

    def _setup_solution_mapping_tab(self):
        layout = QVBoxLayout(self.hotstart_tab)
        g = QGroupBox("Hot Start / Solution Mapping")
        g_layout = QVBoxLayout()

        self.check_hot_start = QCheckBox("Enable Solution Mapping (Hot Start)")
        g_layout.addWidget(self.check_hot_start)

        sep = QFrame()
        sep.setObjectName("hsep")
        sep.setFixedHeight(1)
        g_layout.addWidget(sep)

        # File picker row
        file_row = QHBoxLayout()
        self.input_hot_start_file = QLineEdit()
        self.input_hot_start_file.setReadOnly(True)
        self.input_hot_start_file.setPlaceholderText("Select a checkpoint .npz file…")
        self.btn_hot_start_browse = QPushButton("Browse…")
        file_row.addWidget(QLabel("Source checkpoint (.npz):"))
        file_row.addWidget(self.input_hot_start_file, stretch=1)
        file_row.addWidget(self.btn_hot_start_browse)
        g_layout.addLayout(file_row)

        self.lbl_hot_start_source_info = QLabel("Source info:   —")
        self.lbl_hot_start_source_info.setObjectName("fieldNote")
        self.lbl_hot_start_target_info = QLabel("Target mesh:   —")
        self.lbl_hot_start_target_info.setObjectName("fieldNote")
        g_layout.addWidget(self.lbl_hot_start_source_info)
        g_layout.addWidget(self.lbl_hot_start_target_info)

        # Fallback fields shown only when the file has no embedded metadata
        fallback_row = QHBoxLayout()
        self.lbl_src_nx = QLabel("Source Nx:")
        self.lbl_src_ny = QLabel("Ny:")
        self.input_src_nx = QLineEdit(); self.input_src_nx.setFixedWidth(70)
        self.input_src_ny = QLineEdit(); self.input_src_ny.setFixedWidth(70)
        fallback_row.addWidget(self.lbl_src_nx)
        fallback_row.addWidget(self.input_src_nx)
        fallback_row.addWidget(self.lbl_src_ny)
        fallback_row.addWidget(self.input_src_ny)
        fallback_row.addStretch()
        g_layout.addLayout(fallback_row)
        self._fallback_widgets = [
            self.lbl_src_nx, self.input_src_nx, self.lbl_src_ny, self.input_src_ny]
        for w in self._fallback_widgets:
            w.setVisible(False)

        note = QLabel(
            "ℹ  Lx / Ly of source and target must match.\n"
            "   IBM mask is re-applied on the fine grid after interpolation."
        )
        note.setObjectName("fieldNote")
        note.setWordWrap(True)
        g_layout.addWidget(note)

        g.setLayout(g_layout)
        layout.addWidget(g)
        layout.addStretch()

        # Store references to sub-widgets for toggling
        self._hot_start_sub_widgets = [
            self.input_hot_start_file, self.btn_hot_start_browse,
            self.lbl_hot_start_source_info, self.lbl_hot_start_target_info,
        ]

    def _setup_amr_tab(self):
        layout = QVBoxLayout(self.amr_tab)
        g = QGroupBox("Adaptive Mesh Refinement (AMR)")
        f = QFormLayout()

        self.check_amr = QCheckBox("Enable AMR")
        f.addRow("", self.check_amr)

        self.input_amr_max_level     = QLineEdit("2")
        self.input_amr_patch_nx      = QLineEdit("8")
        self.input_amr_patch_ny      = QLineEdit("8")
        self.input_amr_refine_thr    = QLineEdit("0.3")
        self.input_amr_derefine_thr  = QLineEdit("0.1")
        self.input_amr_refine_every  = QLineEdit("10")
        self.input_amr_init_levels   = QLineEdit("0")
        self.input_amr_buffer_layers = QLineEdit("1")

        self.lbl_amr_max_level    = QLabel("Max refinement levels:")
        self.lbl_amr_patch_nx     = QLabel("Cells per patch (Nx):")
        self.lbl_amr_patch_ny     = QLabel("Cells per patch (Ny):")
        self.lbl_amr_refine_thr   = QLabel("Refine threshold (η):")
        self.lbl_amr_derefine_thr = QLabel("Derefine threshold (η):")
        self.lbl_amr_refine_every = QLabel("Refine every N steps:")
        self.lbl_amr_init_levels  = QLabel("Initial uniform levels:")
        self.lbl_amr_buffer_layers = QLabel("Buffer rings (0=tight):")

        for lbl, widget in [
            (self.lbl_amr_max_level,     self.input_amr_max_level),
            (self.lbl_amr_patch_nx,      self.input_amr_patch_nx),
            (self.lbl_amr_patch_ny,      self.input_amr_patch_ny),
            (self.lbl_amr_refine_thr,    self.input_amr_refine_thr),
            (self.lbl_amr_derefine_thr,  self.input_amr_derefine_thr),
            (self.lbl_amr_refine_every,  self.input_amr_refine_every),
            (self.lbl_amr_init_levels,   self.input_amr_init_levels),
            (self.lbl_amr_buffer_layers, self.input_amr_buffer_layers),
        ]:
            f.addRow(lbl, widget)

        note = QLabel(
            "ℹ  AMR subdivides patches with large density gradients.\n"
            "   Nx, Ny must be multiples of the patch size.\n"
            "   Subcycling: fine patches take 2 sub-steps per coarse step."
        )
        note.setObjectName("fieldNote")
        note.setWordWrap(True)
        f.addRow(note)

        g.setLayout(f)
        layout.addWidget(g)
        layout.addStretch()

        self._amr_sub_widgets = [
            self.lbl_amr_max_level,    self.input_amr_max_level,
            self.lbl_amr_patch_nx,     self.input_amr_patch_nx,
            self.lbl_amr_patch_ny,      self.input_amr_patch_ny,
            self.lbl_amr_refine_thr,    self.input_amr_refine_thr,
            self.lbl_amr_derefine_thr,  self.input_amr_derefine_thr,
            self.lbl_amr_refine_every,  self.input_amr_refine_every,
            self.lbl_amr_init_levels,   self.input_amr_init_levels,
            self.lbl_amr_buffer_layers, self.input_amr_buffer_layers,
        ]

    def _toggle_amr_inputs(self, checked):
        for w in self._amr_sub_widgets:
            w.setEnabled(checked)

    def _update_bc_inputs(self, *args):
        bcs = [self.bc_left.currentText(), self.bc_right.currentText(),
               self.bc_bottom.currentText(), self.bc_top.currentText()]
        if self._is_3d():
            bcs += [self.bc_front.currentText(), self.bc_back.currentText()]
        needs_uin   = ("Velocity Inlet" in bcs) or ("Mass Flow Inlet" in bcs) or ("Moving Wall" in bcs)
        needs_rhoin = ("Mass Flow Inlet" in bcs)
        needs_pout  = ("Pressure Outlet" in bcs)
        self.label_u_in.setVisible(needs_uin); self.input_u_in.setVisible(needs_uin)
        self.label_rho_in.setVisible(needs_rhoin); self.input_rho_in.setVisible(needs_rhoin)
        self.label_p_out.setVisible(needs_pout); self.input_p_out.setVisible(needs_pout)

    def _is_3d(self):
        try:
            return int(self.input_nz.text()) > 1
        except ValueError:
            return False

    def _update_dim_inputs(self, *args):
        """Show the z-direction inputs (Lz, front/back BCs, gz, IBM cz) only for 3-D meshes."""
        three_d = self._is_3d()
        for w in (self.label_Lz, self.input_Lz, self.label_bc_front, self.bc_front,
                  self.label_bc_back, self.bc_back):
            w.setVisible(three_d)
        body = three_d and self.check_bodyforce.isChecked()
        self.label_gz.setVisible(body); self.input_gz.setVisible(body)
        ibm = three_d and self.check_ibm.isChecked()
        self.l_cz.setVisible(ibm); self.input_cz.setVisible(ibm)
        self._update_bc_inputs()

    def _toggle_viscous_inputs(self, checked):
        self.label_mu.setVisible(checked); self.input_mu.setVisible(checked)
        self.label_pr.setVisible(checked); self.input_pr.setVisible(checked)

    def _toggle_bodyforce_inputs(self, checked):
        self.label_gx.setVisible(checked); self.input_gx.setVisible(checked)
        self.label_gy.setVisible(checked); self.input_gy.setVisible(checked)
        self._update_dim_inputs()

    def _toggle_mpi_inputs(self, checked):
        self.label_ncpus.setVisible(checked); self.spin_ncpus.setVisible(checked)

    def _toggle_ibm_inputs(self, checked):
        self.l_shape.setVisible(checked); self.combo_shape.setVisible(checked)
        self.l_rad.setVisible(checked); self.input_radius.setVisible(checked)
        self.l_cx.setVisible(checked); self.input_cx.setVisible(checked)
        self.l_cy.setVisible(checked); self.input_cy.setVisible(checked)
        self._update_dim_inputs()

    def _toggle_muscl_inputs(self, checked):
        self.label_limiter.setVisible(checked)
        self.combo_limiter.setVisible(checked)

    def _toggle_bdf2_inputs(self, text=""):
        is_bdf2 = (text == "BDF2 Dual Time")
        for w in (self.label_dt_phys, self.input_dt_phys,
                  self.label_max_inner, self.input_max_inner,
                  self.label_inner_tol, self.input_inner_tol):
            w.setVisible(is_bdf2)

    def _toggle_hot_start_inputs(self, checked):
        for w in self._hot_start_sub_widgets:
            w.setEnabled(checked)

    def _browse_hot_start_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select source checkpoint", "", "NumPy archives (*.npz)")
        if not path:
            return
        self.input_hot_start_file.setText(path)
        # Probe the file for grid metadata
        try:
            data = np.load(path, allow_pickle=False)
            has_meta = 'nx' in data and 'ny' in data and 'Lx' in data and 'Ly' in data
            if has_meta:
                nx_s = int(data['nx']); ny_s = int(data['ny'])
                Lx_s = float(data['Lx']); Ly_s = float(data['Ly'])
                self.lbl_hot_start_source_info.setText(
                    f"Source info:   {nx_s} × {ny_s},  Lx = {Lx_s},  Ly = {Ly_s}")
                self.lbl_hot_start_source_info.setStyleSheet(f"color: {COLORS['success']};")
                self.input_src_nx.setText(str(nx_s))
                self.input_src_ny.setText(str(ny_s))
            else:
                n_cells = data['U'].shape[0]
                self.lbl_hot_start_source_info.setText(
                    f"⚠  No metadata — enter source Nx / Ny below  "
                    f"(file has {n_cells} cells total).")
                self.lbl_hot_start_source_info.setStyleSheet(f"color: {COLORS['warning']};")
            for w in self._fallback_widgets:
                w.setVisible(not has_meta)
        except Exception as e:
            self.lbl_hot_start_source_info.setText(f"⚠  Could not read file: {e}")
            self.lbl_hot_start_source_info.setStyleSheet(f"color: {COLORS['error']};")

    def _update_hot_start_target_info(self):
        try:
            nx = int(self.input_nx.text())
            ny = int(self.input_ny.text())
            nz = int(self.input_nz.text() or 1)
            dims = f"{nx} × {ny}" + (f" × {nz}" if nz > 1 else "")
            self.lbl_hot_start_target_info.setText(f"Target mesh:   {dims}")
        except ValueError:
            self.lbl_hot_start_target_info.setText("Target mesh:   —")

    def _update_eos_ui(self, text):
        visible = (text == "Stiffened Gas")
        self.label_p_inf.setVisible(visible); self.input_p_inf.setVisible(visible)

    _BC_LABELS = {
        "Inviscid Wall (Slip)": "wall", "Viscous Wall (No-Slip)": "no_slip", "Moving Wall": "moving_wall", 
        "Velocity Inlet": "vel_inlet", "Mass Flow Inlet": "massflow_inlet", 
        "Pressure Outlet": "outlet",
        "Transmissive": "transmissive", 
        "Non-Reflecting Outlet": "nonreflecting",
        "Periodic": "periodic",
    }

    def _apply_test_case_defaults(self, name):
        defaults = _gui_values(case_defaults(name))
        def set_combo(combo, text):
            idx = combo.findText(str(text))
            if idx >= 0: combo.setCurrentIndex(idx)
            elif combo.isEditable(): combo.setEditText(str(text))
        setters = {
            'Lx': lambda v: self.input_Lx.setText(str(v)), 'Ly': lambda v: self.input_Ly.setText(str(v)),
            'nx': lambda v: self.input_nx.setText(str(v)), 'ny': lambda v: self.input_ny.setText(str(v)),
            'nz': lambda v: self.input_nz.setText(str(v)), 'Lz': lambda v: self.input_Lz.setText(str(v)),
            'bc_front': lambda v: set_combo(self.bc_front, v), 'bc_back': lambda v: set_combo(self.bc_back, v),
            'gz': lambda v: self.input_gz.setText(str(v)), 'ibm_cz': lambda v: self.input_cz.setText(str(v)),
            'eos_type': lambda v: set_combo(self.eos_combo, v), 'p_inf': lambda v: self.input_p_inf.setText(str(v)),
            'gamma': lambda v: self.input_gamma.setText(str(v)), 'mu': lambda v: self.input_mu.setText(str(v)),
            'pr': lambda v: self.input_pr.setText(str(v)), 'cfl': lambda v: self.input_cfl.setText(str(v)),
            't_end': lambda v: self.input_t_end.setText(str(v)), 'save_interval': lambda v: self.input_save_interval.setText(str(v)),
            'use_viscous': lambda v: self.check_viscous.setChecked(bool(v)), 'bc_left': lambda v: set_combo(self.bc_left, v),
            'bc_right': lambda v: set_combo(self.bc_right, v), 'bc_bottom': lambda v: set_combo(self.bc_bottom, v),
            'bc_top': lambda v: set_combo(self.bc_top, v), 'riemann_solver': lambda v: set_combo(self.combo_riemann, v),
            'time_scheme': lambda v: set_combo(self.combo_time, v), 'u_in': lambda v: self.input_u_in.setText(str(v)),
            'rho_in': lambda v: self.input_rho_in.setText(str(v)), 'p_out': lambda v: self.input_p_out.setText(str(v)),
            'use_bodyforce': lambda v: self.check_bodyforce.setChecked(bool(v)), 'gx': lambda v: self.input_gx.setText(str(v)),
            'gy': lambda v: self.input_gy.setText(str(v)), 'resume': lambda v: self.check_resume.setChecked(bool(v)),
            'use_mpi': lambda v: self.check_mpi.setChecked(bool(v)), 'n_cpus': lambda v: self.spin_ncpus.setValue(int(v)),
            'use_ibm': lambda v: self.check_ibm.setChecked(bool(v)), 'ibm_shape': lambda v: set_combo(self.combo_shape, v),
            'ibm_radius': lambda v: self.input_radius.setText(str(v)), 'ibm_cx': lambda v: self.input_cx.setText(str(v)),
            'ibm_cy': lambda v: self.input_cy.setText(str(v)),
            'use_muscl': lambda v: self.check_muscl.setChecked(bool(v)),
            'muscl_limiter': lambda v: set_combo(self.combo_limiter, v),
            'dt_phys':        lambda v: self.input_dt_phys.setText(str(v)),
            'max_inner_iter': lambda v: self.input_max_inner.setText(str(v)),
            'inner_tol_rel':  lambda v: self.input_inner_tol.setText(str(v)),
            'use_amr':                lambda v: self.check_amr.setChecked(bool(v)),
            'amr_max_level':          lambda v: self.input_amr_max_level.setText(str(v)),
            'amr_patch_nx':           lambda v: self.input_amr_patch_nx.setText(str(v)),
            'amr_patch_ny':           lambda v: self.input_amr_patch_ny.setText(str(v)),
            'amr_refine_threshold':   lambda v: self.input_amr_refine_thr.setText(str(v)),
            'amr_derefine_threshold': lambda v: self.input_amr_derefine_thr.setText(str(v)),
            'amr_refine_every':       lambda v: self.input_amr_refine_every.setText(str(v)),
            'amr_init_levels':        lambda v: self.input_amr_init_levels.setText(str(v)),
            'amr_buffer_layers':      lambda v: self.input_amr_buffer_layers.setText(str(v)),
        }
        for key, value in defaults.items():
            if key in setters: setters[key](value)

    def _apply_initial_defaults(self):
        self._apply_test_case_defaults(self.combo_testcase.currentText())
        self._toggle_viscous_inputs(self.check_viscous.isChecked())
        self._toggle_bodyforce_inputs(self.check_bodyforce.isChecked())
        self._toggle_mpi_inputs(self.check_mpi.isChecked())
        self._toggle_ibm_inputs(self.check_ibm.isChecked())
        self._update_bc_inputs()
        self._update_eos_ui(self.eos_combo.currentText())
        self._toggle_muscl_inputs(self.check_muscl.isChecked())
        self._toggle_bdf2_inputs(self.combo_time.currentText())
        self._toggle_hot_start_inputs(self.check_hot_start.isChecked())
        self._update_hot_start_target_info()
        self._toggle_amr_inputs(self.check_amr.isChecked())
        self._update_dim_inputs()

    def _collect_params(self):
        """Read every field into ``self.sim_params``; show an error and return False if invalid."""
        try:
            self.sim_params = {
                'test_case': self.combo_testcase.currentText(),
                'Lx': float(self.input_Lx.text()), 'Ly': float(self.input_Ly.text()),
                'nx': int(self.input_nx.text()), 'ny': int(self.input_ny.text()),
                'nz': int(self.input_nz.text()), 'Lz': float(self.input_Lz.text()),
                'bc_front': self._BC_LABELS[self.bc_front.currentText()],
                'bc_back': self._BC_LABELS[self.bc_back.currentText()],
                'gz': float(self.input_gz.text()), 'ibm_cz': float(self.input_cz.text()),
                'eos_type': self.eos_combo.currentText(), 'gamma': float(self.input_gamma.text()),
                'p_inf': float(self.input_p_inf.text()) if self.eos_combo.currentText() == "Stiffened Gas" else 0.0,
                'bc_left': self._BC_LABELS[self.bc_left.currentText()],
                'bc_right': self._BC_LABELS[self.bc_right.currentText()],
                'bc_bottom': self._BC_LABELS[self.bc_bottom.currentText()],
                'bc_top': self._BC_LABELS[self.bc_top.currentText()],
                'riemann_solver': self.combo_riemann.currentText(),
                'time_scheme': self.combo_time.currentText(),
                'cfl': float(self.input_cfl.text()), 't_end': float(self.input_t_end.text()),
                'save_interval': int(self.input_save_interval.text()),
                'use_viscous': self.check_viscous.isChecked(), 'mu': float(self.input_mu.text()),
                'pr': float(self.input_pr.text()), 'u_in': float(self.input_u_in.text()),
                'rho_in': float(self.input_rho_in.text()), 'p_out': float(self.input_p_out.text()),
                'use_bodyforce': self.check_bodyforce.isChecked(), 'gx': float(self.input_gx.text()),
                'gy': float(self.input_gy.text()), 'resume': self.check_resume.isChecked(),
                'use_mpi': self.check_mpi.isChecked(), 'n_cpus': self.spin_ncpus.value(),
                'use_ibm': self.check_ibm.isChecked(), 'ibm_shape': self.combo_shape.currentText(),
                'ibm_radius': float(self.input_radius.text()), 'ibm_cx': float(self.input_cx.text()),
                'ibm_cy': float(self.input_cy.text()),
                'use_muscl': self.check_muscl.isChecked(),
                'muscl_limiter': self.combo_limiter.currentText(),
                'dt_phys':        float(self.input_dt_phys.text()),
                'max_inner_iter': int(self.input_max_inner.text()),
                'inner_tol_rel':  float(self.input_inner_tol.text()),
                'use_hot_start': self.check_hot_start.isChecked(),
                'hot_start_file': self.input_hot_start_file.text(),
                'hot_start_src_nx': int(self.input_src_nx.text()) if self.input_src_nx.text() else 0,
                'hot_start_src_ny': int(self.input_src_ny.text()) if self.input_src_ny.text() else 0,
                'use_amr':                self.check_amr.isChecked(),
                'amr_max_level':          int(self.input_amr_max_level.text()),
                'amr_patch_nx':           int(self.input_amr_patch_nx.text()),
                'amr_patch_ny':           int(self.input_amr_patch_ny.text()),
                'amr_refine_threshold':   float(self.input_amr_refine_thr.text()),
                'amr_derefine_threshold': float(self.input_amr_derefine_thr.text()),
                'amr_refine_every':       int(self.input_amr_refine_every.text()),
                'amr_init_levels':        int(self.input_amr_init_levels.text()),
                'amr_buffer_layers':      int(self.input_amr_buffer_layers.text()),
            }
            if self.sim_params['nz'] < 1:
                QMessageBox.critical(self, "Config Error", "Nz must be at least 1 (1 = 2-D).")
                return False
            if self.sim_params['nz'] > 1 and self.sim_params['use_amr']:
                QMessageBox.critical(self, "Config Error", "AMR is 2-D only: set Nz = 1 or disable AMR.")
                return False
            if self.sim_params['resume'] and self.sim_params['use_amr']:
                QMessageBox.critical(self, "Config Error",
                                     "Resume is not available with AMR (AMR runs write no checkpoint).")
                return False
            for a, b in (('bc_left', 'bc_right'), ('bc_bottom', 'bc_top'), ('bc_front', 'bc_back')):
                if (self.sim_params[a] == 'periodic') != (self.sim_params[b] == 'periodic'):
                    QMessageBox.critical(self, "Config Error",
                                         f"{a} / {b}: periodic must be set on both sides or neither.")
                    return False
            if self.sim_params['use_mpi']:
                if self.sim_params['n_cpus'] > max(self.sim_params['nx'], self.sim_params['ny']):
                    QMessageBox.critical(self, "Config Error", "Ncpus cannot exceed max(Nx, Ny).")
                    return False
            if self.sim_params['use_hot_start']:
                hs_path = self.sim_params['hot_start_file']
                if not hs_path or not os.path.isfile(hs_path):
                    QMessageBox.critical(self, "Config Error",
                                         "Solution Mapping is enabled but no valid .npz file was selected.")
                    return False
                if any(w.isVisible() for w in self._fallback_widgets):
                    nx_ov = self.sim_params['hot_start_src_nx']
                    ny_ov = self.sim_params['hot_start_src_ny']
                    if nx_ov <= 0 or ny_ov <= 0:
                        QMessageBox.critical(self, "Config Error",
                                             "Source Nx and Ny must be positive integers.")
                        return False
            return True
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please check all fields.")
            return False

    def validate_and_accept(self):
        if self._collect_params():
            self.accept()

    def _save_as_case(self):
        """Save the configured run as a new case folder (cases/<name>/case.toml)."""
        if not self._collect_params():
            return
        name, ok = QInputDialog.getText(self, "Save as case",
                                        "Name of the new case (it is written to cases/<name>/case.toml):",
                                        text=f"{self.combo_testcase.currentText()} (copy)")
        name = name.strip()
        if not ok or not name:
            return
        try:
            path = save_case(self.sim_params, name)
        except RuntimeError as exc:
            QMessageBox.critical(self, "Save as case", str(exc))
            return
        self._reload_case_list(select=name)
        QMessageBox.information(self, "Save as case",
                                f"Saved the case '{name}' to\n{path}\n\nEdit the file to change its initial "
                                "conditions; it now appears in the test-case list.")

    def _reload_case_list(self, select=None):
        current = select or self.combo_testcase.currentText()
        self.combo_testcase.blockSignals(True)
        self.combo_testcase.clear()
        self.combo_testcase.addItems(available_cases())
        self.combo_testcase.blockSignals(False)
        idx = self.combo_testcase.findText(current)
        self.combo_testcase.setCurrentIndex(max(idx, 0))
        self._apply_test_case_defaults(self.combo_testcase.currentText())
