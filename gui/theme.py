"""Application theme for the Fenice PyQt6 GUI.

A single, reusable Qt theme: a named colour palette, a Qt Style Sheet (QSS)
built from it, and :func:`apply_theme` to install both on a ``QApplication``.

Design intent — *research campus wayfinding*. A marine rail and warm porcelain
work surface make navigation easy to scan. Geometry, run, and results routes
keep distinct colors across the graph and its readouts; labels and status text
carry the same meaning for users who do not distinguish the colors.

Public API
----------
``COLORS``         dict of role -> hex string (the single source of truth).
``build_stylesheet(colors=COLORS)``  -> the full QSS string.
``apply_theme(app, colors=COLORS)``  install Fusion base + palette + font + QSS.
"""

from string import Template

from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication


# --- Palette -------------------------------------------------------------
# Roles, not raw colours, are referenced throughout the Qt surfaces.
COLORS = {
    # Porcelain work surfaces and the marine navigation rail
    "bg_window":     "#F2F1E9",
    "bg_surface":    "#FCFCF8",
    "bg_panel":      "#E7ECE6",
    "bg_sidebar":    "#173F50",
    "bg_sidebar_hover": "#245568",
    "sidebar_ink":   "#F6F6EF",
    "sidebar_muted": "#D1E0DD",
    "grid":          "#D8E1DA",
    "grid_major":    "#C8D5CD",

    # Hairlines
    "border":        "#CED8D1",
    "border_strong": "#A7B7AF",

    # Ink ramp
    "ink":           "#16333D",
    "ink_soft":      "#3B5961",
    "ink_muted":     "#506970",

    # Actions / focus
    "primary":         "#1B5361",
    "primary_hover":   "#286878",
    "primary_pressed": "#153F49",
    "primary_tint":    "#DCEAE6",
    "on_primary":      "#FFFFFF",
    "action":          "#B64E40",
    "action_hover":    "#A8463A",
    "action_pressed":  "#963F34",
    "on_action":       "#FFFFFF",

    # Typed routes: geometry, execution, results
    "route_geometry":  "#2F7864",
    "route_run":       "#8C5B12",
    "route_results":   "#4D6598",
    "module_solver":   "#B64E40",
    "module_monitor":  "#8C5B12",
    "module_results":  "#4D6598",

    # Semantic state
    "warning": "#83540B",
    "error":   "#A63C35",
    "success": "#286B50",

    # Disabled (contrast intentionally relaxed — non-interactive)
    "disabled_ink": "#899B99",
    "disabled_bg":  "#E9EDE7",

    # Solver log
    "console_bg":   "#142D37",
    "console_ink":  "#E6EEE9",
    "console_dim":  "#A2B7B5",

}


_QSS = Template(r"""
* { font-family: "Segoe UI", "Noto Sans", "Inter", sans-serif; font-size: 13px; }

QDialog, QWidget#rootSurface {
    background-color: $bg_window;
}
QWidget { color: $ink; }

QWidget#mainShell, QWidget#rootSurface { background-color: $bg_window; }
QWidget#moduleSidebar { background-color: $bg_sidebar; }
QLabel#sidebarTitle { color: $sidebar_ink; font-size: 17px; font-weight: 650; }
QLabel#sidebarNote { color: $sidebar_muted; font-size: 12px; }
QLabel#sidebarSection { color: $sidebar_muted; font-size: 11px; font-weight: 650; }
QLabel#canvasTitle { color: $ink; font-size: 18px; font-weight: 650; }
QLabel#canvasSubtitle { color: $ink_muted; font-size: 12px; }
QLabel#emptyStateTitle { color: $ink; font-size: 20px; font-weight: 650; }
QLabel#emptyStateBody { color: $ink_muted; font-size: 13px; }

/* ---- Header ------------------------------------------------------------ */
QWidget#appHeader { background: transparent; }
QLabel#appTitle {
    font-size: 19px;
    font-weight: 600;
    color: $ink;
    letter-spacing: 0.2px;
}
QLabel#appSubtitle {
    font-size: 12px;
    font-weight: 400;
    color: $ink_muted;
}

/* ---- Group boxes ------------------------------------------------------- */
QGroupBox {
    background-color: $bg_surface;
    border: 1px solid $border;
    border-radius: 8px;
    margin-top: 16px;
    padding: 16px 16px 12px 16px;
    font-weight: 600;             /* inherited by ::title; children reset below */
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 1px 6px;
    color: $primary;
    background-color: $bg_window;
}

/* ---- Labels ------------------------------------------------------------ */
QLabel { background: transparent; font-weight: 400; }
QLabel#fieldNote {
    color: $ink_muted;
    font-size: 12px;
}
QLabel#sectionTitle {
    color: $ink;
    font-size: 15px;
    font-weight: 600;
}
QLabel#subsectionTitle { color: $ink_soft; font-size: 13px; font-weight: 650; }
QLabel#statusPill { background-color: $primary_tint; color: $primary; border-radius: 8px; padding: 4px 9px; font-weight: 600; }
QLabel#note { color: $ink_muted; }
QLabel:disabled { color: $disabled_ink; }

/* ---- Text / number inputs --------------------------------------------- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: $bg_surface;
    border: 1px solid $border_strong;
    border-radius: 6px;
    padding: 5px 9px;
    color: $ink;
    font-weight: 400;
    selection-background-color: $primary;
    selection-color: $on_primary;
    min-height: 18px;
}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: $primary_hover;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 2px solid $primary;
    padding: 4px 8px;            /* compensate the +1px border to avoid reflow */
}
QLineEdit[readOnly="true"] {
    background-color: $bg_panel;
    color: $ink_muted;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    background-color: $disabled_bg;
    border-color: $border;
    color: $disabled_ink;
}

/* ---- Combo box --------------------------------------------------------- */
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid $border;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}
QComboBox QAbstractItemView {
    background-color: $bg_surface;
    border: 1px solid $border_strong;
    border-radius: 6px;
    padding: 4px;
    outline: none;
    selection-background-color: $primary_tint;
    selection-color: $ink;
}

/* ---- Check boxes (filled-square checked; zero-asset, unambiguous) ------ */
QCheckBox { spacing: 9px; background: transparent; font-weight: 400; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid $border_strong;
    border-radius: 4px;
    background-color: $bg_surface;
}
QCheckBox::indicator:hover { border-color: $primary_hover; }
QCheckBox::indicator:checked {
    background-color: $primary;
    border-color: $primary;
}
QCheckBox::indicator:checked:hover { background-color: $primary_hover; }
QCheckBox::indicator:disabled {
    border-color: $border;
    background-color: $disabled_bg;
}

/* ---- Tabs -------------------------------------------------------------- */
QTabWidget::pane {
    border: 1px solid $border;
    border-radius: 8px;
    top: -1px;
    background-color: $bg_window;
}
QTabBar { background: transparent; qproperty-drawBase: 0; }
QTabBar::tab {
    background: transparent;
    color: $ink_muted;
    padding: 9px 15px;
    margin: 1px 2px;
    border: 1px solid transparent;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    font-weight: 600;
}
QTabBar::tab:hover { color: $ink; background-color: $bg_panel; }
QTabBar::tab:selected {
    color: $primary;
    background-color: $bg_surface;
    border-color: $border;
    border-bottom-color: $bg_surface;
}
QTabBar::tab:focus { border: 2px solid $primary; }

/* ---- Buttons ----------------------------------------------------------- */
QPushButton {
    background-color: $bg_surface;
    border: 1px solid $border_strong;
    border-radius: 6px;
    padding: 8px 14px;
    min-height: 20px;
    color: $ink;
    font-weight: 600;
}
QPushButton#secondaryButton {
    background-color: $bg_panel;
    border-color: $border;
}
QPushButton#secondaryButton:hover { background-color: $primary_tint; }
QPushButton:hover { border-color: $primary_hover; color: $primary; }
QPushButton:pressed { background-color: $bg_panel; }
QPushButton:disabled {
    color: $disabled_ink;
    border-color: $border;
    background-color: $disabled_bg;
}
QPushButton#primaryButton {
    background-color: $primary;
    border: 1px solid $primary;
    color: $on_primary;
    padding: 8px 24px;
}
QPushButton#primaryButton:hover {
    background-color: $primary_hover;
    border-color: $primary_hover;
    color: $on_primary;
}
QPushButton#primaryButton:pressed { background-color: $primary_pressed; }
QPushButton#secondaryButton:disabled,
QPushButton#primaryButton:disabled,
QPushButton#actionButton:disabled {
    background-color: $disabled_bg;
    border-color: $border;
    color: $disabled_ink;
}
QPushButton#actionButton { background-color: $action; border-color: $action; color: $on_action; font-weight: 650; }
QPushButton#actionButton:hover { background-color: $action_hover; border-color: $action_hover; color: $on_action; }
QPushButton#actionButton:pressed { background-color: $action_pressed; }
QPushButton#sidebarButton { background-color: $bg_sidebar_hover; border-color: #426C78; color: $sidebar_ink; text-align: left; }
QPushButton#sidebarButton:hover { background-color: #2B6576; border-color: #6A929A; color: $sidebar_ink; }
QPushButton#sidebarPrimary { background-color: $action; border-color: $action; color: $on_action; font-weight: 650; }
QPushButton#sidebarPrimary:hover { background-color: $action_hover; border-color: $action_hover; color: $on_action; }
QPushButton#sidebarPrimary:disabled { background-color: #6D7E7E; border-color: #6D7E7E; color: #D7E0DD; }

/* ---- Module palette ---------------------------------------------------- */
QListWidget {
    background-color: $bg_surface;
    border: 1px solid $border;
    border-radius: 8px;
    padding: 4px;
    outline: none;
}
QListWidget::item {
    border-radius: 5px;
    padding: 8px 10px;
}
QListWidget::item:hover { background-color: $bg_panel; }
QListWidget::item:selected {
    background-color: $primary_tint;
    color: $primary;
}
QGraphicsView {
    border: 1px solid $border;
    border-radius: 8px;
}
QWidget#moduleSidebar QListWidget#modulePalette {
    background: transparent;
    border: none;
    padding: 0;
    color: $sidebar_ink;
}
QWidget#moduleSidebar QListWidget#modulePalette::item {
    background: transparent;
    color: $sidebar_ink;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 9px 10px;
    margin: 1px 0;
}
QWidget#moduleSidebar QListWidget#modulePalette::item:hover { background-color: $bg_sidebar_hover; }
QWidget#moduleSidebar QListWidget#modulePalette::item:selected {
    background-color: #2B6576;
    color: $sidebar_ink;
    border-color: #56808A;
}
QSplitter::handle { background-color: $border; }
QSplitter::handle:hover { background-color: $primary; }
QProgressBar { background: $bg_panel; border: none; border-radius: 4px; text-align: center; color: $ink; }
QProgressBar::chunk { background: $route_run; border-radius: 4px; }

/* ---- Thin horizontal rule --------------------------------------------- */
QFrame#hsep { background-color: $border; border: none; }

/* ---- Scrollbars (slim, on-surface) ------------------------------------ */
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical {
    background: transparent; width: 12px; margin: 2px 2px 2px 0;
}
QScrollBar::handle:vertical {
    background: $border_strong; border-radius: 5px; min-height: 32px;
}
QScrollBar::handle:vertical:hover { background: $ink_muted; }
QScrollBar:horizontal {
    background: transparent; height: 12px; margin: 0 2px 2px 2px;
}
QScrollBar::handle:horizontal {
    background: $border_strong; border-radius: 5px; min-width: 32px;
}
QScrollBar::handle:horizontal:hover { background: $ink_muted; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ---- Tooltips ---------------------------------------------------------- */
QToolTip {
    background-color: $ink;
    color: $bg_window;
    border: 1px solid $border_strong;
    padding: 4px 8px;
    border-radius: 4px;
}
QPlainTextEdit#console {
    background-color: $console_bg;
    color: $console_ink;
    border: 1px solid #31505B;
    border-radius: 6px;
    selection-background-color: #346B76;
    selection-color: #FFFFFF;
}
QMenu { background: $bg_surface; border: 1px solid $border_strong; padding: 4px; }
QMenu::item { padding: 6px 24px; border-radius: 4px; }
QMenu::item:selected { background: $primary_tint; color: $ink; }
""")


def build_stylesheet(colors=COLORS):
    """Return the full QSS string with *colors* substituted in."""
    return _QSS.substitute(colors)


def apply_theme(app, colors=COLORS):
    """Install the theme on a ``QApplication``.

    Sets the Fusion base style (so QSS renders consistently across platforms),
    a matching light palette for chrome the stylesheet doesn't paint
    (message boxes, placeholder text), a Segoe UI base font, and the QSS.
    """
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(colors["bg_window"]))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(colors["ink"]))
    pal.setColor(QPalette.ColorRole.Base,            QColor(colors["bg_surface"]))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(colors["bg_panel"]))
    pal.setColor(QPalette.ColorRole.Text,            QColor(colors["ink"]))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(colors["ink_muted"]))
    pal.setColor(QPalette.ColorRole.Button,          QColor(colors["bg_surface"]))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(colors["ink"]))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(colors["primary"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(colors["on_primary"]))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor(colors["ink"]))
    pal.setColor(QPalette.ColorRole.ToolTipText,     QColor(colors["bg_window"]))
    app.setPalette(pal)

    font = QFont("Segoe UI", 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    app.setStyleSheet(build_stylesheet(colors))
