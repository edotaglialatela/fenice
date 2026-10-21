"""Application theme for the Fenice PyQt6 GUI.

A single, reusable Qt theme: a named colour palette, a Qt Style Sheet (QSS)
built from it, and :func:`apply_theme` to install both on a ``QApplication``.

Design intent — *precise instrument*. The palette is derived from the project's
own brand asset (``gui/logo.png``): the deep oxblood of the robe is the
primary/brand accent, the halo gold carries warnings, and a clean bright red is
reserved for errors so the interface never reads as "all error". Surfaces are a
cool, low-chroma neutral — an instrument panel under office light, not the AI
cream default and not the raw-Qt grey it replaces.

Strategy is *restrained* (the product-UI default): tinted neutrals plus one
accent used only for primary actions, current selection, focus, and state — never
decoration. All body/label text is verified to clear WCAG AA (>=4.5:1).

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
# Roles, not raw colours, are referenced everywhere. Values chosen in OKLCH
# intent then pinned to hex (QSS has no oklch()). Contrast of every text role
# against its background is verified >= 4.5:1 (see module docstring).
COLORS = {
    # Cool neutral surfaces (the instrument panel)
    "bg_window":     "#EEF2F6",  # app window / dialog ground
    "bg_surface":    "#FFFFFF",  # inputs, group interiors
    "bg_panel":      "#E4EAF0",  # secondary neutral: tab strip, read-only, hover

    # Hairlines
    "border":        "#CBD5E0",
    "border_strong": "#AEBBC9",

    # Ink ramp (all verified >= 4.5:1 on bg_window/bg_surface)
    "ink":           "#1B2430",  # primary text  (~12:1)
    "ink_soft":      "#3C4A5A",  # subheads
    "ink_muted":     "#51606F",  # notes / secondary (~5.7:1 on bg_window)

    # Brand / primary — oxblood (the robe). White-on-primary ~7.5:1.
    "primary":         "#9A2D3A",
    "primary_hover":   "#B23744",
    "primary_pressed": "#7E2630",
    "primary_tint":    "#F4E3E5",  # selection / soft highlight ground
    "on_primary":      "#FFFFFF",

    # Semantic state — kept distinct from the brand oxblood
    "warning": "#8A5708",  # halo ochre, darkened to ~4.8:1 for text
    "error":   "#C62828",  # clean bright red (~5.6:1), clearly != oxblood
    "success": "#2E7D52",  # muted orb green (~4.7:1)

    # Disabled (contrast intentionally relaxed — non-interactive)
    "disabled_ink": "#9AA7B4",
    "disabled_bg":  "#EDF1F5",

    # Logo gallery mat
    "mat":          "#1B2430",
    "mat_caption":  "#C9D2DC",
    "mat_subtle":   "#8A97A6",
}


_QSS = Template(r"""
* {
    font-family: "Segoe UI", "Inter", system-ui, "Helvetica Neue", sans-serif;
    font-size: 13px;
}

QDialog, QWidget#rootSurface {
    background-color: $bg_window;
}
QWidget { color: $ink; }

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
    padding: 8px 18px;
    margin-right: 3px;
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

/* ---- Buttons ----------------------------------------------------------- */
QPushButton {
    background-color: $bg_surface;
    border: 1px solid $border_strong;
    border-radius: 6px;
    padding: 7px 16px;
    color: $ink;
    font-weight: 600;
}
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

/* ---- Logo gallery panel ------------------------------------------------ */
QWidget#logoPanel {
    background-color: $mat;
    border: 1px solid $border_strong;
    border-radius: 10px;
}
QWidget#logoPanel QLabel { background: transparent; }
QLabel#logoCaption {
    color: $mat_caption;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
QLabel#logoCaptionSub {
    color: $mat_subtle;
    font-size: 11px;
    font-weight: 400;
}

/* ---- Tooltips ---------------------------------------------------------- */
QToolTip {
    background-color: $ink;
    color: $bg_window;
    border: 1px solid $border_strong;
    padding: 4px 8px;
    border-radius: 4px;
}
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
