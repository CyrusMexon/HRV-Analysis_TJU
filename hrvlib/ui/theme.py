"""
Publication-oriented Qt theme for HRV Studio.

Neutral publication theme used for consistent screenshots independent of OS
light/dark mode. The global application stylesheet is defined here.
"""

from pathlib import Path

from PyQt6 import QtGui, QtWidgets


PUBLICATION_LIGHT_THEME = True


_COLORS = {
    "window": "#dfe2e4",
    "sidebar": "#d9e0e5",
    "panel": "#f2f3f4",
    "panel_alt": "#e9ecef",
    "field": "#fbfbfc",
    "field_alt": "#f1f3f4",
    "border": "#aeb5bb",
    "border_dark": "#7f8992",
    "text": "#1f2328",
    "muted_text": "#4e565f",
    "disabled_text": "#868e96",
    "selection": "#c8d7e6",
    "selection_text": "#111111",
    "button": "#d4d8dc",
    "button_hover": "#c8cdd2",
    "button_pressed": "#bac1c7",
    "primary": "#3f6f8f",
    "primary_hover": "#365f7b",
    "primary_pressed": "#2e5169",
    "primary_disabled": "#9fb0bd",
    "plot": "#ffffff",
    "warning": "#b00020",
}

SCROLLBAR_WIDTH_PX = 12
COMBO_DROPDOWN_WIDTH_PX = 22
SPINBOX_BUTTON_WIDTH_PX = 19


def _asset_url(file_name: str) -> str:
    return (Path(__file__).resolve().parent / "assets" / file_name).as_posix()


def _publication_palette() -> QtGui.QPalette:
    colors = _COLORS
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(colors["window"]))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(colors["text"]))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(colors["field"]))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(colors["field_alt"]))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(colors["field"]))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(colors["text"]))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(colors["text"]))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(colors["button"]))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(colors["text"]))
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor(colors["warning"]))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(colors["selection"]))
    palette.setColor(
        QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(colors["selection_text"])
    )
    palette.setColor(
        QtGui.QPalette.ColorGroup.Disabled,
        QtGui.QPalette.ColorRole.Text,
        QtGui.QColor(colors["disabled_text"]),
    )
    palette.setColor(
        QtGui.QPalette.ColorGroup.Disabled,
        QtGui.QPalette.ColorRole.ButtonText,
        QtGui.QColor(colors["disabled_text"]),
    )
    return palette


def publication_stylesheet() -> str:
    c = _COLORS
    chevron_down = _asset_url("chevron_down.svg")
    step_up = _asset_url("step_up.svg")
    step_down = _asset_url("step_down.svg")
    return f"""
    QWidget {{
        background-color: {c["window"]};
        color: {c["text"]};
        selection-background-color: {c["selection"]};
        selection-color: {c["selection_text"]};
    }}

    QMainWindow, QDialog, QMessageBox, QFileDialog {{
        background-color: {c["window"]};
        color: {c["text"]};
    }}

    QMenuBar {{
        background-color: {c["panel"]};
        color: {c["text"]};
        border-bottom: 1px solid {c["border"]};
    }}

    QMenuBar::item {{
        background: transparent;
        padding: 4px 8px;
    }}

    QMenuBar::item:selected, QMenuBar::item:pressed {{
        background-color: {c["selection"]};
        color: {c["selection_text"]};
    }}

    QMenu {{
        background-color: {c["field"]};
        color: {c["text"]};
        border: 1px solid {c["border"]};
    }}

    QMenu::item {{
        padding: 5px 22px 5px 20px;
    }}

    QMenu::item:selected {{
        background-color: {c["selection"]};
        color: {c["selection_text"]};
    }}

    QStatusBar {{
        background-color: {c["panel"]};
        color: {c["muted_text"]};
        border-top: 1px solid {c["border"]};
    }}

    QGroupBox {{
        background-color: {c["panel"]};
        border: 1px solid {c["border"]};
        border-radius: 4px;
        margin-top: 11px;
        padding: 8px 8px 7px 8px;
        font-weight: 600;
        color: {c["text"]};
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 8px;
        padding: 0 4px;
        background-color: {c["panel"]};
    }}

    QFrame, QScrollArea, QTabWidget::pane {{
        background-color: {c["panel"]};
        border: 1px solid {c["border"]};
    }}

    QScrollArea > QWidget > QWidget {{
        background-color: {c["panel"]};
    }}

    QWidget#leftSidebar, QScrollArea#leftSidebarScroll,
    QWidget#leftSidebarContent {{
        background-color: {c["sidebar"]};
    }}

    QScrollArea#leftSidebarScroll {{
        border: none;
    }}

    QScrollArea#leftSidebarScroll > QWidget {{
        background-color: {c["sidebar"]};
    }}

    QTabBar::tab {{
        background-color: {c["button"]};
        color: {c["text"]};
        border: 1px solid {c["border"]};
        border-bottom: none;
        padding: 7px 12px;
        margin-right: 2px;
    }}

    QTabBar::tab:selected {{
        background-color: {c["field"]};
        border-color: {c["border_dark"]};
        font-weight: 600;
    }}

    QTabBar::tab:hover:!selected {{
        background-color: {c["button_hover"]};
    }}

    QLabel {{
        color: {c["text"]};
        background-color: transparent;
    }}

    QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {c["field"]};
        color: {c["text"]};
        border: 1px solid {c["border"]};
        border-radius: 3px;
        padding: 3px 6px;
        min-height: 20px;
    }}

    QComboBox {{
        padding-right: {COMBO_DROPDOWN_WIDTH_PX + 8}px;
    }}

    QSpinBox, QDoubleSpinBox {{
        padding-right: {SPINBOX_BUTTON_WIDTH_PX + 4}px;
    }}

    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {c["border_dark"]};
        background-color: {c["field"]};
    }}

    QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled,
    QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
        background-color: {c["panel_alt"]};
        color: {c["disabled_text"]};
    }}

    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        background-color: {c["panel_alt"]};
        border-left: 1px solid {c["border"]};
        border-top-right-radius: 3px;
        border-bottom-right-radius: 3px;
        width: {COMBO_DROPDOWN_WIDTH_PX}px;
    }}

    QComboBox::drop-down:hover {{
        background-color: {c["button_hover"]};
    }}

    QComboBox::down-arrow {{
        image: url("{chevron_down}");
        width: 9px;
        height: 9px;
    }}

    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        background-color: {c["panel_alt"]};
        width: {SPINBOX_BUTTON_WIDTH_PX}px;
        border-left: 1px solid {c["border"]};
    }}

    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-position: top right;
        border-top-right-radius: 3px;
        border-bottom: 1px solid {c["border"]};
    }}

    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-position: bottom right;
        border-bottom-right-radius: 3px;
    }}

    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {c["button_hover"]};
    }}

    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: url("{step_up}");
        width: 8px;
        height: 6px;
    }}

    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: url("{step_down}");
        width: 8px;
        height: 6px;
    }}

    QPushButton {{
        background-color: {c["button"]};
        color: {c["text"]};
        border: 1px solid {c["border_dark"]};
        border-radius: 4px;
        padding: 5px 10px;
    }}

    QPushButton:hover {{
        background-color: {c["button_hover"]};
    }}

    QPushButton:pressed {{
        background-color: {c["button_pressed"]};
    }}

    QPushButton:disabled {{
        background-color: {c["panel_alt"]};
        color: {c["disabled_text"]};
        border-color: {c["border"]};
    }}

    QPushButton[primaryAction="true"] {{
        background-color: {c["primary"]};
        color: #ffffff;
        border: 1px solid {c["primary_pressed"]};
        font-weight: 600;
    }}

    QPushButton[primaryAction="true"]:hover {{
        background-color: {c["primary_hover"]};
    }}

    QPushButton[primaryAction="true"]:pressed {{
        background-color: {c["primary_pressed"]};
    }}

    QPushButton[primaryAction="true"]:disabled {{
        background-color: {c["primary_disabled"]};
        color: #f3f6f8;
        border-color: {c["border_dark"]};
    }}

    QRadioButton, QCheckBox {{
        color: {c["text"]};
        background-color: transparent;
        spacing: 5px;
    }}

    QRadioButton[segmentedControl="true"] {{
        background-color: {c["field_alt"]};
        color: {c["text"]};
        border: 1px solid {c["border"]};
        border-radius: 4px;
        padding: 4px 9px;
        spacing: 0;
    }}

    QRadioButton[segmentedControl="true"]::indicator {{
        width: 0;
        height: 0;
        margin: 0;
    }}

    QRadioButton[segmentedControl="true"]:hover {{
        background-color: {c["button_hover"]};
    }}

    QRadioButton[segmentedControl="true"]:checked {{
        background-color: {c["selection"]};
        border: 1px solid {c["border_dark"]};
        color: {c["text"]};
        font-weight: 600;
    }}

    QRadioButton[segmentedControl="true"]:disabled {{
        background-color: {c["panel_alt"]};
        color: {c["disabled_text"]};
        border-color: {c["border"]};
    }}

    QTableWidget, QTableView, QListWidget {{
        background-color: {c["field"]};
        alternate-background-color: {c["field_alt"]};
        color: {c["text"]};
        gridline-color: {c["border"]};
        border: 1px solid {c["border"]};
        selection-background-color: {c["selection"]};
        selection-color: {c["selection_text"]};
    }}

    QHeaderView::section {{
        background-color: {c["panel_alt"]};
        color: {c["text"]};
        border: 0;
        border-bottom: 1px solid {c["border"]};
        padding: 5px 6px;
        font-weight: 600;
    }}

    QToolBar {{
        background-color: {c["panel"]};
        border: 1px solid {c["border"]};
        spacing: 3px;
    }}

    QToolButton {{
        background-color: {c["button"]};
        color: {c["text"]};
        border: 1px solid transparent;
        border-radius: 3px;
        padding: 3px;
    }}

    QToolButton:hover {{
        background-color: {c["button_hover"]};
        border-color: {c["border"]};
    }}

    QScrollBar:vertical {{
        background-color: {c["panel"]};
        border: 1px solid {c["border"]};
        width: {SCROLLBAR_WIDTH_PX}px;
        margin: 0;
    }}

    QScrollBar:horizontal {{
        background-color: {c["panel"]};
        border: 1px solid {c["border"]};
        height: {SCROLLBAR_WIDTH_PX}px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background-color: {c["button_pressed"]};
        border-radius: 3px;
        min-height: 18px;
    }}

    QScrollBar::handle:horizontal {{
        background-color: {c["button_pressed"]};
        border-radius: 3px;
        min-width: 18px;
    }}

    QScrollBar::add-line, QScrollBar::sub-line {{
        width: 0;
        height: 0;
        border: none;
        background: transparent;
    }}

    FigureCanvasQTAgg {{
        background-color: {c["plot"]};
        border: 1px solid {c["border"]};
    }}
    """


def apply_publication_theme(app: QtWidgets.QApplication) -> None:
    """Apply a neutral light theme independent of the operating system palette."""
    app.setStyle("Fusion")
    app.setPalette(_publication_palette())
    app.setStyleSheet(publication_stylesheet())


def mark_primary_button(button: QtWidgets.QPushButton) -> None:
    """Mark a semantic primary action for the centralized publication theme."""
    button.setProperty("primaryAction", True)
    button.style().unpolish(button)
    button.style().polish(button)


def mark_segmented_radio(radio: QtWidgets.QRadioButton) -> None:
    """Mark a radio option to render as a compact segmented control."""
    radio.setProperty("segmentedControl", True)
    radio.style().unpolish(radio)
    radio.style().polish(radio)
