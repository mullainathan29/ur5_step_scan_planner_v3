"""Desktop GUI entry point; the CLI remains in main.py."""

from __future__ import annotations

import sys

from PySide6 import QtWidgets

from gui.main_window import MainWindow


def main() -> int:
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
