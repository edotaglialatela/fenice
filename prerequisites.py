"""prerequisites.py -- Install the Python tools around the C++ solver.

The solver itself is C++ (build it with ``.\\cpp\\build.ps1``); Python is used for the
GUI, the validation / post-processing scripts, the regression check and the docs.

Run once on a fresh machine:

    python prerequisites.py

Optional flags
--------------
--no-gui      Skip PyQt6 (headless / server environments).
--no-docs     Skip markdown2 (only needed to rebuild the documentation).

Then:

    .\\cpp\\build.ps1 -Test            # build the solver + unit tests
    python gui/run.py                  # GUI
    python cpp/tests/regression.py     # regression check
"""

import sys
import subprocess
import importlib
import argparse


# ---------------------------------------------------------------------------
# Package groups  (pip_name, import_name, description, required)
# ---------------------------------------------------------------------------

REQUIRED = [
    ("numpy",      "numpy",      "Arrays - validation scripts, regression check, GUI hot-start probe", True),
    ("matplotlib", "matplotlib", "Plotting for validation scripts in validation/",                   True),
]

GUI = [
    ("PyQt6",      "PyQt6",      "Graphical setup dialog  (gui/run.py)",                              True),
]

DOCS = [
    ("markdown2",  "markdown2",  "Markdown renderer for docs/build.py",                               True),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_python_version() -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 9):
        print(f"  ERROR: Python {major}.{minor} detected. This project requires Python 3.9+.")
        print(f"  Current interpreter: {sys.executable}")
        sys.exit(1)
    print(f"  Python {major}.{minor} -- OK")


def _is_installed(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def _pip_install(pip_name: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _install_group(packages: list, group_label: str) -> dict:
    print(f"\n{group_label}")
    print("-" * len(group_label))
    status = {}
    for pip_name, import_name, description, required in packages:
        if _is_installed(import_name):
            print(f"  [OK]  {pip_name:<20}  already installed   ({description})")
            status[pip_name] = "ok"
            continue

        print(f"  [..]  {pip_name:<20}  installing ...      ({description})", end="", flush=True)
        ok = _pip_install(pip_name)
        if ok:
            print(f"\r  [OK]  {pip_name:<20}  installed           ({description})")
            status[pip_name] = "installed"
        else:
            tag   = "[X]" if required else "[!]"
            label = "FAILED (required)" if required else "FAILED (optional)"
            print(f"\r  {tag}  {pip_name:<20}  {label}  ({description})")
            status[pip_name] = "failed"
    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install Fenice prerequisites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--no-gui",   action="store_true", help="Skip PyQt6")
    parser.add_argument("--no-docs",  action="store_true", help="Skip markdown2")
    args = parser.parse_args()

    sep = "=" * 60
    print()
    print(sep)
    print("  Fenice -- Dependency Installer")
    print(sep)

    print("\nPython version")
    print("-" * 14)
    _check_python_version()

    all_status: dict = {}

    all_status.update(_install_group(REQUIRED, "Core (required)"))

    if not args.no_gui:
        all_status.update(_install_group(GUI, "GUI -- gui/run.py"))
    else:
        print("\nGUI  [skipped -- --no-gui]")

    if not args.no_docs:
        all_status.update(_install_group(DOCS, "Documentation -- docs/build.py"))
    else:
        print("\nDocs  [skipped -- --no-docs]")

    # Summary
    print()
    print(sep)
    failed = [k for k, v in all_status.items() if v == "failed"]
    if not failed:
        print("  All packages installed successfully.")
        print()
        print("  Build the solver :  .\\cpp\\build.ps1 -Test")
        print("  Launch the GUI   :  python gui/run.py")
        print("  Regression check :  python cpp/tests/regression.py")
    else:
        print("  ERROR: Some packages failed to install:")
        for f in failed:
            print(f"    - {f}")
        print()
        print("  Check your internet connection and pip setup, then re-run.")
        sys.exit(1)

    print(sep)
    print()


if __name__ == "__main__":
    main()