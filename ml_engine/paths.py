"""Path helpers for the ML engine package.

Purpose:
    Centralize package-relative locations for config, data, reports, output, and pickle files.
Responsibilities:
    Resolve absolute filesystem paths independently of the current working directory.
Inputs:
    Relative filenames or subpaths within the ml_engine package tree.
Outputs:
    Absolute `Path` objects.
Assumptions:
    The repository keeps engine assets under the `ml_engine/` folder.
Limitations:
    This module only resolves paths; it does not validate file contents.
"""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def config_path(*parts: str) -> Path:
    return PACKAGE_ROOT / "config" / Path(*parts)


def data_path(*parts: str) -> Path:
    return PACKAGE_ROOT / "data" / Path(*parts)


def reports_path(*parts: str) -> Path:
    return PACKAGE_ROOT / "reports" / Path(*parts)


def output_path(*parts: str) -> Path:
    return PACKAGE_ROOT / "output" / Path(*parts)


def pickle_path(*parts: str) -> Path:
    return PACKAGE_ROOT / "pickle" / Path(*parts)
