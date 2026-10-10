"""Shared helpers: project paths, config loading and logging."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing or unreadable."""


def resolve_path(path: str | Path) -> Path:
    """Turn a project-relative path from the config into an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> dict:
    """Load config/config.yaml (or a custom file) into a dictionary."""
    cfg_path = resolve_path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{cfg_path} is not valid YAML: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(f"{cfg_path} must contain a YAML mapping at the top level.")
    return cfg


def get_logger(name: str, log_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that prints to the console and, if log_dir is given, to <log_dir>/<name>.log."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:  # already configured
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_dir is not None:
        folder = resolve_path(log_dir)
        folder.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(folder / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger
