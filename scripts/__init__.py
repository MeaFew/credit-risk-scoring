"""Riskscore pipeline scripts package.

These modules are intended to be run as CLI scripts from the project root
(``python scripts/<name>.py`` or via ``make``). Because ``__main__`` modules
cannot use package-relative imports, each script adds the project root to
``sys.path`` once, then imports sibling modules through the ``scripts.``
package namespace (e.g. ``from scripts.metrics_utils import ks_score``).
"""
