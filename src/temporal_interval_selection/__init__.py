"""Temporal interval selection for text-to-video retrieval."""

from .fidelity import EditedFidelity, recover_edited_fidelity
from .gradient_lse import GradientLSESelection, select_gradient_lse_interval
from .results import rebuild_fidelity_report
from .scheduled_attack import PGDConfig, run_scheduled_pgd
from .selectors import (
    IntervalAnalysis,
    SelectedInterval,
    analyze_intervals,
    target_centered_positions,
)
from .temporal import TemporalLoss, temporal_regularizers

__all__ = [
    "EditedFidelity",
    "GradientLSESelection",
    "IntervalAnalysis",
    "PGDConfig",
    "SelectedInterval",
    "TemporalLoss",
    "analyze_intervals",
    "rebuild_fidelity_report",
    "recover_edited_fidelity",
    "run_scheduled_pgd",
    "select_gradient_lse_interval",
    "target_centered_positions",
    "temporal_regularizers",
]
