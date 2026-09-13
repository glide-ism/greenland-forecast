import importlib.util
from pathlib import Path

from .config import (BedConditioningConfig, GlacierConfig, MaternNoise,
                     PriorHyperparams, Schedule, SolverConfig)
from .priors import GlacierPriors
from .problem import GlacierProblem, WhitenedParameters
from .loss import LossTerms, PriorMeans
from .observations import (
    BedSlopeSpec, BedSpec, DhdtSpec, DivideFluxSpec, DomainData, ExtentSpec,
    Observation, SnowlineSpec, SurfaceSpec, VelocitySpec,
    default_observation_specs,
)
from .scheduling import build_step_sequence, merge_times


def load_config(domain_dir):
    """Load and return the `CONFIG` defined in `<domain_dir>/config.py`."""
    domain_path = Path(domain_dir).resolve()
    spec = importlib.util.spec_from_file_location(
        f"_domain_config_{domain_path.name}", domain_path / "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


__all__ = [
    "BedConditioningConfig",
    "GlacierConfig",
    "MaternNoise",
    "PriorHyperparams",
    "Schedule",
    "SolverConfig",
    "GlacierPriors",
    "GlacierProblem",
    "WhitenedParameters",
    "LossTerms",
    "PriorMeans",
    "Observation",
    "DomainData",
    "SurfaceSpec",
    "VelocitySpec",
    "ExtentSpec",
    "BedSpec",
    "SnowlineSpec",
    "DhdtSpec",
    "DivideFluxSpec",
    "BedSlopeSpec",
    "default_observation_specs",
    "build_step_sequence",
    "merge_times",
    "load_config",
]
