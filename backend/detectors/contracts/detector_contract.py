"""Shared detector contract placeholders.

These types define boundaries only. They do not perform detection.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class DetectorManifest:
    """Metadata describing a replaceable detector."""

    detector_id: str
    display_name: str
    modality: str
    version: str
    model_family: str
    input_requirements: Mapping[str, Any] = field(default_factory=dict)
    output_capabilities: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectorInput:
    """Prepared detector input from the forensic layer."""

    analysis_id: str
    media_id: str
    modality: str
    prepared_artifacts: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectorResult:
    """Normalized detector result returned to fusion."""

    detector_id: str
    detector_version: str
    modality: str
    status: str
    label: str
    score: float | None = None
    confidence: str | None = None
    coverage: Mapping[str, Any] = field(default_factory=dict)
    evidence: list[Mapping[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    runtime: Mapping[str, Any] = field(default_factory=dict)


class Detector(Protocol):
    """Protocol every detector adapter must satisfy."""

    manifest: DetectorManifest

    def analyze(self, detector_input: DetectorInput) -> DetectorResult:
        """Analyze prepared forensic artifacts and return a normalized result."""
        ...

