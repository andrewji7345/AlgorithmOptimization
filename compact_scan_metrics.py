#!/usr/bin/env python3
"""Shared schema and metric helpers for compact optimization scans.

The compact ntuple factorizes reconstruction from the event gate. This module
is the single implementation of that factorization used by both the global
evaluator and focused diagnostic tools. Primary efficiencies are unweighted
event fractions; generator weights remain available for labelled follow-ups.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # Permit schema-independent tests on hosts without ROOT I/O packages.
    import awkward as ak
    import uproot
except ImportError:  # pragma: no cover
    ak = None
    uproot = None


METADATA_PATH = "compactScan/Metadata"
EVENTS_PATH = "compactScan/Events"
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
STATUS_NAMES = (
    "valid", "no_selected_jets", "no_selected_constituents", "invalid_com",
    "invalid_thrust", "no_ca_jets", "complexity_guard",
    "no_valid_partition", "numerical_failure",
)
VALID_STATUS = 0
EXPECTED_GATE_RECO_CONSTRAINT = "collectionPtCut<=gatePtCut; nGateJets=0 is ungated"
DEFAULT_BIAS_LIMIT = 0.20
DEFAULT_RESOLUTION_LIMIT = 0.50
DEFAULT_INVALID_LIMIT = 0.20
DEFAULT_TAIL_LIMIT = 0.50
DEFAULT_MIN_VALID_EVENTS = 50
DEFAULT_TAIL_RESPONSE_MIN = 0.70
DEFAULT_TAIL_RESPONSE_MAX = 1.30
DEFAULT_RESPONSE_HIST_MIN = 0.0
DEFAULT_RESPONSE_HIST_MAX = 3.0
DEFAULT_RESPONSE_HIST_BINS = 150

SAMPLE_RE = re.compile(
    r"^(?P<decay>[A-Za-z0-9]+)_(?P<suu_mass>[0-9]+)_(?P<chi_mass>[0-9]+)$"
)
GENERATED_MASS_RE = re.compile(
    r"MSuu-(?P<suu>[0-9]+(?:p[0-9]+)?)_MChi-(?P<chi>[0-9]+(?:p[0-9]+)?)"
)


def _require_root_io() -> None:
    if ak is None or uproot is None:
        raise RuntimeError("compact ROOT I/O requires awkward and uproot")


def _float_key(value: float) -> float:
    return round(float(value), 7)


@dataclass(frozen=True)
class PhysicalityDefinition:
    """Limits whose normalized maximum is physical when at most one.

    Responses use one fixed histogram for scalability and comparability. Every
    estimator setting is written to evaluator outputs.
    """

    bias_limit: float = DEFAULT_BIAS_LIMIT
    resolution_limit: float = DEFAULT_RESOLUTION_LIMIT
    invalid_limit: float = DEFAULT_INVALID_LIMIT
    tail_limit: float = DEFAULT_TAIL_LIMIT
    min_valid_events: int = DEFAULT_MIN_VALID_EVENTS
    tail_response_min: float = DEFAULT_TAIL_RESPONSE_MIN
    tail_response_max: float = DEFAULT_TAIL_RESPONSE_MAX
    response_hist_min: float = DEFAULT_RESPONSE_HIST_MIN
    response_hist_max: float = DEFAULT_RESPONSE_HIST_MAX
    response_hist_bins: int = DEFAULT_RESPONSE_HIST_BINS

    def __post_init__(self) -> None:
        for name in ("bias_limit", "resolution_limit", "invalid_limit", "tail_limit"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(self.min_valid_events) <= 0:
            raise ValueError("min_valid_events must be positive")
        if not 0.0 <= self.tail_response_min < self.tail_response_max:
            raise ValueError("tail response window must satisfy 0 <= min < max")
        if not (
            math.isfinite(self.response_hist_min)
            and math.isfinite(self.response_hist_max)
            and self.response_hist_min < self.response_hist_max
        ):
            raise ValueError("response histogram range must be finite and increasing")
        if int(self.response_hist_bins) < 10:
            raise ValueError("response_hist_bins must be at least 10")

    @property
    def response_edges(self) -> np.ndarray:
        return np.linspace(
            self.response_hist_min, self.response_hist_max,
            self.response_hist_bins + 1, dtype=np.float64,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bias_limit": self.bias_limit,
            "resolution_limit": self.resolution_limit,
            "invalid_limit": self.invalid_limit,
            "tail_limit": self.tail_limit,
            "min_valid_events": self.min_valid_events,
            "tail_response_min": self.tail_response_min,
            "tail_response_max": self.tail_response_max,
            "response_hist_min": self.response_hist_min,
            "response_hist_max": self.response_hist_max,
            "response_hist_bins": self.response_hist_bins,
            "response_hist_estimator": "fixed-bin median and principal-peak FWHM",
        }


@dataclass(frozen=True, order=True)
class ConfigurationKey:
    """Canonical ``(n,Tgate,Tkeep,RAK,RCA,c)`` configuration."""

    n_gate_jets: int
    gate_pt_cut: Optional[float]
    collection_pt_cut: float
    ak_radius: float
    ca_radius: float
    cos_thrust_cut: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_gate_jets", int(self.n_gate_jets))
        if self.n_gate_jets < 0:
            raise ValueError("n_gate_jets must be nonnegative")
        if self.n_gate_jets == 0:
            object.__setattr__(self, "gate_pt_cut", None)
        elif self.gate_pt_cut is None:
            raise ValueError("positive gate multiplicity requires gate_pt_cut")
        for name in ("collection_pt_cut", "ak_radius", "ca_radius", "cos_thrust_cut"):
            value = _float_key(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.gate_pt_cut is not None:
            gate = _float_key(self.gate_pt_cut)
            if not math.isfinite(gate) or gate <= 0.0:
                raise ValueError("gate_pt_cut must be finite and positive")
            object.__setattr__(self, "gate_pt_cut", gate)
            if self.collection_pt_cut > gate + 1.0e-6:
                raise ValueError("collection_pt_cut must not exceed gate_pt_cut")
        if self.collection_pt_cut <= 0.0 or self.ak_radius <= 0.0 or self.ca_radius <= 0.0:
            raise ValueError("collection threshold and radii must be positive")
        if not 0.0 <= self.cos_thrust_cut <= 1.0:
            raise ValueError("cos_thrust_cut must lie in [0, 1]")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_gate_jets": self.n_gate_jets,
            "gate_pt_cut": self.gate_pt_cut,
            "collection_pt_cut": self.collection_pt_cut,
            "ak_radius": self.ak_radius,
            "ca_radius": self.ca_radius,
            "cos_thrust_cut": self.cos_thrust_cut,
        }


def configuration_key(
    n_gate_jets: int, gate_pt_cut: Optional[float], collection_pt_cut: float,
    ak_radius: float, ca_radius: float, cos_thrust_cut: float,
) -> ConfigurationKey:
    return ConfigurationKey(
        n_gate_jets, gate_pt_cut, collection_pt_cut,
        ak_radius, ca_radius, cos_thrust_cut,
    )


def _number_slug(value: float) -> str:
    return format(float(value), ".7g").replace("-", "m").replace(".", "p")


def _parse_number_slug(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))


def configuration_slug(key: ConfigurationKey) -> str:
    gate = "none" if key.gate_pt_cut is None else _number_slug(key.gate_pt_cut)
    return (
        f"ng{key.n_gate_jets}_tg{gate}_tk{_number_slug(key.collection_pt_cut)}"
        f"_ak{_number_slug(key.ak_radius)}_ca{_number_slug(key.ca_radius)}"
        f"_c{_number_slug(key.cos_thrust_cut)}"
    )


CONFIGURATION_SLUG_RE = re.compile(
    r"^ng(?P<n>[0-9]+)_tg(?P<tg>none|[mp0-9]+)_tk(?P<tk>[mp0-9]+)"
    r"_ak(?P<ak>[mp0-9]+)_ca(?P<ca>[mp0-9]+)_c(?P<c>[mp0-9]+)$"
)


def parse_configuration_slug(text: str) -> ConfigurationKey:
    match = CONFIGURATION_SLUG_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"invalid compact configuration slug: {text!r}")
    values = match.groupdict()
    gate = None if values["tg"] == "none" else _parse_number_slug(values["tg"])
    return configuration_key(
        int(values["n"]), gate, _parse_number_slug(values["tk"]),
        _parse_number_slug(values["ak"]), _parse_number_slug(values["ca"]),
        _parse_number_slug(values["c"]),
    )


@dataclass(frozen=True)
class SampleDefinition:
    sample: str
    decay: str
    nominal_suu_mass: float
    nominal_chi_mass: float
    generated_suu_mass: float
    generated_chi_mass: float
    input_list: str

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def parse_sample_name(sample: str) -> Tuple[str, float, float]:
    match = SAMPLE_RE.fullmatch(sample)
    if match is None:
        raise ValueError(f"sample {sample!r} must have form <decay>_<MSuu>_<MChi>")
    values = match.groupdict()
    return values["decay"], float(values["suu_mass"]), float(values["chi_mass"])


def _generated_mass(text: str) -> float:
    return float(text.replace("p", "."))


def load_sample_definitions(
    sample_lists_dir: Any, expected_count: Optional[int] = 114,
) -> Dict[str, SampleDefinition]:
    """Parse generated masses from every non-comment MiniAOD URI."""

    directory = Path(sample_lists_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"sample-list directory does not exist: {directory}")
    paths = sorted(directory.glob("*.txt"))
    if expected_count is not None and len(paths) != int(expected_count):
        raise ValueError(f"found {len(paths)} sample lists; expected {expected_count}")
    definitions: Dict[str, SampleDefinition] = {}
    for path in paths:
        sample = path.stem
        decay, nominal_suu, nominal_chi = parse_sample_name(sample)
        generated = set()
        with path.open() as input_file:
            for raw_line in input_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                match = GENERATED_MASS_RE.search(line)
                if match is None:
                    raise ValueError(f"cannot parse generated masses from {path}: {line}")
                generated.add((_generated_mass(match.group("suu")), _generated_mass(match.group("chi"))))
        if len(generated) != 1:
            raise ValueError(f"{path} contains {len(generated)} generated regimes; expected one")
        generated_suu, generated_chi = next(iter(generated))
        definitions[sample] = SampleDefinition(
            sample, decay, nominal_suu, nominal_chi, generated_suu,
            generated_chi, str(path.resolve()),
        )
    return definitions


def true_chi_mass_for_sample(
    sample: str, definitions: Mapping[str, SampleDefinition],
    overrides: Optional[Mapping[str, float]] = None,
) -> float:
    if overrides and sample in overrides:
        mass = float(overrides[sample])
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"invalid true-mass override for {sample}: {mass}")
        return mass
    if sample not in definitions:
        raise KeyError(f"sample {sample!r} is absent from regime definitions")
    return float(definitions[sample].generated_chi_mass)


@dataclass
class CompactMetadata:
    path: str
    schema_version: int
    sample_name: str
    ak_radius: float
    max_ambiguous_ca_jets: int
    processed_events: int
    sum_weights: float
    sum_weights2: float
    collection_pt_cuts: Tuple[float, ...]
    ca_radii: Tuple[float, ...]
    cos_thrust_cuts: Tuple[float, ...]
    default_gate_jet_counts: Tuple[int, ...]
    default_gate_pt_cuts: Tuple[float, ...]
    status_codes: Tuple[int, ...]
    status_names: Tuple[str, ...]
    base_collection_pt_cut: Tuple[float, ...]
    base_ca_radius: Tuple[float, ...]
    config_ids: Tuple[int, ...]
    config_base_index: Tuple[int, ...]
    config_collection_pt_cut: Tuple[float, ...]
    config_ca_radius: Tuple[float, ...]
    config_cos_thrust: Tuple[float, ...]
    puppi_weighted: bool
    use_jec: bool
    ca_algorithm: str
    mass_objective: str
    gate_reco_constraint: str
    enforce_legacy_radius_constraint: bool
    _config_lookup: Dict[Tuple[float, float, float], int] = field(default_factory=dict, repr=False)

    @property
    def n_configurations(self) -> int:
        return len(self.config_ids)

    @property
    def n_base_configurations(self) -> int:
        return len(self.base_collection_pt_cut)


@dataclass
class EventPayload:
    run: np.ndarray
    lumi: np.ndarray
    event: np.ndarray
    gen_weight: np.ndarray
    ak_jet_pt: Any
    n_ca_jets: np.ndarray
    reco_status: np.ndarray
    n_ambiguous: np.ndarray
    sj1_mass: np.ndarray
    sj2_mass: np.ndarray
    suu_mass: Optional[np.ndarray] = None  # Absent in schema version 1.

    @property
    def n_events(self) -> int:
        return len(self.event)

    @property
    def event_ids(self) -> np.ndarray:
        result = np.empty(self.n_events, dtype=[("run", "<u4"), ("lumi", "<u4"), ("event", "<u8")])
        result["run"], result["lumi"], result["event"] = self.run, self.lumi, self.event
        return result


@dataclass
class MetricBatch:
    keys: Tuple[ConfigurationKey, ...]
    values: Dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.keys)


def _open_root_file(path: Any):
    _require_root_io()
    options = {}
    if "://" not in str(path):
        options["handler"] = uproot.source.file.MemmapSource
    return uproot.open(str(path), **options)


def _one_scalar(array: Any) -> Any:
    if len(array) != 1:
        raise ValueError("metadata branch must contain exactly one entry")
    return ak.to_list(array[0])


def _one_vector(array: Any, cast) -> Tuple[Any, ...]:
    value = _one_scalar(array)
    if not isinstance(value, list):
        raise TypeError("metadata vector branch is not a vector")
    return tuple(cast(item) for item in value)


def load_metadata(path: Any) -> CompactMetadata:
    """Read and validate metadata-authoritative compact schema version 1."""

    names = (
        "schemaVersion", "sampleName", "akRadius", "maxAmbiguousCAJets",
        "puppiWeighted", "useJEC", "caAlgorithm", "massObjective",
        "gateRecoConstraint", "enforceLegacyRadiusConstraint", "collectionPtCuts",
        "caRadii", "cosThrustCuts", "defaultGateJetCounts", "defaultGatePtCuts",
        "statusCodes", "statusNames", "processedEvents", "sumWeights", "sumWeights2",
        "baseCollectionPtCut", "baseCaRadius", "configId", "configBaseIndex",
        "configCollectionPtCut", "configCaRadius", "configCosThrust",
    )
    with _open_root_file(path) as root_file:
        if METADATA_PATH not in root_file or EVENTS_PATH not in root_file:
            raise KeyError(f"{path} is not a compact-scan ROOT file")
        tree = root_file[METADATA_PATH]
        missing = sorted(set(names) - set(tree.keys()))
        if missing:
            raise KeyError(f"{path} metadata is missing: {', '.join(missing)}")
        arrays = tree.arrays(names, entry_start=0, entry_stop=1, library="ak")
        event_entries = int(root_file[EVENTS_PATH].num_entries)
    scalar = lambda name, cast: cast(_one_scalar(arrays[name]))
    vector = lambda name, cast: _one_vector(arrays[name], cast)
    metadata = CompactMetadata(
        path=str(path), schema_version=scalar("schemaVersion", int),
        sample_name=scalar("sampleName", str), ak_radius=scalar("akRadius", float),
        max_ambiguous_ca_jets=scalar("maxAmbiguousCAJets", int),
        processed_events=scalar("processedEvents", int),
        sum_weights=scalar("sumWeights", float), sum_weights2=scalar("sumWeights2", float),
        collection_pt_cuts=vector("collectionPtCuts", float),
        ca_radii=vector("caRadii", float), cos_thrust_cuts=vector("cosThrustCuts", float),
        default_gate_jet_counts=vector("defaultGateJetCounts", int),
        default_gate_pt_cuts=vector("defaultGatePtCuts", float),
        status_codes=vector("statusCodes", int), status_names=vector("statusNames", str),
        base_collection_pt_cut=vector("baseCollectionPtCut", float),
        base_ca_radius=vector("baseCaRadius", float), config_ids=vector("configId", int),
        config_base_index=vector("configBaseIndex", int),
        config_collection_pt_cut=vector("configCollectionPtCut", float),
        config_ca_radius=vector("configCaRadius", float),
        config_cos_thrust=vector("configCosThrust", float),
        puppi_weighted=scalar("puppiWeighted", bool), use_jec=scalar("useJEC", bool),
        ca_algorithm=scalar("caAlgorithm", str), mass_objective=scalar("massObjective", str),
        gate_reco_constraint=scalar("gateRecoConstraint", str),
        enforce_legacy_radius_constraint=scalar("enforceLegacyRadiusConstraint", bool),
    )
    if metadata.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported compact schemaVersion {metadata.schema_version}")
    if metadata.processed_events != event_entries:
        raise ValueError(f"{path}: processedEvents/tree-entry mismatch")
    parse_sample_name(metadata.sample_name)
    if not metadata.puppi_weighted or metadata.use_jec:
        raise ValueError(f"{path}: requires PUPPI weighting and no JEC")
    if metadata.ca_algorithm != "cambridge_y_phi":
        raise ValueError(f"{path}: unexpected CA algorithm {metadata.ca_algorithm!r}")
    if metadata.mass_objective != "abs(m1-m2)/(m1+m2)":
        raise ValueError(f"{path}: unexpected mass objective {metadata.mass_objective!r}")
    if metadata.gate_reco_constraint != EXPECTED_GATE_RECO_CONSTRAINT:
        raise ValueError(
            f"{path}: unexpected gate/reconstruction constraint "
            f"{metadata.gate_reco_constraint!r}"
        )
    if metadata.status_codes != tuple(range(len(STATUS_NAMES))) or metadata.status_names != STATUS_NAMES:
        raise ValueError(f"{path}: unexpected status dictionary")
    lengths = {len(metadata.config_ids), len(metadata.config_base_index),
               len(metadata.config_collection_pt_cut), len(metadata.config_ca_radius),
               len(metadata.config_cos_thrust)}
    if len(lengths) != 1 or not metadata.config_ids:
        raise ValueError(f"{path}: inconsistent or empty config mapping")
    if metadata.config_ids != tuple(range(len(metadata.config_ids))):
        raise ValueError(f"{path}: configId is not contiguous")
    if len(metadata.base_collection_pt_cut) != len(metadata.base_ca_radius):
        raise ValueError(f"{path}: inconsistent base mapping")
    lookup = {}
    for index, (base, pt, ca_radius, cos_cut) in enumerate(zip(
        metadata.config_base_index, metadata.config_collection_pt_cut,
        metadata.config_ca_radius, metadata.config_cos_thrust,
    )):
        if not 0 <= base < len(metadata.base_collection_pt_cut):
            raise ValueError(f"{path}: invalid base index at config {index}")
        if not math.isclose(pt, metadata.base_collection_pt_cut[base], abs_tol=1e-5):
            raise ValueError(f"{path}: pT/base mismatch at config {index}")
        if not math.isclose(ca_radius, metadata.base_ca_radius[base], abs_tol=1e-5):
            raise ValueError(f"{path}: CA/base mismatch at config {index}")
        identity = (_float_key(pt), _float_key(ca_radius), _float_key(cos_cut))
        if identity in lookup:
            raise ValueError(f"{path}: duplicate reconstruction config {identity}")
        lookup[identity] = index
    metadata._config_lookup = lookup
    return metadata


def discover_compact_files(
    input_dir: Optional[Any] = None, paths: Optional[Sequence[Any]] = None,
    pattern: str = "*.root", samples: Optional[Sequence[str]] = None,
) -> List[CompactMetadata]:
    """Discover inputs and reject duplicate ``(sample, AK radius)`` jobs."""

    if (input_dir is None) == (paths is None):
        raise ValueError("provide exactly one of input_dir or paths")
    candidates = (
        sorted(Path(input_dir).glob(pattern))
        if input_dir is not None
        else [str(path) if "://" in str(path) else Path(path) for path in paths or ()]
    )
    requested = set(samples) if samples else None
    result, seen = [], {}
    for path in candidates:
        metadata = load_metadata(path)
        if requested is not None and metadata.sample_name not in requested:
            continue
        identity = (metadata.sample_name, _float_key(metadata.ak_radius))
        if identity in seen:
            raise ValueError(f"duplicate compact sample/AK {identity}: {seen[identity]} and {path}")
        seen[identity] = str(path)
        result.append(metadata)
    result.sort(key=lambda item: (item.sample_name, item.ak_radius, item.path))
    if requested is not None:
        missing = sorted(requested - {item.sample_name for item in result})
        if missing:
            raise FileNotFoundError("no compact files for: " + ", ".join(missing))
    return result


def _regular(array: Any, dtype: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(ak.to_numpy(array), dtype=dtype)
    except Exception as error:
        raise ValueError(f"branch {name} is not regular") from error
    if result.ndim != 2:
        raise ValueError(f"branch {name} has shape {result.shape}; expected 2D")
    return result


def read_event_payload(metadata_or_path: Any, max_events: int = -1) -> EventPayload:
    metadata = metadata_or_path if isinstance(metadata_or_path, CompactMetadata) else load_metadata(metadata_or_path)
    if max_events == 0 or max_events < -1:
        raise ValueError("max_events must be -1 or positive")
    stop = None if max_events < 0 else max_events
    names = ("run", "lumi", "event", "genWeight", "akJetPt", "nCAJets",
             "recoStatus", "nAmbiguous", "sj1Mass", "sj2Mass")
    if metadata.schema_version >= 2:
        names += ("suuMass",)
    with _open_root_file(metadata.path) as root_file:
        tree = root_file[EVENTS_PATH]
        if metadata.schema_version >= 2 and "suuMass" not in tree:
            raise ValueError(f"{metadata.path}: schema version 2 requires suuMass")
        events = tree.arrays(names, entry_stop=stop, library="ak")
    payload = EventPayload(
        np.asarray(ak.to_numpy(events["run"]), dtype=np.uint32),
        np.asarray(ak.to_numpy(events["lumi"]), dtype=np.uint32),
        np.asarray(ak.to_numpy(events["event"]), dtype=np.uint64),
        np.asarray(ak.to_numpy(events["genWeight"]), dtype=np.float64),
        events["akJetPt"], _regular(events["nCAJets"], np.uint16, "nCAJets"),
        _regular(events["recoStatus"], np.uint8, "recoStatus"),
        _regular(events["nAmbiguous"], np.uint16, "nAmbiguous"),
        _regular(events["sj1Mass"], np.float64, "sj1Mass"),
        _regular(events["sj2Mass"], np.float64, "sj2Mass"),
        (_regular(events["suuMass"], np.float64, "suuMass")
         if metadata.schema_version >= 2 else None),
    )
    if payload.reco_status.shape != payload.n_ambiguous.shape or payload.reco_status.shape != payload.sj1_mass.shape or payload.reco_status.shape != payload.sj2_mass.shape:
        raise ValueError(f"{metadata.path}: reconstruction branch shapes differ")
    if payload.reco_status.shape != (payload.n_events, metadata.n_configurations):
        raise ValueError(f"{metadata.path}: reconstruction width mismatch")
    if payload.n_ca_jets.shape != (payload.n_events, metadata.n_base_configurations):
        raise ValueError(f"{metadata.path}: base width mismatch")
    if payload.suu_mass is not None:
        if payload.suu_mass.shape != payload.reco_status.shape:
            raise ValueError(f"{metadata.path}: suuMass shape differs from reconstruction branches")
        valid = payload.reco_status == VALID_STATUS
        if np.any(valid & (~np.isfinite(payload.suu_mass) | (payload.suu_mass < 0))):
            raise ValueError(f"{metadata.path}: valid reconstruction has non-finite/negative suuMass")
        if np.any(~valid & ~np.isnan(payload.suu_mass)):
            raise ValueError(f"{metadata.path}: invalid reconstruction must have NaN suuMass")
    return payload


def gate_jet_multiplicity(payload: EventPayload, gate_pt_cut: float) -> np.ndarray:
    """Count stored-float AK pT values strictly greater than the cut."""

    _require_root_io()
    cut = np.float32(gate_pt_cut)
    if not np.isfinite(cut) or cut <= 0.0:
        raise ValueError("gate_pt_cut must be finite and positive")
    return np.asarray(ak.to_numpy(ak.sum(payload.ak_jet_pt > cut, axis=1)), dtype=np.int32)


def gate_mask(payload: EventPayload, n_gate_jets: int, gate_pt_cut: Optional[float]) -> np.ndarray:
    n_gate_jets = int(n_gate_jets)
    if n_gate_jets < 0:
        raise ValueError("n_gate_jets must be nonnegative")
    if n_gate_jets == 0:
        return np.ones(payload.n_events, dtype=bool)
    if gate_pt_cut is None:
        raise ValueError("positive n_gate_jets requires gate_pt_cut")
    return gate_jet_multiplicity(payload, gate_pt_cut) >= n_gate_jets


def configuration_indices(metadata: CompactMetadata, key: ConfigurationKey) -> Tuple[int, int]:
    if not math.isclose(key.ak_radius, metadata.ak_radius, abs_tol=1e-6):
        raise KeyError(f"AK {key.ak_radius} does not match file AK {metadata.ak_radius}")
    identity = (_float_key(key.collection_pt_cut), _float_key(key.ca_radius), _float_key(key.cos_thrust_cut))
    if identity not in metadata._config_lookup:
        raise KeyError(f"reconstruction configuration is absent: {identity}")
    index = metadata._config_lookup[identity]
    return index, metadata.config_base_index[index]


def _response_bin_indices(values: np.ndarray, definition: PhysicalityDefinition) -> np.ndarray:
    edges = definition.response_edges
    result = np.searchsorted(edges, values, side="right").astype(np.int32)
    core = (values >= edges[0]) & (values <= edges[-1])
    result[core] = np.minimum(result[core], definition.response_hist_bins)
    result[~np.isfinite(values)] = -1
    return result


def fixed_response_histogram(values: np.ndarray, definition: PhysicalityDefinition) -> np.ndarray:
    indices = _response_bin_indices(np.asarray(values, dtype=float), definition)
    return np.bincount(indices[indices >= 0], minlength=definition.response_hist_bins + 2).astype(np.int64)


def fixed_histogram_estimates(
    histograms: np.ndarray, definition: PhysicalityDefinition,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-bin median, principal FWHM, peak, and FWHM/peak."""

    hist = np.asarray(histograms, dtype=np.float64)
    scalar = hist.ndim == 1
    if scalar:
        hist = hist[None, :]
    expected = definition.response_hist_bins + 2
    if hist.ndim != 2 or hist.shape[1] != expected:
        raise ValueError(f"histogram shape must be (N,{expected})")
    total, cumulative = np.sum(hist, axis=1), np.cumsum(hist, axis=1)
    median_bin = np.argmax(cumulative >= (0.5 * total)[:, None], axis=1)
    width = (definition.response_hist_max - definition.response_hist_min) / definition.response_hist_bins
    median = definition.response_hist_min + (median_bin - 0.5) * width
    median = median.astype(float)
    median[(total <= 0) | (median_bin == 0) | (median_bin == definition.response_hist_bins + 1)] = np.nan
    core = hist[:, 1:-1]
    peak_index = np.argmax(core, axis=1)
    peak_count = core[np.arange(len(core)), peak_index]
    peak = definition.response_hist_min + (peak_index + 0.5) * width
    above = core >= (0.5 * peak_count)[:, None]
    bins = np.arange(definition.response_hist_bins)[None, :]
    left = np.max(np.where((bins < peak_index[:, None]) & ~above, bins, -1), axis=1) + 1
    right = np.min(np.where((bins > peak_index[:, None]) & ~above, bins, definition.response_hist_bins), axis=1) - 1
    fwhm = (right - left + 1).astype(float) * width
    invalid = (peak_count <= 0) | (peak <= 0.0)
    peak, fwhm = peak.astype(float), fwhm.astype(float)
    peak[invalid], fwhm[invalid] = np.nan, np.nan
    resolution = np.divide(fwhm, peak, out=np.full_like(fwhm, np.nan), where=np.isfinite(fwhm) & np.isfinite(peak) & (peak > 0))
    return (median[0], fwhm[0], peak[0], resolution[0]) if scalar else (median, fwhm, peak, resolution)


def _safe_divide(numerator: Any, denominator: Any) -> np.ndarray:
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    shape = np.broadcast_shapes(numerator.shape, denominator.shape)
    return np.divide(numerator, denominator, out=np.full(shape, np.nan), where=np.broadcast_to(denominator, shape) > 0)


def physicality_from_arrays(
    mass_bias: Any, fwhm_resolution: Any, invalid_fraction: Any,
    tail_fraction: Any, n_valid_events: Any, definition: PhysicalityDefinition,
) -> Dict[str, np.ndarray]:
    terms = np.broadcast_arrays(
        np.asarray(mass_bias, float) / definition.bias_limit,
        np.asarray(fwhm_resolution, float) / definition.resolution_limit,
        np.asarray(invalid_fraction, float) / definition.invalid_limit,
        np.asarray(tail_fraction, float) / definition.tail_limit,
        np.divide(definition.min_valid_events, np.asarray(n_valid_events, float),
                  out=np.full_like(np.asarray(n_valid_events, float), np.inf),
                  where=np.asarray(n_valid_events, float) > 0),
    )
    score = np.maximum.reduce([np.where(np.isfinite(term), term, np.inf) for term in terms])
    names = ("physicality_bias_term", "physicality_resolution_term",
             "physicality_invalid_term", "physicality_tail_term",
             "physicality_statistics_term")
    result = dict(zip(names, terms))
    result.update(physicality_score=score, physicality_pass=score <= 1.0)
    return result


def physicality_from_metrics(metrics: Mapping[str, Any], definition: PhysicalityDefinition) -> Dict[str, Any]:
    values = physicality_from_arrays(metrics["mass_bias"], metrics["fwhm_resolution"],
                                     metrics["invalid_fraction"], metrics["tail_fraction"],
                                     metrics["n_valid_events"], definition)
    return {name: (bool(value) if name == "physicality_pass" else float(value)) for name, value in values.items()}


def _metrics_from_histograms(
    histograms: np.ndarray, n_events: int, n_gate_events: np.ndarray,
    n_valid_events: np.ndarray, n_valid_all_events: np.ndarray,
    n_tail_events: np.ndarray, n_complexity_guard_events: np.ndarray,
    definition: PhysicalityDefinition,
) -> Dict[str, np.ndarray]:
    median, fwhm, peak, resolution = fixed_histogram_estimates(histograms, definition)
    gate, valid = np.asarray(n_gate_events, float), np.asarray(n_valid_events, float)
    gate_efficiency, reco = _safe_divide(gate, n_events), _safe_divide(valid, gate)
    retention, tail = _safe_divide(valid, n_events), _safe_divide(n_tail_events, valid)
    ungated_reco = _safe_divide(n_valid_all_events, n_events)
    peak_retention = _safe_divide(valid - np.asarray(n_tail_events, float), n_events)
    guard = np.asarray(n_complexity_guard_events, float)
    guard_fraction = _safe_divide(guard, gate)
    other_invalid = np.maximum(gate - valid - guard, 0.0)
    other_invalid_fraction = _safe_divide(other_invalid, gate)
    invalid, bias = 1.0 - reco, np.abs(median - 1.0)
    result = {
        "n_events": np.full(len(histograms), n_events, dtype=np.int64),
        "n_gate_events": np.asarray(n_gate_events, dtype=np.int64),
        "n_valid_events": np.asarray(n_valid_events, dtype=np.int64),
        "n_valid_all_events": np.asarray(n_valid_all_events, dtype=np.int64),
        "n_tail_events": np.asarray(n_tail_events, dtype=np.int64),
        "n_complexity_guard_events": np.asarray(n_complexity_guard_events, dtype=np.int64),
        "n_other_invalid_events": np.asarray(other_invalid, dtype=np.int64),
        "gate_efficiency": gate_efficiency,
        "reco_given_gate_efficiency": reco,
        "ungated_reco_efficiency": ungated_reco,
        "signal_retention": retention,
        "peak_signal_retention": peak_retention,
        "invalid_fraction": invalid,
        "complexity_guard_fraction_given_gate": guard_fraction,
        "other_invalid_fraction_given_gate": other_invalid_fraction,
        "tail_fraction": tail,
        "median_mass_response": median,
        "mass_bias": bias,
        "fwhm_mass_response": fwhm,
        "fwhm_peak_response": peak,
        "fwhm_resolution": resolution,
    }
    result.update(physicality_from_arrays(bias, resolution, invalid, tail, valid, definition))
    return result


def calculate_configuration_metrics(
    metadata: CompactMetadata, payload: EventPayload, key: ConfigurationKey,
    chi_mass: float, physicality: Optional[PhysicalityDefinition] = None,
) -> Dict[str, Any]:
    """Calculate a single selected configuration from event values."""

    definition = physicality or PhysicalityDefinition()
    if not math.isfinite(float(chi_mass)) or chi_mass <= 0:
        raise ValueError("chi_mass must be finite and positive")
    config_index, base_index = configuration_indices(metadata, key)
    gate = gate_mask(payload, key.n_gate_jets, key.gate_pt_cut)
    status = payload.reco_status[:, config_index]
    mass1, mass2 = payload.sj1_mass[:, config_index], payload.sj2_mass[:, config_index]
    finite = np.isfinite(mass1) & np.isfinite(mass2) & (mass1 >= 0) & (mass2 >= 0)
    valid_all, response1, response2 = (status == VALID_STATUS) & finite, mass1 / chi_mass, mass2 / chi_mass
    valid = gate & valid_all
    tail = valid & ((response1 < definition.tail_response_min) | (response1 > definition.tail_response_max) |
                    (response2 < definition.tail_response_min) | (response2 > definition.tail_response_max))
    hist = fixed_response_histogram(np.concatenate((response1[valid], response2[valid])), definition)[None, :]
    arrays = _metrics_from_histograms(
        hist, payload.n_events, np.array([np.count_nonzero(gate)]),
        np.array([np.count_nonzero(valid)]), np.array([np.count_nonzero(valid_all)]),
        np.array([np.count_nonzero(tail)]),
        np.array([np.count_nonzero(gate & (status == 6))]), definition,
    )
    result = {**key.as_dict(), "configuration": configuration_slug(key),
              "config_index": config_index, "base_index": base_index,
              "status_names": metadata.status_names}
    for name, values in arrays.items():
        result[name] = bool(values[0]) if name == "physicality_pass" else values[0].item()
    for code, name in zip(metadata.status_codes, metadata.status_names):
        result[f"status_{name}_all"] = int(np.count_nonzero(status == code))
        result[f"status_{name}_gated"] = int(np.count_nonzero(gate & (status == code)))
    result.update(sum_gen_weight_all=float(np.sum(payload.gen_weight)),
                  sum_gen_weight_gated=float(np.sum(payload.gen_weight[gate])),
                  sum_gen_weight_valid_gated=float(np.sum(payload.gen_weight[valid])))
    result.update(definition.as_dict())
    return result


def enumerate_configuration_keys(metadata: CompactMetadata) -> Tuple[ConfigurationKey, ...]:
    keys = []
    positive_counts = [n for n in metadata.default_gate_jet_counts if n > 0]
    for pt, ca_radius, cos_cut in zip(metadata.config_collection_pt_cut,
                                      metadata.config_ca_radius, metadata.config_cos_thrust):
        keys.append(configuration_key(0, None, pt, metadata.ak_radius, ca_radius, cos_cut))
        for n_gate in positive_counts:
            for gate_pt in metadata.default_gate_pt_cuts:
                if pt <= gate_pt + 1e-6:
                    keys.append(configuration_key(n_gate, gate_pt, pt, metadata.ak_radius, ca_radius, cos_cut))
    return tuple(keys)


def count_configuration_keys(metadata: CompactMetadata) -> int:
    """Count the gate-expanded grid without allocating key objects."""

    positive_gate_counts = sum(value > 0 for value in metadata.default_gate_jet_counts)
    return sum(
        1 + positive_gate_counts * sum(
            pt_cut <= gate_pt + 1.0e-6 for gate_pt in metadata.default_gate_pt_cuts
        )
        for pt_cut in metadata.config_collection_pt_cut
    )


def _histograms_for_columns(bin1, bin2, valid, histogram_size):
    n_events, n_columns = valid.shape
    columns = np.broadcast_to(np.arange(n_columns, dtype=np.int64), (n_events, n_columns))
    counts = np.zeros(histogram_size * n_columns, dtype=np.int64)
    for bins in (bin1, bin2):
        selected = valid & (bins >= 0)
        counts += np.bincount((bins.astype(np.int64) * n_columns + columns)[selected],
                              minlength=histogram_size * n_columns)
    return counts.reshape(histogram_size, n_columns).T


def _joint_gate_histograms(bin1, bin2, valid, buckets, n_thresholds, histogram_size):
    n_events, n_columns = valid.shape
    columns = np.broadcast_to(np.arange(n_columns, dtype=np.int64), (n_events, n_columns))
    counts = np.zeros((n_thresholds + 1) * histogram_size * n_columns, dtype=np.int64)
    for bins in (bin1, bin2):
        selected = valid & (bins >= 0)
        indices = ((buckets[:, None].astype(np.int64) * histogram_size + bins) * n_columns + columns)
        counts += np.bincount(indices[selected], minlength=len(counts))
    by_bucket = counts.reshape(n_thresholds + 1, histogram_size, n_columns)
    return np.cumsum(by_bucket[::-1], axis=0)[::-1]


def iter_configuration_metrics(
    metadata: CompactMetadata, payload: EventPayload, chi_mass: float,
    physicality: Optional[PhysicalityDefinition] = None, chunk_size: int = 64,
) -> Iterator[MetricBatch]:
    """Yield vectorized batches without an event-by-gate-by-config tensor."""

    definition = physicality or PhysicalityDefinition()
    if not math.isfinite(float(chi_mass)) or chi_mass <= 0:
        raise ValueError("chi_mass must be finite and positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    thresholds = np.asarray(metadata.default_gate_pt_cuts, dtype=float)
    positive_counts = [n for n in metadata.default_gate_jet_counts if n > 0]
    masks_by_n, counts_by_n, buckets_by_n = {}, {}, {}
    if len(thresholds):
        multiplicities = np.stack([gate_jet_multiplicity(payload, value) for value in thresholds])
        for n_gate in positive_counts:
            masks = multiplicities >= n_gate
            masks_by_n[n_gate], counts_by_n[n_gate] = masks, np.count_nonzero(masks, axis=1)
            buckets_by_n[n_gate] = np.count_nonzero(masks, axis=0)
    histogram_size, n_events = definition.response_hist_bins + 2, payload.n_events
    for start in range(0, metadata.n_configurations, chunk_size):
        stop = min(start + chunk_size, metadata.n_configurations)
        config_indices = np.arange(start, stop, dtype=np.int32)
        status, mass1, mass2 = payload.reco_status[:, start:stop], payload.sj1_mass[:, start:stop], payload.sj2_mass[:, start:stop]
        valid = (status == VALID_STATUS) & np.isfinite(mass1) & np.isfinite(mass2) & (mass1 >= 0) & (mass2 >= 0)
        response1, response2 = mass1 / chi_mass, mass2 / chi_mass
        bin1, bin2 = _response_bin_indices(response1, definition), _response_bin_indices(response2, definition)
        tail = valid & ((response1 < definition.tail_response_min) | (response1 > definition.tail_response_max) |
                        (response2 < definition.tail_response_min) | (response2 > definition.tail_response_max))
        valid_all_counts = np.count_nonzero(valid, axis=0)
        pt = np.asarray(metadata.config_collection_pt_cut[start:stop])
        ca_radius = np.asarray(metadata.config_ca_radius[start:stop])
        cos_cut = np.asarray(metadata.config_cos_thrust[start:stop])
        keys = [configuration_key(0, None, pt[i], metadata.ak_radius, ca_radius[i], cos_cut[i]) for i in range(stop - start)]
        histograms = [_histograms_for_columns(bin1, bin2, valid, histogram_size)]
        gate_counts = [np.full(stop - start, n_events, dtype=np.int64)]
        guard = status == 6
        valid_counts, valid_all = [valid_all_counts], [valid_all_counts]
        tail_counts = [np.count_nonzero(tail, axis=0)]
        guard_counts = [np.count_nonzero(guard, axis=0)]
        output_indices = [config_indices]
        for n_gate in positive_counts:
            masks = masks_by_n[n_gate]
            cumulative = _joint_gate_histograms(bin1, bin2, valid, buckets_by_n[n_gate], len(thresholds), histogram_size)
            gated_valid = masks.astype(np.int32) @ valid.astype(np.int32)
            gated_tail = masks.astype(np.int32) @ tail.astype(np.int32)
            gated_guard = masks.astype(np.int32) @ guard.astype(np.int32)
            for gate_index, gate_pt in enumerate(thresholds):
                selected = np.flatnonzero(pt <= gate_pt + 1e-6)
                if not len(selected):
                    continue
                keys.extend(configuration_key(n_gate, gate_pt, pt[i], metadata.ak_radius, ca_radius[i], cos_cut[i]) for i in selected)
                # With an advanced column index NumPy moves that axis first,
                # so this expression is already (selected config, histogram).
                histograms.append(cumulative[gate_index + 1, :, selected])
                gate_counts.append(np.full(len(selected), counts_by_n[n_gate][gate_index], dtype=np.int64))
                valid_counts.append(gated_valid[gate_index, selected])
                valid_all.append(valid_all_counts[selected])
                tail_counts.append(gated_tail[gate_index, selected])
                guard_counts.append(gated_guard[gate_index, selected])
                output_indices.append(config_indices[selected])
        values = _metrics_from_histograms(
            np.concatenate(histograms), n_events, np.concatenate(gate_counts),
            np.concatenate(valid_counts), np.concatenate(valid_all),
            np.concatenate(tail_counts), np.concatenate(guard_counts), definition,
        )
        values["config_index"] = np.concatenate(output_indices)
        values["base_index"] = np.asarray([metadata.config_base_index[i] for i in values["config_index"]], dtype=np.int32)
        yield MetricBatch(tuple(keys), values)


def metadata_manifest_row(metadata: CompactMetadata) -> Dict[str, Any]:
    return {
        "sample": metadata.sample_name, "ak_radius": metadata.ak_radius,
        "path": metadata.path, "schema_version": metadata.schema_version,
        "processed_events": metadata.processed_events,
        "sum_weights": metadata.sum_weights, "sum_weights2": metadata.sum_weights2,
        "n_reconstruction_configurations": metadata.n_configurations,
        "n_full_configurations": count_configuration_keys(metadata),
        "collection_pt_cuts": ";".join(map(str, metadata.collection_pt_cuts)),
        "ca_radii": ";".join(map(str, metadata.ca_radii)),
        "cos_thrust_cuts": ";".join(map(str, metadata.cos_thrust_cuts)),
        "gate_jet_counts": ";".join(map(str, metadata.default_gate_jet_counts)),
        "gate_pt_cuts": ";".join(map(str, metadata.default_gate_pt_cuts)),
    }
