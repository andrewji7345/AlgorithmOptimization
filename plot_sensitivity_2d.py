#!/usr/bin/env python3
"""Render and export the fixed two-dimensional sensitivity histograms.

All numerical arrays retain numpy.histogramdd ordering: [Suu bin, average
superjet-mass bin]. Only the plotting call transposes them. The ROOT yield
histograms store the actual sum of squared weights as their bin variances.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _stem(kind: str, name: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:72] or "unnamed"
    digest = hashlib.sha256((kind + "\0" + name).encode()).hexdigest()[:12]
    return f"{kind}_{readable}_{digest}"


def _edges(values: Any, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or len(axis) < 2 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
        raise ValueError(f"{name} must contain finite increasing bin edges")
    return axis


def _histogram(value: Mapping[str, Any], shape: tuple[int, int], name: str) -> dict[str, Any]:
    result = {}
    for field in ("sumw", "sumw2", "entries"):
        array = np.asarray(value[field], dtype=float)
        if array.shape != shape:
            raise ValueError(f"{name}.{field} has shape {array.shape}; expected {shape} in x,y order")
        if field != "sumw" and (np.any(~np.isfinite(array)) or np.any(array < 0)):
            raise ValueError(f"{name}.{field} must be finite and nonnegative")
        if field == "entries":
            if np.any(array != np.floor(array)):
                raise ValueError(f"{name}.entries must be integers")
            array = array.astype(np.int64)
        result[field] = array
    for field, item in value.items():
        if field not in result:
            result[field] = item
    result.setdefault("flow_entries", 0)
    return result


def build_plot_data(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate orientation and totals; prepare plots without changing any yield."""
    parameters = result["parameters"]
    x = _edges(parameters["mass_bin_edges_gev"], "mass_bin_edges_gev")
    y = _edges(parameters["chi_mass_bin_edges_gev"], "chi_mass_bin_edges_gev")
    shape = (len(x) - 1, len(y) - 1)
    background = _histogram(result["background"], shape, "background")
    samples = result["samples"]
    physics_metadata = result.get("comparison_samples", {})
    required = result["required_signals"]
    if not required or len(set(required)) != len(required):
        raise ValueError("required_signals must be nonempty and unique")
    records = []
    categories: dict[str, dict[str, Any]] = {}
    background_sum = {field: np.zeros(shape) for field in ("sumw", "sumw2", "entries")}

    def append(kind: str, name: str, histogram: Mapping[str, Any], **extra: Any) -> None:
        physics = {key: physics_metadata.get(name, {}).get(key) for key in
                   ("generated_suu_mass_gev", "generated_chi_mass_gev", "decay_channel", "dataset")
                   if physics_metadata.get(name, {}).get(key) is not None}
        records.append({"id": _stem(kind, name), "kind": kind, "name": name,
                        "histogram": histogram, "physics_metadata": physics, **extra})

    for name, sample in samples.items():
        hist = _histogram(sample["histogram"], shape, name)
        if sample["kind"] == "background":
            category = str(sample.get("category") or "Uncategorized")
            if category not in categories:
                categories[category] = {field: np.zeros(shape) for field in ("sumw", "sumw2", "entries")}
                categories[category]["flow_entries"] = 0
                categories[category]["members"] = []
            group = categories[category]
            for field in ("sumw", "sumw2", "entries"):
                background_sum[field] += hist[field]
                group[field] += hist[field]
            group["flow_entries"] += int(hist["flow_entries"])
            group["members"].append(name)
            # Preserve available scalar weighted flow diagnostics as well.
            for field in ("flow_sumw", "flow_sumw2"):
                if field in hist:
                    group[field] = group.get(field, 0.0) + float(hist[field])
            append("background_sample", name, hist, category=category)
        elif sample["kind"] == "signal":
            append("signal", name, hist)
        else:
            raise ValueError(f"{name}: unsupported sample kind")
    if not categories:
        raise ValueError("at least one background sample is required")
    for field in ("sumw", "sumw2", "entries"):
        if not np.allclose(background_sum[field], background[field], rtol=1e-10, atol=1e-10, equal_nan=True):
            raise ValueError(f"total background {field} differs from sum of background samples")
    for category, hist in categories.items():
        members = hist.pop("members")
        hist["entries"] = hist["entries"].astype(np.int64)
        append("background_category", category, hist, members=members)
    append("background_total", "Total background", background)
    required_threshold = float(parameters.get("minimum_background_effective_events", 0.0))
    threshold = float(parameters.get("plot_minimum_background_effective_events", max(10.0, required_threshold)))
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("plot_minimum_background_effective_events must be finite and positive")
    b, v = background["sumw"], background["sumw2"]
    neff = np.divide(b * b, v, out=np.full(shape, np.inf), where=v > 0)
    neff[(b == 0) & (v == 0)] = 0
    low_stats = (b > 0) & (neff < threshold)
    for name in required:
        if name not in samples or samples[name]["kind"] != "signal":
            raise ValueError(f"required signal {name!r} has no signal histogram")
        info = result["per_signal"][name]
        q0 = np.asarray(info.get("per_bin_q0", np.full(shape, np.nan)), dtype=float)
        valid = np.asarray(info.get("per_bin_valid", np.isfinite(q0) & (q0 >= 0)), dtype=bool)
        supported = np.asarray(info.get("per_bin_supported", valid), dtype=bool)
        if q0.shape != shape or valid.shape != shape or supported.shape != shape:
            raise ValueError(f"{name}: per-bin score or validity shape differs from histogram")
        if np.any(valid & (~np.isfinite(q0) | (q0 < 0))):
            raise ValueError(f"{name}: valid score cells require finite nonnegative q0")
        # Undefined scores remain undefined; no background floors or clipped yields.
        z = np.full(shape, np.nan)
        z[valid] = np.sqrt(q0[valid])
        append("asimov", name, samples[name]["histogram"], values=z, per_bin_q0=q0,
               per_bin_valid=valid, per_bin_supported=supported, background_effective_events=neff,
               low_background_statistics=low_stats,
               significance=info.get("significance"), feasible=bool(info["feasible"]),
               failure_reasons=list(info.get("failure_reasons", [])))
    return {"schema_version": 1, "array_order": ["suu_mass_bin", "average_superjet_mass_bin"],
            "configuration_slug": result.get("configuration_slug", "unspecified"),
            "purpose": result.get("purpose", "unspecified"),
            "parameters": dict(parameters), "x_edges_gev": x, "y_edges_gev": y,
            "background_effective_events": neff, "low_background_statistics": low_stats,
            "minimum_background_effective_events": required_threshold,
            "plot_minimum_background_effective_events": threshold, "plots": records,
            "notes": ["Yield histograms retain signed weights without clipping.",
                      "JSON null denotes nonfinite or undefined values; validity masks are explicit.",
                      "ROOT yield bin errors use the stored sum of squared event weights.",
                      "ROOT score and diagnostic maps have zero stored error; no error estimate is assigned to these derived quantities.",
                      "ROOT global mass moments use bin centres, not original event coordinates.",
                      "Low-statistics hatching is diagnostic. Its threshold is not the AN 17.5% superbin requirement and does not set objective eligibility.",
                      "The displayed per-cell Asimov score uses the evaluator's finite-MC approximation; it is not a full analysis likelihood.",
                      "A populated cell may be displayed even if another cell makes the complete configuration infeasible."]}


def _th2(name: str, title: str, values: np.ndarray, variance: np.ndarray,
         entries: float, x: np.ndarray, y: np.ndarray) -> Any:
    from uproot.writing.identify import to_TAxis, to_TH2x
    xx, yy = np.meshgrid((x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2, indexing="ij")
    return to_TH2x(
        fName=name, fTitle=title,
        data=np.pad(values, 1).T.ravel().astype(np.float64),
        fEntries=float(entries), fTsumw=float(np.sum(values)), fTsumw2=float(np.sum(variance)),
        fTsumwx=float(np.sum(values * xx)), fTsumwx2=float(np.sum(values * xx * xx)),
        fTsumwy=float(np.sum(values * yy)), fTsumwy2=float(np.sum(values * yy * yy)),
        fTsumwxy=float(np.sum(values * xx * yy)),
        fSumw2=np.pad(variance, 1).T.ravel().astype(np.float64),
        fXaxis=to_TAxis("xaxis", "Reconstructed Suu mass [GeV]", len(x) - 1, x[0], x[-1], x),
        fYaxis=to_TAxis("yaxis", "Average reconstructed superjet mass [GeV]", len(y) - 1, y[0], y[-1], y),
    )


def _write_root(data: Mapping[str, Any], path: Path) -> None:
    import uproot
    x, y = data["x_edges_gev"], data["y_edges_gev"]
    with uproot.recreate(path) as root:
        root["plot_metadata"] = json.dumps(_json_safe({key: value for key, value in data.items() if key != "plots"}), allow_nan=False)
        for record in data["plots"]:
            prefix = record["id"]
            if record["kind"] == "asimov":
                fields = {"z": record["values"], "q0": record["per_bin_q0"],
                          "valid": record["per_bin_valid"].astype(float),
                          "supported": record["per_bin_supported"].astype(float),
                          "low_background_statistics": record["low_background_statistics"].astype(float),
                          "background_effective_events": record["background_effective_events"]}
                for field, values in fields.items():
                    root[f"{prefix}/{field}"] = _th2(field, record["name"] + ": " + field,
                                                       np.asarray(values), np.zeros_like(values), 0, x, y)
            else:
                hist = record["histogram"]
                values, variance = np.asarray(hist["sumw"]), np.asarray(hist["sumw2"])
                root[f"{prefix}/yield"] = _th2(prefix, record["name"], values, variance,
                                               float(np.asarray(hist["entries"]).sum()), x, y)
                root[f"{prefix}/sumw2"] = (variance, x, y)
                root[f"{prefix}/entries"] = (np.asarray(hist["entries"], dtype=float), x, y)
            root[f"{prefix}/metadata"] = json.dumps(_json_safe({
                "kind": record["kind"], "name": record["name"], "category": record.get("category"),
                "members": record.get("members"),
                "physics_metadata": record.get("physics_metadata", {}),
                "flow": {key: value for key, value in record["histogram"].items()
                         if key not in ("sumw", "sumw2", "entries")}}), allow_nan=False)


def _draw_plot(data: Mapping[str, Any], record: Mapping[str, Any], directory: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Patch, Rectangle

    x, y = data["x_edges_gev"], data["y_edges_gev"]
    score = record["kind"] == "asimov"
    values = np.asarray(record["values"] if score else record["histogram"]["sumw"], dtype=float)
    finite = np.isfinite(values)
    positive = finite & (values > 0)
    negative = finite & (values < 0)
    low = np.asarray(data["low_background_statistics"], dtype=bool) if record["kind"] in ("asimov", "background_total") else np.zeros_like(finite)
    invalid = ~finite
    fig, ax = plt.subplots(figsize=(11.4, 9.2))
    fig.subplots_adjust(left=.11, right=.84, bottom=.19, top=.83)
    if score:
        largest = max(float(np.max(values[finite])) if np.any(finite) else 0.0, 1e-9)
        norm, cmap = Normalize(0, largest), plt.get_cmap("viridis").copy()
        shown = np.ma.masked_where(~finite, values)
        colour_label = r"Per-bin Asimov $Z_A$"
    else:
        lower = float(np.min(values[positive])) if np.any(positive) else 1.0
        upper = float(np.max(values[positive])) if np.any(positive) else 10.0
        if upper <= lower:
            lower, upper = lower / math.sqrt(10), upper * math.sqrt(10)
        norm, cmap = LogNorm(lower, upper), plt.get_cmap("viridis").copy()
        shown = np.ma.masked_where(~positive, values)
        colour_label = "Expected events / cell (log colour scale)"
    cmap.set_bad("#f3f3f3")
    mesh = ax.pcolormesh(x, y, shown.T, cmap=cmap, norm=norm,
                         edgecolors="#d0d0d0", linewidth=.4, shading="flat", rasterized=True)
    colorbar_ax = fig.add_axes([.87, .19, .021, .64])
    fig.colorbar(mesh, cax=colorbar_ax, label=colour_label)
    for ix, iy in np.ndindex(values.shape):
        style = None
        if negative[ix, iy]:
            style = {"facecolor": "#f1c1df", "edgecolor": "#82276b", "hatch": "xx"}
        elif invalid[ix, iy]:
            style = {"facecolor": "#e4e4e4", "edgecolor": "#777777", "hatch": "xx"}
        if style:
            ax.add_patch(Rectangle((x[ix], y[iy]), x[ix + 1] - x[ix], y[iy + 1] - y[iy],
                                   linewidth=.25, **style))
        if low[ix, iy]:
            ax.add_patch(Rectangle((x[ix], y[iy]), x[ix + 1] - x[ix], y[iy + 1] - y[iy],
                                   facecolor="none", edgecolor="#c77800", hatch="//", linewidth=.4))
        if values.size <= 400:
            value = values[ix, iy]
            label = f"{value:.2g}" if math.isfinite(value) else "—"
            colour = "black"
            if finite[ix, iy] and (score or positive[ix, iy]) and float(norm(value)) < .5:
                colour = "white"
            ax.text((x[ix] + x[ix + 1]) / 2, (y[iy] + y[iy + 1]) / 2, label,
                    ha="center", va="center", color=colour, fontsize=7 if values.size <= 64 else 5.7)
    ax.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]),
           xlabel=r"Reconstructed Suu mass $m(SJ_1+SJ_2)$ [GeV]",
           ylabel=r"Average superjet mass $(m_{SJ_1}+m_{SJ_2})/2$ [GeV]")
    if len(x) <= 19:
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelrotation=45)
    if len(y) <= 21:
        ax.set_yticks(y)
    ax.tick_params(labelsize=8)
    kind_title = {"signal": "Signal", "background_sample": "Background input sample",
                  "background_category": "Background class", "background_total": "Summed background",
                  "asimov": "Asimov sensitivity"}[record["kind"]]
    fig.text(.11, .955, kind_title, fontsize=16, weight="bold")
    label = record["name"]
    physics = record.get("physics_metadata", {})
    actual_masses = []
    for field, particle in (("generated_suu_mass_gev", "Suu"), ("generated_chi_mass_gev", "chi")):
        if field in physics:
            actual_masses.append(f"m({particle})={float(physics[field]) / 1000:g} TeV")
    if actual_masses:
        label += " | generated " + ", ".join(actual_masses)
    fig.text(.11, .924, textwrap.fill(label, 91), fontsize=11, va="top")
    lumi = data["parameters"].get("luminosity_pb")
    context = "Simulation prediction"
    if lumi is not None:
        context += f" | {float(lumi) / 1000:g} fb⁻¹"
    context += f" | {data['purpose']}"
    fig.text(.11, .865, context, fontsize=9)
    fig.text(.11, .096, textwrap.fill("Configuration: " + str(data["configuration_slug"]), 115), fontsize=8, va="top")
    handles = []
    if np.any(negative):
        handles.append(Patch(facecolor="#f1c1df", edgecolor="#82276b", hatch="xx", label="Negative signed yield"))
    if np.any(invalid):
        handles.append(Patch(facecolor="#e4e4e4", edgecolor="#777777", hatch="xx", label="Undefined / invalid cell"))
    if np.any(low):
        handles.append(Patch(facecolor="none", edgecolor="#c77800", hatch="//",
                             label=f"Diagnostic: background MC effective count < {data['plot_minimum_background_effective_events']:g}"))
    if not score and np.any(finite & (values == 0)):
        handles.append(Patch(facecolor="#f3f3f3", edgecolor="#d0d0d0", label="Zero yield"))
    if handles:
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.105, .047),
                   frameon=False, fontsize=8, ncol=2)
    if score:
        total = record["significance"]
        total_text = f"{float(total):.5g}" if total is not None and math.isfinite(total) else "unavailable"
        footer = f"All-bin Asimov Z: {total_text}; objective {'eligible' if record['feasible'] else 'ineligible'}."
        if record["failure_reasons"]:
            footer += " " + "; ".join(record["failure_reasons"])
    else:
        flow = record["histogram"].get("flow_entries", 0)
        mode = data["parameters"].get("histogram_flow", "unspecified")
        footer = f"Signed yield in displayed bins: {float(np.sum(values)):.6g}. Out-of-range selected entries: {flow}; flow policy: {mode}."
    fig.text(.11, .015, textwrap.fill(footer, 135), fontsize=7.4, va="bottom")
    files = []
    try:
        for suffix in ("png", "pdf"):
            path = directory / f"{record['id']}.{suffix}"
            fig.savefig(path, dpi=160)
            files.append(path.name)
    finally:
        plt.close(fig)
    return files


def plot_2d_outputs(result: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Write PNG/PDF plots, full bin JSON, ROOT TH2s, and a filename manifest."""
    data = build_plot_data(result)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    plot_files = []
    for record in data["plots"]:
        record["files"] = _draw_plot(data, record, directory)
        plot_files.extend(record["files"])
    json_path = directory / "histograms_2d.json"
    json_path.write_text(json.dumps(_json_safe(data), indent=2, allow_nan=False) + "\n")
    root_path = directory / "histograms_2d.root"
    _write_root(data, root_path)
    manifest = {"schema_version": 1, "array_order": data["array_order"],
                "configuration_slug": data["configuration_slug"],
                "plotter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "input_result_sha256": hashlib.sha256(json.dumps(_json_safe(result), sort_keys=True,
                                                                  allow_nan=False).encode()).hexdigest(),
                "histogram_json": json_path.name, "histogram_root": root_path.name,
                "plots": [{key: record[key] for key in ("id", "kind", "name", "files")} for record in data["plots"]],
                "artifacts_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                     for name in [*plot_files, json_path.name, root_path.name]}}
    (directory / "plots_2d_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest
