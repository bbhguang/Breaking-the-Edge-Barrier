"""Experiment orchestration, private degree collection, and original-truth MRE."""
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import ROOT, validate
from .privacy import calibrate_degree, calibrate_wedge

PLAN_FIELDS = ["job_id", "tau", "p", "noise_scale", "projection_seed",
               "sampling_seed", "noise_seed", "shuffle_seed"]


def write_csv(path, rows, fields=None):
    if not rows and fields is None:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def engine_path():
    path = ROOT / "build" / "shuftri_engine"
    if not path.is_file():
        raise ValueError("C++ backend not found; run make in the project directory first")
    return path


def prepare_graph(source, cache_root, name, threads, engine):
    digest = file_digest(source)
    cache = Path(cache_root) / f"{name}-{digest[:16]}"
    stamp = cache / "input.json"
    expected = {"sha256": digest, "format_version": 1}
    try:
        reusable = json.loads(stamp.read_text()) == expected
        saved = json.loads((cache / "metadata.json").read_text())
        reusable = reusable and (cache / "degrees.bin").stat().st_size == 4 * saved["n"]
        graph_bytes = 24 + 8 * (saved["n"] + 1) + 8 * saved["edges"]
        reusable = reusable and (cache / "graph.bin").stat().st_size == graph_bytes
    except (OSError, ValueError, KeyError, TypeError):
        reusable = False
    if not reusable:
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(engine), "prepare", str(source), str(cache), str(threads)], check=True)
        stamp.write_text(json.dumps(expected, indent=2) + "\n")
    metadata = json.loads((cache / "metadata.json").read_text())
    degrees = np.fromfile(cache / "degrees.bin", dtype="<u4")
    if len(degrees) != metadata["n"]:
        raise ValueError("cached degree array has the wrong length")
    if metadata["triangles"] <= 0:
        raise ValueError("original-truth relative error is undefined for a graph with zero triangles")
    return cache, metadata, degrees, digest


def child_seed(sequence):
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def estimate_tau(degrees, epsilon_local, quantile, noise_seed, shuffle_seed):
    if epsilon_local <= 0:
        raise ValueError("degree noise budget must be positive")
    noisy = degrees.astype(np.float64) + np.random.default_rng(noise_seed).laplace(
        0.0, 1.0 / epsilon_local, len(degrees))
    # The collector receives a permuted vector, including in this simulation.
    np.random.default_rng(shuffle_seed).shuffle(noisy)
    value = float(np.quantile(noisy, quantile, method="linear"))
    return min(len(degrees) - 1, max(1, math.floor(value)))


def original_relative_error(estimate, truth):
    if truth <= 0:
        raise ValueError("MRE requires a positive original triangle count")
    if not math.isfinite(estimate):
        raise ValueError("non-finite estimate")
    return abs(estimate - truth) / truth


def run(config, data_root, output_dir, cache_dir, seed=None):
    validate(config)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty; existing results are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    engine = engine_path()
    master_seed = time.time_ns() if seed is None else int(seed)
    if master_seed < 0 or master_seed >= 2 ** 64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": master_seed, "seed_source": "system_time_ns" if seed is None else "explicit",
        "epsilons": config["epsilons"], "trials": config["trials"],
        "delta_total": config["delta_total"],
        "degree_epsilon_fraction": config["degree_epsilon_fraction"],
        "degree_delta_fraction": config["degree_delta_fraction"],
        "degree_quantile": config["degree_quantile"],
        "tau_reuse": "one noisy estimate per dataset and total budget",
        "projection": "independent uniform subsets, regenerated each trial; no symmetrization",
        "truth": "original undirected graph triangle count",
        "communication": "theoretical padded payload bytes, averaged per user; 1 KiB = 1024 bytes",
        "psi": "plaintext set-intersection cardinality simulation",
        "shuffle": "actual random permutation of degree and wedge report vectors",
        "calibration": "divide sampled effective budget by tau, then invert sampling relation",
        "python": platform.python_version(), "numpy": np.__version__,
        "threads": config["threads"], "backend_sha256": file_digest(engine), "datasets": {},
    }
    manifest_file = output / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    all_trials, all_groups, summaries = [], [], []
    try:
        for dataset_index, (name, relative_path) in enumerate(config["datasets"].items()):
            source = Path(data_root) / relative_path
            if not source.is_file():
                raise ValueError(f"input file not found for dataset {name}; check --data-root and configuration")
            print(f"Preparing {name}", flush=True)
            cache, metadata, degrees, digest = prepare_graph(source, cache_dir, name, config["threads"], engine)
            n, truth = int(metadata["n"]), int(metadata["triangles"])
            manifest["datasets"][name] = {**metadata, "input_sha256": digest}
            plans, jobs = [], {}
            for epsilon_index, epsilon in enumerate(config["epsilons"]):
                epsilon = float(epsilon)
                degree_target = config["degree_epsilon_fraction"] * epsilon
                wedge_target = epsilon - degree_target
                delta_degree = config["degree_delta_fraction"] * config["delta_total"]
                delta_wedge = config["delta_total"] - delta_degree
                degree_calibration = calibrate_degree(degree_target, n, delta_degree)
                group_stream = np.random.SeedSequence([master_seed, dataset_index, epsilon_index])
                streams = group_stream.spawn(2 + config["trials"])
                degree_seed, degree_shuffle_seed = child_seed(streams[0]), child_seed(streams[1])
                tau = estimate_tau(degrees, degree_calibration["epsilon_local"],
                                   config["degree_quantile"], degree_seed, degree_shuffle_seed)
                wedge = calibrate_wedge(wedge_target, n, delta_wedge, tau)
                group = {"dataset": name, "epsilon": epsilon, "n": n, "tau": tau,
                         "epsilon_degree_target": degree_target, "epsilon_wedge_target": wedge_target,
                         "delta_degree": delta_degree, "delta_wedge": delta_wedge,
                         "epsilon_degree_local": degree_calibration["epsilon_local"],
                         "epsilon_degree_formula_bound": degree_calibration["epsilon_bound"],
                         "degree_seed": degree_seed, "degree_shuffle_seed": degree_shuffle_seed,
                         **{f"wedge_{key}": value for key, value in wedge.items()}}
                all_groups.append(group)
                for trial in range(config["trials"]):
                    seeds = [child_seed(s) for s in streams[trial + 2].spawn(4)]
                    plan = dict(zip(PLAN_FIELDS, [len(plans), tau, wedge["p"], wedge["noise_scale"], *seeds]))
                    plans.append(plan)
                    jobs[plan["job_id"]] = (group, trial + 1, plan)
            plan_path = output / f"{name}_plans.csv"
            result_path = output / f"{name}_backend.csv"
            write_csv(plan_path, plans, PLAN_FIELDS)
            write_csv(output / "parameters.csv", all_groups)
            manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Running {name}: {len(plans)} trials", flush=True)
            subprocess.run([str(engine), "run", str(cache), str(plan_path), str(result_path),
                            str(config["threads"])], check=True)
            with result_path.open(newline="") as handle:
                backend_rows = list(csv.DictReader(handle))
            if len(backend_rows) != len(plans) or len({r["job_id"] for r in backend_rows}) != len(plans):
                raise ValueError("backend returned missing or duplicate trial results")
            dataset_rows = []
            for result in backend_rows:
                job_id = int(result["job_id"])
                group, trial, plan = jobs[job_id]
                estimate = float(result["estimate"])
                row = {"dataset": name, "epsilon": group["epsilon"], "trial": trial,
                       "tau": group["tau"], "p": plan["p"], "noise_scale": plan["noise_scale"],
                       "epsilon_noise": group["wedge_epsilon_noise"],
                       "triangles_original": truth, "estimate": estimate,
                       "relative_error_original": original_relative_error(estimate, truth),
                       "projected_arcs": int(result["projected_arcs"]),
                       "sampled_instances": int(result["sampled_instances"]),
                       "sampled_wedge_sum": int(result["sampled_wedge_sum"]),
                       "user_outbound_kib": float(result["user_outbound_kib"]),
                       "shuffler_inbound_kib": float(result["shuffler_inbound_kib"]),
                       "expected_user_outbound_kib": float(result["expected_user_outbound_kib"]),
                       **{key: plan[key] for key in PLAN_FIELDS if key.endswith("_seed")}}
                dataset_rows.append(row)
                all_trials.append(row)
            for epsilon in config["epsilons"]:
                selected = [r for r in dataset_rows if r["epsilon"] == float(epsilon)]
                summary = {"dataset": name, "epsilon": float(epsilon), "trials": len(selected),
                           "tau": selected[0]["tau"], "p": selected[0]["p"],
                           "triangles_original": truth,
                           "mre_original": float(np.mean([r["relative_error_original"] for r in selected])),
                           "user_outbound_kib": float(np.mean([r["user_outbound_kib"] for r in selected])),
                           "shuffler_inbound_kib": float(np.mean([r["shuffler_inbound_kib"] for r in selected])),
                           "expected_user_outbound_kib": float(np.mean([r["expected_user_outbound_kib"] for r in selected]))}
                summaries.append(summary)
                print(f"{name} epsilon={epsilon:g} MRE={summary['mre_original']:.6g} tau={summary['tau']} p={summary['p']:.6g}", flush=True)
            write_csv(output / "trials.csv", all_trials)
            write_csv(output / "summary.csv", summaries)
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_type"] = type(exc).__name__
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    return summaries
