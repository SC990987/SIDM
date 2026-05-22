#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import awkward as ak
import numpy as np
from coffea.util import load


SIGNAL_XSEC = 0.001

BACKGROUND_XSECS = {
    "DYJetsToMuMu_M10to50": 7013.0,
    "DYJetsToMuMu_M50": 1976.0,
    "QCD_Pt1000": 1.085,
    "QCD_Pt120To170": 21280.0,
    "QCD_Pt15To20": 2800000.0,
    "QCD_Pt170To300": 7000.0,
    "QCD_Pt20To30": 2527000.0,
    "QCD_Pt300To470": 622.6,
    "QCD_Pt30To50": 1367000.0,
    "QCD_Pt470To600": 58.9,
    "QCD_Pt50To80": 381700.0,
    "QCD_Pt600To800": 18.12,
    "QCD_Pt800To1000": 3.318,
    "QCD_Pt80To120": 87740.0,
    "TTJets": 471.7,
}


def run_cmd(cmd, check=True):
    print("+", " ".join(cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.stdout.strip():
        print(result.stdout.strip())

    if result.stderr.strip():
        print(result.stderr.strip())

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )

    return result


def eos_ls(eos_dir):
    result = run_cmd(
        ["xrdfs", "root://cmseos.fnal.gov", "ls", eos_dir],
        check=True,
    )

    files = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().endswith(".coffea")
    ]

    return files


def eos_mkdir(eos_dir):
    run_cmd(
        ["xrdfs", "root://cmseos.fnal.gov", "mkdir", "-p", eos_dir],
        check=True,
    )


def xrdcp_from_eos(eos_path, local_path):
    """
    eos_path should look like:
        /store/user/scampbel/...
    """
    url = f"root://cmseos.fnal.gov/{eos_path}"
    run_cmd(["xrdcp", "-f", url, str(local_path)], check=True)


def xrdcp_to_eos(local_path, eos_dir):
    basename = os.path.basename(local_path)
    url = f"root://cmseos.fnal.gov/{eos_dir.rstrip('/')}/{basename}"
    run_cmd(["xrdcp", "-f", str(local_path), url], check=True)


def make_jsonable(x):
    """
    Convert common numpy/awkward objects to JSON-safe Python objects.
    If debug branches are already plain Python lists, this leaves them mostly unchanged.
    """
    if isinstance(x, ak.Array):
        return ak.to_list(x)

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, np.generic):
        return x.item()

    if isinstance(x, dict):
        return {str(k): make_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [make_jsonable(v) for v in x]

    return x


def get_debug_dict(sample_output):
    """
    Supports both possible coffea structures:

        out["out"][sample]["debug"]

    and

        out["out"][sample]["out"]["debug"]
    """
    if isinstance(sample_output, dict) and "debug" in sample_output:
        return sample_output["debug"]

    if (
        isinstance(sample_output, dict)
        and "out" in sample_output
        and isinstance(sample_output["out"], dict)
        and "debug" in sample_output["out"]
    ):
        return sample_output["out"]["debug"]

    raise KeyError(
        "Could not find debug dictionary. Expected either "
        'out["out"][sample]["debug"] or out["out"][sample]["out"]["debug"].'
    )


def get_xsec(sample_name, sample_type):
    if sample_type == "signal":
        return SIGNAL_XSEC

    if sample_type == "background":
        if sample_name not in BACKGROUND_XSECS:
            raise KeyError(
                f"No background xsec found for sample '{sample_name}'. "
                f"Known samples: {sorted(BACKGROUND_XSECS)}"
            )
        return BACKGROUND_XSECS[sample_name]

    raise ValueError(f"Unknown sample_type: {sample_type}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy merged .coffea files from EOS and convert their debug outputs to JSON."
        )
    )

    parser.add_argument(
        "--input-eos-dir",
        required=True,
        help=(
            "EOS directory containing merged .coffea files, e.g. "
            "/store/user/scampbel/sidm_condor/SignalMerged_v1"
        ),
    )

    parser.add_argument(
        "--output-json-dir",
        required=True,
        help="Local output directory for JSON files.",
    )

    parser.add_argument(
        "--sample-type",
        choices=["signal", "background"],
        required=True,
        help=(
            "Use 'signal' for fixed xsec=0.001, or 'background' for "
            "BACKGROUND_XSECS lookup."
        ),
    )

    parser.add_argument(
        "--output-eos-dir",
        default=None,
        help="Optional EOS directory to copy JSON outputs to.",
    )

    parser.add_argument(
        "--workdir",
        default="coffea_to_json_tmp",
        help="Local temporary directory for staged .coffea files.",
    )

    parser.add_argument(
        "--keep-local-coffea",
        action="store_true",
        help="Keep staged .coffea files after conversion.",
    )

    args = parser.parse_args()

    input_eos_dir = args.input_eos_dir.rstrip("/")
    output_json_dir = Path(args.output_json_dir)
    workdir = Path(args.workdir)

    output_json_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    if args.output_eos_dir is not None:
        eos_mkdir(args.output_eos_dir.rstrip("/"))

    print("=" * 80)
    print(f"Input EOS directory: {input_eos_dir}")
    print(f"Local JSON output directory: {output_json_dir}")
    print(f"Temporary workdir: {workdir}")
    print(f"sample_type: {args.sample_type}")
    print("=" * 80)

    eos_files = eos_ls(input_eos_dir)

    print(f"Found {len(eos_files)} merged .coffea files on EOS")

    if len(eos_files) == 0:
        raise RuntimeError(f"No .coffea files found in {input_eos_dir}")

    for i, eos_file in enumerate(sorted(eos_files), start=1):
        basename = os.path.basename(eos_file)
        local_coffea = workdir / basename

        print("=" * 80)
        print(f"[{i}/{len(eos_files)}] Processing {basename}")

        if not local_coffea.exists():
            xrdcp_from_eos(eos_file, local_coffea)
        else:
            print(f"Already staged locally: {local_coffea}")

        out = load(local_coffea)

        if "out" not in out:
            raise KeyError(f"{basename} does not contain top-level key 'out'.")

        sample_names = list(out["out"].keys())

        if len(sample_names) != 1:
            raise RuntimeError(
                f"{basename} has {len(sample_names)} samples: {sample_names}"
            )

        sample_name = sample_names[0]
        xsec = get_xsec(sample_name, args.sample_type)

        print(f"sample_name = {sample_name}")
        print(f"xsec = {xsec}")

        sample_output = out["out"][sample_name]
        debug = get_debug_dict(sample_output)

        out_dict = {
            "sample_name": sample_name,
            "xsec": xsec,
        }

        for k, v in debug.items():
            if k == "gen_weights":
                continue
            out_dict[k] = make_jsonable(v)

        if "gen_weights" in debug:
            gen_weights = np.asarray(debug["gen_weights"])
            sumw = float(np.sum(gen_weights))

            # Keep nevents for compatibility with your old plotting workflow.
            # This is the weighted sum of gen weights, not the raw event count.
            out_dict["nevents"] = sumw
            out_dict["sumw"] = sumw

            # Raw number of events before weighting.
            out_dict["nevents_raw"] = int(len(gen_weights))
        else:
            print("WARNING: gen_weights not found, skipping nevents/sumw")

        json_path = output_json_dir / f"{sample_name}.json"

        with open(json_path, "w") as f_out:
            json.dump(out_dict, f_out, indent=4)

        print(f"Wrote {json_path}")

        if args.output_eos_dir is not None:
            xrdcp_to_eos(json_path, args.output_eos_dir.rstrip("/"))

    if not args.keep_local_coffea:
        print(f"Removing temporary workdir: {workdir}")
        shutil.rmtree(workdir, ignore_errors=True)

    print("=" * 80)
    print("Done")


if __name__ == "__main__":
    main()

