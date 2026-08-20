"""Signal-region yields -> Combine counting datacards for the SIDM ABCD analysis.

The merged ``.coffea`` outputs store one ``Hist`` per observable with axes
``(channel, <observable>, abcd_region)``.  ``abcd_region`` is an ``IntCategory``
with ``0 = A``, the signal region of the ABCD plane, so the SR count for a
sample is simply that histogram summed over the observable axis at
``abcd_region == 0`` for the SR channel.

Two details matter for getting the count right:

* The histograms are already scaled to ``lumi * xs`` by
  ``sidm_processor.postprocess``, so the sums are yields, not raw entries.
  Signal has no entry in ``configs/cross_sections.yaml``, so
  ``utilities.get_xs`` falls back to 1 fb -- signal yields (and therefore the
  Combine signal strength ``r``) are relative to a 1 fb reference cross section.
* The observable axis is ``Regular`` and does overflow (a few percent for the
  pt axes), so sums must use ``flow=True``.  With flow included the SR sum
  reproduces the final row of the corresponding cutflow exactly.

The ``Weight`` storage carries the sum of squared weights, so each yield comes
with its MC statistical uncertainty, which is what the per-process MC
statistical nuisance in the datacards is built from.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import hist
from coffea.util import load

# --------------------------------------------------------------------------- #
# Sample locations and channel definitions
# --------------------------------------------------------------------------- #
BKG_DIR = (
    "/eos/uscms/store/user/dlee3/sidm_condor/ABCD_cosmic_veto/"
    "ABCD_landing_10ch_cosmic_veto_v1_bkg_full_merged_samples_v1"
)
SIGNAL_DIR = (
    "/eos/uscms/store/user/dlee3/sidm_condor/ABCD_cosmic_veto/"
    "ABCD_landing_10ch_cosmic_veto_v1_signal_full_merged_samples_v1"
)

# Signal cross section assumed by utilities.get_xs for 2Mu2E/4Mu samples, in pb.
# Combine's `r` is a multiplier on this reference.
SIGNAL_REF_XS_PB = 0.001
LUMI_PB = 59830.0  # 2018, from configs/run_periods.yaml

# ABCD region A -- the signal region of the ABCD plane.
SR_ABCD_REGION = 0


@dataclass(frozen=True)
class Channel:
    """One counting bin: an SR selection plus the histogram to count it with.

    ``hist_name`` is a histogram filled once per selected event in this
    channel, so summing it over its observable axis gives the event yield.
    ``signal_prefix`` is the signal-sample name prefix whose final state this
    channel targets.
    """

    name: str            # datacard bin name
    selection: str       # `channel` axis value in the coffea histograms
    hist_name: str       # histogram filled once per event in this channel
    signal_prefix: str   # "2Mu2E" or "4Mu"


CHANNELS = {
    "SR_2mu2e": Channel(
        name="SR_2mu2e",
        selection="test_SR_2mu2e_spread_cosAlpha_mu_veto",
        hist_name="abcd_2mu2e_mulj_pt",
        signal_prefix="2Mu2E",
    ),
    "SR_4mu": Channel(
        name="SR_4mu",
        selection="test_SR_4mu_spread_cosAlpha_mu_veto",
        hist_name="abcd_4mu_mulj0_pt",
        signal_prefix="4Mu",
    ),
}

# Backgrounds are merged into these groups so no datacard process is empty and
# the card stays readable.  Anything unmatched falls through to "other".
BKG_GROUPS = {
    "QCD": lambda s: s.startswith("QCD"),
    "DY": lambda s: s.startswith("DY"),
    "TT": lambda s: s.startswith("TT"),
    "Diboson": lambda s: s in {"WW", "WZ", "ZZ"},
}


def bkg_group(sample):
    """Map a background sample name onto its datacard process name."""
    for group, matches in BKG_GROUPS.items():
        if matches(sample):
            return group
    return "other"


# `<prefix>_<mMediator>GeV_<mDarkPhoton>GeV_<ctau>mm`, e.g. 2Mu2E_1000GeV_1p2GeV_0p96mm
SIGNAL_NAME = re.compile(
    r"^(?P<prefix>2Mu2E|4Mu)_(?P<mzd>[\dp]+)GeV_(?P<mdp>[\dp]+)GeV_(?P<ctau>[\dp]+)mm$"
)


def parse_signal_name(name):
    """Split a signal sample name into its physics parameters.

    Returns a dict with the final state and the three grid coordinates as
    floats (``p`` is the decimal point in these names), or ``None`` if the
    name does not look like a signal point.
    """
    m = SIGNAL_NAME.match(name)
    if not m:
        return None
    num = lambda s: float(s.replace("p", "."))
    return {
        "final_state": m["prefix"],
        "m_mediator": num(m["mzd"]),
        "m_darkphoton": num(m["mdp"]),
        "ctau": num(m["ctau"]),
    }


# --------------------------------------------------------------------------- #
# Yield extraction
# --------------------------------------------------------------------------- #
@dataclass
class Yield:
    """A yield and its MC statistical uncertainty."""

    value: float
    variance: float

    @property
    def error(self):
        return math.sqrt(max(self.variance, 0.0))

    @property
    def rel_error(self):
        """Fractional MC stat uncertainty; 0 for an empty (or negative) yield."""
        return self.error / self.value if self.value > 0 else 0.0

    def __add__(self, other):
        return Yield(self.value + other.value, self.variance + other.variance)


def sr_yield(sample_out, channel, flow=True):
    """SR (ABCD region A) yield for one sample in one channel.

    ``sample_out`` is ``output["out"][sample]`` from a merged coffea file.
    Returns ``None`` if the file does not contain this channel's histogram.
    """
    h = sample_out["hists"].get(channel.hist_name)
    if h is None:
        return None
    if channel.selection not in list(h.axes["channel"]):
        return None
    sliced = h[{"channel": channel.selection, "abcd_region": hist.loc(SR_ABCD_REGION)}]
    total = sliced.sum(flow=flow)
    return Yield(float(total.value), float(total.variance))


def read_coffea(path):
    """Load a merged coffea file and return ``{sample: sample_output}``."""
    return load(str(path))["out"]


def collect_yields(directory, channels=None, flow=True, progress=None):
    """Extract SR yields for every ``.coffea`` file in ``directory``.

    Returns ``{sample: {channel_name: Yield}}``.  Each merged file holds a
    single sample, but the loop tolerates several.
    """
    channels = channels or CHANNELS
    files = sorted(Path(directory).glob("*.coffea"))
    if not files:
        raise FileNotFoundError(f"no .coffea files under {directory}")

    out = {}
    for i, path in enumerate(files):
        if progress is not None:
            progress(i, len(files), path.name)
        for sample, sample_out in read_coffea(path).items():
            per_channel = {}
            for ch_name, channel in channels.items():
                y = sr_yield(sample_out, channel, flow=flow)
                if y is not None:
                    per_channel[ch_name] = y
            out[sample] = per_channel
    return out


def group_backgrounds(bkg_yields, channels=None):
    """Sum per-sample background yields into datacard process groups.

    Returns ``{channel_name: {process: Yield}}``.
    """
    channels = channels or CHANNELS
    grouped = {ch: {} for ch in channels}
    for sample, per_channel in bkg_yields.items():
        process = bkg_group(sample)
        for ch_name, y in per_channel.items():
            grouped[ch_name][process] = grouped[ch_name].get(process, Yield(0.0, 0.0)) + y
    return grouped


# --------------------------------------------------------------------------- #
# Datacard writing
# --------------------------------------------------------------------------- #
@dataclass
class DatacardConfig:
    """Knobs for the counting-experiment datacards.

    ``lumi_unc`` is the 2018 integrated-luminosity uncertainty (2.5%).
    ``bkg_norm_unc`` is an optional flat normalisation uncertainty applied to
    every background process -- a placeholder for the ABCD closure/transfer
    uncertainty, which is not derivable from the SR count alone.  Set it to
    ``None`` to write a card with only luminosity and MC statistical nuisances.
    ``min_bkg`` floors the total background so Combine has a non-zero
    expectation to build the Asimov dataset from.

    ``mc_stat`` selects how the MC statistical uncertainty on each background
    is modelled.  The SR backgrounds here come from one or two raw simulated
    events, where a log-normal is a bad description of the uncertainty, so the
    default is ``"gmN"``: Combine is told the effective number of raw entries
    ``N = (rate / error)^2`` and the per-entry weight ``alpha = rate / N``, and
    profiles the true rate with a Gamma distribution.  ``"lnN"`` falls back to
    a log-normal built from the same relative error, and ``None`` writes no MC
    statistical nuisance at all.  Signal is always treated with a lnN, since
    its relative MC error is at the sub-percent level.
    """

    lumi_unc: float = 0.025
    signal_unc: float | None = None
    bkg_norm_unc: float | None = None
    mc_stat: str | None = "gmN"
    mc_stat_cap: float = 2.0       # clip runaway lnN values from 1-2 MC events
    min_bkg: float = 1e-4
    min_process_rate: float = 1e-6  # processes at or below this are dropped
    observation: str = "bkg"        # "bkg" -> expected/blinded; or a float


def _fmt(x):
    return f"{x:.6g}"


def build_datacard(signal_name, signal_yield, bkg_processes, channel_name,
                   config=None, floored=None):
    """Render one single-bin counting datacard as text.

    ``bkg_processes`` is ``{process: Yield}``.  Processes at or below
    ``config.min_process_rate`` are dropped -- Combine cannot handle a
    zero-rate process carrying a nuisance.
    """
    config = config or DatacardConfig()

    kept = {
        name: y for name, y in bkg_processes.items()
        if y.value > config.min_process_rate
    }

    # Combine needs something non-zero to normalise the Asimov dataset to.  If
    # every background group is empty, keep one floored process so the card is
    # still usable, and let the caller know via `floored`.
    if not kept:
        kept = {"bkg": Yield(config.min_bkg, config.min_bkg ** 2)}
        if floored is not None:
            floored.append((channel_name, signal_name))

    processes = [("signal", signal_yield)] + list(kept.items())
    names = [name for name, _ in processes]

    # Resolve the MC statistical nuisances first: a gmN row constrains the rate
    # to be exactly N * alpha, so the rate column has to be built from the same
    # rounded numbers that go into the nuisance row.
    rates = [y.value for _, y in processes]
    stat_rows = []
    for i, (name, y) in enumerate(processes):
        if not config.mc_stat or y.rel_error <= 0:
            continue
        cells = ["-"] * len(processes)
        label = f"mcstat_{channel_name}_{name}"
        if config.mc_stat == "gmN" and name != "signal":
            n_raw = max(1, round(1.0 / y.rel_error ** 2))
            alpha = float(_fmt(y.value / n_raw))  # exactly as it is written out
            cells[i] = alpha
            rates[i] = n_raw * alpha
            stat_rows.append((label, f"gmN {n_raw}", cells))
        else:
            cells[i] = 1 + min(y.rel_error, config.mc_stat_cap)
            stat_rows.append((label, "lnN", cells))

    total_bkg = sum(rates[1:])
    observation = total_bkg if config.observation == "bkg" else float(config.observation)

    width = max(14, max(len(n) for n in names) + 2)
    col = lambda vals: "".join(f"{v:<{width}}" for v in vals)
    pad = 40  # keeps the header columns lined up with the nuisance rows

    lines = [
        f"# SIDM ABCD counting datacard -- {signal_name}, {channel_name}",
        f"# signal region = ABCD region A of selection "
        f"{CHANNELS[channel_name].selection}",
        f"# signal normalised to {SIGNAL_REF_XS_PB * 1000:g} fb at "
        f"{LUMI_PB / 1000:g} /fb, so r = sigma / {SIGNAL_REF_XS_PB * 1000:g} fb",
        "imax 1  number of bins",
        f"jmax {len(processes) - 1}  number of background processes",
        "kmax *  number of nuisance parameters",
        "-" * 80,
        f"bin          {channel_name}",
        f"observation  {_fmt(observation)}",
        "-" * 80,
        "bin".ljust(pad) + col([channel_name] * len(processes)),
        "process".ljust(pad) + col(names),
        # Combine's convention: signal <= 0, backgrounds >= 1.
        "process".ljust(pad) + col([str(i) for i in range(len(processes))]),
        "rate".ljust(pad) + col([_fmt(r) for r in rates]),
        "-" * 80,
    ]

    def nuisance(name, kind, values):
        cells = [v if isinstance(v, str) else _fmt(v) for v in values]
        return f"{name:<28}{kind:<12}" + col(cells)

    lines.append(nuisance("lumi_13TeV", "lnN", [1 + config.lumi_unc] * len(processes)))

    if config.signal_unc:
        lines.append(nuisance(
            "signal_norm", "lnN",
            [1 + config.signal_unc] + ["-"] * (len(processes) - 1),
        ))

    if config.bkg_norm_unc:
        lines.append(nuisance(
            "bkg_norm", "lnN",
            ["-"] + [1 + config.bkg_norm_unc] * (len(processes) - 1),
        ))

    lines.extend(nuisance(*row) for row in stat_rows)

    return "\n".join(lines) + "\n"


def write_datacards(signal_yields, bkg_grouped, outdir, channels=None,
                    config=None, only_matching_channel=True):
    """Write one datacard per (signal point, channel) under ``outdir``.

    With ``only_matching_channel`` each signal point is written only for the
    channel targeting its final state (2Mu2E -> SR_2mu2e, 4Mu -> SR_4mu); the
    cross-final-state yields are ~1e-4 events and carry no sensitivity.

    Returns ``(written_paths, floored)`` where ``floored`` lists the
    ``(channel, signal)`` pairs whose background had to be floored.
    """
    channels = channels or CHANNELS
    config = config or DatacardConfig()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written, floored = [], []
    for signal_name in sorted(signal_yields):
        info = parse_signal_name(signal_name)
        for ch_name, channel in channels.items():
            if only_matching_channel and info and info["final_state"] != channel.signal_prefix:
                continue
            sig = signal_yields[signal_name].get(ch_name)
            if sig is None or sig.value <= config.min_process_rate:
                continue
            card = build_datacard(
                signal_name, sig, bkg_grouped.get(ch_name, {}), ch_name,
                config=config, floored=floored,
            )
            path = outdir / f"datacard_{ch_name}_{signal_name}.txt"
            path.write_text(card)
            written.append(path)
    return written, floored
