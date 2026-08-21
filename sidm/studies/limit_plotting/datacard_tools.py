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

from sidm import BASE_DIR

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

# The ABCD plane is built from the two lepton-jet isolation variables (the
# `abcd_iso_base` hist collection).  Region A is the doubly-isolated corner,
# i.e. the signal region; D is the corner diagonally opposite it, so the
# closure relation is A = B*C/D.
SR_ABCD_REGION = 0
ABCD_REGIONS = {0: "A", 1: "B", 2: "C", 3: "D"}


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

    def scaled(self, factor):
        """Yield rescaled by ``factor`` (e.g. to a different luminosity).

        The variance scales as ``factor**2``, so the *relative* MC statistical
        error is unchanged -- which is correct: extrapolating to more
        luminosity does not create more simulated events.
        """
        return Yield(self.value * factor, self.variance * factor ** 2)


class BlindingError(RuntimeError):
    """Raised when an operation would expose the signal region of real data."""


# Primary-dataset prefixes, used only to make the error message specific.  The
# blinding decision does NOT rely on this list -- see is_simulation.
DATA_NAME_HINTS = ("DoubleMuon", "SingleMuon", "EGamma", "SingleElectron",
                   "DoubleEG", "MuonEG", "MET", "JetHT", "Charmonium", "Muon")


def is_simulation(sample_name, sample_out=None, cfg="cross_sections.yaml"):
    """True only if ``sample_name`` can be positively identified as simulation.

    This deliberately does **not** trust ``metadata["is_data"]``: the merge step
    empties that accumulator, so it is an empty set for data and simulation
    alike and would silently report real data as MC.  ``metadata["year"]``
    happens to be filled only for data in the current files, but that is an
    accident of the processor rather than a contract.

    So the test is inverted into an allow-list: a sample counts as simulation
    only if it is a known signal point or carries a cross section in
    ``cross_sections.yaml``.  Anything unrecognised is treated as data and has
    its signal region withheld -- the safe direction to fail in.
    """
    if sample_name.startswith(("2Mu2E", "4Mu")):
        return True
    try:
        from sidm.tools import utilities
        return sample_name in utilities.load_yaml(f"{BASE_DIR}/configs/{cfg}")
    except Exception:
        # If the config cannot be read we cannot prove this is simulation.
        return False


def blinding_reason(sample_name):
    """Human-readable explanation of why a sample's SR is withheld."""
    if any(sample_name.startswith(h) for h in DATA_NAME_HINTS):
        return f"{sample_name!r} looks like a data primary dataset"
    return (f"{sample_name!r} is not a known signal point and has no entry in "
            f"cross_sections.yaml, so it cannot be confirmed to be simulation")


def region_yields(sample_out, channel, sample_name="", flow=True,
                  allow_data_sr=False):
    """Yields in all four ABCD regions for one sample in one channel.

    Returns ``{region_index: Yield}``, or ``None`` if the file does not contain
    this channel's histogram.

    ``flow=True`` is the default and should stay that way: the observable axes
    are ``Regular`` and overflow at the few-percent level, and only with flow
    included does the sum over all four regions reproduce the final cutflow row.

    Blinding: unless the sample is positively identified as simulation (see
    ``is_simulation``) region A -- the signal region -- is omitted from the
    result, so a blinded run simply has no SR key to plot or tabulate.  Control
    regions B, C and D are always returned: they are what the background
    estimate is built from and are not blinded.  Pass ``allow_data_sr=True``
    only when the analysis has genuinely been unblinded.
    """
    h = sample_out["hists"].get(channel.hist_name)
    if h is None:
        return None
    if channel.selection not in list(h.axes["channel"]):
        return None

    blind_sr = not allow_data_sr and not is_simulation(sample_name, sample_out)
    out = {}
    for region in ABCD_REGIONS:
        if region == SR_ABCD_REGION and blind_sr:
            continue
        sliced = h[{"channel": channel.selection, "abcd_region": hist.loc(region)}]
        total = sliced.sum(flow=flow)
        out[region] = Yield(float(total.value), float(total.variance))
    return out


def sr_yield(sample_out, channel, sample_name="", flow=True, allow_data_sr=False):
    """Signal-region (ABCD region A) yield for one sample in one channel.

    Raises ``BlindingError`` unless the sample is known simulation or
    ``allow_data_sr`` is set.
    """
    if not allow_data_sr and not is_simulation(sample_name, sample_out):
        raise BlindingError(
            f"refusing to read the signal region of {channel.name}: "
            f"{blinding_reason(sample_name)}. Pass allow_data_sr=True only "
            f"when the analysis has actually been unblinded."
        )
    regions = region_yields(sample_out, channel, sample_name=sample_name,
                            flow=flow, allow_data_sr=True)
    return None if regions is None else regions[SR_ABCD_REGION]


def read_coffea(path):
    """Load a merged coffea file and return ``{sample: sample_output}``."""
    return load(str(path))["out"]


def collect_yields(directory, channels=None, flow=True, progress=None,
                   allow_data_sr=False):
    """Extract per-region yields for every ``.coffea`` file in ``directory``.

    Returns ``{sample: {channel_name: {region_index: Yield}}}``.  Each merged
    file holds a single sample, but the loop tolerates several.

    Signal-region entries are omitted for any sample not positively identified
    as simulation unless ``allow_data_sr`` is set, so a blinded run simply has
    no region-0 key to plot or tabulate.
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
                y = region_yields(sample_out, channel, sample_name=sample,
                                  flow=flow, allow_data_sr=allow_data_sr)
                if y is not None:
                    per_channel[ch_name] = y
            out[sample] = per_channel
    return out


def sr_only(yields):
    """Reduce ``collect_yields`` output to ``{sample: {channel: Yield}}``.

    Samples or channels with no region-A entry (a blinded data sample) are
    dropped rather than reported as zero.
    """
    out = {}
    for sample, per_channel in yields.items():
        reduced = {ch: regions[SR_ABCD_REGION]
                   for ch, regions in per_channel.items()
                   if SR_ABCD_REGION in regions}
        if reduced:
            out[sample] = reduced
    return out


def group_backgrounds(bkg_yields, channels=None):
    """Sum per-sample background yields into datacard process groups.

    Returns ``{channel_name: {region_index: {process: Yield}}}``.
    """
    channels = channels or CHANNELS
    grouped = {ch: {} for ch in channels}
    for sample, per_channel in bkg_yields.items():
        process = bkg_group(sample)
        for ch_name, regions in per_channel.items():
            for region, y in regions.items():
                per_region = grouped[ch_name].setdefault(region, {})
                per_region[process] = per_region.get(process, Yield(0.0, 0.0)) + y
    return grouped


def total_background(grouped, channel_name, region):
    """Total background Yield in one channel and region."""
    processes = grouped.get(channel_name, {}).get(region, {})
    return sum(processes.values(), Yield(0.0, 0.0))


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

    ``use_theory_xs`` switches the signal normalisation from the 1 fb reference
    to the point's own theory cross section, which turns Combine's signal
    strength from "the limit on sigma in fb" into "the limit relative to
    theory".  The two are related by a constant factor per point, so they carry
    the same information -- but with the theory normalisation the exclusion
    boundary is at r = 1, which is what a reader expects.

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
    # Luminosity the yields should be extrapolated to, in /pb.  The histograms
    # come out of sidm_processor scaled to LUMI_PB (2018), so setting this
    # rescales both signal and background by target/LUMI_PB.  None means "leave
    # the yields at the luminosity they were produced with".
    target_lumi_pb: float | None = None
    # How the signal is normalised.  False keeps the 1 fb reference the
    # histograms were produced with, so Combine's r is the limit on sigma in fb.
    # True rescales each signal point to its own cross section from
    # cross_sections.yaml, so r is directly sigma_limit/sigma_theory and r < 1
    # means the point is excluded.
    use_theory_xs: bool = False
    signal_unc: float | None = None
    bkg_norm_unc: float | None = None
    mc_stat: str | None = "gmN"
    mc_stat_cap: float = 2.0       # clip runaway lnN values from 1-2 MC events
    min_bkg: float = 1e-4
    min_process_rate: float = 1e-6  # processes at or below this are dropped
    observation: str = "bkg"        # "bkg" -> expected/blinded; or a float


def signal_xs_scale(signal_name, config):
    """Factor rescaling a signal yield from the 1 fb reference to the target.

    Returns 1.0 unless ``config.use_theory_xs`` is set, in which case it is
    ``sigma_theory / 1 fb`` for this point.  Raises if the point has no cross
    section, rather than silently falling back to the reference -- a card whose
    normalisation is not what its header claims is worse than no card.
    """
    if config is None or not config.use_theory_xs:
        return 1.0
    from sidm.tools import utilities
    xs_pb = utilities.get_xs(signal_name, use_signal_xs=True)
    if xs_pb is None or xs_pb <= 0:
        raise ValueError(f"no usable theory cross section for {signal_name}")
    # get_xs falls back to the 1 fb reference for unknown signal names, which
    # here would be a silent mis-normalisation, so check the name is really in
    # the config rather than trusting the returned value.
    if signal_name not in utilities.load_yaml(f"{BASE_DIR}/configs/cross_sections.yaml"):
        raise ValueError(
            f"{signal_name} has no entry in cross_sections.yaml; refusing to "
            f"normalise it to a theory cross section it does not have"
        )
    return xs_pb / SIGNAL_REF_XS_PB


def lumi_scale(config):
    """Factor to rescale yields from the production luminosity to the target.

    Returns 1.0 when no target is configured.  This scales the *yields* only:
    MC statistical uncertainties keep their relative size, because the number
    of simulated events does not change with the target luminosity.
    """
    if config is None or config.target_lumi_pb is None:
        return 1.0
    return config.target_lumi_pb / LUMI_PB


def _norm_comment(signal_name, config):
    """The datacard header line describing what the signal is normalised to."""
    lumi_fb = (config.target_lumi_pb or LUMI_PB) / 1000
    if config.use_theory_xs:
        xs_fb = signal_xs_scale(signal_name, config) * SIGNAL_REF_XS_PB * 1000
        return (f"# signal normalised to its theory cross section "
                f"{xs_fb:g} fb at {lumi_fb:g} /fb, so r = sigma_limit/sigma_theory "
                f"and r < 1 means excluded")
    return (f"# signal normalised to {SIGNAL_REF_XS_PB * 1000:g} fb at {lumi_fb:g} /fb, "
            f"so r = sigma / {SIGNAL_REF_XS_PB * 1000:g} fb")


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
        _norm_comment(signal_name, config),
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

    # Widths sized from the content so a long nuisance name can never run into
    # the type column (Combine would then read the first value as the type).
    def nuisance(name, kind, values):
        cells = [v if isinstance(v, str) else _fmt(v) for v in values]
        return f"{name:<{_name_w}}{kind:<{_kind_w}}" + col(cells)

    _all_names = ["lumi_13TeV", "signal_norm", "bkg_norm"] + [r[0] for r in stat_rows]
    _all_kinds = ["lnN"] + [r[1] for r in stat_rows]
    _name_w = max(len(n) for n in _all_names) + 2
    _kind_w = max(len(k) for k in _all_kinds) + 2

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
    """Write one single-bin counting datacard per (signal point, channel).

    Consumes the per-region structures from ``collect_yields`` and
    ``group_backgrounds`` and uses region A only; for the version that predicts
    A from the control regions see ``write_abcd_datacards``.

    With ``only_matching_channel`` each signal point is written only for the
    channel targeting its final state (2Mu2E -> SR_2mu2e, 4Mu -> SR_4mu); the
    cross-final-state yields are ~1e-4 events and carry no sensitivity.

    Returns ``(written_paths, floored)`` where ``floored`` lists the
    ``(channel, signal)`` pairs whose background had to be floored.
    """
    channels = channels or CHANNELS
    config = config or DatacardConfig()
    scale = lumi_scale(config)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written, floored = [], []
    for signal_name in sorted(signal_yields):
        info = parse_signal_name(signal_name)
        for ch_name, channel in channels.items():
            if only_matching_channel and info and info["final_state"] != channel.signal_prefix:
                continue
            sig_regions = signal_yields[signal_name].get(ch_name) or {}
            sig = sig_regions.get(SR_ABCD_REGION)
            if sig is None:
                continue
            sig = sig.scaled(scale * signal_xs_scale(signal_name, config))
            if sig.value <= config.min_process_rate:
                continue
            bkg = {name: y.scaled(scale) for name, y
                   in bkg_grouped.get(ch_name, {}).get(SR_ABCD_REGION, {}).items()}
            card = build_datacard(
                signal_name, sig, bkg, ch_name, config=config, floored=floored,
            )
            path = outdir / f"datacard_{ch_name}_{signal_name}.txt"
            path.write_text(card)
            written.append(path)
    return written, floored


# --------------------------------------------------------------------------- #
# ABCD datacards: the SR background is predicted from the control regions
# --------------------------------------------------------------------------- #
def build_abcd_datacard(signal_name, signal_regions, bkg_regions, channel_name,
                        config=None, warnings=None):
    """Render a four-bin datacard that does the ABCD arithmetic inside Combine.

    ``signal_regions`` and ``bkg_regions`` are ``{region_index: Yield}`` and
    ``{region_index: {process: Yield}}`` respectively.

    Rather than handing Combine a pre-computed ``B*C/D``, each control region
    gets its own freely floating normalisation and region A's background is
    *defined* as the product::

        bNorm, cNorm, dNorm   free rateParams, one per control region
        aNorm = bNorm*cNorm/dNorm

    so the control-region counts constrain the signal-region background inside
    the fit.  The three CR normalisations are unconstrained, which is the point:
    the data (here, MC standing in for it) determines them.

    Signal is entered in **all four** bins at its own per-region yield, not just
    in A.  That matters here -- signal leaks into region C at the tens-of-percent
    level for the high-mass points -- because it lets the fit account for the
    signal that would otherwise inflate the background prediction and quietly
    weaken the limit.
    """
    config = config or DatacardConfig()
    scale = lumi_scale(config)
    sig_scale = scale * signal_xs_scale(signal_name, config)

    bins = [f"{channel_name}_{ABCD_REGIONS[r]}" for r in sorted(ABCD_REGIONS)]
    regions = sorted(ABCD_REGIONS)

    signal_rate, bkg_rate = {}, {}
    for r in regions:
        sig = signal_regions.get(r, Yield(0.0, 0.0)).scaled(sig_scale)
        bkg = sum(bkg_regions.get(r, {}).values(), Yield(0.0, 0.0)).scaled(scale)
        signal_rate[r] = sig
        bkg_rate[r] = bkg

    # A degenerate control region makes B*C/D meaningless, so say so rather than
    # emitting a card that silently divides by ~0.
    for r in (1, 2, 3):
        if bkg_rate[r].value <= config.min_process_rate:
            if warnings is not None:
                warnings.append(
                    f"{channel_name}/{signal_name}: control region "
                    f"{ABCD_REGIONS[r]} is empty ({bkg_rate[r].value:.3g}); "
                    f"the ABCD prediction is undefined"
                )
            return None

    predicted_a = bkg_rate[1].value * bkg_rate[2].value / bkg_rate[3].value

    # Combine multiplies the `rate` column by the rateParam, so the background
    # rate is written as 1 and the entire normalisation lives in the parameter.
    processes = ["signal", "bkg"]
    width = 14
    col = lambda vals: "".join(f"{v:<{width}}" for v in vals)
    pad = 34

    obs = {r: bkg_rate[r].value for r in regions}
    obs[SR_ABCD_REGION] = predicted_a  # blinded: A is the prediction, not MC A

    lines = [
        f"# SIDM ABCD datacard -- {signal_name}, {channel_name}",
        f"# selection {CHANNELS[channel_name].selection}",
        f"# 4 bins = ABCD regions; background in A is bNorm*cNorm/dNorm",
        _norm_comment(signal_name, config),
        f"# MC region A = {bkg_rate[0].value:.4g}, B*C/D = {predicted_a:.4g}",
        f"imax {len(bins)}  number of bins",
        "jmax 1  number of background processes",
        "kmax *  number of nuisance parameters",
        "-" * 96,
        "bin".ljust(pad) + col(bins),
        "observation".ljust(pad) + col([_fmt(obs[r]) for r in regions]),
        "-" * 96,
        "bin".ljust(pad) + col([b for b in bins for _ in processes]),
        "process".ljust(pad) + col([p for _ in bins for p in processes]),
        "process".ljust(pad) + col(["0", "1"] * len(bins)),
        # background rate is 1 everywhere; the rateParams below carry the yields
        "rate".ljust(pad) + col([v for r in regions
                                 for v in (_fmt(signal_rate[r].value), "1")]),
        "-" * 96,
    ]

    # Column widths are computed from the content: a name exactly as long as a
    # fixed width would run into the next column and Combine would read the
    # first value as the nuisance type.
    nuisance_rows = []

    def nuisance(name, kind, values):
        nuisance_rows.append((name, kind,
                              [v if isinstance(v, str) else _fmt(v) for v in values]))

    # Luminosity applies to signal only: the backgrounds are free-floating, so
    # a normalisation nuisance on them would be redundant with the rateParams.
    nuisance(
        "lumi_13TeV", "lnN",
        [v for _ in bins for v in (1 + config.lumi_unc, "-")],
    )
    if config.signal_unc:
        nuisance(
            "signal_norm", "lnN",
            [v for _ in bins for v in (1 + config.signal_unc, "-")],
        )
    if config.mc_stat:
        for r in regions:
            y = signal_rate[r]
            if y.rel_error <= 0:
                continue
            cells = []
            for rr in regions:
                cells += [1 + min(y.rel_error, config.mc_stat_cap)
                          if rr == r else "-", "-"]
            nuisance(
                f"mcstat_{channel_name}_{ABCD_REGIONS[r]}_signal", "lnN", cells)

    if nuisance_rows:
        name_w = max(len(n) for n, _, _ in nuisance_rows) + 2
        kind_w = max(len(k) for _, k, _ in nuisance_rows) + 2
        lines += [f"{n:<{name_w}}{k:<{kind_w}}" + col(c) for n, k, c in nuisance_rows]

    lines.append("-" * 96)
    # One free normalisation per control region, then A defined as their product.
    for r, param in ((1, "bNorm"), (2, "cNorm"), (3, "dNorm")):
        start = bkg_rate[r].value
        lines.append(
            f"{param:<24}rateParam  {bins[r]:<16} bkg  "
            f"{_fmt(start)}  [0,{_fmt(max(start * 50, 1.0))}]"
        )
    lines.append(
        f"{'aNorm':<24}rateParam  {bins[0]:<16} bkg  "
        f"(@0*@1/@2)  bNorm,cNorm,dNorm"
    )

    return "\n".join(lines) + "\n"


def write_abcd_datacards(signal_yields, bkg_grouped, outdir, channels=None,
                         config=None, only_matching_channel=True):
    """Write one four-bin ABCD datacard per (signal point, channel).

    Returns ``(written_paths, warnings)``.
    """
    channels = channels or CHANNELS
    config = config or DatacardConfig()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written, warnings = [], []
    for signal_name in sorted(signal_yields):
        info = parse_signal_name(signal_name)
        for ch_name, channel in channels.items():
            if only_matching_channel and info and info["final_state"] != channel.signal_prefix:
                continue
            sig_regions = signal_yields[signal_name].get(ch_name)
            if not sig_regions:
                continue
            card = build_abcd_datacard(
                signal_name, sig_regions, bkg_grouped.get(ch_name, {}), ch_name,
                config=config, warnings=warnings,
            )
            if card is None:
                continue
            path = outdir / f"datacard_abcd_{ch_name}_{signal_name}.txt"
            path.write_text(card)
            written.append(path)
    return written, warnings
