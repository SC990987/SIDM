# Limit setting from the ABCD signal-region counts

Turns the ABCD-plane yields in the merged coffea outputs into Combine datacards, runs Combine
over them, and plots the expected limits.

Two flavours of datacard are produced: a **counting** card (one bin, region-A background taken
straight from MC) and an **ABCD** card (four bins, region-A background defined inside Combine
as `bNorm*cNorm/dNorm` so the control regions constrain it in the fit). Each is written twice,
with the signal normalised either to the 1 fb reference or to its own theory cross section —
four sets of limits in total:

| directory | method | signal normalised to | `r` means |
|---|---|---|---|
| `limits/` | counting | 1 fb reference | limit on sigma in fb |
| `limits_abcd/` | ABCD | 1 fb reference | limit on sigma in fb |
| `limits_theory/` | counting | its theory sigma | sigma_limit / sigma_theory |
| `limits_abcd_theory/` | ABCD | its theory sigma | sigma_limit / sigma_theory |

Set the normalisation with `DatacardConfig(use_theory_xs=True)`. The two routes agree to a
median of 0.00% (signal strength scales linearly); the few-percent tail is Combine's default 5%
scan tolerance. `run_combine_limits.py` records which normalisation each card used, read from
the card's own header rather than the directory name.

## Contents

| path | what it is |
|---|---|
| `datacard_tools.py` | yield extraction, blinding guard, both datacard writers |
| `make_datacards.ipynb` | reads the coffea files, shows the yield/contamination tables, writes both card sets |
| `limit_plots.ipynb` | reads `limits/limits.csv`, derives `Lxy` and `epsilon^2`, applies theory cross sections, writes `plots/` |
| `NOTES.md` | dated running log of what was done and what was found |
| `reference_limits/` | drop digitised external dark photon contours here; format and provenance in its README |
| `datacards/` | 120 one-bin counting cards (background straight from MC region A) |
| `datacards_abcd/` | 120 four-bin cards, region-A background predicted from B, C, D in the fit |
| `limits/`, `limits_abcd/` | Combine output for each flavour, written by `sidm/scripts/run_combine_limits.py` |
| `plots/` | expected-limit figures (png + pdf), gitignored |
| `sr_yields.pkl` | cached SR yields so the notebook need not re-read EOS, gitignored |
| `slides/` | 24-slide deck: method, nuisances, blinding fail-safe, results (gitignored) |

## Workflow

1. **`make_datacards.ipynb`** — reads the merged coffea files from

   - backgrounds: `.../ABCD_landing_10ch_cosmic_veto_v1_bkg_full_merged_samples_v1`
   - signal: `.../ABCD_landing_10ch_cosmic_veto_v1_signal_full_merged_samples_v1`

   and writes 120 datacards (60 `2Mu2E` points in `SR_2mu2e`, 60 `4Mu` points in `SR_4mu`).

2. **`sidm/scripts/run_combine_limits.py`** — runs `combine -M AsymptoticLimits` over every
   datacard and collects the results into `limits/`:

   ```bash
   python sidm/scripts/run_combine_limits.py -j 8                     # counting cards
   python sidm/scripts/run_combine_limits.py -j 8 \
       --datacards sidm/studies/limit_plotting/datacards_abcd \
       --pattern  'datacard_abcd_*.txt' \
       --outdir   sidm/studies/limit_plotting/limits_abcd             # ABCD cards
   ```

   Combine lives in its own CMSSW release; point `--combine-cmssw` or `$COMBINE_CMSSW_BASE` at
   it (default `/uscms_data/d3/scampbel/CMSSW_14_1_0_pre4`), or pass `--no-cmsenv` if `combine`
   is already on `$PATH`.

3. **`limit_plots.ipynb`** — expected limits vs topology, bound state energy, lifetime
   (as average lab-frame `Lxy`), and kinetic mixing (`epsilon^2` vs dark photon mass), plus the
   theory-cross-section overlays and the `r_theory` exclusion maps.

4. **`slides/`** — the write-up. Rebuild after regenerating the figures with

   ```bash
   cd sidm/studies/limit_plotting/slides && pdflatex limit_setting.tex && pdflatex limit_setting.tex
   ```

   (twice, for the frame numbers; it pulls the pdf figures straight out of `../plots/summary/`)

## Things worth knowing

* **The SR count needs `flow=True`.** The observable axes are `Regular(100, 0, 700)` and
  overflow at the few-percent level. With flow included, the SR sum reproduces the final row
  of the corresponding cutflow *exactly* (verified in both channels, signal and background);
  without it the yield is low by a few percent. The `abcd_region` axis carries no flow
  content, so nothing is lost there.

* **`r` is a cross section in fb.** `utilities.get_xs` returns a 1 fb reference for signal
  unless `use_signal_xs=True`, and `sidm_processor.postprocess` scales the histograms by
  `lumi * xs`. So the datacards are built at 1 fb and Combine's `r` *is* the limit on sigma in
  fb. The theory cross sections are applied afterwards in `limit_plots.ipynb` — no refit, and
  nothing about the coffea processing changes.

* **Two known issues with the theory cross sections.** A duplicate YAML key meant
  `2Mu2E_500GeV_5p0GeV_0p8mm` had no cross section at all (fixed — YAML silently keeps the last
  of a repeated key). And the values are non-monotonic in mass: 0.0499, 1.264, 8.812, 0.3416 fb
  for 200/500/800/1000 GeV. Every expected-excluded point is a consequence of that 800 GeV
  number, so it is worth confirming before any of the exclusion plots are shown outside the
  group.

* **Comparing to the published dark photon limits.** `limit_plots.ipynb` will overlay
  digitised external contours from `reference_limits/`, but **none are shipped** — they are
  published results and must come from a source the group agrees on, with provenance recorded.
  Note also that only 40/120 of our points fall on that plot's canvas (it stops at
  `m_A' = 1 GeV`, so our 1.2 and 5 GeV columns are off the edge), and that those limits
  constrain *direct* dark photon production whereas ours constrain a bound state decaying to
  dark photons. Same plane, different quantity.

* **Every figure names its background estimate.** `plots/by_method/` holds the `Lxy` and
  `m_bound` views for `counting`, for `abcd`, and for the two compared on the same axes, with
  the method in the filename. The older figures under `plots/summary/` are counting-only.

* **How the theory line is drawn.** The cross sections in `cross_sections.yaml` depend only on
  `m_bound` — every `m_ZD` and every lifetime at a given mass shares one value (0.0499, 1.264,
  8.812, 0.3416 fb for 200/500/800/1000 GeV). So on an `Lxy` axis, where `m_bound` is fixed
  within a panel, the theory line is a single **horizontal line**; on an `m_bound` axis the same
  four numbers become a **curve**. No interpolation or fitting — it is those four tabulated
  values joined.

* **`epsilon` on the plots.** Derived as `sqrt(80 / m_ZD / ctau) * 1e-6` with `m_ZD` in GeV and
  `ctau` in mm. Since `ctau` depends on both `m_ZD` and `m_bound`, one `m_ZD` column contains
  points from all four bound state masses at different `epsilon^2`; the panels are kept
  separate because those are different models. The figures are scatters of the simulated
  points, not interpolated contours — the grid has only three `m_ZD` values.

* **The background is still MC.** Every ABCD region rests on 1–7 raw simulated events, which
  is why the counting cards carry a `gmN` nuisance rather than a log-normal. When the
  data-driven prediction exists, substitute it for the MC rate.

* **The MC cannot test ABCD closure.** `B*C/D` and the MC region-A count differ by under
  1 sigma in both channels (0.89 and 0.99), with 45–111% uncertainties. The test has no
  statistical power. In particular, the ABCD cards give limits 3–5x stronger than the counting
  cards only because `B*C/D` happens to land below the MC region-A count — a fluctuation, not
  a gain in sensitivity. Do not quote the improvement.

* **Signal contaminates the control regions.** 19/120 points exceed 10% contamination in some
  control region at the 1 fb reference (32/120 at theory cross sections), concentrated in
  region C and reaching 89%. Not because leakage is large — `S_C/S_A` is only 1–3% for 4Mu —
  but because region C holds just 1.67 background events. This is why the ABCD cards enter
  signal in all four bins rather than only in A.

* **The signal region is blinded by construction.** `datacard_tools` withholds region A for
  any sample it cannot positively identify as simulation. It deliberately does *not* trust
  `metadata["is_data"]`, which the merge step empties, so real data would otherwise read as MC.
  A sample counts as simulation only if it is a known signal point or has a cross section in
  `cross_sections.yaml`; anything else has its SR omitted and `sr_yield` raises
  `BlindingError`. The guard fails toward withholding too much, never toward leaking.

* **The ABCD cards do the arithmetic inside Combine.** `bNorm`, `cNorm` and `dNorm` are
  unconstrained `rateParam`s on the three control regions and region A's background is defined
  as `bNorm*cNorm/dNorm`. Signal is entered in all four bins, not just A, because it leaks into
  region C at the tens-of-percent level for the high-mass points.

* **`target_lumi_pb` extrapolates to a different dataset.** It scales signal and background
  yields by `target/59830` while leaving relative MC statistical errors alone — more
  luminosity does not create more simulated events. Run 3 placeholders are in
  `run_periods.yaml`, unfilled.

* **The cards are blinded.** `observation` is set to the total background, and
  `run_combine_limits.py` passes `--run blind` by default, so only expected limits are
  meaningful. Use `--unblind` once there is real data to unblind to.
