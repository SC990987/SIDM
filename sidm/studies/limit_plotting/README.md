# Limit setting from the ABCD signal-region counts

Turns the counts in the ABCD signal region (region A) of the merged coffea outputs into
Combine counting datacards, runs Combine over them, and plots the expected limits.

## Contents

| path | what it is |
|---|---|
| `datacard_tools.py` | SR yield extraction and datacard writing |
| `make_datacards.ipynb` | reads the coffea files, shows the yield tables, writes `datacards/` |
| `limit_plots.ipynb` | reads `limits/limits.csv`, converts `ctau` to average lab-frame `Lxy`, writes `plots/` |
| `datacards/` | one card per (signal point, channel): 120 cards |
| `limits/limits.csv`, `limits/limits.json` | Combine output, written by `sidm/scripts/run_combine_limits.py` |
| `plots/` | expected-limit figures (png + pdf), gitignored |
| `sr_yields.pkl` | cached SR yields so the notebook need not re-read EOS, gitignored |
| `slides/limit_setting.tex`, `.pdf` | 18-slide deck documenting the method, nuisances, and results |

## Workflow

1. **`make_datacards.ipynb`** — reads the merged coffea files from

   - backgrounds: `.../ABCD_landing_10ch_cosmic_veto_v1_bkg_full_merged_samples_v1`
   - signal: `.../ABCD_landing_10ch_cosmic_veto_v1_signal_full_merged_samples_v1`

   and writes 120 datacards (60 `2Mu2E` points in `SR_2mu2e`, 60 `4Mu` points in `SR_4mu`).

2. **`sidm/scripts/run_combine_limits.py`** — runs `combine -M AsymptoticLimits` over every
   datacard and collects the results into `limits/`:

   ```bash
   python sidm/scripts/run_combine_limits.py -j 8
   ```

   Combine lives in its own CMSSW release; point `--combine-cmssw` or `$COMBINE_CMSSW_BASE` at
   it (default `/uscms_data/d3/scampbel/CMSSW_14_1_0_pre4`), or pass `--no-cmsenv` if `combine`
   is already on `$PATH`.

3. **`limit_plots.ipynb`** — expected limits vs topology, bound state energy, and lifetime.

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

* **`r` is a cross section in fb.** Signal samples have no entry in
  `configs/cross_sections.yaml`, so `utilities.get_xs` falls back to 1 fb and
  `sidm_processor.postprocess` scales the histograms by `lumi * xs`. Combine's signal strength
  is therefore a multiplier on 1 fb.

* **The background is MC, not the ABCD prediction.** The SR background used here comes from one
  or two raw simulated events per process, which is why the cards carry a `gmN` nuisance rather
  than a log-normal. The real analysis predicts the SR background from regions B, C and D in
  data; when that prediction exists, substitute it for the MC rate and set `bkg_norm_unc` in
  `DatacardConfig` to its systematic uncertainty.

* **The cards are blinded.** `observation` is set to the total background, and
  `run_combine_limits.py` passes `--run blind` by default, so only expected limits are
  meaningful. Use `--unblind` once there is real data to unblind to.
