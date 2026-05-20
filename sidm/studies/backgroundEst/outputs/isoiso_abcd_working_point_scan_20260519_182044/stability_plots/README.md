# Iso-Iso ABCD Working Point Stability Plots

These plots use the saved full 2D scan output and do not rerun the scan.

## Working point
- muIsoCut: 0.21
- egmIsoCut: 0.1

## Base cuts
- dPhiCut = 2.0
- pixelCut = 3 with pfMu/pixelHits or dsaMu displacement logic
- mJJCut = 150

## Slices
- muIso scan holds egmIso at nearest saved value 0.10
- egmIso scan holds muIso at nearest saved value 0.21

## Overlay slices
- muIso overlay requested egmIso values [0.09, 0.1, 0.11] and used {0.09: 0.09, 0.1: 0.1, 0.11: 0.11}
- egmIso overlay requested muIso values [0.2, 0.21, 0.22] and used {0.2: 0.2, 0.21: 0.21, 0.22: 0.22}

These figures are intended to show local closure, transfer-factor, and count stability around the selected working point.
