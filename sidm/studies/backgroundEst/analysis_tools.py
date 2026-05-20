import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import hist as Hist
import hist.intervals
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker
import mplhep as hep
from matplotlib.offsetbox import AnchoredText
import re

# --- 1. DYNAMIC PATH HANDLING ---
def setup_project_paths():
    """Finds the 'sidm' directory and adds its parent to sys.path."""
    current = Path.cwd()
    # Search upwards for the 'sidm' directory
    for parent in [current] + list(current.parents):
        if parent.name == 'sidm':
            root = str(parent.parent)
            if root not in sys.path:
                sys.path.insert(1, root)
            return
    print("Warning: 'sidm' directory not found in path hierarchy.")

setup_project_paths()

# Now we can safely import local modules
try:
    from sidm.tools import sidm_processor, utilities
except ImportError:
    print("Error: Could not import sidm.tools. Ensure you are running from within the project tree.")

# --- 2. CONSTANTS & METADATA ---
LUMI = 59830  # pb^-1

# Cross-sections mapped to sample names to avoid index-based errors
BACKGROUND_XSECS = {
    'DYJetsToMuMu_M10to50': 7013.0,
    'DYJetsToMuMu_M50': 1976.0,
    'QCD_Pt1000': 1.085,
    'QCD_Pt120To170': 21280.0,
    'QCD_Pt15To20': 2800000.0,
    'QCD_Pt170To300': 7000.0,
    'QCD_Pt20To30': 2527000.0,
    'QCD_Pt300To470': 622.6,
    'QCD_Pt30To50': 1367000.0,
    'QCD_Pt470To600': 58.9,
    'QCD_Pt50To80': 381700.0,
    'QCD_Pt600To800': 18.12,
    'QCD_Pt800To1000': 3.318,
    'QCD_Pt80To120': 87740.0,
    'TTJets': 471.7
}

# --- 3. DATA LOADING UTILITIES ---
def jsonload(fname):
    """Helper to load a single JSON with error handling."""
    with open(fname) as jsonfile:
        try:
            return json.load(jsonfile)
        except Exception:
            print(f"Failed to load: {fname}")
            return None

def load_samples(idir, sample_ids=None):
    """Loads all JSONs in a directory and filters by sample_ids if provided."""
    fnames = sorted(glob.glob(f"{idir}/*.json"))
    data = np.array([jsonload(f) for f in fnames])
    if sample_ids is not None:
        return data[sample_ids]
    return data

def string_replace_p(string):
    """Replaces 'p' with '.' for float conversion in strings."""
    return string.replace('p', '.') if 'p' in string else string

# --- 4. CORE PHYSICS CALCULATIONS ---
def getAsimovSignificance(s, b):
    """Calculates Asimov significance, handling b <= 0 gracefully."""
    if b > 0:
        term = (s + b) * np.log(1 + s / b) - s
        return np.sqrt(2 * term) if term > 0 else 0
    return 0


def getAsimovSignificanceError(s, b, sigma_s, sigma_b):
    if b <= 0:
        return 0
    term = (s+b)*np.log(1 + s/b) - s
    if term <= 0:
        return 0
    Z = np.sqrt(2*term)
    dZ_ds = np.log(1 + s/b) / Z
    dZ_db = (np.log(1 + s/b) - s/b) / Z
    sigma_Z = np.sqrt((dZ_ds*sigma_s)**2 + (dZ_db*sigma_b)**2)
    return sigma_Z


def getSignalEfficiency(sample, dPhiCut, mJIsoCut, mass, eventCut, muDXYCut, egmIsoCut, pixelCut):
    """Calculates signal efficiency based on various event selection cuts."""
    # Isolation mask
    if muDXYCut is not None and egmIsoCut is not None:
        num_mask = (
            (np.array(sample['egm_isolation']) <= egmIsoCut) &
            (np.array(sample['isolation']) <= mJIsoCut)
        )
    else:
        num_mask = np.ones(len(sample['egm_isolation']), dtype=bool)

    # Displacement logic
    pixelHits = np.array(sample['pixelHits'], dtype=object)
    pixelHits = np.where(pixelHits == None, np.nan, pixelHits).astype(float)
    pixelMask = np.where(np.isnan(pixelHits), False, pixelHits <= pixelCut)
    
    # Selection blocks (Simplified example of your logic)
    if eventCut == 'DSA + pixelHits + dPhi':
        disp_cut = np.where(np.array(sample['pfMu_n']) >= 1, pixelMask, (np.array(sample['dsaMu_n']) >= 1))
        den_mask = disp_cut & (np.array(sample['dPhi']) >= dPhiCut)
        num_mask &= den_mask
    elif eventCut == 'muDXY':
        # Note: You'll need to define muDXYCuts dict or pass it in
        den_mask = (np.array(sample['mu_lj_min_dxy']) >= muDXYCut)
        num_mask &= den_mask
    else:
        den_mask = np.ones(len(sample['dPhi']), dtype=bool)

    num, denom = np.sum(num_mask), np.sum(den_mask)
    return num / denom if denom > 0 else 0

def combineBackgrounds(background_dict):
    hists = list(background_dict.values())
    total_background = hists[0].copy()
    for h in hists[1:]:
        total_background.values()[...] += h.values()
        total_background.variances()[...] += h.variances()
    return total_background

def returnYield(sample, dPhiCut=None, mJIsoCut=None, egmIsoCut=None, dPhiFlag=True, 
                mJIsoFlag=True, egmIsoFlag=True, eventCut='DSA + pixelHits + dPhi', 
                pixelCut=None, invertDisp=False):
    """Returns the scaled yield for a sample given specific cut flags."""
    mask = np.ones(len(sample['dPhi']), dtype=bool)

    if eventCut == 'DSA + pixelHits + dPhi':
        pixelHits = np.array(sample['pixelHits'], dtype=object)
        pixelHits = np.where(pixelHits == None, np.nan, pixelHits).astype(float)
        pixelMask = np.where(np.isnan(pixelHits), False, pixelHits <= pixelCut)
        disp_pass = np.where(np.array(sample['pfMu_n']) >= 1, pixelMask, (np.array(sample['dsaMu_n']) >= 1))
        dPhi_pass = (np.array(sample['dPhi']) >= dPhiCut)
        
        mask &= (disp_pass if not invertDisp else ~disp_pass)
        mask &= dPhi_pass

    # Isolation flags
    if egmIsoCut is not None:
        egm_pass = np.array(sample['egm_isolation']) <= egmIsoCut
        mask &= (egm_pass if egmIsoFlag else ~egm_pass)
    
    if mJIsoCut is not None:
        mJ_pass = np.array(sample['isolation']) <= mJIsoCut
        mask &= (mJ_pass if mJIsoFlag else ~mJ_pass)
    
    scale = LUMI * sample.get('xsec', 0) / sample.get('nevents', 1)
    return scale * np.sum(mask)

# --- 5. ABCD / BACKGROUND ESTIMATION ---
def getObservedCounts(backgrounds, dPhiCut, mJIsoCut, egmIsoCut, muDXYCut, mass, evtCut, pixelCut):
    """Computes yields for ABCD regions across all background samples."""
    # Region definitions based on Iso flags: A (T,T), B (F,T), C (T,F), D (F,F)
    # This logic assumes backgrounds is a list where index -1 is TTJets, etc.
    # Better to use names, but following your structure for now:
    
    def get_abcd(samples):
        a, b, c, d = 0, 0, 0, 0
        for s in samples:
            a += returnYield(s, dPhiCut, mJIsoCut, egmIsoCut, mJIsoFlag=True, egmIsoFlag=True, eventCut=evtCut, pixelCut=pixelCut)
            b += returnYield(s, dPhiCut, mJIsoCut, egmIsoCut, mJIsoFlag=False, egmIsoFlag=True, eventCut=evtCut, pixelCut=pixelCut)
            c += returnYield(s, dPhiCut, mJIsoCut, egmIsoCut, mJIsoFlag=True, egmIsoFlag=False, eventCut=evtCut, pixelCut=pixelCut)
            d += returnYield(s, dPhiCut, mJIsoCut, egmIsoCut, mJIsoFlag=False, egmIsoFlag=False, eventCut=evtCut, pixelCut=pixelCut)
        return a, b, c, d

    dy_a, dy_b, dy_c, dy_d = get_abcd(backgrounds[:2])
    qcd_a, qcd_b, qcd_c, qcd_d = get_abcd(backgrounds[2:14])
    tt_a, tt_b, tt_c, tt_d = get_abcd([backgrounds[-1]])

    A = dy_a + qcd_a + tt_a
    B = dy_b + qcd_b + tt_b
    C = dy_c + qcd_c + tt_c
    D = dy_d + qcd_d + tt_d

    return A, B, C, D, (qcd_a, qcd_b, qcd_c, qcd_d), (dy_a, dy_b, dy_c, dy_d), (tt_a, tt_b, tt_c, tt_d)

# --- 6. PLOTTING ---
def plot_closure_and_significance(closure_matrix, closure_significance, filename, steps=25, low=-0.1, high=0.1):
    """Side-by-side pcolormesh plots of ABCD closure and significance."""
    def edges(c):
        dc = c[1] - c[0]
        return np.concatenate(([c[0] - dc/2], c + dc/2))

    egm_edges = edges(np.linspace(0.0001, 1.0, steps))
    mJ_edges = edges(np.linspace(0.0001, 1.0, steps))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    
    # Closure Plot
    im0 = axes[0].pcolormesh(mJ_edges, egm_edges, np.clip(closure_matrix, -1, None), cmap='viridis')
    axes[0].set_title('Closure')
    plt.colorbar(im0, ax=axes[0], label="Inverted Closure")

    # Significance Plot
    im1 = axes[1].pcolormesh(mJ_edges, egm_edges, closure_significance, cmap='viridis', vmin=-5, vmax=5)
    axes[1].set_title('Closure Significance')
    plt.colorbar(im1, ax=axes[1], label="Significance")

    for i in range(steps):
        for j in range(steps):
            if low <= closure_matrix[i, j] <= high:
                for ax in axes:
                    ax.add_patch(Rectangle((mJ_edges[j], egm_edges[i]), mJ_edges[j+1]-mJ_edges[j], 
                                           egm_edges[i+1]-egm_edges[i], fill=False, edgecolor='red', lw=1.2))

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

nums = np.arange(0,15,1)

# ----- 7. Plot MJ-Iso distribution with a displacement Flag
def returnIsoHist(sample, pixelCut=3, invertDisp=False, egmIsoCut = None, dPhiCut = None):
    """
    Plots MJ Isolation that has the option to add or invert the displacement cut
    pixelCut is set to 3 by default
    """
    lumi = 59830
    h = (
    Hist.new
    .Reg(20, 0.0, 1.0, name="Isolation", label=rf"$\mu$-LJ Isolation")
    .Weight()  # floating-point storage
    )
  
    mask = np.ones(len(sample['isolation']), dtype=bool)
    
    ## adds a egmIso event level cut if applicable
    if (egmIsoCut is not None):
        mask &= (np.array(sample['egm_isolation']) <= egmIsoCut)

    if dPhiCut is not None:
        mask &= (np.array(sample['dPhi']) >= dPhiCut)
        
    ## creates an array of pixelHits
    pixelHits = np.array(sample['pixelHits'], dtype=object)
    
    ## Where we do not have pixelHits it is stored as None and this step converts it to Nan
    pixelHits = np.where(pixelHits == None, np.nan, pixelHits).astype(float)

    ## Now it says if pixelHits is Nan we set that value to false, and then otherwise we apply pixelHits <=3
    pixelMask = np.where(np.isnan(pixelHits), False, pixelHits <= pixelCut)

    ## This is the DSA requirement where nDSA >=1
    dsaCut = (np.array(sample['dsaMu_n']) >= 1)

    ## if we have a pf Muon in the sample, we apply the pixelHits cut, otherwise we apply the DSA requirement
    displacementCut = np.where(np.array(sample['pfMu_n']) >= 1, pixelMask, (np.array(sample['dsaMu_n']) >= 1))

    if invertDisp:
        mask &= (~displacementCut)
    else:
        mask &= displacementCut
    
    dPh = np.array(sample['isolation'])[mask]
    
    h.fill(dPh, weight = np.array(sample['passing_weights'])[mask])
    
    scale = lumi * sample['xsec'] / sample['nevents']
    h *= scale
    
    return h

def getIsoHist(backgrounds, egmIsoCut = 0.2, dPhiCut = None):
    DYHists = {i: returnIsoHist(backgrounds[i], egmIsoCut = 0.2, dPhiCut = dPhiCut) for i in nums[:2]}
    QCDHists = {i: returnIsoHist(backgrounds[i], egmIsoCut = 0.2, dPhiCut = dPhiCut) for i in nums[2:14]}
    TTJets = returnIsoHist(backgrounds[-1], egmIsoCut = 0.2, dPhiCut = dPhiCut)
    DY = combineBackgrounds(DYHists)
    QCD = combineBackgrounds(QCDHists)
    
    DYHistsInv = {i: returnIsoHist(backgrounds[i], invertDisp=True, egmIsoCut = 0.2, dPhiCut = dPhiCut) for i in nums[:2]}
    QCDHistsInv = {i: returnIsoHist(backgrounds[i], invertDisp=True, egmIsoCut = 0.2, dPhiCut = dPhiCut) for i in nums[2:14]}
    TTJetsInv = returnIsoHist(backgrounds[-1], invertDisp=True, egmIsoCut = 0.2, dPhiCut = dPhiCut)
    DYInv = combineBackgrounds(DYHistsInv)
    QCDInv = combineBackgrounds(QCDHistsInv)
    
    totalBackground = {i: returnIsoHist(backgrounds[i], egmIsoCut = 0.2, dPhiCut = dPhiCut) for i in nums}
    total = combineBackgrounds(totalBackground)
    
    totalBackgroundInv = {i: returnIsoHist(backgrounds[i], invertDisp=True, egmIsoCut = 0.2, dPhiCut = dPhiCut) for i in nums}
    totalInv = combineBackgrounds(totalBackgroundInv)

    fig, ax = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)
    hep.histplot(QCD, ax=ax[0], label='QCD', density=True, yerr=True)
    hep.histplot(TTJets, ax=ax[1], label='TTJets', density=True, yerr=True)
    hep.histplot(total, ax=ax[2], label='Total Background', density=True, yerr=True)
    hep.histplot(QCDInv, ax=ax[0], label='QCD Inverted', density=True, yerr=True)
    hep.histplot(TTJetsInv, ax=ax[1], label='TTJets Inverted', density=True, yerr=True)
    hep.histplot(totalInv, ax=ax[2], label='Total Background Inverted', density=True, yerr=True)
    for a in ax:
        hep.cms.label(ax=a)
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[0].set_ylabel('Density')
    # hep.cms.label(ax=ax[0])
    # ax[0].set_title('QCD Counts')
    # hep.hist2dplot(QCDInv, ax=ax[1])
    # hep.cms.label(ax=ax[1])
    # ax[1].set_title('QCD Inv Counts')
    #fig.savefig(f'{odir2}/mJIso.png', facecolor='w', dpi=300)
    return fig


## -------- 8. Mu-LJ Iso vs Displacement Study
def returnYieldmJIsoDisplacement(sample, dPhiCut=None, mJIsoCut=None, egmIsoCut=None, mjIsoFlag = True, 
                                 dispFlag = True, pixelCut=3, invertEGMIso=False):
    lumi = 59830
    
    mask = np.ones(len(sample['dPhi']), dtype=bool)

    ## adds a dPhi event level cut if applicable
    if dPhiCut is not None:
        dPhiMask = (np.array(sample['dPhi']) >= dPhiCut)
        mask &= dPhiMask

    ## adds a egmIso event level cut if applicable
    if (egmIsoCut is not None):
        egmIsoMask = (np.array(sample['egm_isolation']) <= egmIsoCut)
        if invertEGMIso:
            mask &= ~egmIsoMask
        else:
            mask &= egmIsoMask

    pixelHits = np.array(sample['pixelHits'], dtype=object)
    pixelHits = np.where(pixelHits == None, np.nan, pixelHits).astype(float)
    pixelMask = np.where(np.isnan(pixelHits), False, pixelHits <= pixelCut)
    dsaCut = (np.array(sample['dsaMu_n']) >= 1)
    displacementCut = np.where(np.array(sample['pfMu_n']) >= 1, pixelMask, (np.array(sample['dsaMu_n']) >= 1))
    mJIsoMask = (np.array(sample['isolation']) <= mJIsoCut)

    if mjIsoFlag:
        mask &= mJIsoMask
    else:
        mask &= ~mJIsoMask

    if dispFlag:
        mask &= displacementCut
    else:
        mask &= ~displacementCut

    scale = lumi * sample['xsec'] / sample['nevents']

    weights = np.array(sample['passing_weights'])

    
    #### yield would be scale * np.sum(weights[mask])
    #### variance should be scale*scale*np.sum(weights[mask]*weights[mask])
    return scale*np.sum(weights[mask]), scale*scale*np.sum(weights[mask]*weights[mask])


def getSignalCountsmJIsoDisplacement(signal, dPhiCut, mJIsoCut, egmIsoCut, pixelCut=3):
    cuts_base = {
        'dPhi': dPhiCut,
        'mJIso': mJIsoCut,
        'egmIso': egmIsoCut,
        'pixelCut': pixelCut,
    }
    
    A, A_unc2 =  returnYieldmJIsoDisplacement(sample=signal, dPhiCut=cuts_base['dPhi'], mJIsoCut=cuts_base['mJIso'], egmIsoCut=cuts_base['egmIso'], mjIsoFlag = True, dispFlag = True, pixelCut=cuts_base['pixelCut'])
    
    B, B_unc2 =  returnYieldmJIsoDisplacement(sample=signal, dPhiCut=cuts_base['dPhi'], mJIsoCut=cuts_base['mJIso'], egmIsoCut=cuts_base['egmIso'], mjIsoFlag = True, dispFlag = False, pixelCut=cuts_base['pixelCut'])
    
    C, C_unc2 =  returnYieldmJIsoDisplacement(sample=signal, dPhiCut=cuts_base['dPhi'], mJIsoCut=cuts_base['mJIso'], egmIsoCut=cuts_base['egmIso'], mjIsoFlag = False, dispFlag = True, pixelCut=cuts_base['pixelCut'])
    
    D, D_unc2 = returnYieldmJIsoDisplacement(sample=signal, dPhiCut=cuts_base['dPhi'], mJIsoCut=cuts_base['mJIso'], egmIsoCut=cuts_base['egmIso'], mjIsoFlag = False, dispFlag = False, pixelCut=cuts_base['pixelCut'])

    return A, B, C, D, np.sqrt(A_unc2), np.sqrt(B_unc2), np.sqrt(C_unc2), np.sqrt(D_unc2)

# def getObservedCountsdPhiDisplacement(backgrounds, dPhiCut, mJIsoCut, egmIsoCut, invertEgmIso, pixelCut=3):

#     results = {}

#     kwargs_A = {
#     "dPhiCut": dPhiCut,
#     "mJIsoCut": mJIsoCut,
#     "egmIsoCut": egmIsoCut,
#     "mjIsoFlag": True,
#     "dispFlag": True,
#     "pixelCut": pixelCut,
#     "invertEGMIso": invertEgmIso
#     }
#     kwargs_B = {
#     "dPhiCut": dPhiCut,
#     "mJIsoCut": mJIsoCut,
#     "egmIsoCut": egmIsoCut,
#     "mjIsoFlag": True,
#     "dispFlag": False,
#     "pixelCut": pixelCut,
#     "invertEGMIso": invertEgmIso
#     }
#     kwargs_C = {
#     "dPhiCut": dPhiCut,
#     "mJIsoCut": mJIsoCut,
#     "egmIsoCut": egmIsoCut,
#     "mjIsoFlag": False,
#     "dispFlag": True,
#     "pixelCut": pixelCut,
#     "invertEGMIso": invertEgmIso
#     }
#     kwargs_D = {
#     "dPhiCut": dPhiCut,
#     "mJIsoCut": mJIsoCut,
#     "egmIsoCut": egmIsoCut,
#     "mjIsoFlag": False,
#     "dispFlag": False,
#     "pixelCut": pixelCut,
#     "invertEGMIso": invertEgmIso
#     }
#     TTJets_A, TTJets_AUnc2 =  returnYieldmJIsoDisplacement(backgrounds[-1], **kwargs_A)

#     TTJets_B, TTJets_BUnc2 =  returnYieldmJIsoDisplacement(backgrounds[-1], **kwargs_B)

#     TTJets_C, TTJets_CUnc2 =  returnYieldmJIsoDisplacement(backgrounds[-1], **kwargs_C)

#     TTJets_D, TTJets_DUnc2 =  returnYieldmJIsoDisplacement(backgrounds[-1], **kwargs_D)


#     QCD_A = 0
#     QCD_B = 0
#     QCD_C = 0
#     QCD_D = 0

#     QCD_AUnc2 = 0  
#     QCD_BUnc2 = 0
#     QCD_CUnc2 = 0
#     QCD_DUnc2 = 0

#     for ent in backgrounds[2:14]:

#         temp1, temp1Unc = returnYieldmJIsoDisplacement(ent, **kwargs_A)
#         temp2, temp2Unc = returnYieldmJIsoDisplacement(ent, **kwargs_B)
#         temp3, temp3Unc = returnYieldmJIsoDisplacement(ent, **kwargs_C)
#         temp4, temp4Unc = returnYieldmJIsoDisplacement(ent, **kwargs_D)

#         QCD_A += temp1
#         QCD_B += temp2
#         QCD_C += temp3
#         QCD_D += temp4

#         QCD_AUnc2 += temp1Unc
#         QCD_BUnc2 += temp2Unc
#         QCD_CUnc2 += temp3Unc
#         QCD_DUnc2 += temp4Unc

#     DY_A = 0
#     DY_B = 0
#     DY_C = 0
#     DY_D = 0

#     DY_AUnc2 = 0
#     DY_BUnc2 = 0
#     DY_CUnc2 = 0
#     DY_DUnc2 = 0

#     for ent in backgrounds[:2]:

#         temp1, temp1Unc = returnYieldmJIsoDisplacement(ent, **kwargs_A)
#         temp2, temp2Unc = returnYieldmJIsoDisplacement(ent, **kwargs_B)
#         temp3, temp3Unc = returnYieldmJIsoDisplacement(ent, **kwargs_C)
#         temp4, temp4Unc = returnYieldmJIsoDisplacement(ent, **kwargs_D)

#         DY_A += temp1
#         DY_B += temp2
#         DY_C += temp3
#         DY_D += temp4

#         DY_AUnc2 += temp1Unc
#         DY_BUnc2 += temp2Unc
#         DY_CUnc2 += temp3Unc
#         DY_DUnc2 += temp4Unc

#     A = QCD_A + TTJets_A + DY_A
#     B = QCD_B + TTJets_B + DY_B
#     C = QCD_C + TTJets_C + DY_C
#     D = QCD_D + TTJets_D + DY_D

#     AUnc2 = QCD_AUnc2 + TTJets_AUnc2 + DY_AUnc2
#     BUnc2 = QCD_BUnc2 + TTJets_BUnc2 + DY_BUnc2
#     CUnc2 = QCD_CUnc2 + TTJets_CUnc2 + DY_CUnc2
#     DUnc2 = QCD_DUnc2 + TTJets_DUnc2 + DY_DUnc2

#     ## Return Total Background Counts
#     results["ACounts"] = A
#     results["BCounts"] = B
#     results["CCounts"] = C
#     results["DCounts"] = D
#     ## Associated Uncertainties
#     results["A_Unc"] = np.sqrt(AUnc2)
#     results["B_Unc"] = np.sqrt(BUnc2)
#     results["C_Unc"] = np.sqrt(CUnc2)
#     results["D_Unc"] = np.sqrt(DUnc2)
#     ## QCD Counts
#     results["QCD_ACounts"] = QCD_A
#     results["QCD_BCounts"] = QCD_B
#     results["QCD_CCounts"] = QCD_C
#     results["QCD_DCounts"] = QCD_D
#     ## Associated Uncertainties
#     results["QCDA_Unc"] = np.sqrt(QCD_AUnc2)
#     results["QCDB_Unc"] = np.sqrt(QCD_BUnc2)
#     results["QCDC_Unc"] = np.sqrt(QCD_CUnc2)
#     results["QCDD_Unc"] = np.sqrt(QCD_DUnc2)
#     ## TTJets Counts
#     results["TTJets_ACounts"] = TTJets_A
#     results["TTJets_BCounts"] = TTJets_B
#     results["TTJets_CCounts"] = TTJets_C
#     results["TTJets_DCounts"] = TTJets_D
#     ## Associated Uncertainties
#     results["TTJetsA_Unc"] = np.sqrt(TTJets_AUnc2)
#     results["TTJetsB_Unc"] = np.sqrt(TTJets_BUnc2)
#     results["TTJetsC_Unc"] = np.sqrt(TTJets_CUnc2)
#     results["TTJetsD_Unc"] = np.sqrt(TTJets_DUnc2)
#     ## DY Counts
#     results["DY_ACounts"] = DY_A
#     results["DY_BCounts"] = DY_B
#     results["DY_CCounts"] = DY_C
#     results["DY_DCounts"] = DY_D
#     ## Associated Uncertainties
#     results["DYA_Unc"] = np.sqrt(DY_AUnc2)
#     results["DYB_Unc"] = np.sqrt(DY_BUnc2)
#     results["DYC_Unc"] = np.sqrt(DY_CUnc2)
#     results["DYD_Unc"] = np.sqrt(DY_DUnc2)


#     return results
def getObservedCountsmJIsoDisplacement(backgrounds, dPhiCut, mJIsoCut, egmIsoCut, invertEgmIso=False, pixelCut=3):

    results = {}

    # Region definitions
    region_kwargs = {
        "A": {"mjIsoFlag": True,  "dispFlag": True},
        "B": {"mjIsoFlag": True,  "dispFlag": False},
        "C": {"mjIsoFlag": False, "dispFlag": True},
        "D": {"mjIsoFlag": False, "dispFlag": False}
    }

    base_kwargs = {
        "dPhiCut": dPhiCut,
        "mJIsoCut": mJIsoCut,
        "egmIsoCut": egmIsoCut,
        "pixelCut": pixelCut,
        "invertEGMIso": invertEgmIso
    }

    # Background grouping
    ttjets = backgrounds[-1]
    qcd_samples = backgrounds[2:14]
    dy_samples = backgrounds[:2]

    # Storage
    counts = {r: {"QCD":0, "TTJets":0, "DY":0} for r in "ABCD"}
    unc2   = {r: {"QCD":0, "TTJets":0, "DY":0} for r in "ABCD"}

    # --- TTJets ---
    for r, flags in region_kwargs.items():
        val, u = returnYieldmJIsoDisplacement(ttjets, **base_kwargs, **flags)
        counts[r]["TTJets"] = val
        unc2[r]["TTJets"] = u

    # --- QCD ---
    for ent in qcd_samples:
        for r, flags in region_kwargs.items():
            val, u = returnYieldmJIsoDisplacement(ent, **base_kwargs, **flags)
            counts[r]["QCD"] += val
            unc2[r]["QCD"] += u

    # --- DY ---
    for ent in dy_samples:
        for r, flags in region_kwargs.items():
            val, u = returnYieldmJIsoDisplacement(ent, **base_kwargs, **flags)
            counts[r]["DY"] += val
            unc2[r]["DY"] += u

    # --- Totals ---
    for r in "ABCD":

        total = sum(counts[r].values())
        total_unc2 = sum(unc2[r].values())

        results[f"{r}Counts"] = total
        results[f"{r}_Unc"] = np.sqrt(total_unc2)

        # individual backgrounds
        for bkg in ["QCD","TTJets","DY"]:
            results[f"{bkg}_{r}Counts"] = counts[r][bkg]
            results[f"{bkg}{r}_Unc"] = np.sqrt(unc2[r][bkg])

    return results


def returnClosure(sample):
    
    A, B, C, D, AUnc, BUnc, CUnc, DUnc = sample

    prediction = (B * C) / D

    predictionUnc = prediction * np.sqrt(
        (BUnc / B)**2 +
        (CUnc / C)**2 +
        (DUnc / D)**2
    )

    closure = 1 - (prediction / A)

    closureUnc = (prediction / A) * np.sqrt(
        (BUnc / B)**2 +
        (CUnc / C)**2 +
        (DUnc / D)**2 +
        (AUnc / A)**2
    )

    closureSig = (A - prediction) / np.sqrt(AUnc**2 + predictionUnc**2)

    return closure, closureUnc, closureSig

def returnHistIsoVsDisp(sample, dPhiCut=None, mJIsoCut=None, egmIsoCut=None, mJIsoFlag = True, dispFlag = True, pixelCut=3):
    
    lumi = 59830
    h = (
    Hist.new
    .Reg(50, 0.0, 1400.0, name="Isolation", label=rf"$\mu$-LJ Isolation")
    .Weight()  # floating-point storage
    )

    
    mask = np.ones(len(sample['dPhi']), dtype=bool)

    if (egmIsoCut is not None):
        mask &= (np.array(sample['egm_isolation']) <= egmIsoCut)

    if dPhiCut is not None:
        mask &= (np.array(sample['dPhi']) >= dPhiCut)

    pixelHits = np.array(sample['pixelHits'], dtype=object)
    pixelHits = np.where(pixelHits == None, np.nan, pixelHits).astype(float)
    pixelMask = np.where(np.isnan(pixelHits), False, pixelHits <= pixelCut)
    dsaCut = (np.array(sample['dsaMu_n']) >= 1)
    displacementCut = np.where(np.array(sample['pfMu_n']) >= 1, pixelMask, (np.array(sample['dsaMu_n']) >= 1))
    mjIsoMask = (np.array(sample['isolation']) <= mJIsoCut)
    
    if mJIsoFlag:
        mask &= mjIsoMask
    else:
        mask &= ~mjIsoMask

    if dispFlag:
        mask &= displacementCut
    else:
        mask &= ~displacementCut

    
    mJJ = np.array(sample['mJJ'])[mask]
    h.fill(mJJ, weight = np.array(sample['passing_weights'])[mask])
    
    scale = lumi * sample['xsec'] / sample['nevents']
    h *= scale

    print(h.sum(), scale*np.sum(np.array(sample['passing_weights'])[mask]))
    return h
###### ---------- Asimov Significance Grid Search ----------------
def returnBestAsimovSigMJIsoVsDisp(signal, backgrounds, dPhiCut = None, egmIsoCut = 0.2):
    mJIsoCuts = np.linspace(0, 1.0, 100)
    pixelCuts = [2, 3, 4]
    best_asimovSig = -np.inf
    best_values = None
    for mJIsoCut in mJIsoCuts:
        for pixelCut in pixelCuts:
            A, B, C, D, QCD, DY, TTJets = getObservedCountsdPhiDisplacement(backgrounds=backgrounds, dPhiCut=dPhiCut, mJIsoCut=mJIsoCut, egmIsoCut=egmIsoCut, pixelCut=pixelCut)
            closure, closureSig = returnClosure(sample=(A, B, C, D))
            sig_A, sig_B, sig_C, sig_D = getSignalCountsmJIsoDisplacement(signal, dPhiCut=dPhiCut, mJIsoCut=mJIsoCut, egmIsoCut=egmIsoCut, pixelCut=pixelCut)
            asimovSig = getAsimovSignificance(sig_A, A)
            signalEff = sig_A / (sig_A + sig_B + sig_C + sig_D)
            
            if asimovSig > best_asimovSig:
                best_asimovSig = asimovSig
                best_values = (asimovSig, mJIsoCut, pixelCut, (A,B,C,D), (sig_A, sig_B, sig_C, sig_D), signalEff, closure, closureSig)

    return best_values


def returnResults(signal, backgrounds, sampleName = '', dPhiCut = None, egmIsoCut = 0.2, mJIsoCut = 0.1, pixelCut = 2):
    results = {}
    results[sampleName] = {}
    best_values = None
    backgroundResults = getObservedCountsdPhiDisplacement(backgrounds=backgrounds, dPhiCut=dPhiCut, mJIsoCut=mJIsoCut, egmIsoCut=egmIsoCut, pixelCut=pixelCut)

    A = backgroundResults["ACounts"]
    B = backgroundResults["BCounts"]
    C = backgroundResults["CCounts"]
    D = backgroundResults["DCounts"]

    Aunc = backgroundResults["A_Unc"]
    Bunc = backgroundResults["B_Unc"]
    Cunc = backgroundResults["C_Unc"]
    Dunc = backgroundResults["D_Unc"]

    
    closure, closureSig = returnClosure(sample=(A, B, C, D, Aunc, Bunc, Cunc, Dunc))
    sig_A, sig_B, sig_C, sig_D, sig_Aunc, sig_Bunc, sig_Cunc, sig_Dunc = getSignalCountsmJIsoDisplacement(signal, dPhiCut=dPhiCut, mJIsoCut=mJIsoCut, egmIsoCut=egmIsoCut, pixelCut=pixelCut)

    
    asimovSig = getAsimovSignificance(sig_A, A)
    asimovSigErr = getAsimovSignificanceError(s=sig_A, b=A, sigma_s=sig_Aunc, sigma_b=Bunc)
    signalEff = sig_A / (sig_A + sig_B + sig_C + sig_D)
    
    best_asimovSig = asimovSig
    prediction = (B*C)/D
    predictionUnc = prediction*np.sqrt((Bunc/B)**2 + (Cunc/C)**2 + (Dunc/D)**2)

    results[sampleName]['Closure'] = closure
    results[sampleName]['ClosureSignificance'] = closureSig

    results[sampleName]['ACounts_sig'] = sig_A
    results[sampleName]['BCounts_sig'] = sig_B
    results[sampleName]['CCounts_sig'] = sig_C
    results[sampleName]['DCounts_sig'] = sig_D
    
    results[sampleName]['AUnc_sig'] = sig_Aunc
    results[sampleName]['BUnc_sig'] = sig_Bunc
    results[sampleName]['CUnc_sig'] = sig_Cunc
    results[sampleName]['DUnc_sig'] = sig_Dunc

    results[sampleName]['AsimovSig'] = asimovSig
    results[sampleName]['asimovSigErr'] = asimovSigErr
    results[sampleName]['signalEff'] = signalEff

    results[sampleName]['prediction'] = prediction
    results[sampleName]['predictionUnc'] = predictionUnc

    return results[sampleName] | backgroundResults

###### Return Set of Valid Closure points
def returnValidClosurePointsMJIsoVSDisp(backgrounds, dPhiCut = None, egmIsoCut = 0.2):
    mJIsoCuts = np.linspace(0, 1.0, 100)
    pixelCuts = [2, 3, 4]
    closureCuts = []
    for mJIsoCut in mJIsoCuts:
        for pixelCut in pixelCuts:
            A, B, C, D, QCD, DY, TTJets = getObservedCountsdPhiDisplacement(backgrounds=backgrounds, dPhiCut=dPhiCut, mJIsoCut=mJIsoCut, egmIsoCut=egmIsoCut, pixelCut=pixelCut)
            closure, closureSig = returnClosure(sample=(A, B, C, D))
            if abs(closure) > 0.1:
                continue
            closureCuts.append([pixelCut, mJIsoCut, abs(closure), abs(closureSig)])
    return np.array(closureCuts)

###### Closure Significance vs MJ Iso Plot:
def makeClosureSigVsMJIsoPlot(closureCuts):
    pixel2 = closureCuts[closureCuts[:,0] == 2]
    pixel3 = closureCuts[closureCuts[:,0] == 3]
    pixel4 = closureCuts[closureCuts[:,0] == 4]
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.scatter(x=pixel2[:,1], y=pixel2[:,3], label='pixelHits <= 2')
    ax.scatter(x=pixel3[:,1], y=pixel3[:,3], label='pixelHits <= 3')
    ax.scatter(x=pixel4[:,1], y=pixel4[:,3], label='pixelHits <= 4')
    ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
    ax.set_xlabel(rf'$\mu$-LJ Iso')
    ax.set_ylabel('Closure Significance')
    hep.cms.label(ax=ax)

    return fig

###### plot ABCD Plne for MJ Iso VS Disp
def plotABCDPlaneMJIsoVsDisp(signal, backgrounds, mJIsoCut=0.1, egmIsoCut=0.2, dPhiCut=None, pixelCut=3, odir = './', lxy=30):
    nums = np.arange(0,15,1)
    mjIsoCut = mJIsoCut
    
    ##background counts and significance
    A, B, C, D, QCD, DY, TTJets = getObservedCountsdPhiDisplacement(backgrounds=backgrounds, dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, pixelCut=pixelCut)
    closure, closureSig = returnClosure(sample=(A, B, C, D))
    
    
    ##backgrounds region A
    DYHistsA = {i: returnHistIsoVsDisp(sample=backgrounds[i], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = True, dispFlag = True, pixelCut=3) for i in nums[:2]}
    QCDHistsA = {i: returnHistIsoVsDisp(sample=backgrounds[i], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = True, dispFlag = True, pixelCut=3) for i in nums[2:14]}
    TTJetsA = returnHistIsoVsDisp(sample=backgrounds[-1], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = True, dispFlag = True, pixelCut=3)
    DYA = combineBackgrounds(DYHistsA)
    QCDA = combineBackgrounds(QCDHistsA)
    
    ##backgrounds region B
    DYHistsB = {i: returnHistIsoVsDisp(sample=backgrounds[i], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = True, dispFlag = False, pixelCut=3) for i in nums[:2]}
    QCDHistsB = {i: returnHistIsoVsDisp(sample=backgrounds[i], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = True, dispFlag = False, pixelCut=3) for i in nums[2:14]}
    TTJetsB = returnHistIsoVsDisp(sample=backgrounds[-1], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = True, dispFlag = False, pixelCut=3)
    DYB = combineBackgrounds(DYHistsB)
    QCDB = combineBackgrounds(QCDHistsB)
    
    ##backgrounds region C
    DYHistsC = {i: returnHistIsoVsDisp(sample=backgrounds[i], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = False, dispFlag = True, pixelCut=3) for i in nums[:2]}
    QCDHistsC = {i: returnHistIsoVsDisp(sample=backgrounds[i], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = False, dispFlag = True, pixelCut=3) for i in nums[2:14]}
    TTJetsC = returnHistIsoVsDisp(sample=backgrounds[-1], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = False, dispFlag = True, pixelCut=3)
    DYC = combineBackgrounds(DYHistsC)
    QCDC = combineBackgrounds(QCDHistsC)
    
    ##backgrounds region D
    DYHistsD = {i: returnHistIsoVsDisp(sample=backgrounds[i], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = False, dispFlag = False, pixelCut=3) for i in nums[:2]}
    QCDHistsD = {i: returnHistIsoVsDisp(sample=backgrounds[i], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = False, dispFlag = False, pixelCut=3) for i in nums[2:14]}
    TTJetsD = returnHistIsoVsDisp(sample=backgrounds[-1], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = False, dispFlag = False, pixelCut=3)
    DYD = combineBackgrounds(DYHistsD)
    QCDD = combineBackgrounds(QCDHistsD)
    
    for i, ent in enumerate(signal):
        
        sig_A, sig_B, sig_C, sig_D = getSignalCountsmJIsoDisplacement(signal[i], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, pixelCut=pixelCut)
        s_over_b = getAsimovSignificance(s=sig_A, b=A)
        sigHistA = returnHistIsoVsDisp(sample=signal[i], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = True, dispFlag = True, pixelCut=3)
        sigHistB = returnHistIsoVsDisp(sample=signal[i], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = True, dispFlag = False, pixelCut=3)
        sigHistC = returnHistIsoVsDisp(sample=signal[i], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = False, dispFlag = True, pixelCut=3)
        sigHistD = returnHistIsoVsDisp(sample=signal[i], dPhiCut=dPhiCut, mJIsoCut=mjIsoCut, egmIsoCut=egmIsoCut, mJIsoFlag = False, dispFlag = False, pixelCut=3)
        prediction = (B*C)/D
        predictionUnc = prediction*np.sqrt((1/B) + (1/C) + (1/D))
        fig, axes = plt.subplots(2, 2, figsize=(14,10), sharex=True)
        mb = signal[i]['sample_name'].split('_')[1].split('GeV')[0]
        zd = signal[i]['sample_name'].split('_')[2].split('GeV')[0]
        # Inside your loop, instead of putting all that in the label:
        at = AnchoredText(
            f"Mb: {mb} Mzd: {zd} Lxy: {lxy}\nAsimov Sig: {s_over_b:.3f}\nSig Counts: {sig_A:.3f}\nBkg Counts: {A:.3f}\nPred A: {prediction:.3f}\nUnc Pred: {predictionUnc:.3f}\nClosure: {abs(closure):.3f}\nClosure Sig: {abs(closureSig):.3f}",
            prop=dict(size=8), frameon=True, loc='upper right'
        )
        axes[0,0].add_artist(at)
    
        atB = AnchoredText(
            f"Sig Counts: {sig_B:.3f}\nBkg Counts: {B:.3f}",
            prop=dict(size=8), frameon=True, loc='upper right'
        )
        axes[0,1].add_artist(atB)
    
        atC = AnchoredText(
            f"Sig Counts: {sig_C:.3f}\nBkg Counts: {C:.3f}",
            prop=dict(size=8), frameon=True, loc='upper right'
        )
        axes[1,0].add_artist(atC)
    
        atD = AnchoredText(
            f"Sig Counts: {sig_D:.3f}\nBkg Counts: {D:.3f}",
            prop=dict(size=8), frameon=True, loc='upper right'
        )
        axes[1,1].add_artist(atD)
        
        hep.histplot(
        [QCDA, DYA, TTJetsA, sigHistA],
        label=['QCD', 'DY', 'TTJets', f'Signal'],
        stack=True,
        histtype="step",
        #alpha=0.4,
        ax=axes[0,0],
        )
        hep.histplot(
        [QCDB, DYB, TTJetsB, sigHistB],
        label=[f'QCD', 'DY', 'TTJets', f'Signal'],
        stack=True,
        histtype="step",
        #alpha=0.4,
        ax=axes[0,1],
        )
        hep.histplot(
        [QCDC, DYC, TTJetsC, sigHistC],
        label=[f'QCD', 'DY', 'TTJets', f'Signal'],
        stack=True,
        histtype="step",
        #alpha=0.4,
        ax=axes[1,0],
        )
        hep.histplot(
        [QCDD, DYD, TTJetsD, sigHistD],
        label=[f'QCD', 'DY', 'TTJets', f'Signal'],
        stack=True,
        histtype="step",
        #alpha=0.4,
        ax=axes[1,1],
        )
    
        axes[1,1].legend(loc='lower right')
    
        fig.suptitle(f'{signal[i]["sample_name"]} ABCD Plane mJIso vs Displacement')
        plt.savefig(f'{odir}/ABCDPlane_IsoVsDisp_{signal[i]["sample_name"]}.png', facecolor='w', dpi=300)
        plt.show()
        plt.clf()
        #plt.close()


def make_three_panel_plot(variables, dfs, filename, outdir="."):
    """
    Create a 3-row grouped bar chart for given variables and save as PDF.

    variables : list of exactly 3 str
        Column names to plot (e.g., ["best_s_over_b", "sig_A", "bkg_A"])

    Input: str
        Input xlsx filename
    filename : str
        Output PDF file name (e.g., "plots.pdf")
    outdir : str
        Directory to save the output PDF
    """

    

    # -------------------------------------------------
    # 1. Load spreadsheets
    # -------------------------------------------------
    dfs = dfs
    # dfs = {
    #     # "PixelHits <= 4 + dPhi Loose": pd.read_excel('./egmIso_mJIso_DSA_pixelHits4_dPhiLoose.xlsx'),
    #     # "PixelHits <= 2 + dPhi Loose": pd.read_excel('./egmIso_mJIso_DSA_pixelHits2_dPhiLoose.xlsx'),
    #     # "PixelHits <= 4 + dPhi Tight": pd.read_excel('./egmIso_mJIso_DSA_pixelHits4_dPhiTight.xlsx'),
    #     # "PixelHits <= 2 + dPhi Tight": pd.read_excel('./egmIso_mJIso_DSA_pixelHits2_dPhiTight.xlsx'),
    #     # "PixelHits <= 4 + dPhi Loose + mJJ <= 100": pd.read_excel('./egmIso_mJIso_DSA_pixelHits4_dPhiLoose_mJJ.xlsx'),
    #     # "PixelHits <= 4 + dPhi Tight + mJJ <= 100": pd.read_excel('./egmIso_mJIso_DSA_pixelHits4_dPhiTight_mJJ.xlsx'),
    #     # "Loose EGM Iso vs MJ Iso": pd.read_excel('./egmIso_mJIsoLoose.xlsx'),
    #     # "Tight EGM Iso vs MJ Iso": pd.read_excel('./egmIso_mJIsoTight.xlsx'),
    #     "Optimizer Results": pd.read_excel(f'{Input}'),
    #     "Nominal Results": pd.read_excel('./nominalResults.xlsx')
    #     # "dPhi >= 2.5": pd.read_excel('./egmIso_mJIso_dPhi2p5.xlsx'),
    #     # "dPhi >= 2.6": pd.read_excel('./egmIso_mJIso_dPhi2p6.xlsx'),
    #     # "dPhi >= 2.7": pd.read_excel('./egmIso_mJIso_dPhi2p7.xlsx'),
    # }

    # -------------------------------------------------
    # 2. Drop specific unwanted samples
    # -------------------------------------------------
    samples_to_drop = [
        "None"
    ]

    for name, df in dfs.items():
        df["source"] = name
        df.drop(df[df["Sample"].isin(samples_to_drop)].index, inplace=True)

    # -------------------------------------------------
    # 3. Combine into one long table
    # -------------------------------------------------
    combined = pd.concat(dfs.values(), ignore_index=True)

    # -------------------------------------------------
    # 4. Extract mass from sample names
    # -------------------------------------------------
    def extract_mass(sample):
        m = re.search(r'_(\d+)GeV', sample)
        return int(m.group(1)) if m else None

    combined["mass_GeV"] = combined["Sample"].apply(extract_mass)

    # -------------------------------------------------
    # 5. Remove 100 & 150 GeV samples
    # -------------------------------------------------
    combined = combined[~combined["mass_GeV"].isin([100, 150])]

    # -------------------------------------------------
    # 6. Build pivot tables
    # -------------------------------------------------
    pivots = {}
    for var in variables:
        piv = combined.pivot(index="Sample", columns="source", values=var)
        piv = piv.loc[combined["Sample"].unique()]  # keep consistent ordering
        pivots[var] = piv

    # -------------------------------------------------
    # 7. Sort sample order by mass
    # -------------------------------------------------
    sample_masses = {s: extract_mass(s) for s in pivots[variables[0]].index}
    sorted_samples = sorted(sample_masses.keys(), key=lambda s: sample_masses[s])

    for var in variables:
        pivots[var] = pivots[var].loc[sorted_samples]

    # -------------------------------------------------
    # 8. Create 3-row figure instead of 1-row
    # -------------------------------------------------
    sources = pivots[variables[0]].columns
    num_samples = len(sorted_samples)
    x = np.arange(num_samples)
    width = 0.12
    
    fig, axs = plt.subplots(len(variables), 1, figsize=(18, 18), sharex=True)  # <--- stacked vertically

    for ax, var in zip(axs, variables):
        
        pivot = pivots[var]

        for i, src in enumerate(sources):
            bars = ax.bar(x + i*width, pivot[src], width, label=src)

            # Value labels on top
            for b in bars:
                height = b.get_height()
                ax.text(
                    b.get_x() + b.get_width()/2,
                    height,
                    f"{height:.2f}",
                    ha='center', va='bottom',
                    fontsize=6,
                    rotation=90
                )

        ax.set_title(var, fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Bottom subplot gets the xtick labels
    axs[-1].set_xticks(x + width*(len(sources)-1)/2)
    axs[-1].set_xticklabels(sorted_samples, rotation=90, fontsize=8)

    for ax in axs:
        ax.set_ylabel("Value")

    # Legend on last subplot
    axs[-1].legend(title="Source", fontsize=10)

    plt.tight_layout()

    # -------------------------------------------------
    # 9. Save PDF
    # -------------------------------------------------
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, filename)
    plt.savefig(outpath, facecolor='w', dpi=300)
    plt.show()
    plt.close()

    print(f"Saved plot to: {outpath}")



def returnYielddPhiDisplacement(sample, dPhiCut=2.5, mJIsoCut=None, egmIsoCut=None, dPhiFlag = True, dispFlag = True, pixelCut=2):
    lumi = 59830
    
    mask = np.ones(len(sample['dPhi']), dtype=bool)

    if (mJIsoCut is not None) and (egmIsoCut is not None):
        mask &= (np.array(sample['egm_isolation']) <= egmIsoCut) & (np.array(sample['isolation']) <= mJIsoCut)

    pixelHits = np.array(sample['pixelHits'], dtype=object)
    pixelHits = np.where(pixelHits == None, np.nan, pixelHits).astype(float)
    pixelMask = np.where(np.isnan(pixelHits), False, pixelHits <= pixelCut)
    dsaCut = (np.array(sample['dsaMu_n']) >= 1)
    displacementCut = np.where(np.array(sample['pfMu_n']) >= 1, pixelMask, (np.array(sample['dsaMu_n']) >= 1))
    dPhiMask = (np.array(sample['dPhi']) >= dPhiCut)
    
    if dPhiFlag:
        mask &= dPhiMask
    else:
        mask &= ~dPhiMask

    if dispFlag:
        mask &= displacementCut
    else:
        mask &= ~displacementCut

    scale = lumi * sample['xsec'] / sample['nevents']

    weights = np.array(sample['passing_weights'])
   

    return scale*np.sum(weights[mask]), scale*scale*np.sum(weights[mask]*weights[mask])

def getObservedCountsdPhiDisplacement(backgrounds, dPhiCut, mJIsoCut, egmIsoCut, pixelCut=3):

    results = {}

    # Region definitions
    region_kwargs = {
        "A": {"dPhiFlag": True,  "dispFlag": True},
        "B": {"dPhiFlag": True,  "dispFlag": False},
        "C": {"dPhiFlag": False, "dispFlag": True},
        "D": {"dPhiFlag": False, "dispFlag": False}
    }

    base_kwargs = {
        "dPhiCut": dPhiCut,
        "mJIsoCut": mJIsoCut,
        "egmIsoCut": egmIsoCut,
        "pixelCut": pixelCut,
    }

    # Background grouping
    ttjets = backgrounds[-1]
    qcd_samples = backgrounds[2:14]
    dy_samples = backgrounds[:2]

    # Storage
    counts = {r: {"QCD":0, "TTJets":0, "DY":0} for r in "ABCD"}
    unc2   = {r: {"QCD":0, "TTJets":0, "DY":0} for r in "ABCD"}

    # --- TTJets ---
    for r, flags in region_kwargs.items():
        val, u = returnYielddPhiDisplacement(ttjets, **base_kwargs, **flags)
        counts[r]["TTJets"] = val
        unc2[r]["TTJets"] = u

    # --- QCD ---
    for ent in qcd_samples:
        for r, flags in region_kwargs.items():
            val, u = returnYielddPhiDisplacement(ent, **base_kwargs, **flags)
            counts[r]["QCD"] += val
            unc2[r]["QCD"] += u

    # --- DY ---
    for ent in dy_samples:
        for r, flags in region_kwargs.items():
            val, u = returnYielddPhiDisplacement(ent, **base_kwargs, **flags)
            counts[r]["DY"] += val
            unc2[r]["DY"] += u

    # --- Totals ---
    for r in "ABCD":

        total = sum(counts[r].values())
        total_unc2 = sum(unc2[r].values())

        results[f"{r}Counts"] = total
        results[f"{r}_Unc"] = np.sqrt(total_unc2)

        # individual backgrounds
        for bkg in ["QCD","TTJets","DY"]:
            results[f"{bkg}_{r}Counts"] = counts[r][bkg]
            results[f"{bkg}{r}_Unc"] = np.sqrt(unc2[r][bkg])

    return results

def returnHistIsoVsPixel(sample, dPhiCut=None, mJIsoCut=None, egmIsoCut=None,
                        mJIsoFlag=True, dispFlag=True, pixelCut=3):

    lumi = 59830

    h = (
        Hist.new
        .Int(0, 6, name="pixelHits", label="Pixel Hits")  # bins: 0,1,2,3,4,5
        .Reg(50, 0.0, 1.0, name="Isolation", label=r"$\mu$-LJ Isolation")
        .Weight()
    )

    mask = np.ones(len(sample['dPhi']), dtype=bool)

    if egmIsoCut is not None:
        mask &= (np.array(sample['egm_isolation']) <= egmIsoCut)

    if dPhiCut is not None:
        mask &= (np.array(sample['dPhi']) >= dPhiCut)

    pixelHits = np.array(sample['pixelHits'], dtype=object)

    iso = np.array(sample['isolation'])[mask]
    pix = pixelHits[mask]
    weights = np.array(sample['passing_weights'])[mask]
    
    # remove None values
    valid = pix != None
    
    iso = iso[valid]
    pix = pix[valid].astype(int)
    weights = weights[valid]

    h.fill(pix, iso, weight=weights)

    scale = lumi * sample['xsec'] / sample['nevents']
    h *= scale

    return h

def getSignalCountsdPhiDisplacement(signal, dPhiCut, mJIsoCut, egmIsoCut, pixelCut=3):
    cuts_base = {
        'dPhi': dPhiCut,
        'mJIso': mJIsoCut,
        'egmIso': egmIsoCut,
        'pixelCut': pixelCut,
    }
    
    A, A_unc2 =  returnYielddPhiDisplacement(sample=signal, dPhiCut=cuts_base['dPhi'], mJIsoCut=cuts_base['mJIso'], egmIsoCut=cuts_base['egmIso'], dPhiFlag = True, dispFlag = True, pixelCut=cuts_base['pixelCut'])
    
    B, B_unc2 =  returnYielddPhiDisplacement(sample=signal, dPhiCut=cuts_base['dPhi'], mJIsoCut=cuts_base['mJIso'], egmIsoCut=cuts_base['egmIso'], dPhiFlag = True, dispFlag = False, pixelCut=cuts_base['pixelCut'])
    
    C, C_unc2 =  returnYielddPhiDisplacement(sample=signal, dPhiCut=cuts_base['dPhi'], mJIsoCut=cuts_base['mJIso'], egmIsoCut=cuts_base['egmIso'], dPhiFlag = False, dispFlag = True, pixelCut=cuts_base['pixelCut'])
    
    D, D_unc2 = returnYielddPhiDisplacement(sample=signal, dPhiCut=cuts_base['dPhi'], mJIsoCut=cuts_base['mJIso'], egmIsoCut=cuts_base['egmIso'], dPhiFlag = False, dispFlag = False, pixelCut=cuts_base['pixelCut'])

    return A, B, C, D, np.sqrt(A_unc2), np.sqrt(B_unc2), np.sqrt(C_unc2), np.sqrt(D_unc2)


def returnYielddPhiDisplacementMJJ(sample, dPhiCut=2.0, mJIsoCut=None, egmIsoCut=None,
                                   mJJCut=None, dPhiFlag=True, dispFlag=True,
                                   pixelCut=3):
    """Return weighted yield and variance for dPhi/displacement ABCD with an mJJ cut.

    The ABCD axes are dPhi and displacement only. Isolation and mJJ requirements
    are event-level cuts applied before assigning events to A/B/C/D.
    """
    mask = np.ones(len(sample['dPhi']), dtype=bool)

    if mJIsoCut is not None:
        mask &= np.array(sample['isolation']) <= mJIsoCut

    if egmIsoCut is not None:
        mask &= np.array(sample['egm_isolation']) <= egmIsoCut

    if mJJCut is not None:
        mask &= np.array(sample['mJJ']) >= mJJCut

    pixelHits = np.array(sample['pixelHits'], dtype=object)
    pixelHits = np.where(pixelHits == None, np.nan, pixelHits).astype(float)
    pixelMask = np.where(np.isnan(pixelHits), False, pixelHits <= pixelCut)
    displacementCut = np.where(
        np.array(sample['pfMu_n']) >= 1,
        pixelMask,
        np.array(sample['dsaMu_n']) >= 1,
    )
    dPhiMask = np.array(sample['dPhi']) >= dPhiCut

    mask &= dPhiMask if dPhiFlag else ~dPhiMask
    mask &= displacementCut if dispFlag else ~displacementCut

    scale = LUMI * sample['xsec'] / sample['nevents']
    weights = np.array(sample['passing_weights'])

    return scale * np.sum(weights[mask]), scale * scale * np.sum(weights[mask] * weights[mask])


def getObservedCountsdPhiDisplacementMJJ(backgrounds, dPhiCut=2.0, mJIsoCut=None,
                                         egmIsoCut=None, mJJCut=None, pixelCut=3):
    """Compute background ABCD counts for dPhi/displacement with mJJ preselection."""
    results = {}

    region_kwargs = {
        "A": {"dPhiFlag": True, "dispFlag": True},
        "B": {"dPhiFlag": True, "dispFlag": False},
        "C": {"dPhiFlag": False, "dispFlag": True},
        "D": {"dPhiFlag": False, "dispFlag": False},
    }

    base_kwargs = {
        "dPhiCut": dPhiCut,
        "mJIsoCut": mJIsoCut,
        "egmIsoCut": egmIsoCut,
        "mJJCut": mJJCut,
        "pixelCut": pixelCut,
    }

    groups = {
        "DY": backgrounds[:2],
        "QCD": backgrounds[2:14],
        "TTJets": [backgrounds[-1]],
    }

    counts = {r: {bkg: 0.0 for bkg in groups} for r in "ABCD"}
    unc2 = {r: {bkg: 0.0 for bkg in groups} for r in "ABCD"}

    for bkg, samples in groups.items():
        for ent in samples:
            for region, flags in region_kwargs.items():
                val, var = returnYielddPhiDisplacementMJJ(ent, **base_kwargs, **flags)
                counts[region][bkg] += val
                unc2[region][bkg] += var

    for region in "ABCD":
        total = sum(counts[region].values())
        total_unc2 = sum(unc2[region].values())
        results[f"{region}Counts"] = total
        results[f"{region}_Unc"] = np.sqrt(total_unc2)

        for bkg in ["QCD", "TTJets", "DY"]:
            results[f"{bkg}_{region}Counts"] = counts[region][bkg]
            results[f"{bkg}{region}_Unc"] = np.sqrt(unc2[region][bkg])

    return results


def computeABCDMetrics(counts, near_zero=1e-12):
    """Add ABCD prediction, closure, significance, and transfer-factor metrics."""
    out = dict(counts)
    A, B, C, D = (out[f"{r}Counts"] for r in "ABCD")
    A_unc, B_unc, C_unc, D_unc = (out[f"{r}_Unc"] for r in "ABCD")

    invalid = []
    if abs(A) <= near_zero:
        invalid.append("A")
    if abs(C) <= near_zero:
        invalid.append("C")
    if abs(D) <= near_zero:
        invalid.append("D")

    prediction = np.nan
    prediction_unc = np.nan
    closure = np.nan
    closure_unc = np.nan
    closure_sig = np.nan
    tf_ac = np.nan
    tf_ac_unc = np.nan
    tf_bd = np.nan
    tf_bd_unc = np.nan

    if abs(D) > near_zero:
        prediction = B * C / D
        prediction_unc = np.sqrt(
            ((C / D) * B_unc) ** 2
            + ((B / D) * C_unc) ** 2
            + ((B * C / (D * D)) * D_unc) ** 2
        )

    if abs(A) > near_zero and np.isfinite(prediction):
        closure = 1 - prediction / A
        closure_unc = np.sqrt(
            ((prediction / (A * A)) * A_unc) ** 2
            + ((C / (D * A)) * B_unc) ** 2
            + ((B / (D * A)) * C_unc) ** 2
            + ((B * C / (D * D * A)) * D_unc) ** 2
        ) if abs(D) > near_zero else np.nan
        sig_den = np.sqrt(A_unc ** 2 + prediction_unc ** 2)
        if sig_den > near_zero:
            closure_sig = (A - prediction) / sig_den
        else:
            invalid.append("closure_significance_denominator")

    if abs(C) > near_zero:
        tf_ac = A / C
        tf_ac_unc = np.sqrt((A_unc / C) ** 2 + ((A * C_unc) / (C * C)) ** 2)

    if abs(D) > near_zero:
        tf_bd = B / D
        tf_bd_unc = np.sqrt((B_unc / D) ** 2 + ((B * D_unc) / (D * D)) ** 2)

    tf_diff = np.nan
    tf_pull = np.nan
    if np.isfinite(tf_ac) and np.isfinite(tf_bd):
        tf_diff = tf_ac - tf_bd
        tf_unc = np.sqrt(tf_ac_unc ** 2 + tf_bd_unc ** 2)
        if tf_unc > near_zero:
            tf_pull = tf_diff / tf_unc

    out.update({
        "prediction": prediction,
        "predictionUnc": prediction_unc,
        "closure": closure,
        "closureUnc": closure_unc,
        "closureSig": closure_sig,
        "transfer_A_over_C": tf_ac,
        "transfer_A_over_C_Unc": tf_ac_unc,
        "transfer_B_over_D": tf_bd,
        "transfer_B_over_D_Unc": tf_bd_unc,
        "transfer_difference": tf_diff,
        "transfer_pull": tf_pull,
        "status": "valid" if not invalid else "invalid:" + ",".join(sorted(set(invalid))),
    })
    return out


def scanDphiDisplacementMJJIsolation(backgrounds, scan_variable, scan_values,
                                     fixed_mJIsoCut=0.75, fixed_egmIsoCut=0.75,
                                     mJJCut=100, dPhiCut=2.0, pixelCut=3):
    """Scan one isolation cut while keeping the other isolation cut loose."""
    rows = []
    for cut in scan_values:
        if scan_variable == "isolation":
            mJIsoCut = cut
            egmIsoCut = fixed_egmIsoCut
        elif scan_variable == "egm_isolation":
            mJIsoCut = fixed_mJIsoCut
            egmIsoCut = cut
        else:
            raise ValueError("scan_variable must be 'isolation' or 'egm_isolation'")

        counts = getObservedCountsdPhiDisplacementMJJ(
            backgrounds=backgrounds,
            dPhiCut=dPhiCut,
            mJIsoCut=mJIsoCut,
            egmIsoCut=egmIsoCut,
            mJJCut=mJJCut,
            pixelCut=pixelCut,
        )
        metrics = computeABCDMetrics(counts)
        metrics.update({
            "scan_variable": scan_variable,
            "scan_cut": cut,
            "mJIsoCut": mJIsoCut,
            "egmIsoCut": egmIsoCut,
            "mJJCut": mJJCut,
            "dPhiCut": dPhiCut,
            "pixelCut": pixelCut,
        })
        rows.append(metrics)

    return pd.DataFrame(rows)


def plotDphiDisplacementMJJScanThreePanel(df, filename, title=None,
                                           closure_ylim=None, transfer_ylim=None):
    """Save closure, transfer-factor, and ABCD-count panels for one scan."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    x = df["scan_cut"].to_numpy()

    axes[0].errorbar(x, df["closure"], yerr=df["closureUnc"], marker="o", linestyle="-")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("Closure")
    if closure_ylim is not None:
        axes[0].set_ylim(*closure_ylim)

    axes[1].errorbar(
        x, df["transfer_A_over_C"], yerr=df["transfer_A_over_C_Unc"],
        marker="o", linestyle="-", label="A/C",
    )
    axes[1].errorbar(
        x, df["transfer_B_over_D"], yerr=df["transfer_B_over_D_Unc"],
        marker="s", linestyle="-", label="B/D",
    )
    axes[1].set_ylabel("Transfer factor")
    if transfer_ylim is not None:
        axes[1].set_ylim(*transfer_ylim)
    axes[1].legend()

    for region, marker in zip("ABCD", ["o", "s", "^", "D"]):
        axes[2].errorbar(
            x, df[f"{region}Counts"], yerr=df[f"{region}_Unc"],
            marker=marker, linestyle="-", label=region,
        )
    axes[2].set_xlabel(df["scan_variable"].iloc[0])
    axes[2].set_ylabel("Background count")
    axes[2].legend()

    if title:
        fig.suptitle(title)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, facecolor="w", dpi=300)
    plt.close(fig)



def _sample_array_for_cut(sample, key, dtype=float):
    """Return a sample array with None converted to NaN for numerical cuts."""
    values = np.array(sample[key], dtype=object)
    values = np.where(values == None, np.nan, values)
    return values.astype(dtype)


def returnYieldIsoIsoPlane(sample, muIsoCut, egmIsoCut, dPhiCut=2.0, mJJCut=150,
                           pixelCut=3, muIsoFlag=True, egmIsoFlag=True):
    """Return weighted yield and variance for the iso-vs-iso ABCD plane.

    Base cuts are dPhi >= dPhiCut, mJJ >= mJJCut, and the displacement
    requirement: if pfMu_n >= 1 require pixelHits <= pixelCut, otherwise
    require dsaMu_n >= 1. The ABCD axes are only mu-LJ isolation and
    egm-LJ isolation. Region convention is A=(pass muIso, pass egmIso),
    B=(fail muIso, pass egmIso), C=(pass muIso, fail egmIso),
    D=(fail muIso, fail egmIso).
    """
    dphi = _sample_array_for_cut(sample, 'dPhi')
    mask = (dphi >= dPhiCut)
    mask &= _sample_array_for_cut(sample, 'mJJ') >= mJJCut

    pixel_hits = _sample_array_for_cut(sample, 'pixelHits')
    pixel_mask = np.where(np.isnan(pixel_hits), False, pixel_hits <= pixelCut)
    displacement_cut = np.where(
        _sample_array_for_cut(sample, 'pfMu_n') >= 1,
        pixel_mask,
        _sample_array_for_cut(sample, 'dsaMu_n') >= 1,
    )
    mask &= displacement_cut

    mu_iso_pass = _sample_array_for_cut(sample, 'isolation') <= muIsoCut
    egm_iso_pass = _sample_array_for_cut(sample, 'egm_isolation') <= egmIsoCut
    mask &= mu_iso_pass if muIsoFlag else ~mu_iso_pass
    mask &= egm_iso_pass if egmIsoFlag else ~egm_iso_pass

    scale = LUMI * sample['xsec'] / sample['nevents']
    weights = _sample_array_for_cut(sample, 'passing_weights')
    return scale * np.sum(weights[mask]), scale * scale * np.sum(weights[mask] * weights[mask])


def getObservedCountsIsoIsoPlane(backgrounds, muIsoCut, egmIsoCut, dPhiCut=2.0,
                                 mJJCut=150, pixelCut=3):
    """Compute background ABCD counts for the iso-vs-iso ABCD plane."""
    region_kwargs = {
        'A': {'muIsoFlag': True, 'egmIsoFlag': True},
        'B': {'muIsoFlag': False, 'egmIsoFlag': True},
        'C': {'muIsoFlag': True, 'egmIsoFlag': False},
        'D': {'muIsoFlag': False, 'egmIsoFlag': False},
    }
    groups = {
        'DY': backgrounds[:2],
        'QCD': backgrounds[2:14],
        'TTJets': [backgrounds[-1]],
    }
    base_kwargs = {
        'muIsoCut': muIsoCut,
        'egmIsoCut': egmIsoCut,
        'dPhiCut': dPhiCut,
        'mJJCut': mJJCut,
        'pixelCut': pixelCut,
    }
    counts = {region: {bkg: 0.0 for bkg in groups} for region in 'ABCD'}
    unc2 = {region: {bkg: 0.0 for bkg in groups} for region in 'ABCD'}

    for bkg, samples in groups.items():
        for sample in samples:
            for region, flags in region_kwargs.items():
                val, var = returnYieldIsoIsoPlane(sample, **base_kwargs, **flags)
                counts[region][bkg] += val
                unc2[region][bkg] += var

    results = {}
    for region in 'ABCD':
        total = sum(counts[region].values())
        total_unc2 = sum(unc2[region].values())
        results[f'{region}Counts'] = total
        results[f'{region}_Unc'] = np.sqrt(total_unc2)
        for bkg in ['QCD', 'TTJets', 'DY']:
            results[f'{bkg}_{region}Counts'] = counts[region][bkg]
            results[f'{bkg}{region}_Unc'] = np.sqrt(unc2[region][bkg])
    return results


def getSignalCountsIsoIsoPlane(signal, muIsoCut, egmIsoCut, dPhiCut=2.0,
                               mJJCut=150, pixelCut=3):
    """Compute signal A/B/C/D counts and uncertainties for one signal sample."""
    region_kwargs = {
        'A': {'muIsoFlag': True, 'egmIsoFlag': True},
        'B': {'muIsoFlag': False, 'egmIsoFlag': True},
        'C': {'muIsoFlag': True, 'egmIsoFlag': False},
        'D': {'muIsoFlag': False, 'egmIsoFlag': False},
    }
    results = {}
    for region, flags in region_kwargs.items():
        val, var = returnYieldIsoIsoPlane(
            signal,
            muIsoCut=muIsoCut,
            egmIsoCut=egmIsoCut,
            dPhiCut=dPhiCut,
            mJJCut=mJJCut,
            pixelCut=pixelCut,
            **flags,
        )
        results[f'{region}Counts'] = val
        results[f'{region}_Unc'] = np.sqrt(var)
    return results


def _safe_ratio_with_unc(num, den, num_unc, den_unc, near_zero=1e-12):
    if abs(den) <= near_zero:
        return np.nan, np.nan
    ratio = num / den
    unc = np.sqrt((num_unc / den) ** 2 + ((num * den_unc) / (den * den)) ** 2)
    return ratio, unc


def compute_abcd_metrics(counts, near_zero=1e-12):
    """Compute ABCD prediction, closure, transfer factors, and validity status.

    Closure is defined as 1 - prediction / observed_A. Invalid denominator
    cases are stored as NaN and recorded in the status string.
    """
    out = dict(counts)
    A, B, C, D = (out[f'{r}Counts'] for r in 'ABCD')
    A_unc, B_unc, C_unc, D_unc = (out[f'{r}_Unc'] for r in 'ABCD')
    invalid = []
    for label, value in [('A', A), ('B', B), ('C', C), ('D', D)]:
        if abs(value) <= near_zero:
            invalid.append(label)

    prediction = np.nan
    prediction_unc = np.nan
    if abs(D) > near_zero:
        prediction = B * C / D
        prediction_unc = np.sqrt(
            ((C / D) * B_unc) ** 2
            + ((B / D) * C_unc) ** 2
            + ((B * C / (D * D)) * D_unc) ** 2
        )

    closure = np.nan
    closure_unc = np.nan
    closure_sig = np.nan
    if abs(A) > near_zero and np.isfinite(prediction):
        closure = 1 - prediction / A
        if abs(D) > near_zero:
            closure_unc = np.sqrt(
                ((prediction / (A * A)) * A_unc) ** 2
                + ((C / (D * A)) * B_unc) ** 2
                + ((B / (D * A)) * C_unc) ** 2
                + ((B * C / (D * D * A)) * D_unc) ** 2
            )
        sig_den = np.sqrt(A_unc ** 2 + prediction_unc ** 2)
        if sig_den > near_zero:
            closure_sig = (A - prediction) / sig_den
        else:
            invalid.append('closure_significance_denominator')

    tf_ab, tf_ab_unc = _safe_ratio_with_unc(A, B, A_unc, B_unc, near_zero)
    tf_cd, tf_cd_unc = _safe_ratio_with_unc(C, D, C_unc, D_unc, near_zero)
    tf_ac, tf_ac_unc = _safe_ratio_with_unc(A, C, A_unc, C_unc, near_zero)
    tf_bd, tf_bd_unc = _safe_ratio_with_unc(B, D, B_unc, D_unc, near_zero)
    tf_ratio, tf_ratio_unc = _safe_ratio_with_unc(tf_ac, tf_bd, tf_ac_unc, tf_bd_unc, near_zero)

    tf_ac_bd_diff = np.nan
    tf_ac_bd_pull = np.nan
    if np.isfinite(tf_ac) and np.isfinite(tf_bd):
        tf_ac_bd_diff = tf_ac - tf_bd
        tf_unc = np.sqrt(tf_ac_unc ** 2 + tf_bd_unc ** 2)
        if tf_unc > near_zero:
            tf_ac_bd_pull = tf_ac_bd_diff / tf_unc

    out.update({
        'prediction': prediction,
        'predictionUnc': prediction_unc,
        'closure': closure,
        'closureUnc': closure_unc,
        'closureSignificance': closure_sig,
        'tf_AB': tf_ab,
        'tf_AB_Unc': tf_ab_unc,
        'tf_CD': tf_cd,
        'tf_CD_Unc': tf_cd_unc,
        'tf_AC': tf_ac,
        'tf_AC_Unc': tf_ac_unc,
        'tf_BD': tf_bd,
        'tf_BD_Unc': tf_bd_unc,
        'tf_closure_ratio': tf_ratio,
        'tf_closure_ratio_Unc': tf_ratio_unc,
        'tf_AC_minus_BD': tf_ac_bd_diff,
        'tf_AC_minus_BD_pull': tf_ac_bd_pull,
        'status': 'valid' if not invalid else 'invalid:' + ','.join(sorted(set(invalid))),
    })
    return out
