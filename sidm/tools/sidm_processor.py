"""Module to define the base SIDM processor"""

# python
import copy
import numpy as np
# columnar analysis
from coffea import processor
from coffea.nanoevents.methods import nanoaod
from coffea.nanoevents.methods import vector as cvec
import awkward as ak
import fastjet
import vector
# local
from sidm import BASE_DIR
from sidm.tools import selection, cutflow, utilities
from sidm.definitions.hists import hist_defs, counter_defs
from sidm.definitions.objects import preLj_objs, postLj_objs, postLj_objs_MC
import coffea.nanoevents.transforms as tr

def _patched_local2global(stack):
    """
    Original: index,target_offsets,!local2global
    Turn jagged local index into global index
    """
    target_offsets = ak.Array(stack.pop())
    index = ak.Array(stack.pop())
    index = index.mask[index >= 0] + target_offsets[:-1]
    index = index.mask[index < target_offsets[1:]]

    out = ak.flatten(ak.fill_none(index, -1), axis=None)
    out = ak.values_astype(out, np.int64)

    stack.append(out)
tr.local2global = _patched_local2global

class SidmProcessor(processor.ProcessorABC):
    """Class to apply selections, make histograms, and make cutflows

    Accepts NanoEvents records that are assumed to have been produced by FFSchema. Selections are
    chosen by supplying a list of selection names (as defined in selections.yaml), and histograms
    are chosen by providing a list of histogram collection names (as definined in
    hist_collections.yaml).
    """

    def __init__(
        self,
        channel_names,
        hist_collection_names,
        lj_reco_choices=["0.4"],
        selections_cfg="configs/selections.yaml",
        histograms_cfg="configs/hist_collections.yaml",
        unweighted_hist=False,
        verbose=False,
        debug=False,
        debug_branches=None,
        include_default_debug_branches=True,
        debug_suppress_failures=True,
    ):
        self.channel_names = channel_names
        self.hist_collection_names = hist_collection_names
        self.lj_reco_choices = lj_reco_choices
        self.selections_cfg = selections_cfg
        self.histograms_cfg = histograms_cfg
        self.unweighted_hist = unweighted_hist
        self.obj_defs = preLj_objs
        self.postLj_objs = postLj_objs
        self.postLj_objs_MC = postLj_objs_MC
        self.postLj_objs = postLj_objs
        self.postLj_objs_MC = postLj_objs_MC
        self.verbose = verbose

        # Optional debug output.
        #
        # This does not change the standard main-branch processor output.
        # If debug=True, an additional out["debug"] dictionary is added.
        #
        # Users can add their own arrays with:
        #
        # extra_debug_branches = {
        #     "my_array_name": lambda sel_objs, events: sel_objs["ljs"][:, 0].pt,
        # }
        #
        # processor = SidmProcessor(
        #     ...,
        #     debug=True,
        #     debug_branches=extra_debug_branches,
        # )
        self.debug = debug
        self.debug_suppress_failures = debug_suppress_failures

        self.debug_branches = {}
        if include_default_debug_branches:
            self.debug_branches.update(self.default_debug_branches())
        if debug_branches is not None:
            self.debug_branches.update(debug_branches)

    def process(self, events):
        """Apply selections, make histograms and cutflow"""
        is_data = events.metadata["is_data"]
        # create object collections
        # fixme: only include objs used in cuts or hists
        objs = {}
        for obj_name, obj_def in self.obj_defs.items():
            try:
                obj = obj_def(events)
            except AttributeError:
                print(f"Warning: {obj_name} not found in this sample. Skipping.")
                continue
            objs[obj_name] = obj

            # pt order
            objs[obj_name] = self.order(objs[obj_name])

            # use nanoevents.Muon behaviors for dsa muons
            if obj_name == "dsaMuons":
                forms = {f: objs[obj_name][f] for f in objs[obj_name].fields}
                objs[obj_name] = ak.zip(forms, with_name="Muon", behavior=nanoaod.behavior)

            # add lxy attribute to particles with children
            if hasattr(obj, "children"):
                objs[obj_name]["lxy"] = utilities.lxy(objs[obj_name])

            # add dxy wrt beamspot for all objs that don't already have it
            if hasattr(obj, "vx") and not hasattr(obj, "dxy") and "bs" in objs:
                objs[obj_name]["dxy"] = utilities.dxy(objs[obj_name], ref=objs["bs"])

            # add dimension to one-per-event objects to allow independent obj and evt cuts
            # skip objects with no fields
            if objs[obj_name].ndim == 1 and "x" in obj.fields:
                counts = ak.ones_like(objs[obj_name].x, dtype=np.int32)
                objs[obj_name] = ak.unflatten(objs[obj_name], counts)


        cutflows = {}
        counters = {}
        debug_output = {} if self.debug else None

        # define histograms
        hists = self.build_histograms()

        # define pre-lj object, lj, post-lj obj, and event cuts per channel
        # define pre-lj object, lj, post-lj obj, and event cuts per channel
        ch_cuts = self.build_cuts()

        # define event weights
        if not is_data:
            evt_weights = self.obj_defs["weight"](events)
        else:
            evt_weights = ak.broadcast_arrays(1.0, self.obj_defs["met"](events))[0]

        # define event weights
        if not is_data:
            evt_weights = self.obj_defs["weight"](events)
        else:
            evt_weights = ak.broadcast_arrays(1.0, self.obj_defs["met"](events))[0]

        # loop through lj reco choices and channels, treating each lj+channel pair as a unique Selection
        for channel, cuts in ch_cuts.items():
            obj_selection = selection.JaggedSelection(cuts["obj"], self.verbose)
            nested_selection = selection.NestedSelection(cuts["obj"], self.verbose)
            nested_selection = selection.NestedSelection(cuts["obj"], self.verbose)

            for lj_reco in self.lj_reco_choices:
                sel_objs = objs.copy()

                # apply selections on matched_muons within the DSA muons and matched_dsa_muons within the PF muons
                # remove None entries from matched PF or DSA muons before applying cuts
                sel_objs["dsaMuons"]["good_matched_muons"] = nested_selection.apply_obj_cuts(sel_objs, ak.drop_none(sel_objs["dsaMuons"].matched_muons), "muons")
                sel_objs["muons"]["good_matched_dsa_muons"] = nested_selection.apply_obj_cuts(sel_objs, ak.drop_none(sel_objs["muons"].matched_dsa_muons), "dsaMuons")

                # apply pre-LJ object selection
                sel_objs = obj_selection.apply_obj_cuts(sel_objs)
                sel_objs = obj_selection.apply_obj_cuts(sel_objs)

                # reconstruct lepton jets
                sel_objs["ljs"] = self.build_lepton_jets(sel_objs, float(lj_reco))

                # apply obj selection to ljs
                lj_selection = selection.JaggedSelection(cuts["lj"], self.verbose)
                sel_objs = lj_selection.apply_obj_cuts(sel_objs)

                # add post-lj objects to sel_objs
                if not is_data:
                    self.postLj_objs = {**self.postLj_objs, **self.postLj_objs_MC}
                for obj in self.postLj_objs:
                    sel_objs[obj] = self.postLj_objs[obj](sel_objs)
                if not is_data:
                    self.postLj_objs = {**self.postLj_objs, **self.postLj_objs_MC}
                for obj in self.postLj_objs:
                    sel_objs[obj] = self.postLj_objs[obj](sel_objs)

                # apply post-lj obj selection
                postLj_selection = selection.JaggedSelection(cuts["postLj_obj"], self.verbose)
                sel_objs = postLj_selection.apply_obj_cuts(sel_objs)
 
                # build Selection objects and apply event selection
                sel_objs["evt_weights"] = evt_weights
                sel_objs["evt_weights"] = evt_weights
                evt_selection = selection.Selection(cuts["evt"], self.verbose)
                sel_objs = evt_selection.apply_evt_cuts(sel_objs)

                # Optional debug output.
                #
                # This is the only addition to the main processing loop.
                # It saves arrays after the event selection has been applied.
                if self.debug:
                    lj_reco_key = str(lj_reco)
                    if lj_reco_key not in debug_output:
                        debug_output[lj_reco_key] = {}
                    debug_output[lj_reco_key][channel] = self.fill_debug_branches(
                        sel_objs,
                        events,
                    )

                # fill all hists

                # fixme: disable cutflows due to sequential event cut implementation
                # store cutflow in separate dict

                # fixme: disable cutflows due to sequential event cut implementation
                # store cutflow in separate dict
                if lj_reco not in cutflows:
                    cutflows[str(lj_reco)] = {}
                cutflows[str(lj_reco)][channel] = evt_selection.cutflow
                cutflows[str(lj_reco)][channel] = evt_selection.cutflow

                # fill histograms for this channel+lj_reco pair
                sel_objs["ch"] = channel
                sel_objs["lj_reco"] = lj_reco
                hist_weights = sel_objs["evt_weights"]
                sel_objs["ch"] = channel
                sel_objs["lj_reco"] = lj_reco
                hist_weights = sel_objs["evt_weights"]
                if self.unweighted_hist:
                    hist_weights = ak.ones_like(hist_weights)
                for h in hists.values():
                    h.fill(sel_objs, hist_weights, self.verbose)
                    h.fill(sel_objs, hist_weights, self.verbose)

                # Fill counters
                if lj_reco not in counters:
                    counters[lj_reco] = {}
                counters[lj_reco][channel] = {}

                for name, counter in counter_defs.items():
                    try:
                        counters[lj_reco][channel][name] = counter(sel_objs)
                    except (KeyError, AttributeError) as e:
                        print(f"Warning: cannot fill counter {name}. Skipping.")

        # lose lj_reco dimension to cutflows if only one reco was run
        # fixme: disable cutflows due to sequential event cut implemention
        # fixme: disable cutflows due to sequential event cut implemention
        if len(self.lj_reco_choices) == 1:
            cutflows = cutflows[self.lj_reco_choices[0]]

        out = {
            "cutflow": cutflows,
            "hists": {n: h.hist for n, h in hists.items()},  # output hist.Hists, not Histograms
            "counters": counters,
            "metadata": {
                "n_evts": events.metadata["entrystop"] - events.metadata["entrystart"],
                "scaled_sum_weights": ak.sum(evt_weights)/events.metadata["skim_factor"],
                # add sample metadata as set_accumulator to only keep unique values during accumulation
                "year": processor.set_accumulator([events.metadata["year"]]),
                "is_data": processor.set_accumulator([events.metadata["is_data"]]),
            },
        }

        # Optional debug output.
        #
        # This preserves the normal main-branch return shape and only adds one extra key.
        if self.debug:
            out["debug"] = debug_output

        # Optional debug output.
        #
        # This preserves the normal main-branch return shape and only adds one extra key.
        if self.debug:
            out["debug"] = debug_output

        return {events.metadata["dataset"]: out}

    @staticmethod
    def default_debug_branches():
        """Default debug arrays.

        Each entry maps:

            output_name -> function(sel_objs, events)

        The function should return an awkward array, numpy array, list, or scalar.
        It will be converted with ak.to_list before being written to the output.

        To add more arrays without editing the processor internals, pass:

            debug_branches={
                "new_array": lambda sel_objs, events: sel_objs["ljs"][:, 0].pt,
            }

        to SidmProcessor(..., debug=True, debug_branches=debug_branches).
        """

        return {
            # Muon-EGM LJ ABCD variables
            "mu_lj_iso": lambda sel_objs, events: sel_objs["mu_ljs"][:, 0].isolation,
            "egm_lj_iso": lambda sel_objs, events: sel_objs["egm_ljs"][:, 0].isolation,
            "dPhi": lambda sel_objs, events: abs(
                sel_objs["mu_ljs"][:, 0].delta_phi(sel_objs["egm_ljs"][:, 0])
            ),
            "mJJ": lambda sel_objs, events: (
                sel_objs["mu_ljs"][:, 0] + sel_objs["egm_ljs"][:, 0]
            ).mass,
            "dR": lambda sel_objs, events: abs(
                sel_objs["mu_ljs"][:, 0].delta_r(sel_objs["egm_ljs"][:, 0])
            ),
            "deltaEta": lambda sel_objs, events: abs(
                sel_objs["mu_ljs"][:, 0].eta - sel_objs["egm_ljs"][:, 0].eta
            ),

            # Muon LJ details
            "dsaMu_n": lambda sel_objs, events: sel_objs["mu_ljs"][:, 0].dsaMu_n,
            "pfMu_n": lambda sel_objs, events: sel_objs["mu_ljs"][:, 0].pfMu_n,
            "mu_lj_min_dxy": lambda sel_objs, events: ak.min(
                abs(sel_objs["mu_ljs"][:, 0].muons.dxy),
                axis=-1,
            ),
            "mu_lj_max_dxy": lambda sel_objs, events: ak.max(
                abs(sel_objs["mu_ljs"][:, 0].muons.dxy),
                axis=-1,
            ),
            "pixelHits": lambda sel_objs, events: ak.max(
                sel_objs["mu_ljs"][:, 0].pfMuons.trkNumPixelHits,
                axis=-1,
            ),
            "trkHits": lambda sel_objs, events: ak.max(
                sel_objs["mu_ljs"][:, 0].pfMuons.trkNumTrkLayers,
                axis=-1,
            ),

            # Muon LJ kinematics
            "mu_lj_pt": lambda sel_objs, events: sel_objs["mu_ljs"][:, 0].pt,
            "mu_lj_eta": lambda sel_objs, events: sel_objs["mu_ljs"][:, 0].eta,
            "mu_lj_phi": lambda sel_objs, events: sel_objs["mu_ljs"][:, 0].phi,

            # EGM LJ kinematics
            "egm_lj_pt": lambda sel_objs, events: sel_objs["egm_ljs"][:, 0].pt,
            "egm_lj_eta": lambda sel_objs, events: sel_objs["egm_ljs"][:, 0].eta,
            "egm_lj_phi": lambda sel_objs, events: sel_objs["egm_ljs"][:, 0].phi,

            # Generic leading/subleading LJ variables
            "leading_lj_isolation": lambda sel_objs, events: sel_objs["ljs"][:, 0].isolation,
            "subleading_lj_isolation": lambda sel_objs, events: sel_objs["ljs"][:, 1].isolation,
            "4mu_dPhi": lambda sel_objs, events: abs(
                sel_objs["ljs"][:, 0].delta_phi(sel_objs["ljs"][:, 1])
            ),
            "4mu_mJJ": lambda sel_objs, events: (
                sel_objs["ljs"][:, 0] + sel_objs["ljs"][:, 1]
            ).mass,

            # Leading generic LJ constituent counts
            "Leading_pfMu_n": lambda sel_objs, events: sel_objs["ljs"][:, 0].pfMu_n,
            "Leading_dsaMu_n": lambda sel_objs, events: sel_objs["ljs"][:, 0].dsaMu_n,
            "Leading_pixelHits": lambda sel_objs, events: ak.max(
                sel_objs["ljs"][:, 0].pfMuons.trkNumPixelHits,
                axis=-1,
            ),

            # Subleading generic LJ constituent counts
            "SubLeading_pfMu_n": lambda sel_objs, events: sel_objs["ljs"][:, 1].pfMu_n,
            "SubLeading_dsaMu_n": lambda sel_objs, events: sel_objs["ljs"][:, 1].dsaMu_n,
            "SubLeading_pixelHits": lambda sel_objs, events: ak.max(
                sel_objs["ljs"][:, 1].pfMuons.trkNumPixelHits,
                axis=-1,
            ),

            # Weights
            "passing_weights": lambda sel_objs, events: sel_objs["evt_weights"],

            # Generator weights.
            # This will naturally fail for data unless available, and will be skipped
            # when debug_suppress_failures=True.
            "gen_weights": lambda sel_objs, events: events.Generator.weight,
        }

    def fill_debug_branches(self, sel_objs, events):
        """Fill all configured debug branches.

        This method is intentionally generic. The processor does not need to know
        what arrays users want to save. Users only need to provide a dictionary of
        branch functions through debug_branches.
        """

        debug = {}

        for name, branch_func in self.debug_branches.items():
            try:
                debug[name] = self.to_debug_list(branch_func(sel_objs, events))
            except Exception as e:
                if not self.debug_suppress_failures:
                    raise
                print(f"Warning: cannot fill debug branch {name}. Skipping. Error: {e}")
                debug[name] = []

        return debug

    def to_debug_list(self, value):
        """Convert awkward/numpy/list/scalar values into a serializable debug value."""

        try:
            return ak.to_list(value)
        except Exception:
            pass

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, tuple):
            return list(value)

        return value

    def make_vector(self, objs, collection, fields, type_id=None, mass=None):
        shape = ak.ones_like(objs[collection].pt, dtype=np.dtype(int))
        # all objects must have the same fields to later concatenate and cluster them
        # set fields that aren't available for a given object to be -1
        # these additional fields will be removed after clustering anyway
        forms = {f: objs[collection][f] if f in objs[collection].fields else -1*shape for f in fields}
        forms["part_type"] = objs[collection]["type"] if type_id is None else type_id*shape
        forms["mass"] = objs[collection]["mass"] if mass is None else mass*shape
        if type_id == 8:
            forms["trkNumPixelHits"] = 0*shape
            forms["trkNumTrkLayers"] = 0*shape
        if type_id == 4:
            forms["lostHits"] = 999*shape
        return vector.zip(forms)

    def make_constituent(self, consts, type_ids, name, fields):
        """Return array of particles of given type_ids, name, and only specified fields"""
        relevant_consts = consts[ak.any((consts.part_type == x for x in type_ids), axis=0)]
        forms = {f: relevant_consts.__getattr__(f) for f in fields}
        return ak.zip(forms, with_name=name, behavior=nanoaod.behavior)

    def build_lepton_jets(self, objs, lj_reco):
        """Reconstruct lepton jets according to defintion given by lj_reco"""

        # Use electron/muon/photon/dsamuon collections with a custom distance parameter
        collections = ["muons", "dsaMuons", "electrons", "photons"]
        fields = [objs[c].fields for c in collections]

        unsafe_fields = ['muonIdxG','dsaIdxG','matched_muons','matched_dsa_muons','good_matched_muons','good_matched_dsa_muons']

        all_fields = list(set().union(*fields))
        for field in unsafe_fields:
            try:
                all_fields.remove(field)
            except ValueError:
                continue

        muon_inputs = self.make_vector(objs, "muons", all_fields,  type_id=3)
        dsa_inputs = self.make_vector(objs, "dsaMuons", all_fields, type_id=8, mass=0.106)
        ele_inputs = self.make_vector(objs, "electrons", all_fields, type_id=2)
        photon_inputs = self.make_vector(objs, "photons", all_fields, type_id=4)
        lj_inputs = ak.concatenate([muon_inputs, dsa_inputs, ele_inputs, photon_inputs], axis=-1)

        distance_param = abs(lj_reco)
        jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, distance_param)
        cluster = fastjet.ClusterSequence(lj_inputs, jet_def)
        jets = cluster.inclusive_jets()

        # turn lepton jets back into LorentzVectors that match existing structures
        ljs = ak.zip(
            {
                "x": jets.x,
                "y": jets.y,
                "z": jets.z,
                "t": jets.t,
            },
            with_name="LorentzVector",
            behavior=nanoaod.behavior,
        )

        # add fields to access LJ constituents
        consts = cluster.constituents()
        common_fields = list(set(fields[0]).intersection(*fields[1:]))
        ljs["constituents"] = self.make_constituent(consts, [2, 3, 4, 8], "PtEtaPhiMCollection", common_fields)


    ######
        ## FIX ME! Won't be able to access the dsaMuon matches from the LJ constituent muon, and vice versa
        ## (can only access it from the original muon collection in objects)

        objs["dsaMuons"]["mass"] = ak.full_like(objs["dsaMuons"].pt, 0.105712890625)

        safe_pf_fields = list(objs["muons"].fields)
        safe_dsa_fields = list(objs["dsaMuons"].fields) +  ["trkNumPixelHits","trkNumTrkLayers" ]

        for field in unsafe_fields:
            if field in safe_pf_fields:
                safe_pf_fields.remove(field)
            if field in safe_dsa_fields:
                safe_dsa_fields.remove(field)

        extra_muon_fields =  ["trkNumPixelHits","trkNumTrkLayers" ]
        muon_fields = list(set(safe_pf_fields).intersection(safe_dsa_fields)) + extra_muon_fields
       

        ljs["muons"] = self.make_constituent(consts, [3, 8], "Muon", muon_fields)
        ljs["pfMuons"] = self.make_constituent(consts, [3], "Muon", safe_pf_fields)
        ljs["dsaMuons"] = self.make_constituent(consts, [8], "DSAMuon", safe_dsa_fields)
    ######
        extra_egamma_fields  = ["lostHits"]
        safe_electron_fields = list(objs["electrons"].fields)
        safe_photon_fields = list(objs["photons"].fields)
        egamma_fields  =  list(set(safe_electron_fields).intersection(safe_photon_fields)) + extra_egamma_fields
        ljs ["egamma"]  = self.make_constituent(consts, [2, 4], "Egamma", egamma_fields)
        ljs["electrons"] = self.make_constituent(consts, [2], "Electron",safe_electron_fields )
        ljs["photons"] = self.make_constituent(consts, [4], "Photon", safe_photon_fields)

        # define LJ-level quantities

        # number of constituents
        ljs["pfMu_n"] = ak.num(ljs.pfMuons, axis=-1)
        ljs["dsaMu_n"] = ak.num(ljs.dsaMuons, axis=-1)
        ljs["muon_n"] = ak.num(ljs.muons, axis=-1)
        ljs["electron_n"] = ak.num(ljs.electrons, axis=-1)
        ljs["photon_n"] = ak.num(ljs.photons, axis=-1)

        # dRSpread (the maximum dR betwen any pair of constituents in each lepton jet)
        # a) for each constituent, find the dR between it and all other constituents in the same LJ
        # b) flatten that into a list of dRs per LJ
        # c) and then take the maximum dR per LJ, leaving us with a single value per LJ
        ljs["dRSpread"] = ak.max(
            ak.flatten(
                ljs["constituents"].metric_table(ljs["constituents"], axis=2),
                axis=-1,
            ),
            axis=-1,
        )

        # LJ isolation
        ljs["matched_jet"] = ljs.nearest(objs["jets"], threshold=0.4)       
        ljs["lepton_fraction"] =  ljs["matched_jet"].chEmEF + ljs["matched_jet"].neEmEF + ljs["matched_jet"].muEF
        ljs["isolation"] = ak.fill_none((ljs["matched_jet"].energy / ljs.energy) * (1 - (ljs["lepton_fraction"])), 0)
        ljs["dR_matched_jet"] = ljs.delta_r(ljs["matched_jet"])


        # todo: add LJ displacement

        # pt order the new LJs
        ljs = self.order(ljs)

        # return the new LJ collection
        return ljs

    def build_cuts(self):
        """ Make list of pre-lj object, lj, post-lj obj, and event cuts per channel"""

        selection_menu = utilities.load_yaml(f"{BASE_DIR}/{self.selections_cfg}")

        ch_cuts = {}

        for channel in self.channel_names:
            ch_cuts[channel] = {}
            ch_cuts[channel]["obj"] = {}
            ch_cuts[channel]["preLj_obj"] = {}
            ch_cuts[channel]["lj"] = {}
            ch_cuts[channel]["postLj_obj"] = {}
            ch_cuts[channel]["evt"] = {}

            cuts = selection_menu[channel]
            for obj, obj_cuts in cuts["obj_cuts"].items():
                if obj not in ch_cuts[channel]["obj"]:
                    ch_cuts[channel]["obj"][obj] = []
                ch_cuts[channel]["obj"][obj] = utilities.flatten(obj_cuts)

            if "preLj_obj_cuts" in cuts:
                for obj, obj_cuts in cuts["preLj_obj_cuts"].items():
                    ch_cuts[channel]["preLj_obj"][obj] = utilities.flatten(obj_cuts)

            if "postLj_obj_cuts" in cuts:
                for obj, obj_cuts in cuts["postLj_obj_cuts"].items():
                    if obj == "ljs":
                        ch_cuts[channel]["lj"][obj] = utilities.flatten(obj_cuts)
                    else:
                        ch_cuts[channel]["postLj_obj"][obj] = utilities.flatten(obj_cuts)

            if "evt_cuts" in cuts:
                ch_cuts[channel]["evt"] = utilities.flatten(cuts["evt_cuts"])

        return ch_cuts

    def build_histograms(self):
        """Create dictionary of Histogram objects"""
        hist_menu = utilities.load_yaml(f"{BASE_DIR}/{self.histograms_cfg}")

        # build dictionary and create hist.Hist objects
        hists = {}
        for collection in self.hist_collection_names:
            collection = utilities.flatten(hist_menu[collection])
            for hist_name in collection:
                hists[hist_name] = copy.deepcopy(hist_defs[hist_name])

                # Add lj_reco axis only when more than one reco is run
                lj_reco_names = self.lj_reco_choices if len(self.lj_reco_choices) > 1 else None
                hists[hist_name].make_hist(hist_name, self.channel_names, lj_reco_names)

        return hists

    def order(self, obj):
        """Explicitly order objects"""
        # pt order objects with a pt attribute
        if hasattr(obj, "pt"):
            obj = obj[ak.argsort(obj.pt, ascending=False)]

        # fixme: would be good to explicitly order other objects as well
        return obj

    def postprocess(self, accumulator):
        """Modify accumulator after process has run on all chunks"""
        # scale cutflow and hists according to lumi*xs
        for sample, output in accumulator.items():
            if len(output["metadata"]["is_data"]) != 1 or len(output["metadata"]["year"]) != 1:
                print(f"WARNING: {sample} has more than one value for is_data or year. Not scaling histograms or cutflows.")
                continue

            if output["metadata"]["is_data"].pop():
                print(f"{sample} is data. Not scaling histograms or cutflows.")
                continue

            print(f"{sample} is simulation. Scaling histograms or cutflows according to lumi*xs.")
            year = output["metadata"]["year"].pop()
            sum_weights = output["metadata"]["scaled_sum_weights"]
            lumixs_weight = utilities.get_lumixs_weight(sample, year, sum_weights)
            for name in output["cutflow"]:
                accumulator[sample]["cutflow"][name].scale(lumixs_weight)

            if not self.unweighted_hist:
                for name in output["hists"]:
                    accumulator[sample]["hists"][name] *= lumixs_weight