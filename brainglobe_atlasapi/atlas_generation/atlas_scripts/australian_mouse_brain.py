__version__ = "1"
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import SimpleITK as sitk
from rich.progress import track

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

# Copy-paste this script into a new file and fill in the functions to package
# your own atlas.

### Metadata ###
__version__ = 0
ATLAS_NAME = "australian_mouse"
CITATION = "Janke et al. 2015, https://doi.org/10.1016/j.ymeth.2015.01.005"
SPECIES = "Mus musculus"
ATLAS_LINK = "https://imaging.org.au/AMBMC/"
ORIENTATION = "iar"
ROOT_ID = 9999
RESOLUTION = 15
BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

ANNOTATION_URLS = {
    "basalganglia": "https://osf.io/download/xgj4h",
    "cerebellum": "https://osf.io/download/su2a9",
    "cortex": "https://osf.io/download/hr39a",
    "hippocampus": "https://osf.io/download/tjazr",
    "diencephalon": "https://osf.io/download/9atc7",
}
REFERENCE_URL = "https://osf.io/download/g8p6a"
REGION_IDS = {
    "basalganglia": 1001,
    "cerebellum": 1002,
    "cortex": 1003,
    "hippocampus": 1004,
    "diencephalon": 1005,
}
acronym_dict = {
    "diencephalon": {
        "mfb": "Medial forebrain bundle",
        "3N": "oculomotor nucleus",
        "3PC": "oculomotor nucleus, parvicellular part",
        "3V": "third ventricle",
        "4N": "trochlear nucleus",
        "ac": "anterior commissure",
        "acp": "anterior commissure, posterior limb",
        "AD": "anterodorsal thalamic nucleus",
        "AM": "anteromedial thalamic nucleus",
        "AMV": "anteromedial thalamic nucleus, ventral part",
        "AngT": "angular thalamic nucleus",
        "APT": "anterior pretectal nucleus",
        "APTD": "anterior pretectal nucleus, dorsal part",
        "APTV": "anterior pretectal nucleus, ventral part",
        "Aq": "aqueduct",
        "aur": "auditory radiation",
        "AV": "anteroventral thalamic nucleus",
        "AVDM": "anteroventral thalamic nucleus, dorsolateral part",
        "AVVL": "anteroventral thalamic nucleus, ventrolateral part",
        "B9": "B9 serotonin cells",
        "bic": "brachium of the inferior colliculus",
        "bsc": "brachium or the superior colliculus",
        "CA1": "field CA1 of the hippocampus",
        "CA2": "field CA2 of the hippocampus",
        "CA3": "field CA3 of the hippocampus",
        "cc": "corpus callosum",
        "chp": "choroid plexus",
        "CL": "centrolateral thalamic nucleus",
        "CM": "central medial thalamic nucleus",
        "cp": "cerebral peduncle",
        "csc": "commissure of the superior colliculus",
        "D3V": "dorsal 3rd ventricle",
        "DG": "dentate gyrus",
        "Dk": "nucleus of Darkschewitsch",
        "DLG": "dorsal lateral geniculate nucleus",
        "DpG": "deep gray layer of the superior colliculus",
        "DpWh": "deep white layer of the superior colliculus",
        "DR": "dorsal raphe nucleus",
        "eml": "external medullary lamina",
        "EP": "endopeduncular nucleus",
        "Eth": "ethmoid thalamic nucleus",
        "F": "nucleus of the field of Forel",
        "f": "fornix",
        "fi": "fimbria",
        "fr": "fasciculus retroflexus",
        "Gem": "gemini hypothalamic nucleus",
        "GP": "globus pallidus",
        "hbc": "habenular commissure",
        "hif": "hippocampal fissure",
        "IAD": "interanterodorsal thalamic nucleus",
        "IAM": "interanteromedial thalamic nucleus",
        "ic": "internal capsule",
        "IF": "interfascicular nucleus",
        "IGL": "intergeniculate leaflet",
        "IMA": "intermedullary thalamic nucleus",
        "IMD": "intermediodorsal thalamic nucleus",
        "InC": "interstitial nucleus of Cajal",
        "InCSh": "interstitial nucleus of Cajal, shell region",
        "InG": "intermediate layer of the superior colliculus",
        "InWh": "intermediate white layer of the superior colliculus",
        "IPF": "interpeduncular fossa",
        "IPC": "interpeduncular nucleus, caudal part",
        "IPR": "interpeduncular nucleus, rostral part",
        "isRt": "isthmic reticular formation",
        "IVF": "interventricular foramen",
        "JPLH": "juxtaparaventricular part of lateral hypothalamus",
        "LD": "laterodorsal thalamic nucleus",
        "LDDM": "laterodorsal thalamic nucleus, dorsomedial part",
        "LDVL": "laterodorsal thalamic nucleus, ventrolateral part",
        "LHb": "lateral habenular nucleus",
        "LHbM": "lateral habenular nucleus, medial part",
        "LHbL": "lateral habenular nucleus, lateral part",
        "LP": "lateral posterior thalamic nucleus",
        "LPAG": "lateral part of the periaqueductal gray",
        "LPLC": "lateral posterior thalamic nucleus, laterocaudal part",
        "LPLR": "lateral posterior thalamic nucleus, laterorostral part",
        "LPMC": "lateral posterior thalamic nucleus, mediocaudal part",
        "LPMR": "lateral posterior thalamic nucleus, mediorostral part",
        "LT": "lateral terminal nucleus (pretectum)",
        "Lth": "lithoid nucleus",
        "LV": "lateral ventricle",
        "MA3": "medial accessory oculomotor nucleus",
        "MCPC": "magnocellular nucleus of the posterior commissure",
        "mes": "mesencephalon",
        "MD": "mediodorsal thalamic nucleus",
        "MDC": "mediodorsal thalamic nucleus, central part",
        "MDL": "mediodorsal thalamic nucleus, lateral part",
        "MDM": "mediodorsal thalamic nucleus, medial part",
        "MG": "medial geniculate nucleus",
        "MGD": "medial geniculate nucleus, dorsal part",
        "MGM": "medial geniculate nucleus, medial part",
        "MGV": "medial geniculate nucleus, ventral part",
        "MHb": "medial habenular nucleus",
        "ml": "medial lemniscus",
        "mlf": "medial longitudinal fasciculus",
        "mp": "mamillary peduncle",
        "MPT": "medial pretectal nucleus",
        "MT": "medial terminal nucleus",
        "mt": "mammillothalamic tract",
        "ns": "nigrostriatal tract",
        "Op": "optic nerve layer of the superior colliculus",
        "OPC": "oval paracentral thalamic nucleus",
        "OPT": "olivary pretectal nucleus",
        "opt": "optic tract",
        "OT": "nucleus of the optic tract",
        "p1": "prosomere 1",
        "p1Rt": "prosomere 1 reticular formation",
        "p2": "prosomere 2",
        "p3": "prosomere 3",
        "Pa": "paraventricular hypothalamic nucleus",
        "PaF": "parafascicular thalamic nucleus",
        "PAG": "periaqueductal gray",
        "PaPo": "paraventricular hypothalamic nucleus, posterior part",
        "PaR": "pararubral nucleus",
        "PaXi": "paraxiphoid nucleus",
        "PBP": "parabrachial pigmented nucleus of the ventral tegmental area",
        "pc": "paracentral thalamic nucleus",
        "PC": "paracentral thalamic nucleus",
        "PCom": "nucleus of the posterior commissure",
        "PF": "parafascicular thalamic nucleus",
        "PH": "posterior hypothalamus",
        "PIF": "parainterfascicular nucleus of the VTA",
        "PIL": "posterior intralaminar thalamic nucleus",
        "pm": "principal mammillary tract",
        "PN": "paranigral nucleus",
        "Po": "posterior thalamic nuclear group",
        "PoT": "posterior thalamic nuclear group, triangular part",
        "PP": "peripeduncular nucleus",
        "PR": "prerubral field",
        "PrC": "precommissural nucleus",
        "PrEW": "pre-Edinger-Westphal nucleus",
        "PrG": "pregnculate nucleus of the prethalamus",
        "PSTh": "parasubthalamic nucleus",
        "PT": "paratenial thalamic nucleus",
        "PTg": "peduncular tegmental nucleus",
        "PV": "paraventricular thalamic nucleus",
        "PVA": "paraventricular thalamic nucleus, anterior part",
        "PVP": "paraventricular thalamic nucleus, posterior part",
        "Re": "reuniens thalamic nucleus",
        "REth": "retroethmoid nucleus",
        "Rh": "rhomboid thalamic nucleus",
        "RI": "rostral interstitial nucleus",
        "RLi": "rostral linear nucleus",
        "RM": "retromamillary nucleus",
        "RMC": "red nucleus, magnocellular part",
        "RPC": "red nucleus, parvicellular part",
        "RPF": "retroparafascicular nucleus",
        "RRe": "retroreuniens nucleus",
        "RRF/A8": "retrorubral field and A8 dopamine cells",
        "Rt": "reticular nucleus (prethalamus)",
        "Sag": "sagulum nucleus",
        "SC": "superior colliculus",
        "Sc": "scaphoid nucleus",
        "scp": "superior cerebellar peduncle",
        "SG": "suprageniculate nucleus",
        "SM": "stria medullaris",
        "sm": "stria medullaris",
        "SNC": "substantia nigra, compact part",
        "SNCD": "substantia nigra, compact part, dorsal tier",
        "SNL": "substantia nigra, lateral part",
        "SNR": "substantia nigra, reticular part",
        "sox": "supraoptic descussation",
        "SPF": "subparafascicular thalamic nucleus",
        "SPFPC": "subparafascicular thalamic nucleus, parvicellular part",
        "ST": "stria terminalis",
        "STh": "subthalamic nucleus",
        "str": "superior thalamic radiation",
        "Su3": "supraoculomotor periaqueductal gray",
        "Su3C": "supraoculomotor cap",
        "Sub": "submedius thalamic nucleus",
        "SubB": "subbrachial nucleus",
        "SubG": (
            "subgeniculate nucleus of the prethalamus "
            "(ventrolateral nucleus)"
        ),
        "SuG": "superficial gray layer of the superior colliculus",
        "Te": "terete hypothalamic nucleus",
        "TG": "tectal gray",
        "TS": "triangular septal nucleus",
        "VA": "ventral anterior thalamic nucleus",
        "vhc": "ventral hippocampal commissure",
        "VL": "ventrolateral thalamic nucleus",
        "VLi": "ventral linear nucleus",
        "VM": "ventromedial thalamic nucleus",
        "VRe": "ventral reuniens thalamic nucleus",
        "VTA": "ventral tegmental area",
        "VTAR": "ventral tegmental area, rostral part",
        "vtgx": "ventral tegmental decussation",
        "ZIC": "zona incerta, central part",
        "ZIR": "zona incerta, rostral part",
    },
    "basalganglia": {
        "aca": "Anterior limb of anterior commissure",
        "AcbC": "Accumbens nucleus core",
        "AcbSh": "Accumbens nucleus shell",
        "acp": "Posterior limb of posterior commissure",
        "ADC": "Apparent diffusion coefficient",
        "AOP": "Anterior olfactory area",
        "ASt": "Amygdalostriatal transition area",
        "B": "Basal nucleus (Meynert)",
        "cc/ec": "Corpus collosum/external capsule",
        "Ce": "Central amygdaloid nucleus",
        "CPu": "Caudate putamen",
        "DWI": "Diffusion weighted imaging",
        "EA": "Extended amygdala",
        "EP": "Entopeduncular nucleus",
        "f": "Fornix",
        "FA": "Fractional anisotropy",
        "FOD": "fiber orientation distributions",
        "Fu": "Bed nucleus of stria terminalis, fusiform part",
        "GP": "Globus pallidus",
        "HDB": "Nucleus of the horizontal limb of the diagonal band",
        "ic": "Internal capsule",
        "ICjM": "Magna island of Calleja",
        "IEn": "Intermediate nucleus of the endopiriform claustrum",
        "IPAC": (
            "Interstitial nucleus of the post limb of the anterior "
            "commissure"
        ),
        "LAcbSh": "Accumbens nucleus shell, lateral part",
        "LDB": "Lateral nucleus of the horizontal limb of the diagonal band",
        "LH": "Lateral hypothalamus",
        "LPO": "Lateral preoptic area",
        "LSI": "Lateral septal nucleus",
        "LSS": "Lateral striatal stripe",
        "LV": "Lateral Ventricle",
        "MDA": "Minimum deformation atlas",
        "mfb": "Medial forebrain bundle",
        "MS": "Medial septal nucleus",
        "MRI": "Magnetic resonance imaging",
        "ns": "Nigrostriatal bundle",
        "opt": "Optic tract",
        "ROI": "Region of interest",
        "SIB": "Substantia innominate, part B",
        "ST": "Bed nucleus of stria terminalis",
        "st": "Stria terminalis",
        "Tu": "Olfactory tubercle",
        "VDB": "Nucleus of the vertical limb of the diagonal band",
        "VP": "Ventral pallidum",
    },
    "cerebellum": {
        "7Cb": "Lobule 7",
        "8Cb": "Lobule 8",
        "9Cb": "Lobule 9",
        "10Cb": "Lobule 10",
        "Sim": "Simple lobule",
        "Crus1": "Crus 1 of the ansiform lobule",
        "Crus2": "Crus 2 of the ansiform lobule",
        "PM": "Paramedian lobule",
        "Cop": "Copula of the pyramis",
        "PFl": "Paraflocculus",
        "FL": "Flocculus",
        "mlf": "Medial longitudinal fasciculus",
        "scp": "Superior cerebellar peduncle",
        "mcp": "Middle cerebellar peduncle",
        "icp": "Inferior cerebellar peduncle",
        "xscp": "Decussation of the superior cerebellar peduncle",
        "Rbd": "Restiform body",
        "vsc": "Ventral spinocerebellar tract",
        "Med": "Medial cerebellar nucleus",
        "MedDL": "Medial cerebellar nucleus, dorsolateral protuberance",
        "MedL": "Medial cerebellar nucleus, lateral part",
        "Lat": "Lateral cerebellar nucleus",
        "LatPC": "Lateral cerebellar nucleus, parvicellular part",
        "SMV": "Superior medullary velum",
        "DC": "Dorsal cochlear nuclei",
        "VCA": "Ventral cochlear nuclei, anterior part",
        "VCP": "Ventral cochlear nuclei, posterior part",
        "IntA": "Interposed cerebellar nucleus, anterior",
        "IntDL": "Interposed cerebellar nucleus, dorsolateral hump",
        "IntP": "Interposed cerebellar nucleus, posterior",
        "IntPPC": (
            "Interposed cerebellar nucleus, " "posterior parvicellular part"
        ),
        "das": "Dorsal acoustic stria",
    },
    "hippocampus": {
        "CA1-Py": "CA1-Pyramidal cell layer",
        "CA1-Or": "CA1-Pyramidal Oriens layer",
        "CA1-Lmol": "CA1-Lacunosum Moleculare layer",
        "CA1-Rad": "CA1-Radiatum layer",
        "CA2-Py": "CA2-Pyramidal cell layer",
        "CA2-Or": "CA2-Pyramidal Oriens layer",
        "CA2-Lmol": "CA2-Lacunosum Moleculare layer",
        "CA2-Rad": "CA2-Radiatum layer",
        "CA3-Py-inner": "CA3-inner Pyramidal cell layer",
        "CA3-Py-outer": "CA3-outer Pyramidal cell layer",
        "CA3-Py": "CA3-Pyramidal cell layer",
        "CA3-Or": "CA3-Pyramidal Oriens layer",
        "CA3-Lmol": "CA3-Lacunosum Moleculare layer",
        "CA3-Rad": "CA3-Radiatum layer",
        "Or": "Oriens layer",
        "Py": "Pyramidal cell layer",
        "Rad": "Radiatum layer",
        "LMol": "Lacunosum Moleculare layer",
        "Stratum-Lu": "Stratum Lucidum",
        "Dentate-Gyrus-MoDG": "Molecular layer of Dentate Gyrus",
        "Dentate-Gyrus-GrDG": "Granule layer of Dentate Gyrus",
        "Dentate-Gyrus-PoDG": "Polymorph layer of Dentate Gyrus",
        "hif": "hippocampal fissure",
        "FC": "Fasciola Cinereum",
        "PRh": "Perirhinal Cortex",
        "DLEnt": "Dorsolateral Entorhinal area",
        "DIEnt": "Dorsal Intermediate Entorhinal area",
        "VIEnt": "Ventral Intermediate Entorhinal area",
        "MEnt": "Medial Entorhinal area",
        "CEnt": "Caudomedial Entorhinal area",
        "APir": "Amygdalopiriform transition area",
        "PMCo": "Posteromedial cortical amygdaloid area",
        "Pir": "Piriform cortex",
        "DS": "Dorsal Subiculum",
        "VS": "Ventral Subiculum",
        "Post": "Post subiculum",
        "STr": "Subiculum, transition area",
        "PrS": "Presubiculum",
        "PaS": "Parasubiculum",
        "cc": "corpus callosum",
        "ec": "external capsule",
        "cg": "cingulum",
        "LV": "Lateral Ventricle",
        "3 V": "Third Ventricle",
        "alv": "alveus",
        "df": "dorsal fornix",
        "fi": "fimbria",
        "IG": "Indusium Griseum",
        "dhc": "dorsal hippocampal commissure",
        "RSD": "Retrosplenial Dysgranular cortex",
        "RSGa": "Retrosplenial Granular cortex, a region",
        "RSGb": "Retrosplenial Granular cortex, b region",
        "RSGc": "Retrosplenial Granular cortex, c region",
    },
    "cortex": {
        "A24a": "Cingulate cortex, area 24a",
        "A24a'": "Cingulate cortex, area 24a'",
        "A24b": "Cingulate cortex, area 24b",
        "A24b'": "Cingulate cortex, area 24b'",
        "A25": "Cingulate cortex, area 25",
        "A29a": "Cingulate cortex, area 29a",
        "A29b": "Cingulate cortex, area 29b",
        "A29c": "Cingulate cortex, area 29c",
        "A30": "Cingulate cortex, area 30",
        "A32": "Cingulate cortex, area 32",
        "APir": "Ammon's horn/piriform cortex",
        "Au1": "Primary auditory cortex",
        "AuD": "Secondary auditory cortex, dorsal area",
        "AuV": "Secondary auditory cortex, ventral area",
        "cc/ec": "Corpus callosum/external capsule",
        "CEnt": "Caudomedial entorhinal cortex",
        "cg": "Cingulate gyrus",
        "Cl": "Claustrum",
        "CxA": "Cortex-amygdala transition area",
        "DCl": "Dorsal claustrum",
        "DEn": "Dorsal endopiriform nucleus",
        "df": "Dorsal fornix",
        "dhc": "Dorsal hippocampal commissure",
        "DIEnt": "Dorsal intermediate entorhinal cortex",
        "DLEnt": "Dorsolateral entorhinal cortex",
        "DLO": "Dorsolateral orbital cortex",
        "DS": "Dorsal subiculum",
        "DTT": "Dorsal tenia tecta",
        "Ect": "Ectorhinal cortex",
        "fmi": "Forceps minor of the corpus callosum",
        "Fr3": "Frontal cortex, area 3",
        "FrA": "Frontal association cortex",
        "IEn": "Intermediate endopiriform nucleus",
        "Ins": "Insular region, not subdivided",
        "LO": "Lateral orbital cortex",
        "lo": "Lateral orbital cortex (orbital part)",
        "LPtA": "Lateral parietal association cortex",
        "M1": "Primary motor cortex",
        "M2": "Secondary motor cortex",
        "MEnt": "Medial entorhinal cortex",
        "MO": "Medial orbital cortex",
        "MPtA": "Medial parietal association cortex",
        "PaS": "Parasubiculum",
        "Pir": "Piriform cortex",
        "PLCo": "Posterior lateral cortical amygdala",
        "PMCo": "Posterior medial cortical amygdala",
        "Post": "Postsubiculum",
        "PRh": "Perirhinal cortex",
        "PrS": "Presubiculum",
        "PtPR": "Parietal cortex, posterior area, rostral part",
        "RAPir": "Retro-allocortical piriform cortex",
        "S1": "Primary somatosensory cortex",
        "S1BF": "Primary somatosensory cortex, barrel field",
        "S1DZ": "Primary somatosensory cortex, dysgranular zone",
        "S1FL": "Primary somatosensory cortex, forelimb region",
        "S1HL": "Primary somatosensory cortex, hindlimb region",
        "S1J": "Primary somatosensory cortex, jaw region",
        "S1Sh": "Primary somatosensory cortex, shoulder region",
        "S1Tr": "Primary somatosensory cortex, trunk region",
        "S1ULp": "Primary somatosensory cortex, upper lip region",
        "S2": "Secondary somatosensory cortex",
        "STr": "Subiculum transition area",
        "TeA": "Temporal association cortex",
        "V1": "Primary visual cortex",
        "V1B": "Primary visual cortex, binocular area",
        "V1M": "Primary visual cortex, monocular area",
        "V2L": "Secondary visual cortex, lateral area",
        "V2ML": "Secondary visual cortex, mediolateral area",
        "V2MM": "Secondary visual cortex, mediomedial area",
        "VCl": "Visual cortex, lateral area",
        "VEn": "Ventral entorhinal cortex",
        "VIEnt": "Ventral intermediate entorhinal cortex",
        "VO": "Ventrolateral orbital cortex",
        "VTT": "Ventral tenia tecta",
    },
}
TEMPLATE_STRING = "ambmc-c57bl6-label-{}_v0.8{}"


def download_resources():
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)
    ## Download atlas_file
    destination_path = DOWNLOAD_DIR_PATH / "template.nii.tar.gz"
    if not os.path.isfile(destination_path):
        response = requests.get(REFERENCE_URL, stream=True)
        with open(destination_path, "wb") as f:
            for chunk in response.iter_content(
                chunk_size=8192
            ):  # chunk size is 1 kibibyte
                if chunk:
                    f.write(chunk)
        with tarfile.open(destination_path, "r:gz") as tar:
            tar.extractall(path=DOWNLOAD_DIR_PATH)
    for region, url in ANNOTATION_URLS.items():
        destination_path = DOWNLOAD_DIR_PATH / f"{region}.nii.tar.gz"
        if not os.path.isfile(destination_path):
            response = requests.get(url, stream=True)
            with open(destination_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        # Untar the file
        with tarfile.open(destination_path, "r:gz") as tar:
            tar.extractall(path=DOWNLOAD_DIR_PATH)
    return None


def preprocess_annotations():
    """
    The annotations are split amongst multiple files,
    in different formats, and are required to correct values
    in the volumes themselves which are at times overlapping.
    So this preprocessing function must be run before retrieving
    the reference and annotation
    """

    label_list = []
    for region, url in ANNOTATION_URLS.items():
        label_path = (
            DOWNLOAD_DIR_PATH
            / TEMPLATE_STRING.format(region, "-nii")
            / TEMPLATE_STRING.format(region, ".idx")
        )
        if region == "hippocampus":
            total_label = pd.concat(label_list)
            label = pd.read_csv(label_path, sep="\t")
            # correct misformatting
            label["# Hippocampus labels"][
                label["# Hippocampus labels"] == "012 CA2 Py                  "
            ] = "012 CA2-Py                  "
            label = label["# Hippocampus labels"].str.split(" ", expand=True)
            label = label.iloc[3:, :2].reset_index(drop=True)
            label = label.rename(columns={"0": "id", "1": "acronym"})
            # Generate new RGB values
            new_df = pd.DataFrame(columns=["r", "g", "b"])
            while len(new_df) < len(label):
                new_rgb = np.random.randint(
                    0, 256, size=3
                )  # Generate a new RGB value
                if not (
                    (total_label[["r", "g", "b"]] == new_rgb).all(axis=1).any()
                ):  # If the RGB value doesn't exist in df
                    new_df = new_df._append(
                        pd.Series(new_rgb, index=["r", "g", "b"]),
                        ignore_index=True,
                    )  # Add it to new_df
            label[["r", "g", "b"]] = new_df[["r", "g", "b"]]
            label = label.rename(columns={0: "id", 1: "acronym"})
            label["id"] = np.arange(1, 16)
        else:
            label = pd.read_csv(label_path, sep="\t", header=None)
            label = label.rename(
                columns={0: "id", 1: "r", 2: "g", 3: "b", 4: "acronym"}
            )
        label = label[label["acronym"] != "Black (Background)"]
        label = label[label["acronym"] != "White (White)"]
        label["structure_id_path"] = label.apply(
            lambda x: [ROOT_ID, REGION_IDS[region], x["id"]], axis=1
        )
        label["acronym"] = label["acronym"].str.strip()
        # correct typo
        label.loc[label["acronym"] == "Zl", "acronym"] = "ZI"
        label["name"] = label["acronym"].map(acronym_dict[region])
        label.loc[label["name"].isna(), "name"] = label.loc[
            label["name"].isna(), "acronym"
        ]
        label_path = (
            DOWNLOAD_DIR_PATH
            / TEMPLATE_STRING.format(region, "-nii")
            / TEMPLATE_STRING.format(region, "_new.idx")
        )

        label.to_csv(label_path)
        label_list.append(label)


def retrieve_reference_and_annotation():
    """
    Retrieve the desired reference and annotation as two numpy arrays.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array is the
        reference volume, and the second array is the annotation volume.
    """
    filename = (
        DOWNLOAD_DIR_PATH
        / "ambmc-c57bl6-model-symmet_v0.8-nii"
        / "ambmc-c57bl6-model-symmet_v0.8.nii"
    )
    reference = sitk.GetArrayFromImage(sitk.ReadImage(str(filename)))
    ### This part is complex as the atlas segmentations are
    ### Distributed through multiple files which we combine.
    original_origin = np.array([5.07600021, 9.81449986, -3.72600007])
    annotation = np.zeros((499, 1311, 679))
    new_vals = 1
    for region in REGION_IDS.keys():

        filename = (
            DOWNLOAD_DIR_PATH
            / TEMPLATE_STRING.format(region, "-nii")
            / TEMPLATE_STRING.format(region, ".nii")
        )

        label_path = (
            DOWNLOAD_DIR_PATH
            / TEMPLATE_STRING.format(region, "-nii")
            / TEMPLATE_STRING.format(region, "_new.idx")
        )

        label_data = pd.read_csv(label_path)
        img = sitk.ReadImage(filename)
        arr = sitk.GetArrayFromImage(img)
        id_mapping = {}
        # This has to be done because the ids in the labelfile are in
        # correct order, but are not aligned with the volume
        for i, idval in enumerate(label_data["id"]):
            arr[arr == (i + 1)] = new_vals
            id_mapping[idval] = new_vals
            new_vals += 1
            # Assuming annotated_volume is a numpy array
        new_label_data = label_data.copy()
        new_label_data["id"] = new_label_data["id"].map(id_mapping)
        output_path = (
            DOWNLOAD_DIR_PATH
            / TEMPLATE_STRING.format(region, "-nii")
            / TEMPLATE_STRING.format(region, "_renumbered.idx")
        )
        new_label_data.to_csv(output_path)
        origin = np.array(img.GetOrigin())
        spacing = np.array(img.GetSpacing())
        origin_px = np.round((original_origin - origin) / spacing).astype(int)
        segment = annotation[
            origin_px[0] : origin_px[0] + arr.shape[0],
            origin_px[1] : origin_px[1] + arr.shape[1],
            origin_px[2] : origin_px[2] + arr.shape[2],
        ]
        mask = arr != 0
        segment[mask] = arr[mask]
    reference = reference - reference.min()
    reference = reference / reference.max()
    reference = reference * 65535
    reference = reference.astype(np.uint16)
    return annotation, reference


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    If your atlas is asymmetrical, you may want to use a hemisphere map.
    This is an array in the same shape as your template,
    with 0's marking the left hemisphere, and 1's marking the right.

    If your atlas is symmetrical, ignore this function.

    Returns:
        numpy.array or None: A numpy array representing the hemisphere map,
        or None if the atlas is symmetrical.
    """
    return None  # Symmetrical atlas


def retrieve_structure_information():
    """
    This function should return a pandas DataFrame with information about your
    atlas.

    The DataFrame should be in the following format:

    ╭────┬───────────────────┬─────────┬───────────────────┬─────────────────╮
    | id | name              | acronym | structure_id_path | rgb_triplet     |
    |    |                   |         |                   |                 |
    ├────┼───────────────────┼─────────┼───────────────────┼─────────────────┤
    | 997| root              | root    | [997]             | [255, 255, 255] |
    ├────┼───────────────────┼─────────┼───────────────────┼─────────────────┤
    | 8  | Basic cell groups | grey    | [997, 8]          | [191, 218, 227] |
    ├────┼───────────────────┼─────────┼───────────────────┼─────────────────┤
    | 567| Cerebrum          | CH      | [997, 8, 567]     | [176, 240, 255] |
    ╰────┴───────────────────┴─────────┴───────────────────┴─────────────────╯

    Returns:
        pandas.DataFrame: A DataFrame containing the atlas information.
    """

    hierarchical_labels = pd.DataFrame(
        {
            "id": [ROOT_ID, 1001, 1002, 1003, 1004, 1005],
            "acronym": [
                "root",
                "basalganglia",
                "cerebellum",
                "cortex",
                "hippocampus",
                "diencephalon",
            ],
            "name": [
                "root",
                "basalganglia",
                "cerebellum",
                "cortex",
                "hippocampus",
                "diencephalon",
            ],
            "structure_id_path": [
                [ROOT_ID],
                [ROOT_ID, 1001],
                [ROOT_ID, 1002],
                [ROOT_ID, 1003],
                [ROOT_ID, 1004],
                [ROOT_ID, 1005],
            ],
            "rgb_triplet": [
                [255, 255, 255],
                [255, 100, 30],
                [100, 255, 30],
                [30, 100, 255],
                [100, 30, 255],
                [0, 255, 0],
            ],
        }
    )
    label_list = []
    for region in REGION_IDS.keys():
        file_path = (
            DOWNLOAD_DIR_PATH
            / TEMPLATE_STRING.format(region, "-nii")
            / TEMPLATE_STRING.format(region, "_renumbered.idx")
        )
        label_list.append(pd.read_csv(file_path))
    df = pd.concat(label_list).reset_index(drop=True)
    new_df = pd.DataFrame(columns=["r", "g", "b"])
    while len(new_df) < len(
        hierarchical_labels
    ):  # Change 1 to the number of new RGB values you want to generate
        new_rgb = np.random.randint(0, 256, size=3)  # Generate a new RGB value
        if not (
            (df[["r", "g", "b"]] == new_rgb).all(axis=1).any()
        ):  # If the RGB value doesn't exist in df
            new_df = new_df._append(
                pd.Series(new_rgb, index=["r", "g", "b"]),
                ignore_index=True,
            )  # Add it to new_df
    hierarchical_labels[["r", "g", "b"]] = new_df[["r", "g", "b"]]
    hierarchical_labels.loc[
        hierarchical_labels["id"] == 0, ["r", "g", "b"]
    ] = [0, 0, 0]

    rgb = []
    for index, row in df.iterrows():
        temp_rgb = [row["r"], row["g"], row["b"]]
        rgb.append(temp_rgb)

    df = df.drop(columns=["r", "g", "b"])
    df = df.assign(rgb_triplet=rgb)
    total_df = pd.concat([hierarchical_labels, df])
    total_df = total_df[
        ["acronym", "id", "name", "structure_id_path", "rgb_triplet"]
    ]
    # unwanted_structure_id_paths =
    structures = total_df.to_dict("records")

    for s in structures:
        if isinstance(s["structure_id_path"], str):
            s["structure_id_path"] = eval(s["structure_id_path"])
    filtered_structures = []
    ids_to_filter = [
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        62,
        64,
        256,
        257,
        258,
        259,
        260,
        261,
        262,
    ]
    for s in structures:
        if s["id"] not in ids_to_filter:
            filtered_structures.append(s)
    return filtered_structures


def retrieve_or_construct_meshes(annotated_volume, structures):
    """
    This function should return a dictionary of ids and corresponding paths to
    mesh files. Some atlases are packaged with mesh files, in these cases we
    should use these files. Then this function should download those meshes.
    In other cases we need to construct the meshes ourselves. For this we have
    helper functions to achieve this.
    """
    meshes_dir_path = DOWNLOAD_DIR_PATH / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    tree = get_structures_tree(structures)

    labels = np.unique(annotated_volume).astype(np.int32)
    for key, node in tree.nodes.items():
        if key in labels:
            is_label = True
        else:
            is_label = False

        node.data = Region(is_label)

    # Mesh creation
    closing_n_iters = 2  # not used for this atlas
    decimate_fraction = 0  # not used for this atlas

    smooth = False
    start = time.time()

    for node in track(
        tree.nodes.values(),
        total=tree.size(),
        description="Creating meshes",
    ):
        create_region_mesh(
            (
                meshes_dir_path,
                node,
                tree,
                labels,
                annotated_volume,
                ROOT_ID,
                closing_n_iters,
                decimate_fraction,
                smooth,
            )
        )

    print(
        "Finished mesh extraction in: ",
        round((time.time() - start) / 60, 2),
        " minutes",
    )

    # Create meshes dict
    meshes_dict = dict()
    structures_with_mesh = []
    for s in structures:
        # Check if a mesh was created
        mesh_path = meshes_dir_path / f'{s["id"]}.obj'
        if not mesh_path.exists():
            print(f"No mesh file exists for: {s}, ignoring it")
            continue
        else:
            # Check that the mesh actually exists (i.e. not empty)
            if mesh_path.stat().st_size < 512:
                print(f"obj file for {s} is too small, ignoring it.")
                continue

        structures_with_mesh.append(s)
        meshes_dict[s["id"]] = mesh_path

    print(
        f"In the end, {len(structures_with_mesh)} "
        "structures with mesh are kept"
    )
    return meshes_dict


### If the code above this line has been filled correctly, nothing needs to be
### edited below (unless variables need to be passed between the functions).
if __name__ == "__main__":
    BG_ROOT_DIR.mkdir(exist_ok=True)
    download_resources()
    preprocess_annotations()
    annotated_volume, template_volume = retrieve_reference_and_annotation()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict = retrieve_or_construct_meshes(annotated_volume, structures)
    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=(RESOLUTION,) * 3,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=template_volume,
        annotation_stack=annotated_volume,
        structures_list=structures,
        meshes_dict=meshes_dict,
        working_dir=BG_ROOT_DIR,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
    )
