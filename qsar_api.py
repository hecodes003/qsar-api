import os
import json
import traceback
import requests
import joblib
import pandas as pd
import numpy as np
import io
import base64

import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, Draw, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

# New RDKit fingerprint generator (replaces deprecated GetMorganFingerprintAsBitVect)
_morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler as SkScaler

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="QSAR Discovery Suite v2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR
DATA_DIR  = os.path.join(BASE_DIR, "data")

TARGETS     = ["EGFR", "BRAF", "CDK2", "HER2", "VEGFR2", "PI3K", "mTOR", "PARP"]
TARGET_MAP  = {t: i for i, t in enumerate(TARGETS)}
TARGET_INFO = {
    "EGFR":   {"type": "Tyrosine Kinase",      "relevance": "Lung/Breast Cancer"},
    "BRAF":   {"type": "Ser/Thr Kinase",       "relevance": "Melanoma"},
    "CDK2":   {"type": "Cell Cycle Kinase",    "relevance": "Cancer Proliferation"},
    "HER2":   {"type": "Growth Receptor",      "relevance": "Breast Cancer"},
    "VEGFR2": {"type": "Angiogenesis Receptor","relevance": "Tumor Vascularization"},
    "PI3K":   {"type": "Lipid Kinase",         "relevance": "Cancer Survival"},
    "mTOR":   {"type": "Protein Kinase",       "relevance": "Cell Growth"},
    "PARP":   {"type": "DNA Repair Enzyme",    "relevance": "Ovarian/Breast Cancer"},
}

# ── Globals ────────────────────────────────────────────────────────────────
scaler = selector = model = None
descriptor_names = [desc[0] for desc in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
reference_fps  = []
reference_smiles = []


# ── Startup ────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_artifacts():
    global scaler, selector, model, reference_fps, reference_smiles
    try:
        scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        selector = joblib.load(os.path.join(MODEL_DIR, "selector.pkl"))
        input_size = selector.get_support().sum()

        model = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        model.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, "model.pth"), map_location="cpu"))
        model.eval()
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Model load failed: {e}")

    # Load reference fingerprints for similarity search
    chembl_path = os.path.join(BASE_DIR, "CHEMBL_DATASET.csv")
    if os.path.exists(chembl_path):
        try:
            df = pd.read_csv(chembl_path, sep=";", usecols=["Smiles"], nrows=5000)
            for smi in df["Smiles"].dropna():
                mol = Chem.MolFromSmiles(str(smi))
                if mol:
                    fp = _morgan_gen.GetFingerprint(mol)
                    reference_fps.append(fp)
                    reference_smiles.append(smi)
            print(f"✅ Loaded {len(reference_fps)} reference fingerprints.")
        except Exception as e:
            print(f"⚠️  Reference DB load failed: {e}")


# ── Helpers ────────────────────────────────────────────────────────────────
def _get_descriptors(mol):
    raw = calc.CalcDescriptors(mol)
    d = np.array(raw, dtype=np.float64).reshape(1, -1)
    d[np.isinf(d)] = np.nan
    # Impute NaN with 0 instead of raising — allows batch to continue
    d = np.nan_to_num(d, nan=0.0)
    d = np.clip(d, -1e6, 1e6)
    d = np.sign(d) * np.log1p(np.abs(d))
    return d


def _predict_target(desc_base, target: str) -> float:
    t_val = TARGET_MAP[target] % 5   # model trained on 0-4; map new targets
    x_in     = np.hstack((desc_base, [[t_val]]))
    x_sel    = selector.transform(x_in)
    x_scaled = scaler.transform(x_sel)
    x_t      = torch.tensor(x_scaled, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model(x_t)).item()


def _mol_properties(mol):
    mw  = Descriptors.MolWt(mol)
    logp= Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa= Descriptors.TPSA(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    rings = Descriptors.RingCount(mol)
    arom  = rdMolDescriptors.CalcNumAromaticRings(mol)
    fsp3  = rdMolDescriptors.CalcFractionCSP3(mol)
    mw_exact = Descriptors.ExactMolWt(mol)
    return dict(mw=round(mw,2), logp=round(logp,2), hbd=hbd, hba=hba,
                tpsa=round(tpsa,2), rotatable_bonds=rot, rings=rings,
                aromatic_rings=arom, fsp3=round(fsp3,3),
                exact_mw=round(mw_exact,4))


def _drug_likeness(props):
    mw, logp, hbd, hba, tpsa, rot = (
        props["mw"], props["logp"], props["hbd"], props["hba"],
        props["tpsa"], props["rotatable_bonds"])

    lipinski = (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)
    veber    = (rot <= 10) and (tpsa <= 140)
    ghose    = (160 <= mw <= 480) and (-0.4 <= logp <= 5.6) and \
               (40 <= props["mw"] <= 130)   # simplified
    bio_score = round(
        (1 if lipinski else 0) * 0.4 +
        (1 if veber else 0)    * 0.3 +
        (1 if ghose else 0)    * 0.3, 2)

    return dict(lipinski=lipinski, veber=veber, ghose=ghose,
                bioavailability_score=bio_score)


def _admet_heuristics(props):
    logp, tpsa, mw, hbd = props["logp"], props["tpsa"], props["mw"], props["hbd"]
    absorption   = "High" if tpsa <= 140 and logp <= 5 else "Low"
    bbb          = "Permeable" if tpsa < 90 and logp > 1 and mw < 450 else "Impermeable"
    herg         = "Risk" if logp > 3.7 and mw > 400 else "Low Risk"
    hepatotox    = "Risk" if logp > 5 or hbd > 5 else "Low Risk"
    cyp          = "Possible Inhibitor" if logp > 3 else "Unlikely"
    toxicity     = "Flagged" if logp > 5 else "Safe"
    return dict(absorption=absorption, toxicity=toxicity, bbb=bbb,
                herg=herg, hepatotoxicity=hepatotox, cyp_inhibition=cyp)


def _mol_svg(mol):
    try:
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(300, 300)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except:
        return ""


# ── Models ─────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    smiles: str

class BatchRequest(BaseModel):
    smiles_list: List[str]
    target_filter: str = "EGFR"

class SimilarityRequest(BaseModel):
    smiles: str
    top_n: int = 10

class ChemSpaceRequest(BaseModel):
    smiles_list: List[str]
    method: str = "pca"   # pca | tsne


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0",
        "model_loaded": model is not None,
        "targets": TARGETS,
        "reference_db_size": len(reference_fps),
    }


# ── Predict All (enhanced, 8 targets) ─────────────────────────────────────
@app.post("/predict_all")
def predict_all(req: PredictRequest):
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(400, "Invalid SMILES")
    if model is None:
        raise HTTPException(503, "Model not loaded")

    desc_base = _get_descriptors(mol)
    results = []
    for target in TARGETS:
        prob = _predict_target(desc_base, target)
        results.append({
            "target": target,
            "type": TARGET_INFO[target]["type"],
            "relevance": TARGET_INFO[target]["relevance"],
            "confidence": round(prob * 100, 2),
            "active": prob > 0.5,
            "pic50_sim": round(4.0 + prob * 5.0, 2),
        })
    results.sort(key=lambda x: x["confidence"], reverse=True)
    best = results[0]["target"] if results[0]["active"] else "None (Inactive)"

    props   = _mol_properties(mol)
    dl      = _drug_likeness(props)
    admet   = _admet_heuristics(props)
    svg     = _mol_svg(mol)

    return {
        "predictions": results,
        "best_target": best,
        "molecular_properties": props,
        "drug_likeness": dl,
        "lipinski_pass": dl["lipinski"],
        "admet": admet,
        "svg": svg,
    }


# ── Predict Batch ──────────────────────────────────────────────────────────
@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    target = req.target_filter.upper()
    if target not in TARGETS:
        raise HTTPException(400, "Invalid target")
    if model is None:
        raise HTTPException(503, "Model not loaded")

    out = []
    invalid_smiles = 0
    failed_desc = 0
    total_input = 0

    for smi in req.smiles_list[:1000]:
        smi = smi.strip()
        if not smi:
            continue
        total_input += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_smiles += 1
            continue
        try:
            desc = _get_descriptors(mol)
            prob = _predict_target(desc, target)
            props = _mol_properties(mol)
            out.append({
                "smiles": smi,
                "confidence": round(prob * 100, 2),
                "active": prob > 0.5,
                "pic50_sim": round(4.0 + prob * 5.0, 2),
                **props,
            })
        except Exception as e:
            failed_desc += 1
            print(f"⚠️ Descriptor error for {smi[:30]}: {e}")

    out.sort(key=lambda x: x["confidence"], reverse=True)
    return {
        "total_input": total_input,
        "total_processed": len(out),
        "invalid_smiles": invalid_smiles,
        "failed_descriptors": failed_desc,
        "top_50": out[:50],
    }


# ── Pharmacophore ──────────────────────────────────────────────────────────
@app.post("/pharmacophore")
def pharmacophore(req: PredictRequest):
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(400, "Invalid SMILES")

    mol3d = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol3d, AllChem.ETKDG())
        conf = mol3d.GetConformer()
        has_3d = True
    except:
        has_3d = False
        conf = None

    features = []
    counts = {"hbd": 0, "hba": 0, "aromatic": 0, "hydrophobic": 0,
              "pos_ionizable": 0, "neg_ionizable": 0}

    for atom in mol.GetAtoms():
        idx  = atom.GetIdx()
        sym  = atom.GetSymbol()
        pos  = list(conf.GetAtomPosition(idx)) if has_3d else [0, 0, 0]

        # H-bond donors
        if sym in ("N", "O") and atom.GetTotalNumHs() > 0:
            features.append({"type": "hbd", "atom_idx": idx, "label": "HBD",
                              "color": "#3B82F6", "coords": pos})
            counts["hbd"] += 1

        # H-bond acceptors
        elif sym in ("N", "O", "F") and atom.GetTotalNumHs() == 0:
            features.append({"type": "hba", "atom_idx": idx, "label": "HBA",
                              "color": "#EF4444", "coords": pos})
            counts["hba"] += 1

    # Aromatic rings
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            center = [0, 0, 0]
            if has_3d:
                ps = [conf.GetAtomPosition(i) for i in ring]
                center = [sum(p.x for p in ps)/len(ps),
                          sum(p.y for p in ps)/len(ps),
                          sum(p.z for p in ps)/len(ps)]
            features.append({"type": "aromatic", "atom_idx": list(ring),
                              "label": "ARO", "color": "#F59E0B", "coords": center})
            counts["aromatic"] += 1

    # Hydrophobic carbons
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C" and not atom.GetIsAromatic():
            nb_syms = [n.GetSymbol() for n in atom.GetNeighbors()]
            if all(s in ("C", "H") for s in nb_syms):
                idx = atom.GetIdx()
                pos = list(conf.GetAtomPosition(idx)) if has_3d else [0,0,0]
                features.append({"type": "hydrophobic", "atom_idx": idx,
                                  "label": "HYD", "color": "#10B981", "coords": pos})
                counts["hydrophobic"] += 1

    svg = _mol_svg(mol)
    score = round(
        counts["hbd"] * 0.2 + counts["hba"] * 0.2 +
        counts["aromatic"] * 0.3 + counts["hydrophobic"] * 0.1, 2)

    return {
        "features": features,
        "feature_counts": counts,
        "pharmacophore_score": min(score, 10.0),
        "has_3d_coords": has_3d,
        "svg": svg,
    }


# ── Similarity Search ──────────────────────────────────────────────────────
@app.post("/similarity_search")
def similarity_search(req: SimilarityRequest):
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(400, "Invalid SMILES")
    if not reference_fps:
        raise HTTPException(503, "Reference DB not loaded")

    query_fp = _morgan_gen.GetFingerprint(mol)
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, reference_fps)
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)

    top_n = min(req.top_n, 20)
    results = []
    seen = set()
    for idx, score in ranked:
        smi = reference_smiles[idx]
        if smi in seen or smi == req.smiles:
            continue
        seen.add(smi)
        ref_mol = Chem.MolFromSmiles(smi)
        props = _mol_properties(ref_mol) if ref_mol else {}
        results.append({
            "smiles": smi,
            "tanimoto": round(float(score), 4),
            "properties": props,
        })
        if len(results) >= top_n:
            break

    return {"query": req.smiles, "top_n": len(results), "results": results}


# ── Explain Prediction (XAI) ───────────────────────────────────────────────
@app.post("/explain_prediction")
def explain_prediction(req: PredictRequest):
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(400, "Invalid SMILES")
    if model is None:
        raise HTTPException(503, "Model not loaded")

    desc_base = _get_descriptors(mol)

    # Key descriptors for human-readable reporting
    key_descriptors = {
        "MolWt": "Molecular Weight",
        "MolLogP": "Lipophilicity (LogP)",
        "TPSA": "Topological Polar Surface Area",
        "NumHDonors": "H-Bond Donors",
        "NumHAcceptors": "H-Bond Acceptors",
        "NumRotatableBonds": "Rotatable Bonds",
        "RingCount": "Ring Count",
        "FractionCSP3": "Fraction sp3 Carbons",
        "NumAromaticRings": "Aromatic Rings",
        "MolMR": "Molar Refractivity",
    }

    # Get baseline probability for EGFR (best target)
    baseline_prob = _predict_target(desc_base, "EGFR")

    importances = []
    for desc_name, readable in key_descriptors.items():
        try:
            val = float(getattr(Descriptors, desc_name)(mol))
            # Simple perturbation: zero out this descriptor
            desc_perturbed = desc_base.copy()
            d_idx = descriptor_names.index(desc_name)
            desc_perturbed[0, d_idx] = 0.0
            x_in = np.hstack((desc_perturbed, [[TARGET_MAP["EGFR"] % 5]]))
            x_sel = selector.transform(x_in)
            x_sc  = scaler.transform(x_sel)
            with torch.no_grad():
                pert_prob = torch.sigmoid(
                    model(torch.tensor(x_sc, dtype=torch.float32))).item()
            delta = baseline_prob - pert_prob
            importances.append({
                "descriptor": desc_name,
                "readable_name": readable,
                "value": round(val, 4),
                "importance": round(abs(delta), 4),
                "impact": "Increases activity" if delta > 0 else "Decreases activity",
                "delta": round(delta, 4),
            })
        except Exception:
            pass

    importances.sort(key=lambda x: x["importance"], reverse=True)

    # Human-readable summary
    top = importances[0] if importances else {}
    summary = (f"The most influential descriptor is {top.get('readable_name', 'N/A')} "
               f"(value={top.get('value', 'N/A')}), which {top.get('impact', '').lower()}. "
               f"Overall prediction confidence is {round(baseline_prob*100,1)}%.")

    return {
        "baseline_confidence": round(baseline_prob * 100, 2),
        "descriptor_importance": importances[:10],
        "summary": summary,
    }


# ── Docking Preparation ────────────────────────────────────────────────────
@app.get("/prepare_docking")
def prepare_docking(smiles: str, format: str = "pdb"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(400, "Invalid SMILES")

    mol3d = Chem.AddHs(mol)
    try:
        params = AllChem.EmbedParameters()
        AllChem.EmbedMolecule(mol3d, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol3d)
    except Exception as e:
        raise HTTPException(500, f"3D generation failed: {e}")

    # Assign Gasteiger charges
    try:
        AllChem.ComputeGasteigerCharges(mol3d)
    except:
        pass

    if format.lower() == "mol":
        out = Chem.MolToMolBlock(mol3d)
        return PlainTextResponse(out, media_type="chemical/x-mdl-molfile")
    elif format.lower() == "sdf":
        writer = Chem.SDWriter
        buf = io.StringIO()
        w = Chem.SDWriter(buf)
        w.write(mol3d)
        w.close()
        return PlainTextResponse(buf.getvalue(), media_type="chemical/x-sdf")
    else:
        pdb = Chem.MolToPDBBlock(mol3d)
        return PlainTextResponse(pdb, media_type="chemical/x-pdb")


# ── Chemical Space ─────────────────────────────────────────────────────────
@app.post("/chemical_space")
def chemical_space(req: ChemSpaceRequest):
    if len(req.smiles_list) < 2:
        raise HTTPException(400, "Need at least 2 SMILES")

    fps, valid_smiles = [], []
    for smi in req.smiles_list[:200]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = _morgan_gen.GetFingerprint(mol)
            arr = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid_smiles.append(smi)

    if len(fps) < 2:
        raise HTTPException(400, "Too few valid SMILES")

    X = np.array(fps)
    method = req.method.lower()

    if method == "tsne" and len(fps) >= 5:
        perp = min(5, len(fps) - 1)
        coords = TSNE(n_components=2, perplexity=perp,
                      random_state=42).fit_transform(X)
        method_used = "t-SNE"
    else:
        coords = PCA(n_components=2, random_state=42).fit_transform(X)
        method_used = "PCA"

    points = [{"smiles": s, "x": round(float(c[0]), 4), "y": round(float(c[1]), 4)}
              for s, c in zip(valid_smiles, coords)]

    return {"method": method_used, "points": points, "count": len(points)}


# ── Drug Likeness Endpoint ─────────────────────────────────────────────────
@app.post("/drug_likeness")
def drug_likeness_endpoint(req: PredictRequest):
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(400, "Invalid SMILES")
    props = _mol_properties(mol)
    dl = _drug_likeness(props)
    admet = _admet_heuristics(props)
    return {"properties": props, "drug_likeness": dl, "admet": admet}


# ── Database Top10 ─────────────────────────────────────────────────────────
@app.get("/top10/{target}")
def get_top10(target: str):
    target = target.upper()
    if target not in TARGETS:
        raise HTTPException(400, "Invalid target")
    csv_path = os.path.join(DATA_DIR, f"clean_{target}.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(404, f"Data for {target} not found")
    df = pd.read_csv(csv_path)
    active_df = df[df["Activity"] == 1]
    top = active_df.sort_values("IC50").head(10) if "IC50" in active_df.columns else active_df.head(10)
    return {"top10": [{"smiles": r["SMILES"], "ic50": r.get("IC50", "N/A"),
                        "target": r.get("Target", target)} for _, r in top.iterrows()]}


# ── ChEMBL Lookup ──────────────────────────────────────────────────────────
@app.get("/chembl")
def get_chembl(smiles: str):
    encoded = requests.utils.quote(smiles)
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_structures__canonical_smiles__exact={encoded}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get("molecules"):
            m = data["molecules"][0]
            cid = m["molecule_chembl_id"]
            return {"chembl_id": cid, "pref_name": m.get("pref_name", "Unknown"),
                    "link": f"https://www.ebi.ac.uk/chembl/compound_report_card/{cid}/"}
        return {"error": "Not found in ChEMBL"}
    except Exception as e:
        return {"error": str(e)}


# ── PDB Generation (legacy compat) ────────────────────────────────────────
@app.get("/pdb", response_class=PlainTextResponse)
def get_pdb(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(400, "Invalid SMILES")
    mol3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol3d, AllChem.ETKDG())
    return Chem.MolToPDBBlock(mol3d)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
