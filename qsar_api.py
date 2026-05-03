import os
import json
import traceback
import requests
import joblib
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Dict, Any

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw
from rdkit.ML.Descriptors import MoleculeDescriptors

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="QSAR Viva-Ready API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup constants — use relative paths for cloud compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR
DATA_DIR = os.path.join(BASE_DIR, "data")
TARGETS = ["EGFR", "BRAF", "CDK2", "HER2", "VEGFR2"]
TARGET_MAP = {"EGFR":0, "BRAF":1, "CDK2":2, "HER2":3, "VEGFR2":4}

# Load globals
scaler = None
selector = None
model = None

descriptor_names = [desc[0] for desc in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

@app.on_event("startup")
def load_artifacts():
    global scaler, selector, model
    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        selector = joblib.load(os.path.join(MODEL_DIR, "selector.pkl"))
        
        # Original input size depends on the data. Let's infer from selector
        input_size = selector.get_support().sum() 
        
        # Must exactly match qsar_file.py architecture
        model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pth"), map_location=torch.device('cpu')))
        model.eval()
        
        print("Model, scaler, selector, and descriptors loaded successfully.")
    except Exception as e:
        print(f"FAILED TO LOAD MODEL ARTIFACTS: {e}")
        import traceback
        traceback.print_exc()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "selector_loaded": selector is not None,
    }

class PredictRequest(BaseModel):
    smiles: str

class BatchRequest(BaseModel):
    smiles_list: List[str]
    target_filter: str

@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    target = req.target_filter.upper()
    if target not in TARGETS:
        raise HTTPException(status_code=400, detail="Invalid target filter.")
    
    t_val = TARGET_MAP[target]
    successful_predictions = []
    
    # Process up to 1000 items to prevent server locking purely on string length
    s_list = req.smiles_list[:1000]
    
    for smi in s_list:
        smi = smi.strip()
        if not smi: continue
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        
        try:
            raw_desc = calc.CalcDescriptors(mol)
            desc_base = np.array(raw_desc).reshape(1, -1)
            desc_base[np.isinf(desc_base)] = np.nan
            if np.isnan(desc_base).any(): continue

            desc_base = np.clip(desc_base, -1e6, 1e6)
            desc_base = np.sign(desc_base) * np.log1p(np.abs(desc_base))

            x_in = np.hstack((desc_base, [[t_val]]))
            x_sel = selector.transform(x_in)
            x_scaled = scaler.transform(x_sel)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

            with torch.no_grad():
                prob = torch.sigmoid(model(x_tensor)).item()

            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rings = Descriptors.RingCount(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)

            successful_predictions.append({
                "smiles": smi,
                "confidence": prob * 100.0,
                "active": bool(prob > 0.5),
                "pic50_sim": round(4.0 + (prob * 5.0), 2),
                "mw": round(mw, 2),
                "logp": round(logp, 2),
                "tpsa": round(tpsa, 2),
                "hbd": hbd,
                "hba": hba,
                "rings": rings,
                "rotatable_bonds": rot_bonds
            })
        except:
            pass
            
    sorted_res = sorted(successful_predictions, key=lambda x: x["confidence"], reverse=True)
    return {
        "total_processed": len(successful_predictions),
        "top_50": sorted_res[:50]
    }

@app.post("/predict_all")
def predict_all(req: PredictRequest):
    smiles = req.smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES structure.")

    # 1. Feature Extraction
    raw_desc = calc.CalcDescriptors(mol)
    desc_base = np.array(raw_desc).reshape(1, -1)
    
    desc_base[np.isinf(desc_base)] = np.nan
    if np.isnan(desc_base).any():
        raise HTTPException(status_code=400, detail="Molecule produced NaN descriptors.")

    desc_base = np.clip(desc_base, -1e6, 1e6)
    desc_base = np.sign(desc_base) * np.log1p(np.abs(desc_base))

    # 2. Multi-Target Predictions
    results = []
    for target in TARGETS:
        t_val = TARGET_MAP[target]
        x_in = np.hstack((desc_base, [[t_val]]))
        x_sel = selector.transform(x_in)
        x_scaled = scaler.transform(x_sel)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            prob = torch.sigmoid(model(x_tensor)).item()
        
        results.append({
            "target": target,
            "confidence": prob * 100.0,
            "active": bool(prob > 0.5),
            "pic50_sim": round(4.0 + (prob * 5.0), 2)  # Pseudo-pIC50 between 4 and 9
        })
    
    # Sort for ranking
    results = sorted(results, key=lambda x: x["confidence"], reverse=True)
    best_target = results[0]["target"] if results[0]["active"] else "None (Inactive globally)"
    
    # 3. Drug-Likeness (Lipinski) & ADMET
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rings = Descriptors.RingCount(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)

    lipinski_pass = (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)
    
    # Basic ADMET heuristics
    absorption = "High" if tpsa <= 140 else "Low"
    toxicity = "Toxic Flag (High LogP)" if logp > 5 else "Safe"
    
    # 4. 2D SVG
    try:
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(300, 300)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg_text = drawer.GetDrawingText()
    except:
        svg_text = ""

    return {
        "predictions": results,
        "best_target": best_target,
        "molecular_properties": {
            "mw": round(mw, 2),
            "logp": round(logp, 2),
            "hbd": hbd,
            "hba": hba,
            "tpsa": round(tpsa, 2),
            "rings": rings,
            "rotatable_bonds": rot_bonds
        },
        "lipinski_pass": lipinski_pass,
        "admet": {
            "absorption": absorption,
            "toxicity": toxicity
        },
        "svg": svg_text
    }

@app.get("/top10/{target}")
def get_top10(target: str):
    target = target.upper()
    if target not in TARGETS:
        raise HTTPException(status_code=400, detail="Invalid Target")
    
    csv_path = os.path.join(DATA_DIR, f"clean_{target}.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"Data for {target} not found")
        
    df = pd.read_csv(csv_path)
    
    # Filter active and sort by IC50
    active_df = df[df["Activity"] == 1]
    if "IC50" in active_df.columns:
        top_10 = active_df.sort_values(by="IC50").head(10)
    else:
        top_10 = active_df.head(10)
        
    compounds = []
    for _, row in top_10.iterrows():
        compounds.append({
            "smiles": row["SMILES"],
            "ic50": row.get("IC50", "N/A"),
            "target": row.get("Target", target)
        })
    return {"top10": compounds}

@app.get("/chembl")
def get_chembl(smiles: str):
    # Query EBI ChEMBL API for exact SMILES match
    encoded_smiles = requests.utils.quote(smiles)
    ebm_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_structures__canonical_smiles__exact={encoded_smiles}"
    try:
        r = requests.get(ebm_url)
        data = r.json()
        if data.get("molecules") and len(data["molecules"]) > 0:
            chembl_id = data["molecules"][0]["molecule_chembl_id"]
            pref_name = data["molecules"][0].get("pref_name", "Unknown")
            link = f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/"
            return {"chembl_id": chembl_id, "pref_name": pref_name, "link": link}
        else:
            return {"error": "Compound not found in ChEMBL"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/pdb", response_class=PlainTextResponse)
def get_pdb(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES structure.")
    
    try:
        m2 = Chem.AddHs(mol)
        AllChem.EmbedMolecule(m2, AllChem.ETKDG())
        pdb_block = Chem.MolToPDBBlock(m2)
        return pdb_block
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate 3D structure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
