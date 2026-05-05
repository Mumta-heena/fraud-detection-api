
import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import time

app = FastAPI(
    title="Cloud-Based Fraud Detection API",
    description="Real-time fraud detection using Transformer + XGBoost ensemble",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class FraudTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8,
                 num_layers=3, dropout=0.1):
        super(FraudTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),      nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x).squeeze(-1)

DEVICE = torch.device("cpu")

transformer_ieee = FraudTransformer(input_dim=358)
transformer_ieee.load_state_dict(
    torch.load("transformer_ieee.pt", map_location=DEVICE)
)
transformer_ieee.eval()

xgb_ieee = xgb.XGBClassifier()
xgb_ieee.load_model("xgb_ieee.json")

print("Models loaded successfully")

class TransactionRequest(BaseModel):
    features: List[float]
    dataset: str = "ieee"
    threshold: float = 0.5

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    inference_time_ms: float
    model_used: str

def get_risk_level(prob: float) -> str:
    if prob >= 0.8:   return "CRITICAL"
    elif prob >= 0.6: return "HIGH"
    elif prob >= 0.4: return "MEDIUM"
    elif prob >= 0.2: return "LOW"
    else:             return "SAFE"

@app.get("/")
def root():
    return {
        "service": "Cloud-Based Fraud Detection API",
        "status" : "running",
        "models" : ["FraudTransformer", "XGBoost"],
        "datasets": ["IEEE-CIS", "ULB Credit Card"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    start = time.time()
    features = np.array(request.features, dtype=np.float32)
    with torch.no_grad():
        x = torch.FloatTensor(features).unsqueeze(0)
        tf_prob = transformer_ieee(x).item()
    xgb_prob = xgb_ieee.predict_proba(
        features.reshape(1, -1)
    )[0][1]
    final_prob = 0.6 * tf_prob + 0.4 * xgb_prob
    inference_ms = (time.time() - start) * 1000
    return PredictionResponse(
        is_fraud          = final_prob >= request.threshold,
        fraud_probability = round(float(final_prob), 4),
        risk_level        = get_risk_level(final_prob),
        inference_time_ms = round(inference_ms, 2),
        model_used        = "Transformer+XGBoost Ensemble"
    )

@app.post("/predict/batch")
def predict_batch(transactions: List[TransactionRequest]):
    return [predict(t) for t in transactions]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
