# IoT Shield — Network Threat Detection System

Real-time IoT network security threat detection using ML ensemble.

## Project Info
- **Degree**: B.E. CSE 7th Sem, VTU — Global Academy of Technology
- **Course**: CSEP23605
- **Dataset**: IoT-23 (Stratosphere IPS Lab) — 6M rows

## Architecture
| Layer | Description | Tools |
|-------|-------------|-------|
| L1 | Data Collection | IoT-23 CSV + Scapy |
| L2 | Preprocessing | pandas, SMOTE, MinMaxScaler |
| L3 | Feature Engineering | Pearson Correlation, SHAP |
| L4 | ML Detection | Random Forest + XGBoost ensemble |
| L5 | AI Enhancement | Ollama / tinyllama (local LLM) |
| L6 | Dashboard & Alerting | Streamlit + Plotly |

## Results
| Model | Accuracy |
|-------|----------|
| Random Forest | 99.99% |
| XGBoost | 100.00% |
| Ensemble (RF×0.45 + XGB×0.55) | 100.00% |

## Classes
- `benign` — normal traffic
- `ddos` — Distributed Denial of Service
- `malware` — Okiru botnet, C&C traffic
- `portscan` — Horizontal port scanning

## Setup
\`\`\`
conda create -n iotfinal python=3.11
conda activate iotfinal
pip install -r requirements.txt
\`\`\`

## How to Run
```bash
# Terminal 1 — Dashboard
conda activate iotfinal
python -m streamlit run step4_dashboard.py

# Terminal 2 — Simulation mode (no admin needed)
conda activate iotfinal
python step5_simulate.py --reset

# Terminal 3 — Live capture (run as Administrator)
conda activate iotfinal
python step5_live.py --reset 
```

## Base Paper
Maghrabi, L.A. (2024). "Automated Network Intrusion Detection for Internet of Things."
IEEE Access, vol. 12, pp. 30839–30851. DOI: 10.1109/ACCESS.2024.3369237