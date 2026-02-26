from flask import Flask, request, jsonify
import os
import io
import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP, Raw
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

# ───────────────────────────────────────────────
#   CNN Model Definition (simple 1D CNN for byte sequences)
# ───────────────────────────────────────────────
class SimplePacketCNN(nn.Module):
    def __init__(self):
        super(SimplePacketCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 32, 128)   # adjust according to input size
        self.fc2 = nn.Linear(128, 2)         # binary: benign / malicious

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Try to load models (fallback to None if not found)
XGB_MODEL_PATH = "xgboost_model.json"
CNN_MODEL_PATH = "cnn_packet_model.pt"

xgb_model = None
cnn_model = None

if os.path.exists(XGB_MODEL_PATH):
    xgb_model = xgb.Booster()
    xgb_model.load_model(XGB_MODEL_PATH)

if os.path.exists(CNN_MODEL_PATH):
    cnn_model = SimplePacketCNN()
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=torch.device('cpu')))
    cnn_model.eval()

# ───────────────────────────────────────────────
#   Feature extraction from PCAP
# ───────────────────────────────────────────────
def extract_flow_features(packets):
    if not packets:
        return None

    flows = {}  # key: (src, dst, sport, dport, proto)

    for pkt in packets:
        if IP not in pkt:
            continue
        src = pkt[IP].src
        dst = pkt[IP].dst
        proto = pkt[IP].proto
        sport = pkt.sport if hasattr(pkt, 'sport') else 0
        dport = pkt.dport if hasattr(pkt, 'dport') else 0

        key = (src, dst, sport, dport, proto)

        if key not in flows:
            flows[key] = {
                'start_time': pkt.time,
                'times': [],
                'fwd_len': [], 'bwd_len': [],
                'fwd_win': [], 'bwd_win': [],
                'pkt_count': 0, 'byte_count': 0
            }

        flow = flows[key]
        flow['times'].append(pkt.time - flow['start_time'])
        flow['pkt_count'] += 1
        flow['byte_count'] += len(pkt)

        if 'TCP' in pkt:
            if pkt[TCP].flags & 0x02:  # SYN direction approximation
                flow['fwd_win'].append(pkt[TCP].window)
            else:
                flow['bwd_win'].append(pkt[TCP].window)

        payload_len = len(pkt[Raw]) if Raw in pkt else 0
        # Rough fwd/bwd split (can be improved)
        if src < dst:  # simplistic
            flow['fwd_len'].append(payload_len)
        else:
            flow['bwd_len'].append(payload_len)

    if not flows:
        return None

    # For demo: take the flow with most packets
    main_flow = max(flows.values(), key=lambda f: f['pkt_count'])

    duration = main_flow['times'][-1] if main_flow['times'] else 0
    fwd_pkts = len(main_flow['fwd_len'])
    bwd_pkts = len(main_flow['bwd_len'])

    features = {
        'duration': duration,
        'total_fwd_packets': fwd_pkts,
        'total_bwd_packets': bwd_pkts,
        'fwd_packet_lengths_mean': np.mean(main_flow['fwd_len']) if main_flow['fwd_len'] else 0,
        'bwd_packet_lengths_mean': np.mean(main_flow['bwd_len']) if main_flow['bwd_len'] else 0,
        'flow_bytes_per_sec': main_flow['byte_count'] / duration if duration > 0 else 0,
        'flow_packets_per_sec': main_flow['pkt_count'] / duration if duration > 0 else 0,
        'mean_packet_size': main_flow['byte_count'] / main_flow['pkt_count'] if main_flow['pkt_count'] else 0,
        'init_fwd_win_bytes': main_flow['fwd_win'][0] if main_flow['fwd_win'] else 0,
        'init_bwd_win_bytes': main_flow['bwd_win'][0] if main_flow['bwd_win'] else 0,
    }

    return pd.DataFrame([features])

# ───────────────────────────────────────────────
#   Simple byte-level CNN prediction (first packet payload)
# ───────────────────────────────────────────────
def cnn_byte_prediction(packets):
    if not packets or cnn_model is None:
        return 0.0  # neutral

    first_pkt = packets[0]
    if Raw not in first_pkt:
        return 0.0

    payload = bytes(first_pkt[Raw])
    payload = payload[:512]  # truncate
    payload += b'\x00' * (512 - len(payload))  # pad
    arr = np.frombuffer(payload, dtype=np.uint8).astype(np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, 512]

    with torch.no_grad():
        logits = cnn_model(tensor)
        prob = torch.softmax(logits, dim=1)[0][1].item()  # prob malicious

    return prob

# ───────────────────────────────────────────────
#   Main analysis function
# ───────────────────────────────────────────────
def analyze_traffic(file=None, manual_data=None):
    packets = []

    if file:
        try:
            content = file.read()
            packets = rdpcap(io.BytesIO(content))
        except Exception as e:
            return {'threat_level': 'Error', 'details': f'PCAP parsing failed: {str(e)}'}
    elif manual_data:
        keywords = ['malware', 'exploit', 'ransom', 'c2', 'darknet']
        score = sum(1 for kw in keywords if kw in manual_data.lower())
        level = 'High' if score >= 3 else 'Medium' if score >= 1 else 'Low'
        return {'threat_level': level, 'details': f'Keyword-based analysis (manual input): {score} suspicious terms found.'}

    if not packets:
        return {'threat_level': 'Low', 'details': 'No valid packets found.'}

    # ── XGBoost flow-based prediction ──
    xgb_threat = 'Low'
    xgb_prob = 0.0

    features_df = extract_flow_features(packets)
    if features_df is not None and xgb_model is not None:
        try:
            dmatrix = xgb.DMatrix(features_df)
            pred = xgb_model.predict(dmatrix)
            xgb_prob = float(pred[0]) if len(pred) > 0 else 0.0
            xgb_threat = 'High' if xgb_prob > 0.7 else 'Medium' if xgb_prob > 0.4 else 'Low'
        except Exception as e:
            xgb_threat = 'Error'
            xgb_prob = str(e)

    # ── CNN byte-based prediction ──
    cnn_prob = cnn_byte_prediction(packets)
    cnn_threat = 'High' if cnn_prob > 0.7 else 'Medium' if cnn_prob > 0.4 else 'Low'

    # ── Simple combination ──
    threats = [xgb_threat, cnn_threat]
    if 'High' in threats:
        final_level = 'High'
    elif 'Medium' in threats:
        final_level = 'Medium'
    else:
        final_level = 'Low'

    details = (
        f"XGBoost flow threat: {xgb_threat} (prob {xgb_prob:.3f})\n"
        f"CNN byte threat (first pkt): {cnn_threat} (prob {cnn_prob:.3f})\n"
        f"Analyzed {len(packets)} packets."
    )

    return {'threat_level': final_level, 'details': details}

# ───────────────────────────────────────────────
#   API endpoint for analysis (called by frontend)
# ───────────────────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('file')
    manual_data = request.form.get('data', '')

    result = analyze_traffic(file=file, manual_data=manual_data)
    return jsonify(result)

# ───────────────────────────────────────────────
#   Serve the frontend HTML at root URL
# ───────────────────────────────────────────────
@app.route('/')
def home():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "<h1 style='color:red; text-align:center; margin-top:80px;'>Error: index.html not found<br>Make sure index.html exists in the same folder as app.py</h1>", 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)