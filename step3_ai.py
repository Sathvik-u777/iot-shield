"""
step3_ai.py — IoT Shield AI Enhancement Layer
Tests Ollama connection and explains threats using local LLM.
Run: python step3_ai.py
"""

import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "tinyllama"   # change to phi3:mini if you have enough RAM

def explain_threat(label, src_ip, dst_ip, confidence, proto):
    prompt = (
        f"You are a network security analyst. Explain this IoT network alert in 2-3 sentences:\n"
        f"- Threat type: {label.upper()}\n"
        f"- Source IP: {src_ip}\n"
        f"- Destination IP: {dst_ip}\n"
        f"- Protocol: {proto.upper()}\n"
        f"- Confidence: {confidence:.1f}%\n"
        f"What does this mean and what action should be taken?"
    )
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=30)
        resp.raise_for_status()
        return resp.json().get("response", "No response from model.")
    except Exception as e:
        return f"[Ollama error] {e}"

if __name__ == "__main__":
    print("=" * 60)
    print("  IoT Shield — AI Layer Test")
    print("=" * 60)
    print(f"\n[*] Testing Ollama with model: {MODEL}")

    result = explain_threat(
        label="ddos",
        src_ip="192.168.1.100",
        dst_ip="10.0.0.1",
        confidence=97.3,
        proto="udp"
    )
    print(f"\n[*] AI Response:\n{result}")
    print("\n[*] AI layer working. Run step4_dashboard.py next.")