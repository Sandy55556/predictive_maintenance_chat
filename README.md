# Predictive Maintenance Chatbot - Demo

Lightweight Streamlit demo for the Siemens Mobility architecture design challenge.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
streamlit run app.py
```

Without an API key, the app runs in demo mode with placeholder responses (UI still fully functional).

## Structure

- `app.py` - main Streamlit app (4 tabs)
- `generate_data.py` - regenerates dummy data in `data/`
- `data/` - dummy assets, manuals, telemetry, failure patterns, risk scores, repair logs

## Tabs

1. **Manual Q&A** - RAG-style chat grounded in maintenance manual excerpts (capability 1)
2. **Live Diagnosis** - correlates 30-day telemetry trends with known failure patterns (capability 2)
3. **Predictive Insights** - fleet-wide risk dashboard with failure forecasts (capability 3)
4. **Work Orders (CMMS)** - auto-generated work orders from diagnoses (capability 4)

6 of the 12 dummy assets have injected anomaly trends matching specific failure
patterns (gearbox wear, switch blade gap, insulated joint degradation, ballast
contamination, sensor bracket loosening, junction box moisture) - try those for
the most interesting diagnosis results. The other 6 are nominal baselines.
