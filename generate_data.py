"""
Generates dummy data for the Predictive Maintenance Chatbot demo:
- assets.json: list of rail assets (point machines, track circuits, axle counters)
- manuals.json: maintenance manual chunks per asset type (for RAG)
- repair_logs.json: historical repair log entries
- failure_patterns.json: known fault signatures used for diagnosis
- telemetry.csv: time-series telemetry data per asset (normal + anomalous)
- risk_scores.json: predictive maintenance risk scores per asset
"""

import json
import csv
import random
import datetime as dt

random.seed(42)

OUT = "data"

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
ASSET_TYPES = ["Point Machine", "Track Circuit", "Axle Counter"]

LOCATIONS = [
    "Wuppertal Hbf - Track 3",
    "Wuppertal Vohwinkel - Yard A",
    "Solingen Hbf - Track 1",
    "Remscheid Hbf - Track 2",
    "Hagen Hbf - Junction North",
    "Schwelm - Track 4",
    "Velbert - Branch Line",
    "Gevelsberg - Track 1",
]

MANUFACTURERS = {
    "Point Machine": ["Siemens S700K", "Siemens Hi-Motion"],
    "Track Circuit": ["Siemens FTGS", "Siemens EBI Track 200"],
    "Axle Counter": ["Siemens ACM200", "Siemens ECM2000"],
}

assets = []
asset_id_counter = 1000

for i in range(12):
    a_type = ASSET_TYPES[i % len(ASSET_TYPES)]
    asset = {
        "asset_id": f"AST-{asset_id_counter}",
        "asset_type": a_type,
        "manufacturer": random.choice(MANUFACTURERS[a_type]),
        "location": LOCATIONS[i % len(LOCATIONS)],
        "install_date": (dt.date(2015, 1, 1) + dt.timedelta(days=random.randint(0, 3000))).isoformat(),
        "last_inspection": (dt.date(2026, 1, 1) + dt.timedelta(days=random.randint(0, 150))).isoformat(),
    }
    assets.append(asset)
    asset_id_counter += 1

with open(f"{OUT}/assets.json", "w") as f:
    json.dump(assets, f, indent=2)

# ---------------------------------------------------------------------------
# Manuals (RAG knowledge base chunks)
# ---------------------------------------------------------------------------
manuals = [
    {
        "doc_id": "MAN-PM-001",
        "asset_type": "Point Machine",
        "title": "Siemens S700K - Switch Blade Adjustment Procedure",
        "language": "EN",
        "content": (
            "Section 4.2: Switch Blade Gap Adjustment. The maximum permissible gap between "
            "the switch blade and stock rail is 4mm under load. If the gap exceeds this "
            "value, the point machine drive rod must be inspected for wear. Loosen the "
            "locking nut on the drive rod (torque spec: 45 Nm), adjust the rod length using "
            "the turnbuckle, and re-torque. Verify gap with a feeler gauge at three points "
            "along the blade. A gap exceeding 4mm can cause derailment risk and must be "
            "treated as a Priority 1 defect."
        ),
    },
    {
        "doc_id": "MAN-PM-002",
        "asset_type": "Point Machine",
        "title": "Siemens S700K - Motor Current Draw Diagnostics",
        "language": "EN",
        "content": (
            "Section 6.1: Normal Operating Current. Under normal conditions, the S700K "
            "drive motor draws 2.1 to 2.8 amps during a switch operation, with a peak "
            "inrush current of up to 4.5 amps for the first 150ms. Sustained current draw "
            "above 3.5 amps for more than 500ms indicates excessive mechanical resistance, "
            "typically caused by debris in the switch bed, ice buildup, or a failing "
            "gearbox bearing. Section 6.2: If current draw trends upward over multiple "
            "operations (more than 15% increase week over week), schedule a gearbox "
            "lubrication and bearing inspection within 7 days."
        ),
    },
    {
        "doc_id": "MAN-PM-003",
        "asset_type": "Point Machine",
        "title": "Siemens S700K - Switching Time Specifications",
        "language": "EN",
        "content": (
            "Section 3.4: Switching Time Tolerances. The S700K point machine completes a "
            "full switch operation (lock to lock) in 4.0 to 5.5 seconds under normal "
            "ambient conditions (-10C to 40C). Switching times exceeding 6.5 seconds "
            "indicate mechanical resistance and require immediate inspection. Switching "
            "times below 3.5 seconds may indicate a locking mechanism fault and the asset "
            "must be taken out of service until inspected, as this can result in failure "
            "to fully lock the switch blades."
        ),
    },
    {
        "doc_id": "MAN-TC-001",
        "asset_type": "Track Circuit",
        "title": "Siemens FTGS - Insulated Joint Inspection",
        "language": "EN",
        "content": (
            "Section 2.3: Insulated Rail Joint (IRJ) Resistance. The minimum acceptable "
            "insulation resistance for an IRJ is 1 megaohm when dry and 0.5 megaohm in wet "
            "conditions. Readings below this threshold indicate joint contamination or "
            "physical damage to the insulation material. A failing IRJ can cause false "
            "track circuit occupancy or, in severe cases, loss of train detection. "
            "Replacement of the insulation material should be scheduled within 14 days of "
            "a sub-threshold reading; readings below 0.1 megaohm require replacement "
            "within 48 hours."
        ),
    },
    {
        "doc_id": "MAN-TC-002",
        "asset_type": "Track Circuit",
        "title": "Siemens FTGS - Receiver Voltage Thresholds",
        "language": "EN",
        "content": (
            "Section 5.1: Receiver Input Voltage. Normal receiver input voltage for the "
            "FTGS track circuit ranges from 1.8V to 2.4V when the track is unoccupied. A "
            "gradual decline in receiver voltage over several weeks (more than 0.1V per "
            "week) typically indicates rising ballast resistance issues, often due to "
            "vegetation growth or ballast contamination. Voltage drops below 1.2V will "
            "trigger a false occupancy signal and must be investigated immediately, "
            "including a visual ballast inspection."
        ),
    },
    {
        "doc_id": "MAN-AC-001",
        "asset_type": "Axle Counter",
        "title": "Siemens ACM200 - Sensor Head Alignment",
        "language": "EN",
        "content": (
            "Section 3.1: Sensor Head Alignment Tolerance. The ACM200 wheel sensor head "
            "must be aligned within 2mm of the rail running surface centerline and at a "
            "height of 25mm to 32mm above the rail foot. Misalignment beyond these "
            "tolerances can cause missed axle counts or double-counting. Section 3.2: "
            "Vibration Sensitivity. The sensor includes an internal vibration dampener "
            "rated for normal traffic vibration. Vibration readings sustained above 4.5g "
            "may indicate a loosening sensor bracket and require torque-check of the "
            "M10 mounting bolts (spec: 25 Nm)."
        ),
    },
    {
        "doc_id": "MAN-AC-002",
        "asset_type": "Axle Counter",
        "title": "Siemens ACM200 - Count Discrepancy Troubleshooting",
        "language": "EN",
        "content": (
            "Section 7.2: Count Discrepancy Resolution. If the axle counter reports a "
            "count discrepancy alarm, first verify the sensor head alignment per Section "
            "3.1. If alignment is within tolerance, check for moisture ingress in the "
            "junction box, which can cause intermittent signal noise. Persistent count "
            "discrepancies after these checks require a sensor head replacement. Do not "
            "reset the section counter to clear a discrepancy without confirming no train "
            "is occupying the section, as this can mask a genuine occupancy and create a "
            "serious safety hazard."
        ),
    },
    {
        "doc_id": "MAN-PM-004",
        "asset_type": "Point Machine",
        "title": "Siemens S700K - Wartung der Weichenantriebe (DE)",
        "language": "DE",
        "content": (
            "Abschnitt 4.5: Schmierung der Antriebskomponenten. Die Gleitflaechen des "
            "Antriebsgestaenges muessen alle 6 Monate mit dem freigegebenen Schmierfett "
            "(Spezifikation Siemens LF-200) behandelt werden. Bei Temperaturen unter -5C "
            "kann die Schmierung verhaerten und zu erhoehtem Motorstrom fuehren. Eine "
            "Inspektion auf Verhaertung sollte vor dem Winterbetrieb durchgefuehrt werden. "
            "Eine unzureichende Schmierung ist eine haeufige Ursache fuer ansteigenden "
            "Motorstrom und verlaengerte Schaltzeiten."
        ),
    },
]

with open(f"{OUT}/manuals.json", "w") as f:
    json.dump(manuals, f, indent=2)

# ---------------------------------------------------------------------------
# Failure patterns (used by the diagnosis subsystem)
# ---------------------------------------------------------------------------
failure_patterns = [
    {
        "pattern_id": "FP-001",
        "asset_type": "Point Machine",
        "name": "Gearbox bearing wear",
        "signature": "Motor current draw trending upward >15%/week, switching time increasing toward 6s",
        "recommended_action": "Schedule gearbox lubrication and bearing inspection within 7 days (Manual MAN-PM-002, Sec 6.2)",
        "estimated_repair_minutes": 90,
        "spare_parts": ["Gearbox bearing kit (P/N S700K-BRG-04)", "Siemens LF-200 grease cartridge"],
    },
    {
        "pattern_id": "FP-002",
        "asset_type": "Point Machine",
        "name": "Switch blade gap out of tolerance",
        "signature": "Switching time intermittently exceeds 6.5s, current spikes irregular",
        "recommended_action": "Inspect and adjust switch blade gap per Manual MAN-PM-001, Sec 4.2. Priority 1 if gap >4mm.",
        "estimated_repair_minutes": 60,
        "spare_parts": ["Drive rod turnbuckle assembly", "Feeler gauge set (calibration check)"],
    },
    {
        "pattern_id": "FP-003",
        "asset_type": "Track Circuit",
        "name": "Insulated joint degradation",
        "signature": "Receiver voltage declining gradually >0.1V/week, IRJ resistance below 1 megaohm",
        "recommended_action": "Schedule IRJ insulation replacement within 14 days (Manual MAN-TC-001, Sec 2.3)",
        "estimated_repair_minutes": 120,
        "spare_parts": ["Insulated joint kit - composite", "End post insulation set"],
    },
    {
        "pattern_id": "FP-004",
        "asset_type": "Track Circuit",
        "name": "Ballast contamination / vegetation",
        "signature": "Receiver voltage below 1.2V, sudden drop rather than gradual decline",
        "recommended_action": "Immediate visual ballast inspection and vegetation clearance (Manual MAN-TC-002, Sec 5.1)",
        "estimated_repair_minutes": 180,
        "spare_parts": ["Ballast cleaning equipment (no parts required)"],
    },
    {
        "pattern_id": "FP-005",
        "asset_type": "Axle Counter",
        "name": "Sensor bracket loosening",
        "signature": "Vibration readings sustained above 4.5g, occasional count discrepancy alarms",
        "recommended_action": "Torque-check M10 mounting bolts to 25 Nm and re-verify alignment (Manual MAN-AC-001, Sec 3.2)",
        "estimated_repair_minutes": 45,
        "spare_parts": ["M10 bolt set (stainless)", "Alignment gauge"],
    },
    {
        "pattern_id": "FP-006",
        "asset_type": "Axle Counter",
        "name": "Junction box moisture ingress",
        "signature": "Intermittent count discrepancy alarms, alignment within tolerance",
        "recommended_action": "Inspect junction box for moisture, dry and reseal; replace sensor head if discrepancies persist (Manual MAN-AC-002, Sec 7.2)",
        "estimated_repair_minutes": 75,
        "spare_parts": ["Junction box seal kit", "Sensor head (ACM200) - contingency"],
    },
]

with open(f"{OUT}/failure_patterns.json", "w") as f:
    json.dump(failure_patterns, f, indent=2)

# ---------------------------------------------------------------------------
# Telemetry time series
# ---------------------------------------------------------------------------
# For each asset, generate 30 days of daily telemetry.
# A subset of assets get an "anomalous" trend matching one of the failure patterns.

rows = []
today = dt.date(2026, 6, 15)

# Assign anomaly profiles to a few assets for narrative variety
anomaly_assignments = {
    "AST-1000": "FP-001",  # Point Machine - gearbox bearing wear (gradual current rise)
    "AST-1003": "FP-002",  # Point Machine - switch blade gap
    "AST-1001": "FP-003",  # Track Circuit - insulated joint degradation
    "AST-1007": "FP-004",  # Track Circuit - ballast contamination (sudden drop)
    "AST-1002": "FP-005",  # Axle Counter - sensor bracket loosening
    "AST-1005": "FP-006",  # Axle Counter - junction box moisture
}

for asset in assets:
    asset_id = asset["asset_id"]
    a_type = asset["asset_type"]
    anomaly = anomaly_assignments.get(asset_id)

    for day_offset in range(30, 0, -1):
        date = today - dt.timedelta(days=day_offset)
        day_index = 30 - day_offset  # 0 .. 29, increases over time

        if a_type == "Point Machine":
            base_current = 2.4
            base_switch_time = 4.6
            if anomaly == "FP-001":
                # gradual current increase ~15-20% per week -> over 30 days roughly +60-80%
                current = base_current * (1 + 0.022 * day_index) + random.uniform(-0.05, 0.05)
                switch_time = base_switch_time + 0.04 * day_index + random.uniform(-0.1, 0.1)
            elif anomaly == "FP-002":
                # intermittent spikes in switching time near end of period
                spike = 2.0 if (day_index > 22 and day_index % 3 == 0) else 0
                current = base_current + random.uniform(-0.1, 0.1)
                switch_time = base_switch_time + spike + random.uniform(-0.1, 0.1)
            else:
                current = base_current + random.uniform(-0.1, 0.1)
                switch_time = base_switch_time + random.uniform(-0.15, 0.15)

            rows.append({
                "asset_id": asset_id, "date": date.isoformat(), "asset_type": a_type,
                "metric": "motor_current_amps", "value": round(current, 3),
            })
            rows.append({
                "asset_id": asset_id, "date": date.isoformat(), "asset_type": a_type,
                "metric": "switching_time_seconds", "value": round(switch_time, 3),
            })

        elif a_type == "Track Circuit":
            base_voltage = 2.1
            if anomaly == "FP-003":
                voltage = base_voltage - 0.018 * day_index + random.uniform(-0.02, 0.02)
            elif anomaly == "FP-004":
                # sudden drop in last 5 days
                voltage = base_voltage + random.uniform(-0.03, 0.03)
                if day_index > 25:
                    voltage = 1.05 + random.uniform(-0.05, 0.05)
            else:
                voltage = base_voltage + random.uniform(-0.03, 0.03)

            rows.append({
                "asset_id": asset_id, "date": date.isoformat(), "asset_type": a_type,
                "metric": "receiver_voltage_volts", "value": round(voltage, 3),
            })

        elif a_type == "Axle Counter":
            base_vibration = 2.5
            if anomaly == "FP-005":
                vibration = base_vibration + 0.07 * day_index + random.uniform(-0.1, 0.1)
            else:
                vibration = base_vibration + random.uniform(-0.2, 0.2)

            discrepancy = 0
            if anomaly == "FP-006" and day_index > 18 and day_index % 4 == 0:
                discrepancy = 1
            if anomaly == "FP-005" and day_index > 24 and day_index % 5 == 0:
                discrepancy = 1

            rows.append({
                "asset_id": asset_id, "date": date.isoformat(), "asset_type": a_type,
                "metric": "vibration_g", "value": round(vibration, 3),
            })
            rows.append({
                "asset_id": asset_id, "date": date.isoformat(), "asset_type": a_type,
                "metric": "count_discrepancy_flag", "value": discrepancy,
            })

with open(f"{OUT}/telemetry.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["asset_id", "date", "asset_type", "metric", "value"])
    writer.writeheader()
    writer.writerows(rows)

# ---------------------------------------------------------------------------
# Predictive risk scores
# ---------------------------------------------------------------------------
risk_scores = []
for asset in assets:
    asset_id = asset["asset_id"]
    anomaly = anomaly_assignments.get(asset_id)

    if anomaly in ("FP-001", "FP-003", "FP-005"):
        # gradually worsening -> medium-high risk, predicted failure in 5-12 days
        risk = round(random.uniform(0.62, 0.85), 2)
        days_to_failure = random.randint(5, 12)
    elif anomaly in ("FP-002", "FP-004", "FP-006"):
        # intermittent/sudden -> high risk, sooner failure window
        risk = round(random.uniform(0.75, 0.93), 2)
        days_to_failure = random.randint(2, 6)
    else:
        risk = round(random.uniform(0.05, 0.25), 2)
        days_to_failure = random.randint(30, 90)

    risk_scores.append({
        "asset_id": asset_id,
        "asset_type": asset["asset_type"],
        "location": asset["location"],
        "risk_score": risk,
        "predicted_days_to_failure": days_to_failure,
        "matched_pattern": anomaly,
    })

with open(f"{OUT}/risk_scores.json", "w") as f:
    json.dump(risk_scores, f, indent=2)

# ---------------------------------------------------------------------------
# Repair logs (historical)
# ---------------------------------------------------------------------------
repair_logs = [
    {
        "log_id": "RL-2031",
        "asset_id": "AST-1006",
        "asset_type": "Point Machine",
        "date": "2026-03-12",
        "issue": "Switching time exceeded 6.5s during cold weather",
        "action_taken": "Inspected gearbox lubrication, found hardened grease, re-lubricated with LF-200 per Manual MAN-PM-004",
        "technician": "T. Hoffmann",
        "resolution_minutes": 85,
    },
    {
        "log_id": "RL-2032",
        "asset_id": "AST-1004",
        "asset_type": "Track Circuit",
        "date": "2026-04-02",
        "issue": "Receiver voltage drifted to 1.4V over 3 weeks",
        "action_taken": "Replaced insulated joint, resistance restored to 1.3 megaohm",
        "technician": "S. Becker",
        "resolution_minutes": 130,
    },
    {
        "log_id": "RL-2033",
        "asset_id": "AST-1009",
        "asset_type": "Axle Counter",
        "date": "2026-02-18",
        "issue": "Intermittent count discrepancy alarms",
        "action_taken": "Found moisture in junction box, dried and resealed; no further alarms after 2 weeks monitoring",
        "technician": "T. Hoffmann",
        "resolution_minutes": 70,
    },
    {
        "log_id": "RL-2034",
        "asset_id": "AST-1000",
        "asset_type": "Point Machine",
        "date": "2025-11-22",
        "issue": "Motor current trending up, gearbox bearing found worn at inspection",
        "action_taken": "Replaced gearbox bearing kit, current returned to baseline 2.3A",
        "technician": "M. Klein",
        "resolution_minutes": 95,
    },
]

with open(f"{OUT}/repair_logs.json", "w") as f:
    json.dump(repair_logs, f, indent=2)

print("Data generation complete.")
print(f"Assets: {len(assets)}")
print(f"Manuals: {len(manuals)}")
print(f"Failure patterns: {len(failure_patterns)}")
print(f"Telemetry rows: {len(rows)}")
print(f"Risk scores: {len(risk_scores)}")
print(f"Repair logs: {len(repair_logs)}")
