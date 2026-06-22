"""
Generates maintenance manual PDFs for the predictive maintenance demo.
Run: python generate_pdfs.py
Creates docs/*.pdf — then run ingest.py to build the RAG embeddings index.
"""

from fpdf import FPDF
import os

os.makedirs("docs", exist_ok=True)

BLUE = (0, 100, 161)
LIGHT_BLUE = (220, 237, 248)
GRAY = (120, 120, 120)
WARN_BG = (255, 243, 205)
WARN_TEXT = (133, 100, 4)


class ManualPDF(FPDF):
    _doc_id = ""
    _doc_title = ""
    _revision = ""
    _rev_date = ""

    def setup(self, doc_id, doc_title, revision="Rev C", rev_date="2025-01-15"):
        self._doc_id = doc_id
        self._doc_title = doc_title
        self._revision = revision
        self._rev_date = rev_date
        self.alias_nb_pages()

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 5, f"Siemens Mobility  |  {self._doc_id}  |  {self._doc_title}",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLUE)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_draw_color(0, 0, 0)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 10,
                  f"Page {self.page_no()}/{{nb}}  -  CONFIDENTIAL  -  Authorised maintenance personnel only",
                  align="C")

    def title_page(self, title, subtitle, asset_type, lang="EN"):
        self.add_page()
        self.set_fill_color(*BLUE)
        self.rect(0, 0, 220, 20, "F")
        self.set_xy(10, 6)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, "SIEMENS MOBILITY  -  Rail Infrastructure & Maintenance Division")

        self.set_y(45)
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*BLUE)
        self.multi_cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 12)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 7, subtitle, new_x="LMARGIN", new_y="NEXT")
        self.ln(10)

        y0 = self.get_y()
        self.set_fill_color(*LIGHT_BLUE)
        self.set_draw_color(*BLUE)
        self.rect(10, y0, 190, 50, "DF")
        self.set_xy(18, y0 + 5)
        for label, value in [
            ("Document ID:", self._doc_id),
            ("Asset Type:", asset_type),
            ("Language:", lang),
            ("Revision:", f"{self._revision}  |  {self._rev_date}"),
            ("Status:", "APPROVED FOR USE"),
        ]:
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(30, 30, 30)
            self.cell(55, 8, label)
            self.set_font("Helvetica", "", 10)
            self.cell(0, 8, value, new_x="LMARGIN", new_y="NEXT")
            self.set_x(18)

        self.set_fill_color(*BLUE)
        self.rect(0, 272, 220, 30, "F")
        self.set_xy(10, 277)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(255, 255, 255)
        self.cell(0, 5, "Copyright Siemens Mobility GmbH. All rights reserved.", align="C")

    def h1(self, text):
        self.ln(5)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*BLUE)
        self.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLUE)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_draw_color(0, 0, 0)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def h2(self, text):
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def bullets(self, items):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        for item in items:
            self.cell(8, 5, "-")
            self.multi_cell(0, 5, item, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def warn(self, text):
        self.ln(2)
        y = self.get_y()
        n = max(1, len(text) // 80 + 1)
        h = n * 5 + 12
        self.set_fill_color(*WARN_BG)
        self.set_draw_color(255, 193, 7)
        self.rect(10, y, 190, h, "DF")
        self.set_xy(15, y + 4)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*WARN_TEXT)
        self.multi_cell(180, 5, f"WARNING: {text}", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.set_draw_color(0, 0, 0)
        self.ln(3)

    def note(self, text):
        self.ln(2)
        y = self.get_y()
        n = max(1, len(text) // 80 + 1)
        h = n * 5 + 12
        self.set_fill_color(245, 245, 245)
        self.set_draw_color(180, 180, 180)
        self.rect(10, y, 190, h, "DF")
        self.set_xy(15, y + 4)
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(180, 5, f"NOTE: {text}", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.set_draw_color(0, 0, 0)
        self.ln(3)

    def table(self, headers, rows):
        col_w = 185 // len(headers)
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(*BLUE)
        self.set_text_color(255, 255, 255)
        for h in headers:
            self.cell(col_w, 7, h, border=1, fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 9)
        for i, row in enumerate(rows):
            self.set_fill_color(*LIGHT_BLUE) if i % 2 == 0 else self.set_fill_color(255, 255, 255)
            for cell in row:
                self.cell(col_w, 6, str(cell), border=1, fill=True)
            self.ln()
        self.ln(4)

    def figure(self, caption, h=38):
        self.ln(2)
        y = self.get_y()
        self.set_fill_color(248, 248, 248)
        self.set_draw_color(200, 200, 200)
        self.rect(15, y, 180, h, "DF")
        self.set_xy(15, y + h / 2 - 5)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(160, 160, 160)
        self.multi_cell(180, 5, f"[Schematic: {caption}]", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(80, 80, 80)
        self.cell(0, 5, f"Figure: {caption}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.set_draw_color(0, 0, 0)
        self.ln(3)


# ---------------------------------------------------------------------------
# MAN-PM-001  Switch Blade Adjustment
# ---------------------------------------------------------------------------
def make_man_pm_001():
    pdf = ManualPDF()
    pdf.setup("MAN-PM-001", "Switch Blade Adjustment Procedure")
    pdf.title_page(
        "Siemens S700K Point Machine\nSwitch Blade Adjustment Procedure",
        "Inspection, Measurement and Adjustment of Switch Blade Gap",
        "Point Machine",
    )

    pdf.add_page()
    pdf.h1("1. Scope and Safety")
    pdf.body(
        "This procedure covers inspection, measurement, and adjustment of the switch blade gap "
        "on Siemens S700K and S700H point machines on mainline and yard tracks. "
        "Track Protection Procedures (TPP) must be active throughout. Only technicians holding "
        "a current Siemens S700K Level-2 Maintenance Certificate may perform this work."
    )
    pdf.warn(
        "Ensure the track section is fully isolated under TPP before beginning any work. "
        "Failure to comply may result in serious injury or death from moving train traffic."
    )
    pdf.h2("1.1 Tools and Materials Required")
    pdf.bullets([
        "Calibrated feeler gauge set, 0.1 mm - 10 mm (calibration certificate required, within 6 months)",
        "Torque wrench, 20 - 80 Nm range (calibration within 6 months)",
        "Drive rod adjustment tool  P/N S700K-ADJ-02",
        "27 mm open-end spanner",
        "Siemens LF-200 grease cartridge (if lubrication is required during adjustment)",
        "PPE: high-visibility vest, safety boots, hard hat, gloves",
        "Camera or mobile device for photographic record",
    ])

    pdf.h1("2. Switch Blade Gap Specifications")
    pdf.body(
        "The switch blade gap is the clearance between the switch blade toe and the stock rail "
        "when the point machine has completed a full throw and the locking indicator confirms the "
        "locked position. Measure at three points along the blade: 100 mm from the toe, "
        "at the mid-point, and 100 mm from the heel. Record all three readings."
    )
    pdf.table(
        ["Condition", "Gap Value", "Required Action"],
        [
            ["Normal - serviceable", "0.0 - 3.4 mm", "Record and continue in service"],
            ["Pre-emptive adjustment", "3.5 - 3.9 mm", "Schedule adjustment within 7 days"],
            ["Mandatory adjustment", "4.0 mm (at limit)", "Adjust within 48 hours"],
            ["Out of service", "> 4.0 mm", "Withdraw immediately - Priority 1 defect"],
            ["Drive rod play (acceptable)", "< 1.0 mm", "No action required"],
            ["Drive rod play (excessive)", "> 1.0 mm", "Replace drive rod assembly"],
        ]
    )
    pdf.warn(
        "A confirmed gap exceeding 4 mm creates a risk of wheel flange damage and potential "
        "derailment. The asset MUST be immediately withdrawn from service and a Priority 1 "
        "work order raised before the track is returned to traffic."
    )
    pdf.figure("Switch blade gap measurement positions - toe, mid, heel", h=38)

    pdf.add_page()
    pdf.h1("3. Adjustment Procedure")
    pdf.body(
        "If the measured gap requires adjustment, proceed through all steps in sequence. "
        "Do not return the asset to service until Section 4 verification is complete."
    )
    pdf.bullets([
        "Step 1: Using the 27 mm spanner, break the locking nut on the drive rod turnbuckle. "
        "Nominal break-free torque should be approximately 45 Nm. Record if it breaks free at "
        "a significantly lower value - this indicates prior under-torque and must be noted.",
        "Step 2: Turn the turnbuckle clockwise (viewed from motor end) to reduce the gap, or "
        "counter-clockwise to increase it. Each full rotation changes blade position by approximately 1.5 mm.",
        "Step 3: Re-measure the gap at all three points. Variation across measurement points must "
        "not exceed +/- 0.3 mm. If variation is greater, inspect the blade for twist or bend before continuing.",
        "Step 4: When the gap is within 1.0 - 3.4 mm across all measurement points, hold the "
        "turnbuckle and re-torque the locking nut to exactly 45 Nm.",
        "Step 5: Re-measure gap after torquing. If the gap shifted by more than 0.2 mm during "
        "locking, the threads are likely worn - replace the turnbuckle assembly and repeat.",
    ])
    pdf.note(
        "Apply a thin coat of Siemens LF-200 grease to the drive rod sliding surfaces if the "
        "rod shows signs of dry running (bright, unlubricated contact marks). Do not over-grease "
        "as excess grease attracts ballast and causes premature wear."
    )

    pdf.h1("4. Verification and Sign-off")
    pdf.body(
        "After adjustment, complete all verification steps below before returning to service. "
        "Record results on Maintenance Record Sheet MRS-S700K-05 and upload to the CMMS."
    )
    pdf.bullets([
        "Perform 5 complete switch operations (normal-to-reverse and reverse-to-normal). "
        "Switching time must be within 4.0 - 5.5 seconds on all 5 operations.",
        "Observe motor current draw during switching. Steady-state current must be 2.1 - 2.8 A. "
        "Peak inrush current up to 4.5 A for the first 150 ms is normal and acceptable.",
        "Re-measure blade gap after the 5 operations to confirm no settling has occurred.",
        "Photograph all three gap measurement points before and after adjustment. Tag with "
        "asset ID, timestamp, and technician ID. Attach to the work order in CMMS.",
    ])
    pdf.note(
        "If the gap shifts by more than 0.5 mm after 5 verification operations, the turnbuckle "
        "or locking nut threads may be worn. Escalate to a Level-3 engineer before returning "
        "the asset to service."
    )

    pdf.output("docs/MAN-PM-001.pdf")
    print("  Created docs/MAN-PM-001.pdf")


# ---------------------------------------------------------------------------
# MAN-PM-002  Motor Current Draw Diagnostics
# ---------------------------------------------------------------------------
def make_man_pm_002():
    pdf = ManualPDF()
    pdf.setup("MAN-PM-002", "Motor Current Draw Diagnostics", "Rev B", "2024-09-01")
    pdf.title_page(
        "Siemens S700K Point Machine\nMotor Current Draw Diagnostics",
        "Normal Operating Parameters, Fault Indicators, and Escalation Criteria",
        "Point Machine",
    )

    pdf.add_page()
    pdf.h1("1. Overview")
    pdf.body(
        "Motor current draw is a primary health indicator for the S700K point machine. "
        "Changes in current consumption reflect the mechanical load on the drive gearbox and "
        "can provide early warning of bearing wear, lubrication failure, debris ingestion, "
        "or ice build-up in the switch bed. This document defines normal operating parameters "
        "and the criteria for scheduling preventive maintenance based on current trending."
    )
    pdf.figure("S700K motor current waveform during a normal switch operation", h=40)

    pdf.h1("2. Normal Operating Parameters")
    pdf.body(
        "Under normal operating conditions at ambient temperatures between -10 C and +40 C, "
        "the S700K drive motor exhibits the following current profile during a full switch operation:"
    )
    pdf.table(
        ["Phase", "Current", "Duration", "Status"],
        [
            ["Inrush (start)", "Up to 4.5 A peak", "First 150 ms", "Normal - motor acceleration"],
            ["Steady-state operation", "2.1 - 2.8 A", "Remainder of throw", "Normal"],
            ["Locked position (holding)", "0.4 - 0.8 A", "Continuous", "Normal"],
            ["Sustained > 3.5 A", "> 3.5 A", "> 500 ms", "FAULT - excessive resistance"],
            ["Sustained > 4.5 A", "> 4.5 A", "Any duration", "FAULT - immediate inspection"],
        ]
    )
    pdf.warn(
        "Sustained motor current above 3.5 A for more than 500 ms during normal operations "
        "indicates excessive mechanical resistance. Log the fault and schedule inspection "
        "within 7 days. Do not wait for the current to trip the overload protection."
    )

    pdf.h1("3. Trending Analysis and Preventive Maintenance Trigger")
    pdf.body(
        "Single-event high current readings may result from transient causes such as debris "
        "in the switch bed or low-temperature grease hardening. However, a consistent upward "
        "trend in steady-state current across multiple switch operations is a reliable indicator "
        "of progressive gearbox bearing wear. The CMMS telemetry dashboard plots daily mean "
        "current and flags the following thresholds:"
    )
    pdf.table(
        ["Trending Condition", "Threshold", "Recommended Action"],
        [
            ["Week-on-week increase", "> 15% per week", "Schedule gearbox inspection within 7 days (Sec 6.2)"],
            ["30-day cumulative increase", "> 40%", "Expedite gearbox lubrication and bearing check"],
            ["30-day cumulative increase", "> 60%", "Treat as high-risk; bearing replacement likely required"],
            ["Single spike (transient)", "> 3.5 A, non-recurring", "Check switch bed for debris or ice; monitor"],
        ]
    )
    pdf.body(
        "Section 6.2 of this manual details the gearbox lubrication and bearing inspection "
        "procedure. Spare parts required: Gearbox bearing kit P/N S700K-BRG-04 and one "
        "Siemens LF-200 grease cartridge. Estimated repair time: 90 minutes."
    )
    pdf.note(
        "Low ambient temperatures (below -5 C) can cause LF-200 grease to harden, resulting "
        "in elevated current even on healthy bearings. Cross-reference ambient temperature logs "
        "before scheduling maintenance for cold-weather current spikes. See MAN-PM-004 for "
        "winter lubrication procedures."
    )

    pdf.output("docs/MAN-PM-002.pdf")
    print("  Created docs/MAN-PM-002.pdf")


# ---------------------------------------------------------------------------
# MAN-PM-003  Switching Time Specifications
# ---------------------------------------------------------------------------
def make_man_pm_003():
    pdf = ManualPDF()
    pdf.setup("MAN-PM-003", "Switching Time Specifications", "Rev B", "2024-06-15")
    pdf.title_page(
        "Siemens S700K Point Machine\nSwitching Time Specifications",
        "Operating Tolerances, Fault Conditions, and Environmental Factors",
        "Point Machine",
    )

    pdf.add_page()
    pdf.h1("1. Overview")
    pdf.body(
        "Switching time is the elapsed time from the initiation of a switch command to the "
        "confirmation of the locked position (lock-to-lock time). It is measured by the S700K "
        "internal controller and logged to the CMMS telemetry system. Switching time is an "
        "important diagnostic indicator: values outside the normal range indicate either "
        "mechanical resistance (high time) or a locking mechanism fault (low time)."
    )

    pdf.h1("2. Normal Operating Range")
    pdf.body(
        "The following values apply at ambient temperatures between -10 C and +40 C. "
        "Times are measured from command initiation to locking relay energisation."
    )
    pdf.table(
        ["Condition", "Switching Time", "Action"],
        [
            ["Normal operation", "4.0 - 5.5 seconds", "No action required"],
            ["Elevated - monitor", "5.5 - 6.4 seconds", "Log and monitor trend over 7 days"],
            ["High - inspect within 48h", "6.5 seconds", "Inspect for mechanical resistance"],
            ["Critically high - withdraw", "> 6.5 seconds", "Withdraw from service; Priority 2 defect"],
            ["Abnormally fast", "< 3.5 seconds", "Withdraw from service; possible locking fault - Priority 1"],
        ]
    )
    pdf.warn(
        "A switching time below 3.5 seconds may indicate a failure of the locking mechanism. "
        "The switch blades may not be achieving full lock, creating a derailment risk. "
        "Withdraw the asset from service immediately and escalate to a Level-3 engineer."
    )

    pdf.h1("3. Common Causes of Elevated Switching Time")
    pdf.bullets([
        "Gearbox bearing wear - typically presents as a gradual increase over weeks, "
        "correlated with rising motor current (cross-reference MAN-PM-002).",
        "Switch bed debris or fouling - sudden onset; inspect switch bed for ballast, "
        "vegetation, ice, or foreign objects.",
        "Drive rod misalignment or damage - check rod for bending or wear at joints.",
        "Hardened lubrication (winter) - cold grease increases friction; see MAN-PM-004.",
        "Blade obstruction - check for deformation or track geometry issues preventing full throw.",
    ])

    pdf.h1("4. Switching Time Under Extreme Temperatures")
    pdf.body(
        "The S700K is rated for ambient temperatures from -25 C to +70 C, but switching "
        "times outside the -10 C to +40 C normal range may exhibit the following behaviour:"
    )
    pdf.table(
        ["Temperature Range", "Expected Effect", "Additional Action"],
        [
            ["Below -10 C", "Up to 1.0 s increase is acceptable", "Inspect lubrication pre-winter (MAN-PM-004)"],
            ["-10 C to +40 C", "Normal range applies", "Standard monitoring"],
            ["Above +40 C", "Minor decrease possible", "Inspect for thermal expansion of components if > 10%"],
        ]
    )
    pdf.note(
        "If switching time repeatedly exceeds 6.0 seconds in cold weather despite correct "
        "lubrication, consider installing a point heating system. Raise a capital works request "
        "through the infrastructure planning team."
    )

    pdf.output("docs/MAN-PM-003.pdf")
    print("  Created docs/MAN-PM-003.pdf")


# ---------------------------------------------------------------------------
# MAN-PM-004  Wartung Weichenantriebe (DE)
# ---------------------------------------------------------------------------
def make_man_pm_004():
    pdf = ManualPDF()
    pdf.setup("MAN-PM-004", "Wartung der Weichenantriebe", "Rev A", "2024-11-01")
    pdf.title_page(
        "Siemens S700K Weichenantrieb\nWartung der Antriebskomponenten",
        "Schmierung, Winterbetrieb und Fehlerdiagnose",
        "Point Machine",
        lang="DE",
    )

    pdf.add_page()
    pdf.h1("1. Geltungsbereich und Sicherheitshinweise")
    pdf.body(
        "Dieses Dokument beschreibt die Wartung der Schmier- und Antriebskomponenten des "
        "Siemens S700K Weichenantriebs. Die Arbeiten duerfen nur von zertifizierten Technikern "
        "(Siemens S700K Zertifikat Level 2 oder hoeher) und mit gueltiger Gleissicherungsausbildung "
        "durchgefuehrt werden. Der Streckenabschnitt muss waehrend der gesamten Arbeit "
        "durch Sicherungsposten oder technische Sicherung gesichert sein."
    )
    pdf.warn(
        "Vor Beginn der Arbeiten ist sicherzustellen, dass der betroffene Gleisabschnitt "
        "gemaess den geltenden Gleissicherungsvorschriften gesperrt ist. Nichteinhaltung "
        "kann zu schweren Verletzungen oder zum Tod fuehren."
    )

    pdf.h1("2. Schmierung der Antriebskomponenten")
    pdf.body(
        "Die Gleitflaechen des Antriebsgestaenges sowie alle beweglichen Verbindungen muessen "
        "gemaess dem folgenden Wartungsplan mit dem freigegebenen Schmiermittel behandelt werden. "
        "Nur Siemens LF-200 (Bestellnummer SIE-LF200-500G) oder ein baugleich freigegebenes "
        "Aequivalent darf verwendet werden. Die Verwendung anderer Schmiermittel kann zu "
        "Materialunvertraeglichkeit und vorzeitigem Verschleiss fuehren."
    )
    pdf.table(
        ["Bauteil", "Schmierstoff", "Intervall", "Auftragsmenge"],
        [
            ["Antriebsgestaenge Gleitflaechen", "Siemens LF-200", "6 Monate", "Duenner Film, ca. 5 g"],
            ["Gelenke Spurstange", "Siemens LF-200", "6 Monate", "2-3 Tropfen pro Gelenk"],
            ["Getriebe (Inspektion)", "Siemens LF-200", "12 Monate", "Laut Getsriebefuellung pruefen"],
            ["Verschlussbolzen", "Siemens LF-200", "6 Monate", "Leichter Auftrag"],
        ]
    )
    pdf.note(
        "Bei Temperaturen unter -5 C kann das Schmierfett LF-200 aushaerten und zu einem "
        "erhoehten Motorstrom fuehren, ohne dass ein mechanischer Defekt vorliegt. "
        "Eine Inspektion auf Schmierungshaertung sollte daher stets vor dem Winterbetrieb "
        "(Oktober/November) durchgefuehrt werden."
    )

    pdf.add_page()
    pdf.h1("3. Inspektion vor Winterbetrieb")
    pdf.body(
        "Die Winterinspektion ist jaehrlich im Oktober oder November vor dem Einsetzen von "
        "Frostperioden durchzufuehren. Sie umfasst folgende Pruefpunkte:"
    )
    pdf.bullets([
        "Schmiermittel auf Antriebsgestaenge und Gelenkverbindungen erneuern (LF-200 gemaess Abschnitt 2).",
        "Weichenbett auf Ablagerungen, Vegetation oder Gegengstaende pruefen und reinigen.",
        "Weichenheizung (sofern vorhanden) auf Funktion pruefen. Sollwert-Temperatur laut Herstellervorgabe.",
        "Schaltzeitenmessung durchfuehren: Bei -10 C bis +5 C ist eine Verlaengerung der Schaltzeit "
        "um bis zu 1,0 Sekunde tolerierbar (Normalbereich dann 4,0 bis 6,5 Sekunden).",
        "Motorstrom kontrollieren: Kaltes Schmierfett kann den Strom um bis zu 20 % erhoehen. "
        "Wenn der Strom dauerhaft ueber 3,5 A liegt, Schmierung sofort erneuern.",
    ])

    pdf.h1("4. Symptome und Fehlerbehebung")
    pdf.body(
        "Folgende haeufige Stoerungsbilder sind im Winterbetrieb zu beachten:"
    )
    pdf.table(
        ["Symptom", "Wahrscheinliche Ursache", "Massnahme"],
        [
            ["Strom > 3,5 A, Schaltzeit erhoeht", "Schmierfett ausgehaertet", "LF-200 auftragen, Strom neu messen"],
            ["Schaltzeit > 6,5 s", "Eisbildung im Weichenbett", "Eis entfernen, Heizung pruefen"],
            ["Strom hoch, kein Eis", "Lagererverschleiss Getriebe", "Getriebe inspizieren (MAN-PM-002)"],
            ["Keine Schaltbestaetigung", "Verriegelungsmechanismus blockiert", "Eskalation Level-3 Techniker"],
        ]
    )
    pdf.warn(
        "Ersatzteile fuer die Winterwartung muessen VOR dem Winterbetrieb bevorratet sein. "
        "Benoetigt werden: LF-200 Schmierfett (min. 2 Kartuschen pro Antrieb), "
        "Getriebelager-Satz P/N S700K-BRG-04 (als Notfallvorrat)."
    )

    pdf.output("docs/MAN-PM-004.pdf")
    print("  Created docs/MAN-PM-004.pdf")


# ---------------------------------------------------------------------------
# MAN-TC-001  Insulated Joint Inspection
# ---------------------------------------------------------------------------
def make_man_tc_001():
    pdf = ManualPDF()
    pdf.setup("MAN-TC-001", "Insulated Joint Inspection", "Rev D", "2025-03-01")
    pdf.title_page(
        "Siemens FTGS Track Circuit\nInsulated Rail Joint (IRJ) Inspection",
        "Resistance Measurement, Fault Classification, and Replacement Criteria",
        "Track Circuit",
    )

    pdf.add_page()
    pdf.h1("1. Purpose and Background")
    pdf.body(
        "The Insulated Rail Joint (IRJ) provides electrical isolation between adjacent track "
        "circuit sections. Degradation of the insulation material allows current to leak between "
        "sections, reducing the voltage received by the track circuit receiver. In severe cases "
        "this causes false occupancy signals (section reported occupied when clear) or, more "
        "critically, the masking of genuine occupancy (train present but not detected). "
        "Regular IRJ resistance measurement is therefore a safety-critical maintenance activity."
    )
    pdf.figure("IRJ cross-section: end posts, fish plates, and insulation material", h=40)

    pdf.h1("2. IRJ Resistance Specifications")
    pdf.body(
        "Measure IRJ resistance using a calibrated insulation resistance tester (megohmmeter) "
        "at 500 V DC. Take measurements in both dry and wet conditions where possible. "
        "Both fish plate joints at a given IRJ location must be measured separately."
    )
    pdf.table(
        ["Condition", "Minimum Resistance", "Required Action"],
        [
            ["Dry conditions - serviceable", ">= 1.0 Mohm", "No action; re-test at next scheduled visit"],
            ["Wet conditions - serviceable", ">= 0.5 Mohm", "No action; monitor trend"],
            ["At risk - schedule replacement", "0.1 - 0.5 Mohm (dry)", "Schedule IRJ replacement within 14 days"],
            ["Critical - urgent replacement", "< 0.1 Mohm", "Replace within 48 hours - Priority 1"],
        ]
    )
    pdf.warn(
        "Resistance below 0.1 Mohm in any conditions constitutes a critical failure. "
        "A false clear condition may exist. Notify the control centre immediately, consider "
        "imposing a temporary speed restriction, and arrange replacement within 48 hours."
    )

    pdf.h1("3. Inspection Procedure")
    pdf.body(
        "Disconnect the bonding wires at both ends of the IRJ before taking measurements to "
        "prevent parallel paths from adjacent bonding corrupting the reading. Reconnect and "
        "verify continuity after measurement."
    )
    pdf.bullets([
        "Step 1: Obtain a line blockage and apply track protection for the affected section.",
        "Step 2: Visually inspect the IRJ. Look for cracked or missing end posts, "
        "deteriorated fish plate insulation sleeves, mechanical damage, or vegetation "
        "growing through the joint.",
        "Step 3: Disconnect rail bonding wires at both sides of the IRJ.",
        "Step 4: Apply 500 V DC from the megohmmeter between one fish plate and the rail. "
        "Allow 60 seconds for the reading to stabilise before recording.",
        "Step 5: Repeat for the second fish plate.",
        "Step 6: Reconnect bonding wires and test track circuit operation (occupied/clear cycle).",
        "Step 7: Record readings, date, ambient conditions, and technician ID in the CMMS.",
    ])

    pdf.h1("4. Replacement Criteria and Parts")
    pdf.body(
        "When resistance is below threshold, schedule or execute IRJ insulation replacement "
        "using the standard insulated joint kit. The composite kit includes glass-fibre "
        "end posts, insulation sleeves, and hardware and is rated for 760 mm rail.",
    )
    pdf.table(
        ["Spare Part", "Part Number", "Notes"],
        [
            ["Insulated joint kit - composite", "IRJ-KIT-760-CF", "Covers one joint (both fish plates)"],
            ["End post insulation set (pair)", "IRJ-EP-760-SET", "Replace if cracked or missing"],
            ["Fish plate insulation sleeve", "IRJ-SLV-760-4PK", "4 per kit; replace all when replacing joint"],
        ]
    )
    pdf.note(
        "Estimated replacement time is 120 minutes per joint by a two-person team. Allow "
        "additional time for track circuit commissioning after replacement (see MAN-TC-002 "
        "for receiver voltage verification procedure)."
    )

    pdf.output("docs/MAN-TC-001.pdf")
    print("  Created docs/MAN-TC-001.pdf")


# ---------------------------------------------------------------------------
# MAN-TC-002  Receiver Voltage Thresholds
# ---------------------------------------------------------------------------
def make_man_tc_002():
    pdf = ManualPDF()
    pdf.setup("MAN-TC-002", "Receiver Voltage Thresholds", "Rev C", "2024-08-20")
    pdf.title_page(
        "Siemens FTGS Track Circuit\nReceiver Voltage Thresholds",
        "Normal Voltage Range, Ballast Diagnostics, and Fault Investigation",
        "Track Circuit",
    )

    pdf.add_page()
    pdf.h1("1. FTGS Track Circuit Overview")
    pdf.body(
        "The Siemens FTGS (Frequency-shift Track Circuit with Guard Signal) operates by "
        "transmitting an audio-frequency signal through the rail and measuring the received "
        "voltage at the far end. The voltage received depends on the resistance of the ballast "
        "and rail path. A high ballast resistance reduces leakage and results in a higher "
        "received voltage. A low ballast resistance (due to contamination or moisture) "
        "increases leakage, reducing receiver voltage. When receiver voltage drops below the "
        "occupancy threshold, the system reports the section as occupied regardless of "
        "whether a train is present."
    )
    pdf.figure("FTGS track circuit transmission and receiver configuration", h=38)

    pdf.h1("2. Receiver Voltage Specifications")
    pdf.table(
        ["Condition", "Receiver Voltage", "Interpretation / Action"],
        [
            ["Normal - unoccupied", "1.8 - 2.4 V", "Track clear; no action"],
            ["Normal - occupied (train shunt)", "< 0.15 V", "Train present; normal operation"],
            ["Low - monitor", "1.2 - 1.8 V", "Monitor trend; inspect ballast within 14 days"],
            ["Critical - false occupancy risk", "< 1.2 V", "Investigate immediately; inspect ballast"],
            ["Transmitter or bonding fault", "< 0.5 V (no train)", "Check transmitter output and bonding wires"],
        ]
    )
    pdf.warn(
        "A receiver voltage below 1.2 V with no train present will trigger a false occupancy "
        "signal. This may cause unnecessary train stops and delay. Notify the control centre "
        "and carry out a ballast inspection within 24 hours."
    )

    pdf.h1("3. Identifying the Failure Mode from Voltage Trend")
    pdf.body(
        "The shape of the voltage decline provides important diagnostic information:"
    )
    pdf.bullets([
        "Gradual decline over weeks (more than 0.1 V per week consistently): Typically indicates "
        "progressive insulated joint degradation (cross-reference MAN-TC-001) or slowly increasing "
        "ballast resistance from contamination. Schedule IRJ resistance measurement and visual "
        "ballast inspection.",
        "Sudden drop over 2-5 days to below 1.2 V: Likely ballast contamination from a vegetation "
        "overgrowth event, flooding, or track maintenance that disturbed the ballast profile. "
        "Carry out urgent visual inspection and vegetation clearance.",
        "Intermittent dips followed by recovery: Possible loose bonding wire or intermittent IRJ "
        "contact. Inspect all bonding connections along the section.",
        "Stable low voltage with no trend: Transmitter output level may have drifted. Measure "
        "transmitter output and adjust if outside specification.",
    ])

    pdf.h1("4. Ballast Inspection Procedure")
    pdf.body(
        "A visual ballast inspection must be carried out by a track geometer or P-way technician. "
        "Key items to assess and clear:"
    )
    pdf.bullets([
        "Vegetation (grass, weeds) growing through ballast within the section - cut and treat.",
        "Mud or fine material contamination of the ballast voids - ballast cleaning or replacement required.",
        "Water pooling or poor drainage adjacent to the track - raise a drainage defect order.",
        "Evidence of recent track maintenance (tamping, stoneblowing) that may have disturbed "
        "ballast continuity - allow 48 hours for ballast to settle and re-test voltage.",
    ])
    pdf.note(
        "After clearance, allow one wet-weather cycle (rainfall event) before concluding the "
        "ballast resistance has returned to normal. Continue daily voltage monitoring for "
        "at least 7 days post-clearance before closing the defect."
    )

    pdf.output("docs/MAN-TC-002.pdf")
    print("  Created docs/MAN-TC-002.pdf")


# ---------------------------------------------------------------------------
# MAN-AC-001  Sensor Head Alignment
# ---------------------------------------------------------------------------
def make_man_ac_001():
    pdf = ManualPDF()
    pdf.setup("MAN-AC-001", "Sensor Head Alignment", "Rev B", "2025-02-10")
    pdf.title_page(
        "Siemens ACM200 Axle Counter\nSensor Head Alignment",
        "Alignment Tolerances, Mounting Torques, and Vibration Monitoring",
        "Axle Counter",
    )

    pdf.add_page()
    pdf.h1("1. Sensor Head Overview")
    pdf.body(
        "The ACM200 wheel sensor head detects passing wheel flanges using an inductive "
        "electromagnetic field. Correct alignment relative to the rail running surface is "
        "critical for reliable axle counting. Misalignment can result in missed counts "
        "(wheel passes undetected) or double-counts (single axle counted twice), both of "
        "which create safety-critical track occupancy errors. The sensor is mounted on an "
        "adjustable bracket bolted to the rail web."
    )
    pdf.figure("ACM200 sensor head mounting on rail web - alignment axes", h=40)

    pdf.h1("2. Alignment Specifications")
    pdf.table(
        ["Alignment Parameter", "Required Value", "Measurement Method"],
        [
            ["Lateral position (centre of rail)", "+/- 2 mm from running surface centreline", "Alignment gauge"],
            ["Vertical height above rail foot", "25 mm to 32 mm", "Height gauge"],
            ["Longitudinal angle to rail", "90 degrees +/- 2 deg", "Digital angle gauge"],
            ["Cable strain relief clearance", "> 50 mm free cable", "Visual inspection"],
        ]
    )
    pdf.warn(
        "Misalignment beyond the tolerances above must be corrected immediately. "
        "An out-of-tolerance sensor should be treated as a provisional fault until "
        "re-aligned and a full axle count verification cycle has been completed."
    )

    pdf.h1("3. Alignment Procedure")
    pdf.bullets([
        "Step 1: Obtain track protection and ensure the line is blocked.",
        "Step 2: Loosen the four M10 mounting bolts on the sensor bracket (do not fully remove). "
        "Slacken only enough to allow lateral and vertical adjustment.",
        "Step 3: Using the alignment gauge, position the sensor at the correct lateral offset "
        "(within +/- 2 mm of the rail centreline). Tighten the lateral locking bolt hand-tight.",
        "Step 4: Using the height gauge, set the sensor at 28 mm above the rail foot "
        "(mid-point of the 25 - 32 mm tolerance). This centre-point setting provides the "
        "best margin against track movement and sensor drift.",
        "Step 5: Verify angular alignment with the digital angle gauge. Adjust if needed.",
        "Step 6: Torque all four M10 mounting bolts to 25 Nm in a cross pattern. "
        "Recheck all three alignment dimensions after torquing.",
        "Step 7: Perform a full axle count verification: have a track vehicle (or manually "
        "simulate with a test axle) pass through the section. Confirm the count matches expected.",
    ])
    pdf.note(
        "Mounting bolt torque of 25 Nm is critical. Under-torqued brackets can loosen under "
        "traffic vibration, leading to progressive misalignment. Check torque at every "
        "scheduled maintenance visit (every 6 months minimum)."
    )

    pdf.h1("4. Vibration Monitoring")
    pdf.body(
        "The ACM200 sensor includes an internal MEMS vibration sensor. Sustained vibration "
        "readings above 4.5 g indicate that the mounting bracket has loosened or that an "
        "abnormal vibration source is present (e.g. flat-spotted wheel on a frequent train path). "
        "The following action thresholds apply:"
    )
    pdf.table(
        ["Vibration Level", "Duration", "Action"],
        [
            ["< 3.5 g", "Any", "Normal - no action required"],
            ["3.5 - 4.5 g", "Sustained (> 5 min)", "Check bracket bolt torques at next visit"],
            ["> 4.5 g", "Any sustained reading", "Inspect and torque-check M10 bolts to 25 Nm within 48 hours"],
            ["> 6.0 g", "Any reading", "Inspect immediately; may indicate structural bracket failure"],
        ]
    )

    pdf.output("docs/MAN-AC-001.pdf")
    print("  Created docs/MAN-AC-001.pdf")


# ---------------------------------------------------------------------------
# MAN-AC-002  Count Discrepancy Troubleshooting
# ---------------------------------------------------------------------------
def make_man_ac_002():
    pdf = ManualPDF()
    pdf.setup("MAN-AC-002", "Count Discrepancy Troubleshooting", "Rev C", "2025-04-01")
    pdf.title_page(
        "Siemens ACM200 Axle Counter\nCount Discrepancy Troubleshooting",
        "Diagnostic Procedure, Moisture Ingress Protocol, and Safety Constraints",
        "Axle Counter",
    )

    pdf.add_page()
    pdf.h1("1. Count Discrepancy Alarm")
    pdf.body(
        "A count discrepancy alarm is generated when the ACM200 evaluator detects that the "
        "number of axles counted entering a track section does not match the number counted "
        "leaving it. This condition means the system cannot confirm that the section is "
        "clear and will hold the section as occupied until the discrepancy is resolved by "
        "an authorised technician. Count discrepancies must be investigated and resolved "
        "before the section counter is reset."
    )
    pdf.warn(
        "DO NOT reset the section counter to clear a count discrepancy without first "
        "physically confirming that no train or vehicle is present in the section. "
        "Resetting a counter while a train is occupying the section masks a genuine "
        "occupancy and can authorise a conflicting train movement - a potentially fatal error."
    )

    pdf.h1("2. Initial Diagnostic Steps")
    pdf.body(
        "Follow these steps in sequence. Stop and resolve if a cause is identified at any stage; "
        "do not proceed to later steps without clearing earlier ones."
    )
    pdf.bullets([
        "Step 1: Confirm with the control centre that no train or on-track machine is "
        "authorised in the section. Obtain a physical occupation check from the last "
        "signaller to handle train movements through the section.",
        "Step 2: Physically walk the section to confirm it is clear. This is mandatory "
        "before any counter reset is authorised.",
        "Step 3: Check sensor head alignment per MAN-AC-001 Section 3. Even minor "
        "misalignment can cause intermittent missed counts, especially at high speed.",
        "Step 4: Inspect the junction box for moisture ingress (see Section 3 below).",
        "Step 5: Review the evaluator event log for recurring discrepancy patterns. "
        "Single-event discrepancies may indicate a transient cause; recurring patterns "
        "at regular intervals suggest a timing or interference issue.",
    ])

    pdf.h1("3. Moisture Ingress in Junction Box")
    pdf.body(
        "Moisture in the sensor junction box is the most common non-alignment cause of count "
        "discrepancy alarms. Water ingress creates intermittent leakage paths that introduce "
        "noise into the sensor signal, causing spurious counts. Inspect as follows:"
    )
    pdf.bullets([
        "Open the junction box lid. Inspect for visible water droplets, condensation, "
        "corrosion on terminals, or white mineral deposits (salt from evaporated water).",
        "If moisture is present: dry the interior with compressed air (clean, dry, oil-free). "
        "Replace the cable gland seals and junction box gasket using the standard seal kit "
        "(P/N ACM200-SEAL-KIT). Do not use silicone sealant as a substitute for proper gaskets.",
        "Inspect the cable entry glands. Loose or cracked glands are the most common ingress "
        "point. Replace glands if cracked or if the cable is not firmly gripped.",
        "After drying and resealing, monitor the system for 24 hours before closing the defect.",
        "If discrepancies persist after resealing, replace the sensor head unit "
        "(P/N ACM200-SH-ASSY) as an internal seal failure is likely.",
    ])
    pdf.table(
        ["Spare Part", "Part Number", "Notes"],
        [
            ["Junction box seal kit", "ACM200-SEAL-KIT", "Includes gasket, 4 cable gland seals"],
            ["Sensor head assembly (contingency)", "ACM200-SH-ASSY", "Replace if moisture persists after resealing"],
            ["M10 bolt set (stainless)", "ACM200-M10-SS-4PK", "Replace if corroded"],
        ]
    )
    pdf.note(
        "After any count discrepancy resolution, conduct a minimum 3-axle verification count "
        "before handing back to the control centre. Document the cause, corrective action, "
        "and verification count result on the CMMS work order."
    )

    pdf.output("docs/MAN-AC-002.pdf")
    print("  Created docs/MAN-AC-002.pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating PDF manuals...")
    make_man_pm_001()
    make_man_pm_002()
    make_man_pm_003()
    make_man_pm_004()
    make_man_tc_001()
    make_man_tc_002()
    make_man_ac_001()
    make_man_ac_002()
    print(f"\nDone. 8 PDFs written to docs/")
    print("Next step: run  python ingest.py  to build the RAG embeddings index.")
