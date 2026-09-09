import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sqlite3
import os
import shutil
import re
from datetime import datetime

import numpy as np
import cv2
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis


# =========================================================
# SETTINGS
# =========================================================

DATABASE_PATH = os.path.join(
    "database",
    "criminal_intelligence.db"
)

FACES_FOLDER = "faces"
PHOTOS_FOLDER = "criminal_photos"

WINDOW_WIDTH = 1250
WINDOW_HEIGHT = 760

MODEL_NAME = "buffalo_l"

MATCH_THRESHOLD = 0.45

CAMERA_INDEX = 0

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

PERSON_ID_PATTERN = re.compile(
    r"^P\d{3}$"
)

FIR_ID_PATTERN = re.compile(
    r"^FIR\d{3}$"
)

PHONE_PATTERN = re.compile(
    r"^\d{10}$"
)

VEHICLE_PATTERN = re.compile(
    r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$"
)


# =========================================================
# GLOBALS
# =========================================================

root = None

face_app = None

camera = None

camera_running = False



# =========================================================
# WINDOW CONTROLS / ORIENTATION
# =========================================================

def enable_window_controls(win):
    try:
        win.resizable(True, True)
        win.minsize(760, 520)
        win.bind("<Control-Shift-F>", lambda e: toggle_maximize(win))
        win.bind("<Escape>", lambda e: restore_window(win))
        win.bind("<Control-w>", lambda e: safe_close_window(win))
    except Exception:
        pass


def toggle_maximize(win):
    try:
        if win.state() == "zoomed":
            restore_window(win)
        else:
            win.state("zoomed")
    except Exception:
        pass


def restore_window(win):
    try:
        win.state("normal")
        win.update_idletasks()
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        w = min(max(win.winfo_width(), 900), sw - 60)
        h = min(max(win.winfo_height(), 620), sh - 100)
        x = max(20, (sw - w) // 2)
        y = max(20, (sh - h) // 2)
        win.geometry(f"{w}x{h}+{x}+{y}")
    except Exception:
        pass


def safe_close_window(win):
    try:
        win.grab_release()
    except Exception:
        pass
    try:
        win.destroy()
    except Exception:
        pass


def enhance_all_windows():
    if root is None:
        return
    try:
        for win in root.winfo_children():
            if isinstance(win, tk.Toplevel):
                enable_window_controls(win)
                title = re.sub(
                    r"\s*[-|:]*\s*STEP\s*\d+(?:\.\d+)*\s*[-|:]*\s*",
                    " ",
                    win.title(),
                    flags=re.IGNORECASE
                ).strip(" -|:")
                if title:
                    win.title(title)
                if win.state() != "zoomed":
                    win.update_idletasks()
                    sw = win.winfo_screenwidth()
                    sh = win.winfo_screenheight()
                    w = min(max(win.winfo_width(), 900), sw - 40)
                    h = min(max(win.winfo_height(), 620), sh - 80)
                    x = max(20, min(win.winfo_x(), sw - w - 20))
                    y = max(20, min(win.winfo_y(), sh - h - 40))
                    win.geometry(f"{w}x{h}+{x}+{y}")
    except Exception:
        pass
    root.after(500, enhance_all_windows)


# =========================================================
# CLEAN WHITE UI — LAYOUT ENHANCEMENT
# =========================================================

def enhance_all_windows():
    """Keep secondary windows resizable, centered and free of STEP/branding text."""
    if root is None:
        return
    try:
        for win in root.winfo_children():
            if not isinstance(win, tk.Toplevel):
                continue
            try:
                win.resizable(True, True)
                win.minsize(780, 540)
                win.bind("<Control-Shift-F>", lambda e, w=win: toggle_maximize(w))
                win.bind("<Escape>", lambda e, w=win: restore_window(w))
            except Exception:
                pass

            title = re.sub(
                r"\s*[-|:]*\s*STEP\s*\d+(?:\.\d+)*\s*[-|:]*\s*",
                " ", win.title(), flags=re.IGNORECASE
            ).strip(" -|:")
            if title:
                win.title(title)

            try:
                if win.state() != "zoomed":
                    win.update_idletasks()
                    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
                    w = min(max(win.winfo_width(), 900), sw - 50)
                    h = min(max(win.winfo_height(), 620), sh - 90)
                    x = max(20, min(win.winfo_x(), sw - w - 20))
                    y = max(20, min(win.winfo_y(), sh - h - 50))
                    win.geometry(f"{w}x{h}+{x}+{y}")
            except Exception:
                pass
    except Exception:
        pass
    try:
        root.after(700, enhance_all_windows)
    except Exception:
        pass


# =========================================================
# STEP 5 — SUSPICIOUS PATTERN DETECTION
# =========================================================

def detect_suspicious_patterns():
    """Generate explainable investigator-assistance patterns."""
    graph = _person_network_graph()
    patterns = []

    if not graph:
        return patterns

    # Pattern A: multiple persons connected to the same non-person entity.
    entity_people = {}
    for person, neighbors in graph.items():
        if str(person[0]).upper() != "PERSON":
            continue
        for entity in neighbors:
            if str(entity[0]).upper() == "PERSON":
                continue
            entity_people.setdefault(entity, set()).add(person)

    for entity, people in entity_people.items():
        count = len(people)
        if count < 3:
            continue

        entity_type = str(entity[0]).upper()
        label = get_entity_label(entity[0], entity[1])
        severity = "HIGH" if count >= 5 else "MEDIUM"

        patterns.append({
            "type": f"SHARED {entity_type}",
            "severity": severity,
            "score": min(95, 45 + count * 10),
            "title": f"{count} persons linked to same {entity_type.lower()}",
            "details": f"{label} is connected to {count} person entities.",
            "persons": [get_entity_label(p[0], p[1]) for p in people],
        })

    # Pattern B: unusually high connectivity.
    person_nodes = [
        n for n in graph
        if str(n[0]).upper() == "PERSON"
    ]

    degrees = {
        n: len(graph.get(n, set()))
        for n in person_nodes
    }

    if degrees:
        values = sorted(degrees.values())
        median = values[len(values) // 2]
        threshold = max(8, median * 2)

        for node, degree in degrees.items():
            if degree >= threshold:
                patterns.append({
                    "type": "HIGH CONNECTIVITY",
                    "severity": "HIGH" if degree >= threshold * 1.5 else "MEDIUM",
                    "score": min(99, 55 + degree * 3),
                    "title": "Unusually high network connectivity",
                    "details": (
                        f"{get_entity_label(node[0], node[1])} has "
                        f"{degree} direct network connections."
                    ),
                    "persons": [get_entity_label(node[0], node[1])],
                })

    # Pattern C: two persons sharing multiple non-person entities.
    for i, p1 in enumerate(person_nodes):
        for p2 in person_nodes[i + 1:]:
            shared = (
                graph.get(p1, set()) &
                graph.get(p2, set())
            )
            shared = [
                x for x in shared
                if str(x[0]).upper() != "PERSON"
            ]

            if len(shared) >= 2:
                n1 = get_entity_label(p1[0], p1[1])
                n2 = get_entity_label(p2[0], p2[1])
                patterns.append({
                    "type": "MULTIPLE SHARED ENTITIES",
                    "severity": "HIGH" if len(shared) >= 3 else "MEDIUM",
                    "score": min(92, 50 + len(shared) * 12),
                    "title": "Persons share multiple network entities",
                    "details": (
                        f"{n1} and {n2} share "
                        f"{len(shared)} linked entities."
                    ),
                    "persons": [n1, n2],
                })

    patterns.sort(
        key=lambda x: (
            {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x["severity"], 3),
            -x["score"]
        )
    )
    return patterns


def get_suspicious_pattern_summary(limit=8):
    patterns = detect_suspicious_patterns()

    if not patterns:
        return "No suspicious network patterns detected from current data."

    lines = ["SUSPICIOUS PATTERNS", ""]
    for i, pattern in enumerate(patterns[:limit], 1):
        lines.extend([
            f"{i}. [{pattern['severity']}] {pattern['type']} "
            f"— Score {pattern['score']}/100",
            f"   {pattern['title']}",
            f"   {pattern['details']}",
            "",
        ])

    return "\n".join(lines)


# =========================================================
# STEP 6 — RISK / THREAT SCORING
# =========================================================

def calculate_person_risk(person_node):
    """
    Calculate an explainable 0-100 network risk indicator.

    IMPORTANT:
    This is an investigative prioritization score only.
    It does not establish guilt or criminality.
    """
    graph = _person_network_graph()
    if person_node not in graph:
        return {
            "person": get_entity_label(person_node[0], person_node[1]),
            "score": 0,
            "level": "LOW",
            "factors": [],
        }

    neighbors = graph.get(person_node, set())
    connections = len(neighbors)

    # Network influence.
    influencers = calculate_network_influencers(limit=max(10, len(graph)))
    influence_row = next(
        (
            r for r in influencers
            if str(r["person_id"]).upper() == str(person_node[1]).upper()
        ),
        None
    )
    influence = influence_row["influence_score"] if influence_row else 0

    # Suspicious patterns involving this person.
    patterns = detect_suspicious_patterns()
    related_patterns = [
        p for p in patterns
        if get_entity_label(person_node[0], person_node[1]) in p.get("persons", [])
    ]

    high_patterns = sum(
        1 for p in related_patterns
        if p.get("severity") == "HIGH"
    )
    medium_patterns = sum(
        1 for p in related_patterns
        if p.get("severity") == "MEDIUM"
    )

    # Shared non-person entities.
    shared_entity_count = sum(
        1 for n in neighbors
        if str(n[0]).upper() != "PERSON"
    )

    # Normalize transparent components.
    connection_component = min(25, connections * 2)
    influence_component = round(influence * 0.25)
    pattern_component = min(
        30,
        high_patterns * 10 + medium_patterns * 5
    )
    shared_component = min(20, shared_entity_count * 4)

    score = min(
        100,
        connection_component +
        influence_component +
        pattern_component +
        shared_component
    )

    if score >= 75:
        level = "HIGH"
    elif score >= 45:
        level = "MEDIUM"
    else:
        level = "LOW"

    factors = [
        f"{connections} network connections",
        f"{shared_entity_count} linked non-person entities",
        f"influence score {influence}/100",
        f"{len(related_patterns)} suspicious pattern(s)",
    ]

    if high_patterns:
        factors.append(f"{high_patterns} high-severity pattern(s)")
    if medium_patterns:
        factors.append(f"{medium_patterns} medium-severity pattern(s)")

    return {
        "person": get_entity_label(person_node[0], person_node[1]),
        "person_id": str(person_node[1]),
        "score": int(score),
        "level": level,
        "factors": factors,
        "connections": connections,
        "influence": influence,
        "suspicious_patterns": len(related_patterns),
        "high_patterns": high_patterns,
        "medium_patterns": medium_patterns,
    }


def calculate_all_person_risks():
    graph = _person_network_graph()
    people = [
        node for node in graph
        if str(node[0]).upper() == "PERSON"
    ]

    rows = [
        calculate_person_risk(person)
        for person in people
    ]

    rows.sort(
        key=lambda x: (-x["score"], x["person"].lower())
    )
    return rows


def get_risk_summary(limit=10):
    rows = calculate_all_person_risks()

    if not rows:
        return "No person network data available for risk scoring."

    lines = ["RISK / THREAT PRIORITIZATION", ""]
    for i, row in enumerate(rows[:limit], 1):
        lines.append(
            f"{i}. {row['person']} — "
            f"{row['score']}/100 — {row['level']}"
        )
        lines.append(
            f"   Connections: {row['connections']} | "
            f"Influence: {row['influence']}/100 | "
            f"Patterns: {row['suspicious_patterns']}"
        )
        lines.append(
            "   Factors: " + "; ".join(row["factors"])
        )
        lines.append("")

    return "\n".join(lines)


def get_person_risk_by_id(person_id):
    """Convenience lookup for profile/search screens."""
    graph = _person_network_graph()
    for node in graph:
        if (
            str(node[0]).upper() == "PERSON" and
            str(node[1]).upper() == str(person_id).upper()
        ):
            return calculate_person_risk(node)
    return None


# =========================================================
# STEP 8 — INTELLIGENCE DASHBOARD
# =========================================================

def get_intelligence_dashboard_data():
    """Collect existing analytics into one dashboard-safe structure."""
    try:
        graph = _person_network_graph()
    except Exception:
        graph = {}

    people = [
        n for n in graph
        if str(n[0]).upper() == "PERSON"
    ]

    try:
        influencers = calculate_network_influencers(
            limit=max(5, len(people))
        )
    except Exception:
        influencers = []

    try:
        patterns = detect_suspicious_patterns()
    except Exception:
        patterns = []

    try:
        anomalies = detect_network_anomalies()
    except Exception:
        anomalies = []

    try:
        risks = calculate_all_person_risks()
    except Exception:
        risks = []

    high_risk = sum(1 for r in risks if r["level"] == "HIGH")
    medium_risk = sum(1 for r in risks if r["level"] == "MEDIUM")
    low_risk = sum(1 for r in risks if r["level"] == "LOW")

    high_anomaly = sum(
        1 for a in anomalies
        if a["anomaly_level"] == "HIGH"
    )

    return {
        "people": len(people),
        "relationships": sum(len(v) for v in graph.values()) // 2,
        "influencers": influencers[:5],
        "patterns": patterns[:8],
        "anomalies": anomalies[:8],
        "risks": risks[:8],
        "risk_counts": {
            "high": high_risk,
            "medium": medium_risk,
            "low": low_risk,
        },
        "high_anomaly": high_anomaly,
    }


def open_intelligence_dashboard():
    """Open the unified investigator intelligence dashboard."""
    win = tk.Toplevel(root)
    win.title("Intelligence Dashboard")
    win.geometry("1250x760")
    win.minsize(900, 600)
    win.resizable(True, True)

    outer = tk.Frame(
        win,
        bg=THEME.get("bg", "#F5F7FA")
    )
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer,
        bg=THEME.get("surface", "#FFFFFF"),
        highlightthickness=1,
        highlightbackground=THEME.get("border", "#E2E8F0")
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="CRIMINAL INTELLIGENCE",
        font=("Segoe UI", 18, "bold"),
        bg=THEME.get("surface", "#FFFFFF"),
        fg=THEME.get("text", "#111827"),
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text="Network analysis • risk prioritization • suspicious-pattern detection • ML anomalies",
        font=("Segoe UI", 9),
        bg=THEME.get("surface", "#FFFFFF"),
        fg=THEME.get("muted", "#6B7280"),
    ).pack(anchor="w", padx=18, pady=(0, 14))

    body = tk.Frame(
        outer,
        bg=THEME.get("bg", "#F5F7FA")
    )
    body.pack(fill="both", expand=True, padx=16, pady=(0, 16))

    canvas = tk.Canvas(
        body,
        bg=THEME.get("bg", "#F5F7FA"),
        highlightthickness=0
    )
    scrollbar = ttk.Scrollbar(
        body,
        orient="vertical",
        command=canvas.yview
    )
    content = tk.Frame(
        canvas,
        bg=THEME.get("bg", "#F5F7FA")
    )

    content.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas_window = canvas.create_window(
        (0, 0),
        window=content,
        anchor="nw"
    )

    def resize_content(event):
        canvas.itemconfigure(
            canvas_window,
            width=event.width
        )

    canvas.bind("<Configure>", resize_content)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def card(parent, title, value, subtitle=""):
        frame = tk.Frame(
            parent,
            bg=THEME.get("surface", "#FFFFFF"),
            highlightthickness=1,
            highlightbackground=THEME.get("border", "#E2E8F0"),
        )
        frame.pack(side="left", fill="both", expand=True, padx=5)

        tk.Label(
            frame,
            text=title,
            font=("Segoe UI", 9, "bold"),
            bg=THEME.get("surface", "#FFFFFF"),
            fg=THEME.get("muted", "#6B7280"),
        ).pack(anchor="w", padx=14, pady=(12, 2))

        tk.Label(
            frame,
            text=str(value),
            font=("Segoe UI", 23, "bold"),
            bg=THEME.get("surface", "#FFFFFF"),
            fg=THEME.get("text", "#111827"),
        ).pack(anchor="w", padx=14)

        if subtitle:
            tk.Label(
                frame,
                text=subtitle,
                font=("Segoe UI", 8),
                bg=THEME.get("surface", "#FFFFFF"),
                fg=THEME.get("muted", "#6B7280"),
            ).pack(anchor="w", padx=14, pady=(0, 12))

        return frame

    def section(parent, title):
        frame = tk.Frame(
            parent,
            bg=THEME.get("surface", "#FFFFFF"),
            highlightthickness=1,
            highlightbackground=THEME.get("border", "#E2E8F0"),
        )
        frame.pack(fill="x", pady=7)

        tk.Label(
            frame,
            text=title,
            font=("Segoe UI", 11, "bold"),
            bg=THEME.get("surface", "#FFFFFF"),
            fg=THEME.get("text", "#111827"),
        ).pack(anchor="w", padx=14, pady=(12, 7))

        return frame

    data = get_intelligence_dashboard_data()

    stats = tk.Frame(
        content,
        bg=THEME.get("bg", "#F5F7FA")
    )
    stats.pack(fill="x", pady=(0, 7))

    card(stats, "PERSONS", data["people"], "network person nodes")
    card(stats, "RELATIONSHIPS", data["relationships"], "detected network links")
    card(stats, "HIGH RISK", data["risk_counts"]["high"], "priority persons")
    card(stats, "HIGH ANOMALIES", data["high_anomaly"], "ML/statistical signals")

    # Key individuals.
    sec = section(content, "KEY INDIVIDUALS / NETWORK INFLUENCE")

    if not data["influencers"]:
        tk.Label(
            sec,
            text="No person network data available.",
            bg=THEME.get("surface", "#FFFFFF"),
            fg=THEME.get("muted", "#6B7280"),
        ).pack(anchor="w", padx=14, pady=(0, 14))
    else:
        tree = ttk.Treeview(
            sec,
            columns=("rank", "person", "connections", "influence", "role"),
            show="headings",
            height=min(6, max(1, len(data["influencers"]))),
        )

        headings = {
            "rank": "#",
            "person": "Person",
            "connections": "Connections",
            "influence": "Influence",
            "role": "Role",
        }

        for col, heading in headings.items():
            tree.heading(col, text=heading)
            tree.column(col, anchor="w", width=120)

        for i, row in enumerate(data["influencers"], 1):
            tree.insert(
                "",
                "end",
                values=(
                    i,
                    row["name"],
                    row["connections"],
                    f"{row['influence_score']}/100",
                    row["role"],
                )
            )

        tree.pack(fill="x", padx=14, pady=(0, 14))

    # Risk prioritization.
    sec = section(content, "RISK PRIORITIZATION")

    if not data["risks"]:
        tk.Label(
            sec,
            text="No person risk data available.",
            bg=THEME.get("surface", "#FFFFFF"),
            fg=THEME.get("muted", "#6B7280"),
        ).pack(anchor="w", padx=14, pady=(0, 14))
    else:
        risk_text = tk.Text(
            sec,
            height=min(12, max(4, len(data["risks"]) * 2)),
            wrap="word",
            font=("Consolas", 9),
            bg=THEME.get("surface2", "#F8FAFC"),
            fg=THEME.get("text", "#111827"),
            relief="flat",
            padx=12,
            pady=10,
        )
        risk_text.pack(fill="x", padx=14, pady=(0, 14))

        for i, row in enumerate(data["risks"], 1):
            risk_text.insert(
                "end",
                f"{i:02d}. {row['person']}   "
                f"{row['score']:>3}/100   {row['level']}\n"
            )
            risk_text.insert(
                "end",
                "    " + " • ".join(row["factors"]) + "\n\n"
            )

        risk_text.configure(state="disabled")

    # Suspicious patterns.
    sec = section(content, "SUSPICIOUS PATTERNS")

    if not data["patterns"]:
        tk.Label(
            sec,
            text="No suspicious network patterns detected.",
            bg=THEME.get("surface", "#FFFFFF"),
            fg=THEME.get("muted", "#6B7280"),
        ).pack(anchor="w", padx=14, pady=(0, 14))
    else:
        for pattern in data["patterns"]:
            row = tk.Frame(
                sec,
                bg=THEME.get("surface2", "#F8FAFC"),
                highlightthickness=1,
                highlightbackground=THEME.get("border", "#E2E8F0"),
            )
            row.pack(fill="x", padx=14, pady=4)

            tk.Label(
                row,
                text=f"{pattern['severity']}  •  {pattern['score']}/100",
                font=("Segoe UI", 9, "bold"),
                bg=THEME.get("surface2", "#F8FAFC"),
                fg=THEME.get("text", "#111827"),
                width=20,
                anchor="w",
            ).pack(side="left", padx=10, pady=9)

            tk.Label(
                row,
                text=f"{pattern['type']} — {pattern['details']}",
                font=("Segoe UI", 9),
                bg=THEME.get("surface2", "#F8FAFC"),
                fg=THEME.get("text", "#111827"),
                anchor="w",
                justify="left",
            ).pack(side="left", fill="x", expand=True, padx=5, pady=9)

    # ML anomalies.
    sec = section(content, "ML / NETWORK ANOMALIES")

    if not data["anomalies"]:
        tk.Label(
            sec,
            text="Not enough data for anomaly detection.",
            bg=THEME.get("surface", "#FFFFFF"),
            fg=THEME.get("muted", "#6B7280"),
        ).pack(anchor="w", padx=14, pady=(0, 14))
    else:
        for anomaly in data["anomalies"]:
            row = tk.Frame(
                sec,
                bg=THEME.get("surface2", "#F8FAFC"),
            )
            row.pack(fill="x", padx=14, pady=3)

            tk.Label(
                row,
                text=(
                    f"{anomaly['person']}  |  "
                    f"{anomaly['anomaly_score']}/100  |  "
                    f"{anomaly['anomaly_level']}"
                ),
                font=("Segoe UI", 9, "bold"),
                bg=THEME.get("surface2", "#F8FAFC"),
                fg=THEME.get("text", "#111827"),
            ).pack(side="left", padx=10, pady=8)

            reasons = ", ".join(anomaly["reasons"]) or "No strong deviation"
            tk.Label(
                row,
                text=f"Signals: {reasons}",
                font=("Segoe UI", 9),
                bg=THEME.get("surface2", "#F8FAFC"),
                fg=THEME.get("muted", "#6B7280"),
            ).pack(side="left", padx=8, pady=8)

    tk.Label(
        content,
        text="Analytical scores are prioritization signals and do not establish guilt or criminality.",
        font=("Segoe UI", 8, "italic"),
        bg=THEME.get("bg", "#F5F7FA"),
        fg=THEME.get("muted", "#6B7280"),
    ).pack(anchor="w", pady=(8, 18))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    return win


# =========================================================
# STEP 7 — ML / ANOMALY DETECTION
# =========================================================

def build_person_feature_matrix():
    """
    Build a dependency-free feature matrix from the current network.

    Features:
      1. total connections
      2. person-to-person connections
      3. linked non-person entities
      4. network influence score
      5. suspicious-pattern count
      6. high-severity pattern count
    """
    graph = _person_network_graph()
    people = [
        n for n in graph
        if str(n[0]).upper() == "PERSON"
    ]

    influencers = {
        str(r["person_id"]).upper(): r
        for r in calculate_network_influencers(
            limit=max(10, len(people))
        )
    }

    patterns = detect_suspicious_patterns()
    rows = []

    for node in people:
        label = get_entity_label(node[0], node[1])
        neighbors = graph.get(node, set())

        person_links = sum(
            1 for n in neighbors
            if str(n[0]).upper() == "PERSON"
        )
        entity_links = sum(
            1 for n in neighbors
            if str(n[0]).upper() != "PERSON"
        )

        related = [
            p for p in patterns
            if label in p.get("persons", [])
        ]

        high = sum(
            1 for p in related
            if p.get("severity") == "HIGH"
        )

        influence = influencers.get(
            str(node[1]).upper(), {}
        ).get("influence_score", 0)

        rows.append({
            "person_id": str(node[1]),
            "person": label,
            "connections": len(neighbors),
            "person_links": person_links,
            "entity_links": entity_links,
            "influence": float(influence),
            "patterns": len(related),
            "high_patterns": high,
        })

    return rows


def detect_network_anomalies():
    """
    Dependency-free statistical anomaly detector.

    Uses robust median/MAD-style deviation per feature and combines
    the signals into an explainable anomaly score.
    """
    rows = build_person_feature_matrix()
    if len(rows) < 3:
        return []

    features = [
        "connections",
        "person_links",
        "entity_links",
        "influence",
        "patterns",
        "high_patterns",
    ]

    medians = {}
    scales = {}

    for feature in features:
        values = sorted(
            float(r[feature]) for r in rows
        )
        mid = len(values) // 2
        if len(values) % 2:
            median = values[mid]
        else:
            median = (values[mid - 1] + values[mid]) / 2.0

        deviations = sorted(
            abs(v - median) for v in values
        )
        dmid = len(deviations) // 2
        if len(deviations) % 2:
            mad = deviations[dmid]
        else:
            mad = (
                deviations[dmid - 1] +
                deviations[dmid]
            ) / 2.0

        medians[feature] = median
        scales[feature] = max(mad, 1.0)

    results = []

    for row in rows:
        signals = {}

        for feature in features:
            deviation = abs(
                float(row[feature]) - medians[feature]
            )
            # Robust normalized deviation.
            signals[feature] = min(
                100.0,
                (deviation / (3.0 * scales[feature])) * 100.0
            )

        # Strongest signals receive more weight.
        score = round(
            signals["connections"] * 0.25 +
            signals["person_links"] * 0.15 +
            signals["entity_links"] * 0.20 +
            signals["influence"] * 0.15 +
            signals["patterns"] * 0.15 +
            signals["high_patterns"] * 0.10
        )

        if score >= 70:
            level = "HIGH"
        elif score >= 45:
            level = "MEDIUM"
        else:
            level = "LOW"

        top_signals = sorted(
            signals.items(),
            key=lambda x: -x[1]
        )[:3]

        reasons = []
        for feature, signal in top_signals:
            if signal >= 35:
                reasons.append(
                    feature.replace("_", " ").title()
                )

        results.append({
            **row,
            "anomaly_score": score,
            "anomaly_level": level,
            "reasons": reasons,
        })

    results.sort(
        key=lambda x: (
            -x["anomaly_score"],
            x["person"].lower()
        )
    )
    return results


def get_anomaly_summary(limit=10):
    rows = detect_network_anomalies()

    if not rows:
        return (
            "Not enough network data for statistical anomaly "
            "detection (minimum: 3 person nodes)."
        )

    lines = ["ML / NETWORK ANOMALIES", ""]

    for i, row in enumerate(rows[:limit], 1):
        reason = ", ".join(row["reasons"]) or "No strong deviation"
        lines.extend([
            f"{i}. {row['person']} — "
            f"{row['anomaly_score']}/100 — "
            f"{row['anomaly_level']}",
            f"   Reasons: {reason}",
            f"   Connections: {row['connections']} | "
            f"Influence: {row['influence']}/100 | "
            f"Patterns: {row['patterns']}",
            "",
        ])

    return "\n".join(lines)


def get_person_anomaly(person_id):
    for row in detect_network_anomalies():
        if str(row["person_id"]).upper() == str(person_id).upper():
            return row
    return None


# =========================================================
# STEP 9 — PERSON INTELLIGENCE PROFILE
# =========================================================

def get_person_profile_data(person_id):
    """Collect available database + network intelligence for one person."""
    profile = {
        "person_id": str(person_id),
        "name": str(person_id),
        "details": {},
        "connections": [],
        "influence": None,
        "risk": None,
        "anomaly": None,
        "patterns": [],
    }

    # Basic person record — adapt to whichever columns exist.
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(persons)")
        columns = [row[1] for row in cur.fetchall()]

        if "person_id" in columns:
            cur.execute(
                "SELECT * FROM persons WHERE person_id = ? LIMIT 1",
                (person_id,)
            )
            row = cur.fetchone()
            if row:
                profile["details"] = dict(zip(columns, row))

                for key in ("name", "full_name", "person_name"):
                    if key in profile["details"] and profile["details"][key]:
                        profile["name"] = str(profile["details"][key])
                        break

        conn.close()
    except Exception:
        pass

    graph = _person_network_graph()
    node = None

    for candidate in graph:
        if (
            str(candidate[0]).upper() == "PERSON" and
            str(candidate[1]).upper() == str(person_id).upper()
        ):
            node = candidate
            break

    if node is not None:
        for neighbor in graph.get(node, set()):
            profile["connections"].append({
                "type": str(neighbor[0]),
                "id": str(neighbor[1]),
                "label": get_entity_label(neighbor[0], neighbor[1]),
            })

        profile["connections"].sort(
            key=lambda x: (x["type"], x["label"].lower())
        )

        try:
            profile["risk"] = calculate_person_risk(node)
        except Exception:
            pass

        try:
            profile["anomaly"] = get_person_anomaly(person_id)
        except Exception:
            pass

        try:
            all_influencers = calculate_network_influencers(
                limit=max(10, len(graph))
            )
            profile["influence"] = next(
                (
                    r for r in all_influencers
                    if str(r["person_id"]).upper()
                    == str(person_id).upper()
                ),
                None
            )
        except Exception:
            pass

        try:
            label = get_entity_label(node[0], node[1])
            profile["patterns"] = [
                p for p in detect_suspicious_patterns()
                if label in p.get("persons", [])
            ]
        except Exception:
            pass

    return profile


def open_person_intelligence_profile(person_id=None):
    """
    Open a complete investigator profile.
    If no ID is supplied, ask for a Person ID.
    """
    if person_id is None:
        from tkinter import simpledialog
        person_id = simpledialog.askstring(
            "Person Intelligence",
            "Enter Person ID:",
            parent=root
        )

    if not person_id:
        return None

    profile = get_person_profile_data(person_id)

    win = tk.Toplevel(root)
    win.title("Person Intelligence Profile")
    win.geometry("1100x720")
    win.minsize(850, 560)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text=profile["name"],
        font=("Segoe UI", 20, "bold"),
        bg=surface,
        fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text=f"Person ID: {profile['person_id']}",
        font=("Segoe UI", 9),
        bg=surface,
        fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    content = tk.Frame(outer, bg=bg)
    content.pack(fill="both", expand=True, padx=16, pady=(0, 16))

    canvas = tk.Canvas(content, bg=bg, highlightthickness=0)
    scrollbar = ttk.Scrollbar(
        content, orient="vertical", command=canvas.yview
    )
    body = tk.Frame(canvas, bg=bg)

    body.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas_id = canvas.create_window((0, 0), window=body, anchor="nw")

    canvas.bind(
        "<Configure>",
        lambda e: canvas.itemconfigure(canvas_id, width=e.width)
    )
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def make_section(title):
        frame = tk.Frame(
            body, bg=surface,
            highlightthickness=1,
            highlightbackground=border
        )
        frame.pack(fill="x", pady=6)

        tk.Label(
            frame,
            text=title,
            font=("Segoe UI", 11, "bold"),
            bg=surface,
            fg=text_color
        ).pack(anchor="w", padx=14, pady=(12, 8))

        return frame

    # Summary cards.
    cards = tk.Frame(body, bg=bg)
    cards.pack(fill="x", pady=(0, 6))

    risk = profile["risk"] or {}
    influence = profile["influence"] or {}
    anomaly = profile["anomaly"] or {}

    metrics = [
        ("RISK", f"{risk.get('score', 0)}/100",
         risk.get("level", "LOW")),
        ("INFLUENCE", f"{influence.get('influence_score', 0)}/100",
         influence.get("role", "—")),
        ("ANOMALY", f"{anomaly.get('anomaly_score', 0)}/100",
         anomaly.get("anomaly_level", "LOW")),
        ("CONNECTIONS", str(len(profile["connections"])),
         "network links"),
    ]

    for title, value, sub in metrics:
        box = tk.Frame(
            cards, bg=surface,
            highlightthickness=1,
            highlightbackground=border
        )
        box.pack(side="left", fill="both", expand=True, padx=4)

        tk.Label(
            box, text=title,
            font=("Segoe UI", 8, "bold"),
            bg=surface, fg=muted
        ).pack(anchor="w", padx=12, pady=(10, 2))

        tk.Label(
            box, text=value,
            font=("Segoe UI", 18, "bold"),
            bg=surface, fg=text_color
        ).pack(anchor="w", padx=12)

        tk.Label(
            box, text=sub,
            font=("Segoe UI", 8),
            bg=surface, fg=muted
        ).pack(anchor="w", padx=12, pady=(0, 10))

    # Database details.
    details_section = make_section("BASIC INFORMATION")

    details = profile["details"]
    if details:
        for key, value in details.items():
            if value in (None, ""):
                continue

            row = tk.Frame(details_section, bg=surface)
            row.pack(fill="x", padx=14, pady=2)

            tk.Label(
                row,
                text=str(key).replace("_", " ").title(),
                width=22,
                anchor="w",
                font=("Segoe UI", 9, "bold"),
                bg=surface,
                fg=muted
            ).pack(side="left")

            tk.Label(
                row,
                text=str(value),
                anchor="w",
                justify="left",
                font=("Segoe UI", 9),
                bg=surface,
                fg=text_color
            ).pack(side="left", fill="x", expand=True)

        tk.Frame(details_section, bg=surface, height=8).pack()
    else:
        tk.Label(
            details_section,
            text="No additional database details available.",
            bg=surface, fg=muted
        ).pack(anchor="w", padx=14, pady=(0, 14))

    # Network connections.
    conn_section = make_section("NETWORK CONNECTIONS")

    if profile["connections"]:
        tree = ttk.Treeview(
            conn_section,
            columns=("type", "entity"),
            show="headings",
            height=min(10, len(profile["connections"]))
        )
        tree.heading("type", text="TYPE")
        tree.heading("entity", text="CONNECTED ENTITY")
        tree.column("type", width=160, anchor="w")
        tree.column("entity", width=620, anchor="w")

        for item in profile["connections"]:
            tree.insert(
                "", "end",
                values=(item["type"], item["label"])
            )

        tree.pack(fill="x", padx=14, pady=(0, 14))
    else:
        tk.Label(
            conn_section,
            text="No network connections found.",
            bg=surface, fg=muted
        ).pack(anchor="w", padx=14, pady=(0, 14))

    # Risk explanation.
    risk_section = make_section("RISK EXPLANATION")
    if risk:
        tk.Label(
            risk_section,
            text=f"Risk Level: {risk.get('level', 'LOW')}   "
                 f"Score: {risk.get('score', 0)}/100",
            font=("Segoe UI", 10, "bold"),
            bg=surface, fg=text_color
        ).pack(anchor="w", padx=14)

        for factor in risk.get("factors", []):
            tk.Label(
                risk_section,
                text=f"• {factor}",
                font=("Segoe UI", 9),
                bg=surface, fg=text_color
            ).pack(anchor="w", padx=22, pady=1)

        tk.Frame(risk_section, bg=surface, height=10).pack()
    else:
        tk.Label(
            risk_section,
            text="Risk score unavailable for this person.",
            bg=surface, fg=muted
        ).pack(anchor="w", padx=14, pady=(0, 14))

    # Suspicious patterns.
    pattern_section = make_section("SUSPICIOUS PATTERNS")

    if profile["patterns"]:
        for p in profile["patterns"]:
            tk.Label(
                pattern_section,
                text=(
                    f"[{p['severity']}] {p['type']} — "
                    f"{p['score']}/100\n"
                    f"  {p['details']}"
                ),
                justify="left",
                anchor="w",
                font=("Segoe UI", 9),
                bg=surface, fg=text_color
            ).pack(fill="x", padx=14, pady=4)
    else:
        tk.Label(
            pattern_section,
            text="No suspicious patterns currently linked to this person.",
            bg=surface, fg=muted
        ).pack(anchor="w", padx=14, pady=(0, 14))

    tk.Label(
        body,
        text=(
            "Scores are analytical prioritization signals and do not "
            "establish guilt or criminality."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", pady=(8, 14))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    return win


# =========================================================
# STEP 10 — EXPLAINABLE INTELLIGENCE + INVESTIGATION TIMELINE
# =========================================================

def get_person_evidence_explanation(person_id):
    """
    Convert existing network signals into a human-readable explanation.
    No new evidence is invented; explanations are derived from current data.
    """
    profile = get_person_profile_data(person_id)
    risk = profile.get("risk") or {}
    influence = profile.get("influence") or {}
    anomaly = profile.get("anomaly") or {}
    patterns = profile.get("patterns") or []

    explanation = {
        "person": profile.get("name", str(person_id)),
        "person_id": str(person_id),
        "summary": [],
        "evidence": [],
        "timeline": [],
    }

    explanation["summary"].append(
        f"Network connections: {len(profile.get('connections', []))}"
    )

    if influence:
        explanation["summary"].append(
            f"Influence score: {influence.get('influence_score', 0)}/100"
        )
        explanation["evidence"].append({
            "category": "NETWORK INFLUENCE",
            "finding": (
                f"{influence.get('role', 'Connected person')} with "
                f"{influence.get('connections', 0)} direct connections."
            ),
            "score": influence.get("influence_score", 0),
        })

    if risk:
        explanation["summary"].append(
            f"Risk prioritization: {risk.get('score', 0)}/100 "
            f"({risk.get('level', 'LOW')})"
        )
        for factor in risk.get("factors", []):
            explanation["evidence"].append({
                "category": "RISK FACTOR",
                "finding": factor,
                "score": risk.get("score", 0),
            })

    if anomaly:
        explanation["summary"].append(
            f"Network anomaly: {anomaly.get('anomaly_score', 0)}/100 "
            f"({anomaly.get('anomaly_level', 'LOW')})"
        )
        for reason in anomaly.get("reasons", []):
            explanation["evidence"].append({
                "category": "ANOMALY SIGNAL",
                "finding": reason,
                "score": anomaly.get("anomaly_score", 0),
            })

    for pattern in patterns:
        explanation["evidence"].append({
            "category": pattern.get("type", "PATTERN"),
            "finding": pattern.get("details", ""),
            "score": pattern.get("score", 0),
        })

    # Timeline is constructed only from currently available records.
    # If a dated event table exists, use it; otherwise show analysis events.
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name IN "
            "('firs','transactions','cdrs','events','reports')"
        )
        tables = {row[0] for row in cur.fetchall()}
        conn.close()
    except Exception:
        tables = set()

    for table in sorted(tables):
        explanation["timeline"].append({
            "date": "AVAILABLE RECORD",
            "source": table.upper(),
            "event": f"Records available in {table} for investigation review.",
        })

    # Always provide analysis milestones so the investigator can see
    # how the current score was derived.
    explanation["timeline"].extend([
        {
            "date": "CURRENT ANALYSIS",
            "source": "NETWORK",
            "event": "Relationship graph evaluated.",
        },
        {
            "date": "CURRENT ANALYSIS",
            "source": "INFLUENCE",
            "event": "Network influence calculated.",
        },
        {
            "date": "CURRENT ANALYSIS",
            "source": "PATTERN ENGINE",
            "event": "Suspicious network patterns evaluated.",
        },
        {
            "date": "CURRENT ANALYSIS",
            "source": "ANOMALY ENGINE",
            "event": "Statistical network anomaly signals evaluated.",
        },
        {
            "date": "CURRENT ANALYSIS",
            "source": "RISK ENGINE",
            "event": "Explainable prioritization score calculated.",
        },
    ])

    return explanation


def format_person_explanation(person_id):
    data = get_person_evidence_explanation(person_id)
    lines = [
        "EXPLAINABLE INTELLIGENCE",
        "",
        f"Person: {data['person']}",
        f"Person ID: {data['person_id']}",
        "",
        "WHY THE SYSTEM FLAGGED THIS PERSON",
    ]

    for item in data["summary"]:
        lines.append(f"• {item}")

    lines.extend(["", "EVIDENCE / SIGNALS"])

    if not data["evidence"]:
        lines.append("• No analytical signals available.")
    else:
        for item in data["evidence"]:
            lines.append(
                f"• [{item['category']}] {item['finding']} "
                f"| score {item['score']}/100"
            )

    return "\n".join(lines)


def open_person_explainable_intelligence(person_id=None):
    """Dedicated explainability window for an investigator."""
    if person_id is None:
        from tkinter import simpledialog
        person_id = simpledialog.askstring(
            "Explainable Intelligence",
            "Enter Person ID:",
            parent=root
        )

    if not person_id:
        return None

    data = get_person_evidence_explanation(person_id)

    win = tk.Toplevel(root)
    win.title("Explainable Intelligence")
    win.geometry("1100x720")
    win.minsize(850, 560)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    surface2 = THEME.get("surface2", "#F8FAFC")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="EXPLAINABLE INTELLIGENCE",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text=f"{data['person']}  •  Person ID: {data['person_id']}",
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    body = tk.Frame(outer, bg=bg)
    body.pack(fill="both", expand=True, padx=16, pady=(0, 16))

    canvas = tk.Canvas(body, bg=bg, highlightthickness=0)
    scroll = ttk.Scrollbar(
        body, orient="vertical", command=canvas.yview
    )
    content = tk.Frame(canvas, bg=bg)
    content.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    cid = canvas.create_window((0, 0), window=content, anchor="nw")
    canvas.bind(
        "<Configure>",
        lambda e: canvas.itemconfigure(cid, width=e.width)
    )
    canvas.configure(yscrollcommand=scroll.set)

    canvas.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    def section(title):
        frame = tk.Frame(
            content, bg=surface,
            highlightthickness=1,
            highlightbackground=border
        )
        frame.pack(fill="x", pady=6)
        tk.Label(
            frame, text=title,
            font=("Segoe UI", 11, "bold"),
            bg=surface, fg=text_color
        ).pack(anchor="w", padx=14, pady=(12, 8))
        return frame

    # Why flagged.
    sec = section("WHY THIS PERSON WAS FLAGGED")
    for item in data["summary"]:
        tk.Label(
            sec, text=f"• {item}",
            font=("Segoe UI", 9),
            bg=surface, fg=text_color,
            anchor="w", justify="left"
        ).pack(fill="x", padx=18, pady=2)
    tk.Frame(sec, bg=surface, height=10).pack()

    # Evidence.
    sec = section("EVIDENCE / ANALYTICAL SIGNALS")
    if data["evidence"]:
        for item in data["evidence"]:
            row = tk.Frame(sec, bg=surface2)
            row.pack(fill="x", padx=14, pady=3)

            tk.Label(
                row,
                text=f"{item['category']}  |  {item['score']}/100",
                font=("Segoe UI", 8, "bold"),
                bg=surface2, fg=text_color,
                width=28, anchor="w"
            ).pack(side="left", padx=9, pady=8)

            tk.Label(
                row,
                text=item["finding"],
                font=("Segoe UI", 9),
                bg=surface2, fg=text_color,
                anchor="w", justify="left"
            ).pack(side="left", fill="x", expand=True, padx=5, pady=8)
    else:
        tk.Label(
            sec,
            text="No analytical signals available.",
            bg=surface, fg=muted
        ).pack(anchor="w", padx=14, pady=(0, 14))

    # Timeline.
    sec = section("INVESTIGATION TIMELINE")
    timeline = ttk.Treeview(
        sec,
        columns=("date", "source", "event"),
        show="headings",
        height=min(12, max(5, len(data["timeline"])))
    )
    timeline.heading("date", text="TIME / STAGE")
    timeline.heading("source", text="SOURCE")
    timeline.heading("event", text="EVENT")
    timeline.column("date", width=170, anchor="w")
    timeline.column("source", width=160, anchor="w")
    timeline.column("event", width=600, anchor="w")

    for item in data["timeline"]:
        timeline.insert(
            "", "end",
            values=(
                item["date"],
                item["source"],
                item["event"]
            )
        )

    timeline.pack(fill="x", padx=14, pady=(0, 14))

    tk.Label(
        content,
        text=(
            "Explainability uses currently available database/network signals. "
            "Analytical scores are prioritization aids, not proof of guilt."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", pady=(8, 14))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    return win


# =========================================================
# STEP 11 — DATA IMPORT + ENTITY EXTRACTION
# =========================================================

def _safe_normalize(value):
    return " ".join(str(value or "").strip().split())


def extract_entities_from_text(text_value):
    """
    Lightweight, dependency-free entity extraction for prototype/demo use.
    Extracts phone numbers, vehicle-like identifiers, and common location/
    organization/person labels from structured text.
    """
    import re

    text_value = _safe_normalize(text_value)
    entities = {
        "phones": [],
        "vehicles": [],
        "locations": [],
        "organizations": [],
        "persons": [],
    }

    # Indian/international phone-like patterns.
    for match in re.findall(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)", text_value):
        phone = re.sub(r"\D", "", match)
        if phone.startswith("91") and len(phone) == 12:
            phone = phone[2:]
        if phone not in entities["phones"]:
            entities["phones"].append(phone)

    # Common Indian vehicle registration shape.
    for match in re.findall(
        r"\b[A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,3}[-\s]?\d{1,4}\b",
        text_value.upper()
    ):
        vehicle = re.sub(r"[\s-]+", "-", match)
        if vehicle not in entities["vehicles"]:
            entities["vehicles"].append(vehicle)

    # Simple explicit labels: "Location: Patna", "Organization: XYZ".
    patterns = {
        "locations": r"(?:location|place|address)\s*[:=-]\s*([^,;|]+)",
        "organizations": r"(?:organization|organisation|company|org)\s*[:=-]\s*([^,;|]+)",
        "persons": r"(?:person|name|suspect|individual)\s*[:=-]\s*([^,;|]+)",
    }

    for key, pattern in patterns.items():
        for match in re.findall(pattern, text_value, flags=re.I):
            value = _safe_normalize(match)
            if value and value.lower() not in {
                x.lower() for x in entities[key]
            }:
                entities[key].append(value)

    return entities


def import_structured_file(file_path):
    """
    Import CSV or TXT data into a normalized in-memory record list.
    Existing database schema is not modified automatically.
    """
    import csv
    from pathlib import Path

    path = Path(file_path)
    suffix = path.suffix.lower()
    records = []

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                clean = {
                    str(k).strip(): _safe_normalize(v)
                    for k, v in row.items()
                    if k is not None
                }
                records.append(clean)

    elif suffix in {".txt", ".log"}:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = _safe_normalize(line)
                if line:
                    records.append({"text": line})

    else:
        raise ValueError(
            "Supported prototype import formats: CSV, TXT, LOG"
        )

    return records


def extract_entities_from_records(records):
    """Extract entities from every imported record."""
    result = []

    for index, record in enumerate(records, 1):
        combined = " | ".join(
            str(v) for v in record.values()
            if v not in (None, "")
        )

        entities = extract_entities_from_text(combined)

        # Structured fields get priority.
        field_map = {
            "phone": "phones",
            "phone_number": "phones",
            "mobile": "phones",
            "vehicle": "vehicles",
            "vehicle_number": "vehicles",
            "location": "locations",
            "place": "locations",
            "organization": "organizations",
            "organisation": "organizations",
            "company": "organizations",
            "person": "persons",
            "person_name": "persons",
            "name": "persons",
        }

        for field, target in field_map.items():
            if field in {str(k).lower() for k in record}:
                original_key = next(
                    (k for k in record if str(k).lower() == field),
                    None
                )
                value = _safe_normalize(record.get(original_key))
                if value and value.lower() not in {
                    x.lower() for x in entities[target]
                }:
                    entities[target].append(value)

        result.append({
            "record_number": index,
            "source": record,
            "entities": entities,
        })

    return result


def open_data_import_entity_extraction():
    """Professional import/extraction window."""
    from tkinter import filedialog, messagebox

    win = tk.Toplevel(root)
    win.title("Data Import & Entity Extraction")
    win.geometry("1100x720")
    win.minsize(850, 560)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    surface2 = THEME.get("surface2", "#F8FAFC")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="DATA IMPORT & ENTITY EXTRACTION",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text="Import prototype crime/intelligence records and extract entities",
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=(0, 8))
    toolbar.columnconfigure(0, weight=1)

    status = tk.StringVar(value="No file imported.")

    result_frame = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    result_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

    tree = ttk.Treeview(
        result_frame,
        columns=("record", "persons", "phones", "vehicles",
                 "locations", "organizations"),
        show="headings"
    )

    for col, heading, width in [
        ("record", "Record", 80),
        ("persons", "Persons", 180),
        ("phones", "Phones", 150),
        ("vehicles", "Vehicles", 150),
        ("locations", "Locations", 180),
        ("organizations", "Organizations", 220),
    ]:
        tree.heading(col, text=heading)
        tree.column(col, width=width, anchor="w")

    scroll = ttk.Scrollbar(
        result_frame, orient="vertical", command=tree.yview
    )
    tree.configure(yscrollcommand=scroll.set)
    tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
    scroll.pack(side="right", fill="y", padx=(0, 10), pady=10)

    def import_file():
        path = filedialog.askopenfilename(
            parent=win,
            title="Select intelligence data",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("Log files", "*.log"),
                ("All supported files", "*.csv *.txt *.log"),
            ]
        )
        if not path:
            return

        try:
            records = import_structured_file(path)
            extracted = extract_entities_from_records(records)

            for item in tree.get_children():
                tree.delete(item)

            totals = {
                "persons": set(),
                "phones": set(),
                "vehicles": set(),
                "locations": set(),
                "organizations": set(),
            }

            for item in extracted:
                entities = item["entities"]

                for key in totals:
                    totals[key].update(
                        str(x).lower() for x in entities[key]
                    )

                tree.insert(
                    "", "end",
                    values=(
                        item["record_number"],
                        ", ".join(entities["persons"]) or "—",
                        ", ".join(entities["phones"]) or "—",
                        ", ".join(entities["vehicles"]) or "—",
                        ", ".join(entities["locations"]) or "—",
                        ", ".join(entities["organizations"]) or "—",
                    )
                )

            status.set(
                f"Imported {len(records)} record(s)  •  "
                f"Persons {len(totals['persons'])}  •  "
                f"Phones {len(totals['phones'])}  •  "
                f"Vehicles {len(totals['vehicles'])}  •  "
                f"Locations {len(totals['locations'])}  •  "
                f"Organizations {len(totals['organizations'])}"
            )

        except Exception as exc:
            messagebox.showerror(
                "Import Error",
                f"Could not import the selected file.\n\n{exc}",
                parent=win
            )

    tk.Button(
        toolbar,
        text="IMPORT CSV / TXT",
        command=import_file,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=7
    ).pack(side="left")

    tk.Label(
        toolbar,
        textvariable=status,
        font=("Segoe UI", 9),
        bg=bg, fg=muted
    ).pack(side="left", padx=14)

    tk.Label(
        outer,
        text="Prototype extraction is rule-based; review extracted entities before committing them to the database.",
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    return win


# =========================================================
# STEP 12 — DATABASE SAVE + AUTOMATIC RELATIONSHIPS
# =========================================================

def _table_columns(conn, table_name):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cur.fetchall()]


def _find_first_column(columns, candidates):
    lower = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def save_extracted_entities_to_database(extracted_records):
    """
    Save extracted entities when compatible tables exist.

    The function is schema-aware: it inspects existing SQLite tables and
    only writes to tables/columns that are already present. It does not
    silently alter the existing schema.
    """
    conn = get_connection()
    cur = conn.cursor()

    summary = {
        "persons": 0,
        "phones": 0,
        "vehicles": 0,
        "locations": 0,
        "organizations": 0,
        "relationships": 0,
    }

    table_map = {
        "persons": ["persons", "people", "person"],
        "phones": ["phones", "phone_numbers", "phone"],
        "vehicles": ["vehicles", "vehicle"],
        "locations": ["locations", "places", "location"],
        "organizations": ["organizations", "organisations", "organization"],
        "relationships": [
            "relationships", "relations", "entity_relationships"
        ],
    }

    existing = {}
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    for row in cur.fetchall():
        existing[str(row[0]).lower()] = row[0]

    resolved = {}
    columns = {}
    for key, candidates in table_map.items():
        table = next(
            (existing[c.lower()] for c in candidates if c.lower() in existing),
            None
        )
        resolved[key] = table
        columns[key] = _table_columns(conn, table) if table else []

    def insert_entity(kind, value):
        table = resolved.get(kind)
        cols = columns.get(kind, [])
        if not table or not cols or not value:
            return False

        id_col = _find_first_column(
            cols, ["id", f"{kind[:-1]}_id", "entity_id"]
        )
        name_col = _find_first_column(
            cols,
            ["name", "full_name", "person_name", "phone",
             "phone_number", "number", "vehicle", "vehicle_number",
             "location", "place", "organization", "organization_name"]
        )

        if not name_col:
            return False

        # Avoid duplicates where possible.
        try:
            cur.execute(
                f"SELECT 1 FROM {table} WHERE {name_col} = ? LIMIT 1",
                (value,)
            )
            if cur.fetchone():
                return False
        except Exception:
            pass

        insert_cols = [name_col]
        insert_values = [value]

        # Common optional source column.
        source_col = _find_first_column(
            cols, ["source", "data_source", "source_type"]
        )
        if source_col:
            insert_cols.append(source_col)
            insert_values.append("IMPORTED")

        placeholders = ", ".join("?" for _ in insert_values)
        quoted_cols = ", ".join(insert_cols)

        try:
            cur.execute(
                f"INSERT INTO {table} ({quoted_cols}) "
                f"VALUES ({placeholders})",
                tuple(insert_values)
            )
            return True
        except Exception:
            return False

    # Save unique entities.
    entity_values = {
        "persons": set(),
        "phones": set(),
        "vehicles": set(),
        "locations": set(),
        "organizations": set(),
    }

    for item in extracted_records:
        entities = item.get("entities", {})
        for kind in entity_values:
            for value in entities.get(kind, []):
                clean = _safe_normalize(value)
                if clean:
                    entity_values[kind].add(clean)

    for kind, values in entity_values.items():
        for value in values:
            if insert_entity(kind, value):
                summary[kind] += 1

    conn.commit()
    conn.close()

    # Relationship generation is performed separately so that existing
    # relationship-engine logic remains the single source of truth.
    try:
        relationship_result = run_automatic_relationship_engine()
        if isinstance(relationship_result, dict):
            summary["relationships"] = sum(
                int(v or 0)
                for k, v in relationship_result.items()
                if "link" in str(k).lower()
                or "relationship" in str(k).lower()
            )
    except Exception:
        pass

    return summary


def open_import_save_workflow():
    """Import → extract → review → save → rebuild relationships."""
    from tkinter import filedialog, messagebox

    win = tk.Toplevel(root)
    win.title("Import, Extract & Save")
    win.geometry("1150x750")
    win.minsize(900, 600)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="IMPORT → EXTRACT → SAVE → RELATE",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text="Review extracted entities before writing them to the database.",
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=(0, 8))

    records_holder = {"value": []}
    extracted_holder = {"value": []}
    status = tk.StringVar(value="Select a CSV/TXT/LOG file.")

    table_frame = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    table_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    tree = ttk.Treeview(
        table_frame,
        columns=("record", "persons", "phones", "vehicles",
                 "locations", "organizations"),
        show="headings"
    )

    for col, heading, width in [
        ("record", "Record", 80),
        ("persons", "Persons", 180),
        ("phones", "Phones", 150),
        ("vehicles", "Vehicles", 150),
        ("locations", "Locations", 180),
        ("organizations", "Organizations", 220),
    ]:
        tree.heading(col, text=heading)
        tree.column(col, width=width, anchor="w")

    sy = ttk.Scrollbar(
        table_frame, orient="vertical", command=tree.yview
    )
    tree.configure(yscrollcommand=sy.set)
    tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
    sy.pack(side="right", fill="y", padx=(0, 10), pady=10)

    def choose():
        path = filedialog.askopenfilename(
            parent=win,
            title="Select intelligence data",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("Log files", "*.log"),
                ("Supported files", "*.csv *.txt *.log"),
            ]
        )
        if not path:
            return

        try:
            records = import_structured_file(path)
            extracted = extract_entities_from_records(records)

            records_holder["value"] = records
            extracted_holder["value"] = extracted

            for item in tree.get_children():
                tree.delete(item)

            for item in extracted:
                e = item["entities"]
                tree.insert(
                    "", "end",
                    values=(
                        item["record_number"],
                        ", ".join(e["persons"]) or "—",
                        ", ".join(e["phones"]) or "—",
                        ", ".join(e["vehicles"]) or "—",
                        ", ".join(e["locations"]) or "—",
                        ", ".join(e["organizations"]) or "—",
                    )
                )

            status.set(
                f"{len(records)} record(s) extracted. "
                "Review the table, then click SAVE & BUILD LINKS."
            )
        except Exception as exc:
            messagebox.showerror(
                "Import Error",
                str(exc),
                parent=win
            )

    def save():
        extracted = extracted_holder["value"]
        if not extracted:
            messagebox.showwarning(
                "Nothing to Save",
                "Import and review a file first.",
                parent=win
            )
            return

        if not messagebox.askyesno(
            "Confirm Database Write",
            "Save the extracted entities to the existing database "
            "and rebuild automatic relationships?",
            parent=win
        ):
            return

        try:
            summary = save_extracted_entities_to_database(extracted)

            status.set(
                "Saved: "
                f"Persons {summary['persons']} • "
                f"Phones {summary['phones']} • "
                f"Vehicles {summary['vehicles']} • "
                f"Locations {summary['locations']} • "
                f"Organizations {summary['organizations']}"
            )

            messagebox.showinfo(
                "Save Complete",
                "Entities saved where compatible tables were found.\n\n"
                f"Persons: {summary['persons']}\n"
                f"Phones: {summary['phones']}\n"
                f"Vehicles: {summary['vehicles']}\n"
                f"Locations: {summary['locations']}\n"
                f"Organizations: {summary['organizations']}\n\n"
                "Automatic relationship processing was requested.",
                parent=win
            )
        except Exception as exc:
            messagebox.showerror(
                "Database Error",
                f"Could not save the imported data.\n\n{exc}",
                parent=win
            )

    tk.Button(
        toolbar,
        text="IMPORT",
        command=choose,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=7
    ).pack(side="left", padx=(0, 8))

    tk.Button(
        toolbar,
        text="SAVE & BUILD LINKS",
        command=save,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=7
    ).pack(side="left")

    tk.Label(
        toolbar,
        textvariable=status,
        font=("Segoe UI", 9),
        bg=bg, fg=muted
    ).pack(side="left", padx=14)

    tk.Label(
        outer,
        text=(
            "Only compatible existing tables/columns are written. "
            "Review imported entities before committing them."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    return win


# =========================================================
# STEP 13 — INTERACTIVE RELATIONSHIP GRAPH
# =========================================================

def get_relationship_graph_data():
    """Return the current person/entity graph in a UI-friendly format."""
    graph = _person_network_graph()
    nodes = set(graph.keys())
    edges = set()

    for source, neighbors in graph.items():
        nodes.add(source)
        for target in neighbors:
            nodes.add(target)
            edge = tuple(sorted((source, target), key=lambda x: (str(x[0]), str(x[1]))))
            edges.add(edge)

    node_rows = []
    for node in nodes:
        node_type = str(node[0]).upper()
        node_id = str(node[1])
        label = get_entity_label(node_type, node_id)

        if node_type == "PERSON":
            category = "PERSON"
        elif node_type in {"PHONE", "PHONE_NUMBER"}:
            category = "PHONE"
        elif node_type == "VEHICLE":
            category = "VEHICLE"
        elif node_type in {"LOCATION", "PLACE"}:
            category = "LOCATION"
        elif node_type in {"ORGANIZATION", "ORGANISATION"}:
            category = "ORGANIZATION"
        else:
            category = node_type

        node_rows.append({
            "key": node,
            "type": category,
            "id": node_id,
            "label": label,
            "degree": len(graph.get(node, set())),
        })

    node_rows.sort(key=lambda x: (-x["degree"], x["label"].lower()))
    return node_rows, list(edges)


def open_relationship_graph():
    """Open a dependency-free interactive relationship graph."""
    from tkinter import messagebox

    win = tk.Toplevel(root)
    win.title("Relationship Network Graph")
    win.geometry("1250x780")
    win.minsize(900, 600)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    surface2 = THEME.get("surface2", "#F8FAFC")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="RELATIONSHIP NETWORK GRAPH",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text="Explore people and connected intelligence entities",
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=(0, 8))

    status = tk.StringVar(value="Loading network...")
    selected = {"key": None}

    graph_frame = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    graph_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    canvas = tk.Canvas(
        graph_frame,
        bg=surface,
        highlightthickness=0
    )
    canvas.pack(fill="both", expand=True)

    nodes, edges = get_relationship_graph_data()

    # Limit only the visual canvas when extremely large; underlying graph is untouched.
    display_nodes = nodes[:80]
    display_keys = {n["key"] for n in display_nodes}
    display_edges = [
        e for e in edges
        if e[0] in display_keys and e[1] in display_keys
    ]

    positions = {}
    radius = 28

    def draw_graph():
        canvas.delete("all")

        width = max(canvas.winfo_width(), 800)
        height = max(canvas.winfo_height(), 500)

        if not display_nodes:
            canvas.create_text(
                width / 2,
                height / 2,
                text="No relationship data available.",
                font=("Segoe UI", 13),
                fill=text_color
            )
            status.set("No network data.")
            return

        # Circular layout keeps the graph stable and avoids overlapping
        # coordinates without requiring external graph libraries.
        import math

        cx = width / 2
        cy = height / 2
        ring_radius = max(120, min(width, height) * 0.34)

        positions.clear()

        for i, item in enumerate(display_nodes):
            angle = (2 * math.pi * i / len(display_nodes)) - math.pi / 2
            x = cx + ring_radius * math.cos(angle)
            y = cy + ring_radius * math.sin(angle)
            positions[item["key"]] = (x, y)

        # Edges first.
        for a, b in display_edges:
            if a not in positions or b not in positions:
                continue
            x1, y1 = positions[a]
            x2, y2 = positions[b]
            canvas.create_line(
                x1, y1, x2, y2,
                fill="#CBD5E1",
                width=1,
                tags=("edge",)
            )

        # Nodes.
        for item in display_nodes:
            key = item["key"]
            x, y = positions[key]
            canvas.create_oval(
                x - radius, y - radius,
                x + radius, y + radius,
                fill="#FFFFFF",
                outline="#94A3B8",
                width=2,
                tags=("node", str(key))
            )

            short = item["label"]
            if len(short) > 16:
                short = short[:15] + "…"

            canvas.create_text(
                x, y - 2,
                text=short,
                font=("Segoe UI", 8, "bold"),
                fill=text_color,
                width=48,
                tags=("node", str(key))
            )

            canvas.create_text(
                x, y + 13,
                text=item["type"],
                font=("Segoe UI", 6),
                fill=muted,
                tags=("node", str(key))
            )

        status.set(
            f"Showing {len(display_nodes)} nodes • "
            f"{len(display_edges)} relationships"
        )

    def select_node(event):
        nearest = None
        nearest_distance = 10**9

        for key, (x, y) in positions.items():
            d = ((event.x - x) ** 2 + (event.y - y) ** 2) ** 0.5
            if d < nearest_distance:
                nearest_distance = d
                nearest = key

        if nearest is None or nearest_distance > radius + 8:
            selected["key"] = None
            return

        selected["key"] = nearest

        if str(nearest[0]).upper() == "PERSON":
            label = get_entity_label(nearest[0], nearest[1])
            status.set(
                f"Selected: {label} • Person ID: {nearest[1]}"
            )
        else:
            status.set(
                f"Selected: {get_entity_label(nearest[0], nearest[1])} "
                f"• Type: {nearest[0]}"
            )

    def open_selected():
        key = selected["key"]
        if not key:
            messagebox.showinfo(
                "Select Entity",
                "Click a node first.",
                parent=win
            )
            return

        if str(key[0]).upper() == "PERSON":
            open_person_intelligence_profile(key[1])
        else:
            messagebox.showinfo(
                "Entity",
                f"Type: {key[0]}\n"
                f"ID: {key[1]}\n"
                f"Label: {get_entity_label(key[0], key[1])}",
                parent=win
            )

    def refresh():
        nonlocal display_nodes, display_edges
        new_nodes, new_edges = get_relationship_graph_data()
        display_nodes = new_nodes[:80]
        display_keys = {n["key"] for n in display_nodes}
        display_edges = [
            e for e in new_edges
            if e[0] in display_keys and e[1] in display_keys
        ]
        draw_graph()

    canvas.bind("<Button-1>", select_node)
    canvas.bind("<Configure>", lambda e: draw_graph())

    tk.Button(
        toolbar,
        text="REFRESH",
        command=refresh,
        font=("Segoe UI", 9, "bold"),
        padx=13, pady=6
    ).pack(side="left", padx=(0, 7))

    tk.Button(
        toolbar,
        text="OPEN SELECTED PROFILE",
        command=open_selected,
        font=("Segoe UI", 9, "bold"),
        padx=13, pady=6
    ).pack(side="left")

    tk.Label(
        toolbar,
        textvariable=status,
        font=("Segoe UI", 9),
        bg=bg, fg=muted
    ).pack(side="left", padx=14)

    tk.Label(
        outer,
        text=(
            "Click a node to select it. Person nodes can be opened in "
            "the full intelligence profile."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    win.after(100, draw_graph)
    return win


# =========================================================
# STEP 14 — SEARCH & INVESTIGATION WORKSPACE
# =========================================================

def search_investigation_entities(query):
    """Search the existing network for a person or any connected entity."""
    query = _safe_normalize(query).lower()
    if not query:
        return []

    graph = _person_network_graph()
    results = []

    for node in graph:
        node_type = str(node[0])
        node_id = str(node[1])
        label = str(get_entity_label(node_type, node_id))

        haystack = f"{node_type} {node_id} {label}".lower()

        if query in haystack:
            results.append({
                "key": node,
                "type": node_type,
                "id": node_id,
                "label": label,
                "connections": len(graph.get(node, set())),
            })

    results.sort(
        key=lambda x: (-x["connections"], x["label"].lower())
    )
    return results


def open_investigation_workspace():
    """Unified investigator search workspace."""
    from tkinter import messagebox

    win = tk.Toplevel(root)
    win.title("Investigation Workspace")
    win.geometry("1200x760")
    win.minsize(900, 600)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    selected = {"item": None}
    results_holder = {"items": []}

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="INVESTIGATION WORKSPACE",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text="Search across the current intelligence network",
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    search_bar = tk.Frame(outer, bg=bg)
    search_bar.pack(fill="x", padx=16, pady=(0, 10))

    query_var = tk.StringVar()
    query_entry = tk.Entry(
        search_bar,
        textvariable=query_var,
        font=("Segoe UI", 11),
        relief="solid",
        bd=1
    )
    query_entry.pack(side="left", fill="x", expand=True, ipady=7)

    status = tk.StringVar(value="Enter a name, ID, phone, vehicle, location or organization.")

    results_frame = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    results_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    tree = ttk.Treeview(
        results_frame,
        columns=("type", "id", "label", "connections"),
        show="headings",
        selectmode="browse"
    )

    for col, heading, width in [
        ("type", "TYPE", 150),
        ("id", "ID", 220),
        ("label", "ENTITY", 470),
        ("connections", "CONNECTIONS", 130),
    ]:
        tree.heading(col, text=heading)
        tree.column(col, width=width, anchor="w")

    sy = ttk.Scrollbar(
        results_frame, orient="vertical", command=tree.yview
    )
    tree.configure(yscrollcommand=sy.set)
    tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
    sy.pack(side="right", fill="y", padx=(0, 10), pady=10)

    def search():
        items = search_investigation_entities(query_var.get())
        results_holder["items"] = items
        selected["item"] = None

        for item in tree.get_children():
            tree.delete(item)

        for item in items:
            tree.insert(
                "", "end",
                values=(
                    item["type"],
                    item["id"],
                    item["label"],
                    item["connections"]
                )
            )

        status.set(f"{len(items)} matching entity/entities found.")

    def on_select(event=None):
        selection = tree.selection()
        if not selection:
            selected["item"] = None
            return

        index = tree.index(selection[0])
        if 0 <= index < len(results_holder["items"]):
            selected["item"] = results_holder["items"][index]
            item = selected["item"]
            status.set(
                f"Selected: {item['label']} • "
                f"{item['type']} • {item['connections']} connection(s)"
            )

    def open_profile():
        item = selected["item"]
        if not item:
            messagebox.showinfo(
                "Select Entity",
                "Select an entity from the results first.",
                parent=win
            )
            return

        if str(item["type"]).upper() == "PERSON":
            open_person_intelligence_profile(item["id"])
        else:
            messagebox.showinfo(
                "Entity Details",
                f"Type: {item['type']}\n"
                f"ID: {item['id']}\n"
                f"Entity: {item['label']}\n"
                f"Connections: {item['connections']}",
                parent=win
            )

    def open_graph():
        item = selected["item"]
        if item and str(item["type"]).upper() == "PERSON":
            open_relationship_graph()
        else:
            open_relationship_graph()

    tk.Button(
        search_bar,
        text="SEARCH",
        command=search,
        font=("Segoe UI", 9, "bold"),
        padx=16, pady=7
    ).pack(side="left", padx=(8, 5))

    tk.Button(
        search_bar,
        text="OPEN PROFILE",
        command=open_profile,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=7
    ).pack(side="left", padx=5)

    tk.Button(
        search_bar,
        text="NETWORK GRAPH",
        command=open_graph,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=7
    ).pack(side="left", padx=(5, 0))

    query_entry.bind("<Return>", lambda e: search())
    tree.bind("<<TreeviewSelect>>", on_select)

    tk.Label(
        outer,
        textvariable=status,
        font=("Segoe UI", 9),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 3))

    tk.Label(
        outer,
        text=(
            "Search is performed against the currently available relationship "
            "network. Person results can be opened in the full intelligence profile."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    query_entry.focus_set()
    return win


# =========================================================
# STEP 15 — FIR / CDR / TRANSACTION / REPORT EVIDENCE VIEWER
# =========================================================

def _existing_evidence_tables():
    """Detect common evidence tables without changing the current schema."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    names = {str(row[0]).lower(): row[0] for row in cur.fetchall()}
    conn.close()

    aliases = {
        "FIR": ["firs", "fir", "fir_records"],
        "CDR": ["cdrs", "cdr", "call_records", "call_detail_records"],
        "TRANSACTION": [
            "transactions", "transaction", "financial_transactions"
        ],
        "REPORT": ["reports", "report", "intelligence_reports"],
        "EVENT": ["events", "event"],
    }

    resolved = {}
    for label, candidates in aliases.items():
        resolved[label] = next(
            (names[c.lower()] for c in candidates if c.lower() in names),
            None
        )
    return resolved


def _evidence_columns(table):
    if not table:
        return []
    conn = get_connection()
    try:
        return _table_columns(conn, table)
    finally:
        conn.close()


def _find_evidence_person_column(columns):
    return _find_first_column(
        columns,
        [
            "person_id", "suspect_id", "individual_id",
            "person", "person_name", "name",
            "accused", "subject", "entity_id"
        ]
    )


def _read_evidence_for_person(person_id):
    """
    Read available evidence rows related to a person.
    Matching is deliberately conservative and schema-aware.
    """
    resolved = _existing_evidence_tables()
    evidence = []

    conn = get_connection()
    cur = conn.cursor()

    for evidence_type, table in resolved.items():
        if not table:
            continue

        columns = _table_columns(conn, table)
        person_col = _find_evidence_person_column(columns)

        if not person_col:
            continue

        # Only select rows where the existing person/entity column matches.
        try:
            cur.execute(
                f"SELECT * FROM {table} WHERE {person_col} = ? "
                f"ORDER BY rowid DESC LIMIT 100",
                (person_id,)
            )
            rows = cur.fetchall()
        except Exception:
            rows = []

        for row in rows:
            record = dict(zip(columns, row))

            date_col = _find_first_column(
                columns,
                [
                    "date", "event_date", "timestamp",
                    "created_at", "datetime", "time"
                ]
            )

            ref_col = _find_first_column(
                columns,
                [
                    "fir_number", "case_id", "record_id",
                    "cdr_id", "transaction_id", "report_id",
                    "id"
                ]
            )

            summary_parts = []
            for key in columns:
                value = record.get(key)
                if value in (None, ""):
                    continue
                if str(key).lower() in {
                    str(person_col).lower(),
                    str(date_col).lower() if date_col else "",
                }:
                    continue
                summary_parts.append(
                    f"{str(key).replace('_', ' ').title()}: {value}"
                )
                if len(summary_parts) >= 4:
                    break

            evidence.append({
                "type": evidence_type,
                "table": table,
                "date": str(record.get(date_col, "—")) if date_col else "—",
                "reference": str(record.get(ref_col, "—")) if ref_col else "—",
                "summary": " • ".join(summary_parts) or "Record available.",
                "record": record,
            })

    conn.close()

    evidence.sort(
        key=lambda x: (
            str(x["date"]) == "—",
            str(x["date"])
        ),
        reverse=True
    )
    return evidence


def open_evidence_viewer(person_id=None):
    """Open FIR/CDR/transaction/report evidence for a selected person."""
    from tkinter import messagebox, simpledialog

    if person_id is None:
        person_id = simpledialog.askstring(
            "Evidence Viewer",
            "Enter Person ID:",
            parent=root
        )

    if not person_id:
        return None

    win = tk.Toplevel(root)
    win.title("Investigation Evidence Viewer")
    win.geometry("1200x760")
    win.minsize(900, 600)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="INVESTIGATION EVIDENCE",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    person_label = str(person_id)
    try:
        person_label = get_entity_label("PERSON", person_id)
    except Exception:
        pass

    tk.Label(
        header,
        text=f"{person_label}  •  Person ID: {person_id}",
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=(0, 8))

    filter_var = tk.StringVar(value="ALL")
    status = tk.StringVar(value="Loading evidence...")

    table_frame = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    table_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    tree = ttk.Treeview(
        table_frame,
        columns=("type", "date", "reference", "summary"),
        show="headings",
        selectmode="browse"
    )

    for col, heading, width in [
        ("type", "SOURCE", 140),
        ("date", "DATE / TIME", 170),
        ("reference", "REFERENCE", 180),
        ("summary", "EVIDENCE SUMMARY", 620),
    ]:
        tree.heading(col, text=heading)
        tree.column(col, width=width, anchor="w")

    sy = ttk.Scrollbar(
        table_frame, orient="vertical", command=tree.yview
    )
    tree.configure(yscrollcommand=sy.set)
    tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
    sy.pack(side="right", fill="y", padx=(0, 10), pady=10)

    evidence_holder = {"items": []}

    def load():
        evidence_holder["items"] = _read_evidence_for_person(person_id)

        for item in tree.get_children():
            tree.delete(item)

        active = filter_var.get()
        visible = [
            x for x in evidence_holder["items"]
            if active == "ALL" or x["type"] == active
        ]

        for item in visible:
            tree.insert(
                "", "end",
                values=(
                    item["type"],
                    item["date"],
                    item["reference"],
                    item["summary"]
                )
            )

        status.set(
            f"{len(visible)} evidence record(s) shown • "
            f"{len(evidence_holder['items'])} total available"
        )

    def show_selected():
        selection = tree.selection()
        if not selection:
            messagebox.showinfo(
                "Select Evidence",
                "Select an evidence record first.",
                parent=win
            )
            return

        index = tree.index(selection[0])
        active = filter_var.get()
        visible = [
            x for x in evidence_holder["items"]
            if active == "ALL" or x["type"] == active
        ]

        if not (0 <= index < len(visible)):
            return

        item = visible[index]

        detail = "\n".join(
            f"{str(k).replace('_', ' ').title()}: {v}"
            for k, v in item["record"].items()
            if v not in (None, "")
        )

        detail_win = tk.Toplevel(win)
        detail_win.title(f"{item['type']} Evidence Detail")
        detail_win.geometry("850x600")
        detail_win.minsize(650, 450)
        detail_win.resizable(True, True)

        detail_frame = tk.Frame(detail_win, bg=surface)
        detail_frame.pack(fill="both", expand=True, padx=14, pady=14)

        text_box = tk.Text(
            detail_frame,
            wrap="word",
            font=("Consolas", 10),
            bg=surface,
            fg=text_color,
            relief="solid",
            bd=1
        )
        text_box.pack(fill="both", expand=True)
        text_box.insert("1.0", detail)
        text_box.configure(state="disabled")

        try:
            enable_window_controls(detail_win)
        except Exception:
            pass

    def set_filter(value):
        filter_var.set(value)
        load()

    for value in ("ALL", "FIR", "CDR", "TRANSACTION", "REPORT", "EVENT"):
        tk.Button(
            toolbar,
            text=value,
            command=lambda v=value: set_filter(v),
            font=("Segoe UI", 8, "bold"),
            padx=10, pady=5
        ).pack(side="left", padx=2)

    tk.Button(
        toolbar,
        text="VIEW SELECTED",
        command=show_selected,
        font=("Segoe UI", 9, "bold"),
        padx=12, pady=6
    ).pack(side="left", padx=(10, 2))

    tk.Button(
        toolbar,
        text="REFRESH",
        command=load,
        font=("Segoe UI", 9, "bold"),
        padx=12, pady=6
    ).pack(side="left", padx=2)

    tk.Label(
        toolbar,
        textvariable=status,
        font=("Segoe UI", 9),
        bg=bg, fg=muted
    ).pack(side="left", padx=12)

    tk.Label(
        outer,
        text=(
            "Evidence is read from compatible existing tables only. "
            "No database schema changes are made by this viewer."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    load()
    return win


# =========================================================
# STEP 16 — SUSPICIOUS ACTIVITY & ALERT ENGINE
# =========================================================

def generate_suspicious_alerts():
    """
    Generate explainable alerts from the existing database/network.
    This is a prototype intelligence layer; it does not modify evidence.
    """
    alerts = []
    seen = set()

    try:
        graph = _person_network_graph()
    except Exception:
        graph = {}

    # Network-based alerts.
    for node, neighbors in graph.items():
        if str(node[0]).upper() != "PERSON":
            continue

        degree = len(neighbors)
        if degree >= 5:
            key = ("HIGH_CONNECTIVITY", str(node[1]))
            if key not in seen:
                seen.add(key)
                alerts.append({
                    "type": "HIGH CONNECTIVITY",
                    "severity": "HIGH" if degree >= 10 else "MEDIUM",
                    "entity_type": "PERSON",
                    "entity_id": str(node[1]),
                    "entity": get_entity_label("PERSON", node[1]),
                    "reason": (
                        f"Person has {degree} detected network connections."
                    ),
                    "score": min(100, 45 + degree * 5),
                })

    # Repeated shared-entity patterns.
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {str(r[0]).lower(): r[0] for r in cur.fetchall()}

        # Check common relationship tables for duplicate/high-frequency links.
        rel_table = next(
            (
                tables[x] for x in (
                    "relationships", "relations", "entity_relationships"
                ) if x in tables
            ),
            None
        )

        if rel_table:
            cols = _table_columns(conn, rel_table)
            source_col = _find_first_column(
                cols, ["source_id", "from_id", "person_id", "entity_a", "source"]
            )
            target_col = _find_first_column(
                cols, ["target_id", "to_id", "entity_b", "target"]
            )

            if source_col and target_col:
                cur.execute(
                    f"""
                    SELECT {source_col}, {target_col}, COUNT(*) AS c
                    FROM {rel_table}
                    GROUP BY {source_col}, {target_col}
                    HAVING COUNT(*) >= 3
                    ORDER BY c DESC
                    LIMIT 100
                    """
                )

                for source, target, count in cur.fetchall():
                    key = ("REPEATED_RELATIONSHIP", str(source), str(target))
                    if key in seen:
                        continue
                    seen.add(key)

                    alerts.append({
                        "type": "REPEATED RELATIONSHIP",
                        "severity": "MEDIUM" if count < 6 else "HIGH",
                        "entity_type": "RELATIONSHIP",
                        "entity_id": f"{source} → {target}",
                        "entity": f"{source} → {target}",
                        "reason": (
                            f"The same relationship appears {count} times "
                            "in the relationship data."
                        ),
                        "score": min(100, 50 + count * 6),
                    })
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Evidence-table anomaly checks where numeric amount/duration columns exist.
    try:
        resolved = _existing_evidence_tables()
        conn = get_connection()
        cur = conn.cursor()

        for evidence_type, table in resolved.items():
            if not table:
                continue

            cols = _table_columns(conn, table)
            amount_col = _find_first_column(
                cols,
                [
                    "amount", "transaction_amount", "value",
                    "total_amount", "duration", "call_duration"
                ]
            )
            person_col = _find_evidence_person_column(cols)

            if not amount_col or not person_col:
                continue

            # Numeric conversion is performed by SQLite where possible.
            cur.execute(
                f"""
                SELECT {person_col}, {amount_col}
                FROM {table}
                WHERE {amount_col} IS NOT NULL
                ORDER BY CAST({amount_col} AS REAL) DESC
                LIMIT 20
                """
            )

            for person_id, amount in cur.fetchall():
                try:
                    numeric = float(str(amount).replace(",", "").strip())
                except Exception:
                    continue

                # Prototype threshold: only flag clearly large values.
                if evidence_type == "TRANSACTION" and numeric >= 100000:
                    key = ("LARGE_TRANSACTION", str(person_id), str(amount))
                    if key in seen:
                        continue
                    seen.add(key)

                    alerts.append({
                        "type": "LARGE TRANSACTION",
                        "severity": "HIGH",
                        "entity_type": "PERSON",
                        "entity_id": str(person_id),
                        "entity": get_entity_label("PERSON", person_id),
                        "reason": (
                            f"A financial transaction value of {numeric:,.2f} "
                            "was found in the available data."
                        ),
                        "score": 85,
                    })

                if evidence_type == "CDR" and numeric >= 1800:
                    key = ("LONG_CALL", str(person_id), str(amount))
                    if key in seen:
                        continue
                    seen.add(key)

                    alerts.append({
                        "type": "LONG CALL",
                        "severity": "MEDIUM",
                        "entity_type": "PERSON",
                        "entity_id": str(person_id),
                        "entity": get_entity_label("PERSON", person_id),
                        "reason": (
                            f"A call duration of {numeric:g} seconds "
                            "was found in the available CDR data."
                        ),
                        "score": 65,
                    })

        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass

    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    alerts.sort(
        key=lambda x: (
            severity_order.get(x["severity"], 9),
            -int(x["score"])
        )
    )
    return alerts


def open_suspicious_alerts():
    """Open the investigator-facing suspicious activity console."""
    from tkinter import messagebox

    win = tk.Toplevel(root)
    win.title("Suspicious Activity Alerts")
    win.geometry("1250x780")
    win.minsize(900, 600)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="SUSPICIOUS ACTIVITY & ALERTS",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text="Explainable alerts generated from available intelligence data",
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=(0, 8))

    status = tk.StringVar(value="Generating alerts...")
    alerts_holder = {"items": []}

    table_frame = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    table_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    tree = ttk.Treeview(
        table_frame,
        columns=(
            "severity", "type", "entity",
            "score", "reason"
        ),
        show="headings",
        selectmode="browse"
    )

    for col, heading, width in [
        ("severity", "SEVERITY", 100),
        ("type", "ALERT TYPE", 190),
        ("entity", "ENTITY", 260),
        ("score", "SCORE", 90),
        ("reason", "REASON", 520),
    ]:
        tree.heading(col, text=heading)
        tree.column(col, width=width, anchor="w")

    sy = ttk.Scrollbar(
        table_frame, orient="vertical", command=tree.yview
    )
    tree.configure(yscrollcommand=sy.set)
    tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
    sy.pack(side="right", fill="y", padx=(0, 10), pady=10)

    def load():
        alerts = generate_suspicious_alerts()
        alerts_holder["items"] = alerts

        for item in tree.get_children():
            tree.delete(item)

        for alert in alerts:
            tree.insert(
                "", "end",
                values=(
                    alert["severity"],
                    alert["type"],
                    alert["entity"],
                    alert["score"],
                    alert["reason"]
                )
            )

        high = sum(x["severity"] == "HIGH" for x in alerts)
        medium = sum(x["severity"] == "MEDIUM" for x in alerts)

        status.set(
            f"{len(alerts)} alert(s) • "
            f"High: {high} • Medium: {medium}"
        )

    def open_selected():
        selection = tree.selection()
        if not selection:
            messagebox.showinfo(
                "Select Alert",
                "Select an alert first.",
                parent=win
            )
            return

        index = tree.index(selection[0])
        if not (0 <= index < len(alerts_holder["items"])):
            return

        alert = alerts_holder["items"][index]

        detail = (
            f"Alert Type: {alert['type']}\n"
            f"Severity: {alert['severity']}\n"
            f"Entity Type: {alert['entity_type']}\n"
            f"Entity: {alert['entity']}\n"
            f"Entity ID: {alert['entity_id']}\n"
            f"Risk Score: {alert['score']}\n\n"
            f"Reason:\n{alert['reason']}"
        )

        if alert["entity_type"] == "PERSON":
            if messagebox.askyesno(
                "Alert Details",
                detail + "\n\nOpen this person's intelligence profile?",
                parent=win
            ):
                open_person_intelligence_profile(alert["entity_id"])
        else:
            messagebox.showinfo(
                "Alert Details",
                detail,
                parent=win
            )

    def open_profile():
        selection = tree.selection()
        if not selection:
            messagebox.showinfo(
                "Select Alert",
                "Select a person alert first.",
                parent=win
            )
            return

        index = tree.index(selection[0])
        if not (0 <= index < len(alerts_holder["items"])):
            return

        alert = alerts_holder["items"][index]
        if alert["entity_type"] == "PERSON":
            open_person_intelligence_profile(alert["entity_id"])
        else:
            messagebox.showinfo(
                "Not a Person",
                "The selected alert is not linked to a person.",
                parent=win
            )

    tk.Button(
        toolbar,
        text="REFRESH ALERTS",
        command=load,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=6
    ).pack(side="left", padx=(0, 7))

    tk.Button(
        toolbar,
        text="VIEW ALERT",
        command=open_selected,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=6
    ).pack(side="left", padx=3)

    tk.Button(
        toolbar,
        text="OPEN PROFILE",
        command=open_profile,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=6
    ).pack(side="left", padx=3)

    tk.Label(
        toolbar,
        textvariable=status,
        font=("Segoe UI", 9),
        bg=bg, fg=muted
    ).pack(side="left", padx=12)

    tk.Label(
        outer,
        text=(
            "Alerts are analytical indicators for investigation and should "
            "be verified against source evidence before action."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    load()
    return win


# =========================================================
# STEP 17 — NLP ENTITY EXTRACTION & RELATIONSHIP DISCOVERY
# =========================================================

def nlp_extract_entities(text_value):
    """
    Lightweight, dependency-free NLP-style extraction for prototype data.
    Extracts people, phones, vehicles, locations and organizations using
    patterns plus conservative context rules.
    """
    import re

    text_value = str(text_value or "")
    people = set()
    phones = set()
    vehicles = set()
    locations = set()
    organizations = set()

    # Phone numbers: Indian/mobile-like and generic long digit formats.
    for m in re.findall(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)", text_value):
        phones.add(re.sub(r"[\s-]", "", m))

    # Vehicle registration examples: BR01AB1234 / DL01CA1234 etc.
    for m in re.findall(
        r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{1,4}\b",
        text_value.upper()
    ):
        vehicles.add(re.sub(r"\s+", "", m))

    # Organizations with common legal/business suffixes.
    org_pattern = (
        r"\b([A-Z][A-Za-z0-9&.\- ]{2,60}"
        r"(?:Ltd|Limited|Pvt\.?\s*Ltd|Private Limited|"
        r"LLP|Inc|Corporation|Corp|Company|Foundation|"
        r"Trust|Bank|University|Institute|Agency))\b"
    )
    for m in re.findall(org_pattern, text_value):
        organizations.add(" ".join(m.split()))

    # Location phrases after common location indicators.
    loc_pattern = (
        r"\b(?:at|near|from|in|towards|around|nearby)\s+"
        r"([A-Z][A-Za-z0-9 .'\-]{2,50}?"
        r"(?:Station|Road|Rd|Street|St|Nagar|Chowk|"
        r"Market|Airport|Junction|District|City|Village|"
        r"Colony|Park|Temple|Mall|Bridge))\b"
    )
    for m in re.findall(loc_pattern, text_value):
        locations.add(" ".join(m.split()).strip(" ,."))

    # Conservative person-name patterns following relationship/action verbs.
    person_patterns = [
        r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})",
        r"\b(?:person|suspect|accused|subject|individual|named)\s+"
        r"([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})",
    ]
    for pattern in person_patterns:
        for m in re.findall(pattern, text_value):
            people.add(" ".join(m.split()))

    # Capitalized name pairs; exclude obvious place/org suffixes.
    for m in re.findall(
        r"\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})\b",
        text_value
    ):
        if not any(
            token.lower() in m.lower()
            for token in (
                "Station", "Road", "Street", "Market", "Airport",
                "District", "University", "Institute", "Company"
            )
        ):
            people.add(m)

    return {
        "people": sorted(people),
        "phones": sorted(phones),
        "vehicles": sorted(vehicles),
        "locations": sorted(locations),
        "organizations": sorted(organizations),
    }


def nlp_extract_relationships(text_value, entities):
    """Discover simple explicit relationships expressed in text."""
    import re

    text_value = str(text_value or "")
    people = entities.get("people", [])
    relationships = []

    relationship_verbs = [
        ("met", "MET"),
        ("contacted", "CONTACTED"),
        ("called", "CALLED"),
        ("visited", "VISITED"),
        ("worked with", "WORKED_WITH"),
        ("associated with", "ASSOCIATED_WITH"),
        ("linked to", "LINKED_TO"),
        ("spoke with", "SPOKE_WITH"),
        ("travelled with", "TRAVELLED_WITH"),
        ("traveled with", "TRAVELED_WITH"),
        ("transferred to", "TRANSFERRED_TO"),
        ("sent to", "SENT_TO"),
    ]

    for left in people:
        for right in people:
            if left == right:
                continue

            for phrase, relation in relationship_verbs:
                pattern = (
                    rf"\b{re.escape(left)}\b\s+"
                    rf"{re.escape(phrase)}\s+"
                    rf"\b{re.escape(right)}\b"
                )
                if re.search(pattern, text_value, flags=re.I):
                    relationships.append({
                        "source": left,
                        "relation": relation,
                        "target": right,
                    })

    # Deduplicate while preserving order.
    unique = []
    seen = set()
    for r in relationships:
        key = (r["source"], r["relation"], r["target"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


def open_nlp_extraction_workspace():
    """Interactive NLP extraction workspace for FIR/report text."""
    from tkinter import messagebox

    win = tk.Toplevel(root)
    win.title("NLP Entity & Relationship Extraction")
    win.geometry("1250x800")
    win.minsize(900, 600)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="NLP ENTITY & RELATIONSHIP EXTRACTION",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text="Extract structured intelligence from unstructured FIR / report text",
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    content = tk.Frame(outer, bg=bg)
    content.pack(fill="both", expand=True, padx=16)

    left = tk.Frame(
        content, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    left.pack(side="left", fill="both", expand=True, padx=(0, 7))

    right = tk.Frame(
        content, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    right.pack(side="left", fill="both", expand=True, padx=(7, 0))

    tk.Label(
        left, text="SOURCE TEXT",
        font=("Segoe UI", 10, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=12, pady=(12, 6))

    input_box = tk.Text(
        left, wrap="word",
        font=("Segoe UI", 10),
        bg=surface, fg=text_color,
        relief="solid", bd=1
    )
    input_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    sample = (
        "Ravi Kumar met Mohan Singh near Patna Railway Station. "
        "Ravi contacted Mohan using 9876543210. "
        "The vehicle BR01AB1234 was seen near the station."
    )
    input_box.insert("1.0", sample)

    output = tk.Text(
        right, wrap="word",
        font=("Consolas", 9),
        bg=surface, fg=text_color,
        relief="solid", bd=1
    )
    output.pack(fill="both", expand=True, padx=12, pady=(12, 8))

    status = tk.StringVar(value="Ready.")
    result_holder = {"entities": {}, "relationships": []}

    def extract():
        source = input_box.get("1.0", "end").strip()
        if not source:
            messagebox.showinfo(
                "NLP Extraction",
                "Enter FIR/report text first.",
                parent=win
            )
            return

        entities = nlp_extract_entities(source)
        relationships = nlp_extract_relationships(source, entities)
        result_holder["entities"] = entities
        result_holder["relationships"] = relationships

        output.configure(state="normal")
        output.delete("1.0", "end")

        labels = [
            ("PEOPLE", "people"),
            ("PHONES", "phones"),
            ("VEHICLES", "vehicles"),
            ("LOCATIONS", "locations"),
            ("ORGANIZATIONS", "organizations"),
        ]

        for title, key in labels:
            output.insert("end", f"{title}\n")
            output.insert("end", "-" * 45 + "\n")
            values = entities.get(key, [])
            output.insert(
                "end",
                "\n".join(values) if values else "None detected"
            )
            output.insert("end", "\n\n")

        output.insert("end", "RELATIONSHIPS\n")
        output.insert("end", "-" * 45 + "\n")

        if relationships:
            for rel in relationships:
                output.insert(
                    "end",
                    f"{rel['source']} --[{rel['relation']}]--> "
                    f"{rel['target']}\n"
                )
        else:
            output.insert("end", "No explicit person-to-person relationship detected.\n")

        output.configure(state="disabled")

        total = sum(len(v) for v in entities.values())
        status.set(
            f"Extraction complete • {total} entities • "
            f"{len(relationships)} relationship(s)"
        )

    def copy_results():
        output.configure(state="normal")
        data = output.get("1.0", "end").strip()
        output.configure(state="disabled")
        if not data:
            return
        win.clipboard_clear()
        win.clipboard_append(data)
        status.set("Extracted intelligence copied to clipboard.")

    def clear_all():
        input_box.delete("1.0", "end")
        output.configure(state="normal")
        output.delete("1.0", "end")
        output.configure(state="disabled")
        result_holder["entities"] = {}
        result_holder["relationships"] = []
        status.set("Cleared.")

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=8)

    tk.Button(
        toolbar, text="EXTRACT",
        command=extract,
        font=("Segoe UI", 9, "bold"),
        padx=16, pady=7
    ).pack(side="left", padx=(0, 6))

    tk.Button(
        toolbar, text="COPY RESULTS",
        command=copy_results,
        font=("Segoe UI", 9, "bold"),
        padx=13, pady=7
    ).pack(side="left", padx=3)

    tk.Button(
        toolbar, text="CLEAR",
        command=clear_all,
        font=("Segoe UI", 9, "bold"),
        padx=13, pady=7
    ).pack(side="left", padx=3)

    tk.Label(
        toolbar, textvariable=status,
        font=("Segoe UI", 9),
        bg=bg, fg=muted
    ).pack(side="left", padx=12)

    tk.Label(
        outer,
        text=(
            "Prototype NLP uses conservative pattern-based extraction. "
            "Detected entities should be verified against source records."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    return win


# =========================================================
# STEP 18 — CDR ANALYSIS ENGINE
# =========================================================

def _resolve_cdr_table():
    tables = _existing_evidence_tables()
    return tables.get("CDR")


def analyze_cdr_records(person_id=None):
    """
    Analyze compatible CDR records without changing the database.
    Supports common caller/receiver, duration and timestamp column names.
    """
    table = _resolve_cdr_table()
    if not table:
        return {
            "available": False,
            "reason": "No compatible CDR table found.",
            "records": [],
            "contacts": [],
            "summary": {},
        }

    conn = get_connection()
    try:
        columns = _table_columns(conn, table)
        caller = _find_first_column(
            columns,
            ["caller", "caller_id", "from_number", "source",
             "source_number", "phone_from", "calling_number"]
        )
        receiver = _find_first_column(
            columns,
            ["receiver", "receiver_id", "to_number", "target",
             "target_number", "phone_to", "called_number"]
        )
        duration = _find_first_column(
            columns,
            ["duration", "call_duration", "duration_seconds",
             "seconds", "call_time"]
        )
        timestamp = _find_first_column(
            columns,
            ["timestamp", "datetime", "date_time", "call_time",
             "date", "event_date", "created_at"]
        )

        if not caller or not receiver:
            return {
                "available": False,
                "reason": "CDR table exists but caller/receiver columns were not detected.",
                "records": [],
                "contacts": [],
                "summary": {},
            }

        selected = [caller, receiver]
        for col in (duration, timestamp):
            if col and col not in selected:
                selected.append(col)

        query = f"SELECT {', '.join(selected)} FROM {table}"
        params = ()

        if person_id is not None:
            query += f" WHERE {caller} = ? OR {receiver} = ?"
            params = (str(person_id), str(person_id))

        query += " LIMIT 5000"

        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()

        contacts = {}
        records = []
        total_duration = 0.0

        for row in rows:
            data = dict(zip(selected, row))
            a = str(data.get(caller, "") or "")
            b = str(data.get(receiver, "") or "")

            try:
                d = float(str(data.get(duration, 0) or 0).replace(",", ""))
            except Exception:
                d = 0.0

            total_duration += d

            pair = tuple(sorted((a, b)))
            if pair[0] and pair[1]:
                if pair not in contacts:
                    contacts[pair] = {
                        "caller_a": pair[0],
                        "caller_b": pair[1],
                        "calls": 0,
                        "duration": 0.0,
                    }
                contacts[pair]["calls"] += 1
                contacts[pair]["duration"] += d

            records.append({
                "caller": a,
                "receiver": b,
                "duration": d,
                "timestamp": str(data.get(timestamp, "—")) if timestamp else "—",
            })

        contact_rows = list(contacts.values())
        contact_rows.sort(
            key=lambda x: (-x["calls"], -x["duration"])
        )

        repeated = [x for x in contact_rows if x["calls"] >= 5]
        long_calls = [x for x in records if x["duration"] >= 1800]

        return {
            "available": True,
            "reason": "",
            "records": records,
            "contacts": contact_rows,
            "summary": {
                "total_calls": len(records),
                "unique_contacts": len(contact_rows),
                "total_duration": total_duration,
                "repeated_contacts": len(repeated),
                "long_calls": len(long_calls),
            },
        }
    finally:
        conn.close()


def open_cdr_analysis():
    """Open the CDR analysis dashboard."""
    from tkinter import messagebox, simpledialog

    person_id = simpledialog.askstring(
        "CDR Analysis",
        "Enter Person ID (leave blank for all CDRs):",
        parent=root
    )

    if person_id == "":
        person_id = None

    win = tk.Toplevel(root)
    win.title("CDR Communication Analysis")
    win.geometry("1250x780")
    win.minsize(900, 600)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="CDR COMMUNICATION ANALYSIS",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    scope = "ALL CDR RECORDS" if person_id is None else f"PERSON ID: {person_id}"
    tk.Label(
        header,
        text=scope,
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    summary_frame = tk.Frame(outer, bg=bg)
    summary_frame.pack(fill="x", padx=16, pady=(0, 10))

    cards = {}
    for title, key in [
        ("TOTAL CALLS", "total_calls"),
        ("UNIQUE CONTACTS", "unique_contacts"),
        ("REPEATED CONTACTS", "repeated_contacts"),
        ("LONG CALLS", "long_calls"),
    ]:
        card = tk.Frame(
            summary_frame, bg=surface,
            highlightthickness=1,
            highlightbackground=border
        )
        card.pack(side="left", fill="x", expand=True, padx=3)

        tk.Label(
            card, text=title,
            font=("Segoe UI", 8, "bold"),
            bg=surface, fg=muted
        ).pack(pady=(10, 2))

        value = tk.StringVar(value="0")
        cards[key] = value

        tk.Label(
            card, textvariable=value,
            font=("Segoe UI", 17, "bold"),
            bg=surface, fg=text_color
        ).pack(pady=(0, 10))

    table_frame = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    table_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    tree = ttk.Treeview(
        table_frame,
        columns=("contact_a", "contact_b", "calls", "duration"),
        show="headings",
        selectmode="browse"
    )

    for col, heading, width in [
        ("contact_a", "CALLER / ENTITY A", 300),
        ("contact_b", "RECEIVER / ENTITY B", 300),
        ("calls", "CALL COUNT", 120),
        ("duration", "TOTAL DURATION (SEC)", 180),
    ]:
        tree.heading(col, text=heading)
        tree.column(col, width=width, anchor="w")

    sy = ttk.Scrollbar(
        table_frame, orient="vertical", command=tree.yview
    )
    tree.configure(yscrollcommand=sy.set)
    tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
    sy.pack(side="right", fill="y", padx=(0, 10), pady=10)

    status = tk.StringVar(value="Analyzing CDR...")
    analysis_holder = {"data": None}

    def load():
        data = analyze_cdr_records(person_id)
        analysis_holder["data"] = data

        for item in tree.get_children():
            tree.delete(item)

        if not data["available"]:
            for key in cards:
                cards[key].set("—")
            status.set(data["reason"])
            messagebox.showwarning(
                "CDR Analysis",
                data["reason"],
                parent=win
            )
            return

        summary = data["summary"]
        for key, variable in cards.items():
            variable.set(str(summary.get(key, 0)))

        for contact in data["contacts"]:
            tree.insert(
                "", "end",
                values=(
                    contact["caller_a"],
                    contact["caller_b"],
                    contact["calls"],
                    f"{contact['duration']:.0f}"
                )
            )

        status.set(
            f"{summary['total_calls']} calls analyzed • "
            f"{summary['unique_contacts']} unique contacts"
        )

    def show_selected():
        selection = tree.selection()
        if not selection:
            messagebox.showinfo(
                "Select Contact",
                "Select a contact row first.",
                parent=win
            )
            return

        data = analysis_holder["data"]
        index = tree.index(selection[0])

        if not data or index >= len(data["contacts"]):
            return

        contact = data["contacts"][index]

        messagebox.showinfo(
            "Communication Pattern",
            f"Entity A: {contact['caller_a']}\n"
            f"Entity B: {contact['caller_b']}\n"
            f"Call Count: {contact['calls']}\n"
            f"Total Duration: {contact['duration']:.0f} seconds\n\n"
            f"Interpretation:\n"
            f"This pair has the highest available communication "
            f"frequency for the current result set.",
            parent=win
        )

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=(0, 8))

    tk.Button(
        toolbar,
        text="REFRESH ANALYSIS",
        command=load,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=6
    ).pack(side="left", padx=(0, 6))

    tk.Button(
        toolbar,
        text="VIEW CONTACT",
        command=show_selected,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=6
    ).pack(side="left")

    tk.Label(
        toolbar,
        textvariable=status,
        font=("Segoe UI", 9),
        bg=bg, fg=muted
    ).pack(side="left", padx=12)

    tk.Label(
        outer,
        text=(
            "Communication frequency is an analytical indicator only; "
            "source CDR records should be verified before investigative action."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    load()
    return win


# =========================================================
# STEP 19 — FINANCIAL TRANSACTION ANALYSIS
# =========================================================

def _resolve_transaction_table():
    tables = _existing_evidence_tables()
    return tables.get("TRANSACTION")


def analyze_financial_transactions(person_id=None):
    """Analyze compatible financial transaction records read-only."""
    table = _resolve_transaction_table()
    if not table:
        return {
            "available": False,
            "reason": "No compatible transaction table found.",
            "records": [],
            "pairs": [],
            "summary": {},
        }

    conn = get_connection()
    try:
        columns = _table_columns(conn, table)

        sender = _find_first_column(
            columns,
            ["sender", "sender_id", "from_account", "source_account",
             "payer", "account_from", "from_id"]
        )
        receiver = _find_first_column(
            columns,
            ["receiver", "receiver_id", "to_account", "target_account",
             "payee", "account_to", "to_id"]
        )
        amount = _find_first_column(
            columns,
            ["amount", "transaction_amount", "value", "total_amount",
             "amount_inr", "transaction_value"]
        )
        timestamp = _find_first_column(
            columns,
            ["timestamp", "datetime", "date_time", "transaction_date",
             "date", "event_date", "created_at"]
        )
        tx_type = _find_first_column(
            columns,
            ["type", "transaction_type", "mode", "method", "category"]
        )

        if not sender or not receiver or not amount:
            return {
                "available": False,
                "reason": (
                    "Transaction table exists but sender, receiver or "
                    "amount columns were not detected."
                ),
                "records": [],
                "pairs": [],
                "summary": {},
            }

        selected = [sender, receiver, amount]
        for col in (timestamp, tx_type):
            if col and col not in selected:
                selected.append(col)

        cur = conn.cursor()
        cur.execute(f"SELECT {', '.join(selected)} FROM {table} LIMIT 10000")
        rows = cur.fetchall()

        records = []
        pairs = {}

        for row in rows:
            data = dict(zip(selected, row))
            s = str(data.get(sender, "") or "")
            r = str(data.get(receiver, "") or "")

            try:
                value = float(
                    str(data.get(amount, 0) or 0)
                    .replace(",", "")
                    .replace("₹", "")
                    .strip()
                )
            except Exception:
                continue

            record = {
                "sender": s,
                "receiver": r,
                "amount": value,
                "timestamp": (
                    str(data.get(timestamp, "—"))
                    if timestamp else "—"
                ),
                "type": (
                    str(data.get(tx_type, "—"))
                    if tx_type else "—"
                ),
            }

            if person_id is not None:
                pid = str(person_id)
                if pid not in (s, r):
                    continue

            records.append(record)

            pair = (s, r)
            if pair not in pairs:
                pairs[pair] = {
                    "sender": s,
                    "receiver": r,
                    "count": 0,
                    "total": 0.0,
                    "max": 0.0,
                }

            pairs[pair]["count"] += 1
            pairs[pair]["total"] += value
            pairs[pair]["max"] = max(pairs[pair]["max"], value)

        pair_rows = list(pairs.values())
        pair_rows.sort(key=lambda x: (-x["total"], -x["count"]))

        high_value = [x for x in records if x["amount"] >= 100000]
        repeated = [x for x in pair_rows if x["count"] >= 3]

        return {
            "available": True,
            "reason": "",
            "records": records,
            "pairs": pair_rows,
            "summary": {
                "transactions": len(records),
                "unique_pairs": len(pair_rows),
                "total_value": sum(x["amount"] for x in records),
                "high_value": len(high_value),
                "repeated_pairs": len(repeated),
            },
        }
    finally:
        conn.close()


def open_financial_transaction_analysis():
    """Open investigator-facing financial transaction analysis."""
    from tkinter import messagebox, simpledialog

    person_id = simpledialog.askstring(
        "Financial Analysis",
        "Enter Person/Account ID (leave blank for all transactions):",
        parent=root
    )

    if person_id == "":
        person_id = None

    win = tk.Toplevel(root)
    win.title("Financial Transaction Analysis")
    win.geometry("1300x800")
    win.minsize(900, 600)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="FINANCIAL TRANSACTION ANALYSIS",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    scope = (
        "ALL TRANSACTIONS"
        if person_id is None
        else f"PERSON / ACCOUNT ID: {person_id}"
    )

    tk.Label(
        header,
        text=scope,
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    summary_frame = tk.Frame(outer, bg=bg)
    summary_frame.pack(fill="x", padx=16, pady=(0, 10))

    cards = {}
    for title, key in [
        ("TRANSACTIONS", "transactions"),
        ("UNIQUE PAIRS", "unique_pairs"),
        ("HIGH-VALUE", "high_value"),
        ("REPEATED PAIRS", "repeated_pairs"),
    ]:
        card = tk.Frame(
            summary_frame, bg=surface,
            highlightthickness=1,
            highlightbackground=border
        )
        card.pack(side="left", fill="x", expand=True, padx=3)

        tk.Label(
            card, text=title,
            font=("Segoe UI", 8, "bold"),
            bg=surface, fg=muted
        ).pack(pady=(10, 2))

        variable = tk.StringVar(value="0")
        cards[key] = variable

        tk.Label(
            card, textvariable=variable,
            font=("Segoe UI", 17, "bold"),
            bg=surface, fg=text_color
        ).pack(pady=(0, 10))

    tk.Label(
        outer,
        text="TOP MONEY FLOW PAIRS",
        font=("Segoe UI", 10, "bold"),
        bg=bg, fg=text_color
    ).pack(anchor="w", padx=18, pady=(2, 6))

    table_frame = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    table_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    tree = ttk.Treeview(
        table_frame,
        columns=("sender", "receiver", "count", "total", "maximum"),
        show="headings",
        selectmode="browse"
    )

    for col, heading, width in [
        ("sender", "SENDER / SOURCE", 260),
        ("receiver", "RECEIVER / TARGET", 260),
        ("count", "COUNT", 90),
        ("total", "TOTAL VALUE", 170),
        ("maximum", "MAX TRANSACTION", 170),
    ]:
        tree.heading(col, text=heading)
        tree.column(col, width=width, anchor="w")

    sy = ttk.Scrollbar(
        table_frame, orient="vertical", command=tree.yview
    )
    tree.configure(yscrollcommand=sy.set)
    tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
    sy.pack(side="right", fill="y", padx=(0, 10), pady=10)

    status = tk.StringVar(value="Analyzing transactions...")
    holder = {"data": None}

    def load():
        data = analyze_financial_transactions(person_id)
        holder["data"] = data

        for item in tree.get_children():
            tree.delete(item)

        if not data["available"]:
            for variable in cards.values():
                variable.set("—")
            status.set(data["reason"])
            messagebox.showwarning(
                "Financial Analysis",
                data["reason"],
                parent=win
            )
            return

        summary = data["summary"]

        for key, variable in cards.items():
            variable.set(str(summary.get(key, 0)))

        for pair in data["pairs"]:
            tree.insert(
                "", "end",
                values=(
                    pair["sender"],
                    pair["receiver"],
                    pair["count"],
                    f"₹ {pair['total']:,.2f}",
                    f"₹ {pair['max']:,.2f}"
                )
            )

        status.set(
            f"{summary['transactions']} transactions • "
            f"Total value: ₹ {summary['total_value']:,.2f}"
        )

    def show_selected():
        selection = tree.selection()
        if not selection:
            messagebox.showinfo(
                "Select Transaction Flow",
                "Select a transaction pair first.",
                parent=win
            )
            return

        data = holder["data"]
        index = tree.index(selection[0])

        if not data or index >= len(data["pairs"]):
            return

        pair = data["pairs"][index]

        messagebox.showinfo(
            "Financial Flow Details",
            f"Sender / Source: {pair['sender']}\n"
            f"Receiver / Target: {pair['receiver']}\n"
            f"Transaction Count: {pair['count']}\n"
            f"Total Value: ₹ {pair['total']:,.2f}\n"
            f"Maximum Transaction: ₹ {pair['max']:,.2f}\n\n"
            "Analytical note:\n"
            "Repeated or high-value flows are indicators for review, "
            "not conclusions of wrongdoing.",
            parent=win
        )

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=(0, 8))

    tk.Button(
        toolbar, text="REFRESH ANALYSIS",
        command=load,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=6
    ).pack(side="left", padx=(0, 6))

    tk.Button(
        toolbar, text="VIEW FLOW",
        command=show_selected,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=6
    ).pack(side="left")

    tk.Label(
        toolbar, textvariable=status,
        font=("Segoe UI", 9),
        bg=bg, fg=muted
    ).pack(side="left", padx=12)

    tk.Label(
        outer,
        text=(
            "Financial indicators are analytical leads and must be verified "
            "against the underlying transaction records."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    load()
    return win


# =========================================================
# STEP 20 — ADVANCED GRAPH ANALYTICS
# =========================================================

def calculate_graph_analytics():
    """
    Calculate degree, betweenness-style bridge scores, density and
    connected communities from the existing relationship graph.
    Uses only the standard library.
    """
    try:
        graph = _person_network_graph()
    except Exception:
        graph = {}

    # Convert the existing graph to a simple undirected adjacency map.
    adj = {}
    for node, neighbors in graph.items():
        adj.setdefault(node, set())
        for neighbor in neighbors:
            adj.setdefault(neighbor, set())
            adj[node].add(neighbor)
            adj[neighbor].add(node)

    person_nodes = [
        n for n in adj
        if isinstance(n, tuple) and len(n) >= 2
        and str(n[0]).upper() == "PERSON"
    ]

    degree = {
        n: len(adj.get(n, set()))
        for n in person_nodes
    }

    # Approximate betweenness using shortest-path dependency for small/
    # prototype graphs. Non-person nodes are retained as intermediaries.
    nodes = list(adj.keys())
    betweenness = {n: 0.0 for n in nodes}

    from collections import deque

    for source in nodes:
        stack = []
        predecessors = {v: [] for v in nodes}
        sigma = {v: 0.0 for v in nodes}
        distance = {v: -1 for v in nodes}

        sigma[source] = 1.0
        distance[source] = 0
        queue = deque([source])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj.get(v, set()):
                if distance[w] < 0:
                    distance[w] = distance[v] + 1
                    queue.append(w)
                if distance[w] == distance[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        dependency = {v: 0.0 for v in nodes}
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                if sigma[w]:
                    dependency[v] += (
                        sigma[v] / sigma[w]
                    ) * (1.0 + dependency[w])
            if w != source:
                betweenness[w] += dependency[w]

    # Normalize undirected betweenness.
    n = len(nodes)
    if n > 2:
        scale = 1.0 / ((n - 1) * (n - 2) / 2.0)
        betweenness = {
            k: v * scale for k, v in betweenness.items()
        }

    # Connected components as lightweight communities.
    communities = []
    visited = set()

    for node in nodes:
        if node in visited:
            continue

        component = []
        queue = [node]
        visited.add(node)

        while queue:
            current = queue.pop()
            component.append(current)
            for nxt in adj.get(current, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        communities.append(component)

    communities.sort(key=len, reverse=True)

    ranked_people = []
    for node in person_nodes:
        ranked_people.append({
            "node": node,
            "label": get_entity_label("PERSON", node[1]),
            "degree": degree.get(node, 0),
            "betweenness": betweenness.get(node, 0.0),
        })

    ranked_people.sort(
        key=lambda x: (
            -x["degree"],
            -x["betweenness"]
        )
    )

    bridges = sorted(
        (
            {
                "node": n,
                "label": (
                    get_entity_label("PERSON", n[1])
                    if isinstance(n, tuple) and len(n) >= 2
                    and str(n[0]).upper() == "PERSON"
                    else str(n)
                ),
                "betweenness": score,
                "degree": len(adj.get(n, set())),
            }
            for n, score in betweenness.items()
            if score > 0
        ),
        key=lambda x: -x["betweenness"]
    )

    edges = sum(len(v) for v in adj.values()) // 2
    possible = n * (n - 1) // 2
    density = (edges / possible) if possible else 0.0

    return {
        "nodes": nodes,
        "edges": edges,
        "density": density,
        "person_count": len(person_nodes),
        "ranked_people": ranked_people,
        "bridges": bridges,
        "communities": communities,
    }


def open_advanced_graph_analytics():
    """Open advanced network analytics without modifying the database."""
    from tkinter import messagebox

    win = tk.Toplevel(root)
    win.title("Advanced Graph Analytics")
    win.geometry("1300x820")
    win.minsize(950, 620)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="ADVANCED GRAPH ANALYTICS",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text="Centrality, bridge detection and connected community analysis",
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    summary_frame = tk.Frame(outer, bg=bg)
    summary_frame.pack(fill="x", padx=16, pady=(0, 10))

    cards = {}
    for title, key in [
        ("NETWORK NODES", "nodes"),
        ("NETWORK EDGES", "edges"),
        ("PERSON NODES", "person_count"),
        ("COMMUNITIES", "communities"),
    ]:
        card = tk.Frame(
            summary_frame, bg=surface,
            highlightthickness=1,
            highlightbackground=border
        )
        card.pack(side="left", fill="x", expand=True, padx=3)

        tk.Label(
            card, text=title,
            font=("Segoe UI", 8, "bold"),
            bg=surface, fg=muted
        ).pack(pady=(10, 2))

        value = tk.StringVar(value="0")
        cards[key] = value

        tk.Label(
            card, textvariable=value,
            font=("Segoe UI", 17, "bold"),
            bg=surface, fg=text_color
        ).pack(pady=(0, 10))

    notebook = ttk.Notebook(outer)
    notebook.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    ranking_tab = tk.Frame(notebook, bg=surface)
    bridges_tab = tk.Frame(notebook, bg=surface)
    community_tab = tk.Frame(notebook, bg=surface)

    notebook.add(ranking_tab, text="Key Individuals")
    notebook.add(bridges_tab, text="Bridge Individuals")
    notebook.add(community_tab, text="Communities")

    ranking_tree = ttk.Treeview(
        ranking_tab,
        columns=("person", "degree", "between"),
        show="headings"
    )

    for col, heading, width in [
        ("person", "PERSON", 420),
        ("degree", "DEGREE", 140),
        ("between", "BETWEENNESS", 180),
    ]:
        ranking_tree.heading(col, text=heading)
        ranking_tree.column(col, width=width, anchor="w")

    ranking_tree.pack(fill="both", expand=True, padx=12, pady=12)

    bridge_tree = ttk.Treeview(
        bridges_tab,
        columns=("person", "degree", "between"),
        show="headings"
    )

    for col, heading, width in [
        ("person", "ENTITY", 420),
        ("degree", "DEGREE", 140),
        ("between", "BRIDGE SCORE", 180),
    ]:
        bridge_tree.heading(col, text=heading)
        bridge_tree.column(col, width=width, anchor="w")

    bridge_tree.pack(fill="both", expand=True, padx=12, pady=12)

    community_tree = ttk.Treeview(
        community_tab,
        columns=("community", "size", "members"),
        show="headings"
    )

    for col, heading, width in [
        ("community", "COMMUNITY", 140),
        ("size", "SIZE", 100),
        ("members", "MEMBERS", 850),
    ]:
        community_tree.heading(col, text=heading)
        community_tree.column(col, width=width, anchor="w")

    community_tree.pack(fill="both", expand=True, padx=12, pady=12)

    status = tk.StringVar(value="Calculating graph metrics...")
    holder = {"data": None}

    def load():
        data = calculate_graph_analytics()
        holder["data"] = data

        for tree in (ranking_tree, bridge_tree, community_tree):
            for item in tree.get_children():
                tree.delete(item)

        cards["nodes"].set(str(len(data["nodes"])))
        cards["edges"].set(str(data["edges"]))
        cards["person_count"].set(str(data["person_count"]))
        cards["communities"].set(str(len(data["communities"])))

        for row in data["ranked_people"][:100]:
            ranking_tree.insert(
                "", "end",
                values=(
                    row["label"],
                    row["degree"],
                    f"{row['betweenness']:.4f}"
                )
            )

        for row in data["bridges"][:100]:
            bridge_tree.insert(
                "", "end",
                values=(
                    row["label"],
                    row["degree"],
                    f"{row['betweenness']:.4f}"
                )
            )

        for index, community in enumerate(
            data["communities"][:100], start=1
        ):
            labels = []
            for node in community:
                if (
                    isinstance(node, tuple)
                    and len(node) >= 2
                    and str(node[0]).upper() == "PERSON"
                ):
                    labels.append(get_entity_label("PERSON", node[1]))
                else:
                    labels.append(str(node))

            community_tree.insert(
                "", "end",
                values=(
                    f"Community {index}",
                    len(community),
                    ", ".join(labels)
                )
            )

        status.set(
            f"{len(data['nodes'])} nodes • "
            f"{data['edges']} edges • "
            f"Density: {data['density']:.4f}"
        )

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=(0, 8))

    tk.Button(
        toolbar,
        text="REFRESH ANALYTICS",
        command=load,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=6
    ).pack(side="left")

    tk.Label(
        toolbar,
        textvariable=status,
        font=("Segoe UI", 9),
        bg=bg, fg=muted
    ).pack(side="left", padx=12)

    tk.Label(
        outer,
        text=(
            "Graph metrics identify structural importance and connectivity; "
            "they are analytical indicators and not proof of criminal activity."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    load()
    return win


# =========================================================
# STEP 21 — ML RISK / ANOMALY PREDICTION PIPELINE
# =========================================================

def build_ml_features():
    """Build person-level features from the existing relationship graph."""
    try:
        graph = _person_network_graph()
    except Exception:
        graph = {}

    people = [
        node for node in graph
        if isinstance(node, tuple)
        and len(node) >= 2
        and str(node[0]).upper() == "PERSON"
    ]

    features = []
    for node in people:
        neighbors = graph.get(node, set()) or set()
        degree = len(neighbors)

        phone_links = 0
        vehicle_links = 0
        location_links = 0
        organization_links = 0
        fir_links = 0

        for item in neighbors:
            kind = str(item[0]).upper() if isinstance(item, tuple) and item else ""
            if "PHONE" in kind:
                phone_links += 1
            elif "VEHICLE" in kind:
                vehicle_links += 1
            elif "LOCATION" in kind:
                location_links += 1
            elif "ORGANIZATION" in kind:
                organization_links += 1
            elif "FIR" in kind:
                fir_links += 1

        features.append({
            "id": str(node[1]),
            "name": get_entity_label("PERSON", node[1]),
            "degree": degree,
            "phone_links": phone_links,
            "vehicle_links": vehicle_links,
            "location_links": location_links,
            "organization_links": organization_links,
            "fir_links": fir_links,
            "network_score": (
                degree * 2
                + phone_links
                + vehicle_links
                + location_links
                + organization_links * 2
                + fir_links * 2
            ),
        })

    return features


def run_ml_prediction():
    """
    Run a prototype ML anomaly pipeline.

    IsolationForest is used when scikit-learn is available. If it is not
    installed, a transparent percentile-based fallback keeps the UI usable.
    """
    features = build_ml_features()

    if not features:
        return {
            "available": False,
            "method": "No person-level graph features available",
            "rows": [],
        }

    numeric_keys = [
        "degree",
        "phone_links",
        "vehicle_links",
        "location_links",
        "organization_links",
        "fir_links",
        "network_score",
    ]

    matrix = [
        [float(row[key]) for key in numeric_keys]
        for row in features
    ]

    method = "Isolation Forest"

    try:
        from sklearn.ensemble import IsolationForest

        if len(matrix) >= 5:
            model = IsolationForest(
                n_estimators=150,
                contamination="auto",
                random_state=42
            )
            predictions = model.fit_predict(matrix)
            anomaly_scores = model.decision_function(matrix)

            rows = []
            for row, prediction, score in zip(
                features, predictions, anomaly_scores
            ):
                # Lower decision_function values are more anomalous.
                normalized = max(
                    0.0,
                    min(100.0, 50.0 - float(score) * 100.0)
                )

                if prediction == -1 or normalized >= 65:
                    label = "ANOMALOUS"
                else:
                    label = "NORMAL"

                rows.append({
                    **row,
                    "ml_score": round(normalized, 2),
                    "prediction": label,
                })

            rows.sort(key=lambda x: -x["ml_score"])

            return {
                "available": True,
                "method": method,
                "rows": rows,
            }

    except Exception:
        method = "Rule-based ML fallback"

    # Dependency-free fallback for small/demo environments.
    values = [r["network_score"] for r in features]
    ordered = sorted(values)
    if ordered:
        threshold_index = max(0, int(len(ordered) * 0.75) - 1)
        threshold = ordered[threshold_index]
    else:
        threshold = 0

    rows = []
    maximum = max(values) if values else 1

    for row in features:
        raw = row["network_score"]
        score = (
            (raw / maximum) * 100.0
            if maximum > 0 else 0.0
        )

        # Top quartile or clearly high connectivity becomes an anomaly lead.
        label = (
            "ANOMALOUS"
            if raw >= threshold and raw > 0
            else "NORMAL"
        )

        rows.append({
            **row,
            "ml_score": round(score, 2),
            "prediction": label,
        })

    rows.sort(key=lambda x: -x["ml_score"])

    return {
        "available": True,
        "method": method,
        "rows": rows,
    }


def open_ml_prediction_workspace():
    """Open the ML prediction and feature-inspection workspace."""
    from tkinter import messagebox

    win = tk.Toplevel(root)
    win.title("ML Risk & Anomaly Prediction")
    win.geometry("1350x820")
    win.minsize(950, 620)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="ML RISK & ANOMALY PREDICTION",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text="Person-level feature engineering and anomaly scoring",
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    summary = tk.Frame(outer, bg=bg)
    summary.pack(fill="x", padx=16, pady=(0, 10))

    cards = {}
    for title, key in [
        ("PERSONS", "persons"),
        ("ANOMALOUS", "anomalous"),
        ("NORMAL", "normal"),
        ("METHOD", "method"),
    ]:
        card = tk.Frame(
            summary, bg=surface,
            highlightthickness=1,
            highlightbackground=border
        )
        card.pack(side="left", fill="x", expand=True, padx=3)

        tk.Label(
            card, text=title,
            font=("Segoe UI", 8, "bold"),
            bg=surface, fg=muted
        ).pack(pady=(10, 2))

        value = tk.StringVar(value="—")
        cards[key] = value

        tk.Label(
            card, textvariable=value,
            font=("Segoe UI", 14, "bold"),
            bg=surface, fg=text_color
        ).pack(pady=(0, 10))

    table_frame = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    table_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    tree = ttk.Treeview(
        table_frame,
        columns=(
            "name", "degree", "phone", "vehicle",
            "location", "organization", "fir",
            "score", "prediction"
        ),
        show="headings",
        selectmode="browse"
    )

    for col, heading, width in [
        ("name", "PERSON", 220),
        ("degree", "DEGREE", 80),
        ("phone", "PHONE", 80),
        ("vehicle", "VEHICLE", 90),
        ("location", "LOCATION", 100),
        ("organization", "ORG", 80),
        ("fir", "FIR", 70),
        ("score", "ML SCORE", 100),
        ("prediction", "PREDICTION", 120),
    ]:
        tree.heading(col, text=heading)
        tree.column(col, width=width, anchor="w")

    sy = ttk.Scrollbar(
        table_frame, orient="vertical", command=tree.yview
    )
    sx = ttk.Scrollbar(
        table_frame, orient="horizontal", command=tree.xview
    )
    tree.configure(
        yscrollcommand=sy.set,
        xscrollcommand=sx.set
    )

    tree.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=(10, 0))
    sy.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=(10, 0))
    sx.grid(row=1, column=0, sticky="ew", padx=(10, 0), pady=(0, 10))

    table_frame.grid_rowconfigure(0, weight=1)
    table_frame.grid_columnconfigure(0, weight=1)

    holder = {"data": None}
    status = tk.StringVar(value="Ready.")

    def load():
        data = run_ml_prediction()
        holder["data"] = data

        for item in tree.get_children():
            tree.delete(item)

        if not data["available"]:
            cards["persons"].set("0")
            cards["anomalous"].set("0")
            cards["normal"].set("0")
            cards["method"].set("—")
            status.set("No person-level features available.")
            return

        rows = data["rows"]
        anomalous = sum(
            row["prediction"] == "ANOMALOUS"
            for row in rows
        )

        cards["persons"].set(str(len(rows)))
        cards["anomalous"].set(str(anomalous))
        cards["normal"].set(str(len(rows) - anomalous))
        cards["method"].set(data["method"])

        for row in rows:
            tree.insert(
                "", "end",
                values=(
                    row["name"],
                    row["degree"],
                    row["phone_links"],
                    row["vehicle_links"],
                    row["location_links"],
                    row["organization_links"],
                    row["fir_links"],
                    f"{row['ml_score']:.2f}",
                    row["prediction"],
                )
            )

        status.set(
            f"{len(rows)} persons scored • "
            f"{anomalous} anomaly lead(s)"
        )

    def show_features():
        selection = tree.selection()
        if not selection:
            messagebox.showinfo(
                "Select Person",
                "Select a person first.",
                parent=win
            )
            return

        data = holder["data"]
        index = tree.index(selection[0])

        if not data or index >= len(data["rows"]):
            return

        row = data["rows"][index]

        messagebox.showinfo(
            "ML Feature Explanation",
            f"Person: {row['name']}\n"
            f"Prediction: {row['prediction']}\n"
            f"ML Score: {row['ml_score']:.2f}\n\n"
            f"Network degree: {row['degree']}\n"
            f"Phone links: {row['phone_links']}\n"
            f"Vehicle links: {row['vehicle_links']}\n"
            f"Location links: {row['location_links']}\n"
            f"Organization links: {row['organization_links']}\n"
            f"FIR links: {row['fir_links']}\n\n"
            "The score is an analytical model output and should be "
            "validated against source evidence.",
            parent=win
        )

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=(0, 8))

    tk.Button(
        toolbar,
        text="RUN ML",
        command=load,
        font=("Segoe UI", 9, "bold"),
        padx=16, pady=7
    ).pack(side="left", padx=(0, 6))

    tk.Button(
        toolbar,
        text="VIEW FEATURES",
        command=show_features,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=7
    ).pack(side="left")

    tk.Label(
        toolbar,
        textvariable=status,
        font=("Segoe UI", 9),
        bg=bg, fg=muted
    ).pack(side="left", padx=12)

    tk.Label(
        outer,
        text=(
            "ML output is a prioritization aid, not a determination of "
            "criminality. Verify anomalies with source evidence."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    load()
    return win


# =========================================================
# STEP 22 — MULTI-SOURCE UNIFIED INTELLIGENCE
# =========================================================

def build_unified_intelligence(person_id=None):
    """Combine relationship, CDR, financial and ML indicators for a person."""
    people = build_ml_features()

    if person_id is not None:
        people = [
            p for p in people
            if str(p["id"]) == str(person_id)
        ]

    cdr_data = analyze_cdr_records(person_id)
    fin_data = analyze_financial_transactions(person_id)

    cdr_summary = cdr_data.get("summary", {}) if cdr_data.get("available") else {}
    fin_summary = fin_data.get("summary", {}) if fin_data.get("available") else {}

    results = []
    for person in people:
        network_score = float(person.get("network_score", 0))

        call_count = float(cdr_summary.get("total_calls", 0))
        repeated_contacts = float(cdr_summary.get("repeated_contacts", 0))
        high_value = float(fin_summary.get("high_value", 0))
        repeated_pairs = float(fin_summary.get("repeated_pairs", 0))

        unified_score = (
            network_score
            + min(call_count, 50) * 0.5
            + repeated_contacts * 2
            + high_value * 3
            + repeated_pairs * 2
        )

        if unified_score >= 80:
            priority = "HIGH"
        elif unified_score >= 35:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        results.append({
            **person,
            "call_count": int(call_count),
            "repeated_contacts": int(repeated_contacts),
            "high_value_transactions": int(high_value),
            "repeated_financial_pairs": int(repeated_pairs),
            "unified_score": round(unified_score, 2),
            "priority": priority,
        })

    results.sort(key=lambda x: -x["unified_score"])

    return {
        "available": bool(results),
        "rows": results,
        "sources": {
            "relationship_graph": True,
            "cdr": bool(cdr_data.get("available")),
            "financial": bool(fin_data.get("available")),
            "ml_features": bool(people),
        },
    }


def open_unified_intelligence_workspace():
    """Open a single investigator view across multiple intelligence sources."""
    from tkinter import messagebox, simpledialog

    person_id = simpledialog.askstring(
        "Unified Intelligence",
        "Enter Person ID (leave blank for all persons):",
        parent=root
    )

    if person_id == "":
        person_id = None

    win = tk.Toplevel(root)
    win.title("Unified Intelligence")
    win.geometry("1450x830")
    win.minsize(1000, 620)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="UNIFIED INTELLIGENCE",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text="Relationship + CDR + financial + ML indicators in one view",
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    summary = tk.Frame(outer, bg=bg)
    summary.pack(fill="x", padx=16, pady=(0, 10))

    cards = {}
    for title, key in [
        ("PERSONS", "persons"),
        ("HIGH PRIORITY", "high"),
        ("MEDIUM", "medium"),
        ("LOW", "low"),
    ]:
        card = tk.Frame(
            summary, bg=surface,
            highlightthickness=1,
            highlightbackground=border
        )
        card.pack(side="left", fill="x", expand=True, padx=3)

        tk.Label(
            card, text=title,
            font=("Segoe UI", 8, "bold"),
            bg=surface, fg=muted
        ).pack(pady=(10, 2))

        variable = tk.StringVar(value="—")
        cards[key] = variable

        tk.Label(
            card, textvariable=variable,
            font=("Segoe UI", 17, "bold"),
            bg=surface, fg=text_color
        ).pack(pady=(0, 10))

    table_frame = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    table_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    tree = ttk.Treeview(
        table_frame,
        columns=(
            "person", "network", "calls", "repeat_calls",
            "high_value", "repeat_fin", "score", "priority"
        ),
        show="headings",
        selectmode="browse"
    )

    for col, heading, width in [
        ("person", "PERSON", 240),
        ("network", "NETWORK", 100),
        ("calls", "CALLS", 80),
        ("repeat_calls", "REPEATED CALLS", 120),
        ("high_value", "HIGH-VALUE TX", 120),
        ("repeat_fin", "REPEATED TX", 110),
        ("score", "UNIFIED SCORE", 120),
        ("priority", "PRIORITY", 110),
    ]:
        tree.heading(col, text=heading)
        tree.column(col, width=width, anchor="w")

    sy = ttk.Scrollbar(
        table_frame, orient="vertical", command=tree.yview
    )
    sx = ttk.Scrollbar(
        table_frame, orient="horizontal", command=tree.xview
    )
    tree.configure(
        yscrollcommand=sy.set,
        xscrollcommand=sx.set
    )

    tree.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=(10, 0))
    sy.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=(10, 0))
    sx.grid(row=1, column=0, sticky="ew", padx=(10, 0), pady=(0, 10))

    table_frame.grid_rowconfigure(0, weight=1)
    table_frame.grid_columnconfigure(0, weight=1)

    holder = {"data": None}
    status = tk.StringVar(value="Ready.")

    def load():
        data = build_unified_intelligence(person_id)
        holder["data"] = data

        for item in tree.get_children():
            tree.delete(item)

        rows = data["rows"]

        high = sum(r["priority"] == "HIGH" for r in rows)
        medium = sum(r["priority"] == "MEDIUM" for r in rows)
        low = sum(r["priority"] == "LOW" for r in rows)

        cards["persons"].set(str(len(rows)))
        cards["high"].set(str(high))
        cards["medium"].set(str(medium))
        cards["low"].set(str(low))

        for row in rows:
            tree.insert(
                "", "end",
                values=(
                    row["name"],
                    row["degree"],
                    row["call_count"],
                    row["repeated_contacts"],
                    row["high_value_transactions"],
                    row["repeated_financial_pairs"],
                    f"{row['unified_score']:.2f}",
                    row["priority"],
                )
            )

        active = [
            name for name, enabled in data["sources"].items()
            if enabled
        ]
        status.set(
            f"{len(rows)} persons • Sources: {', '.join(active)}"
        )

    def show_selected():
        selection = tree.selection()
        if not selection:
            messagebox.showinfo(
                "Select Person",
                "Select a person first.",
                parent=win
            )
            return

        data = holder["data"]
        index = tree.index(selection[0])

        if not data or index >= len(data["rows"]):
            return

        row = data["rows"][index]

        messagebox.showinfo(
            "Unified Intelligence Detail",
            f"Person: {row['name']}\n"
            f"Priority: {row['priority']}\n"
            f"Unified Score: {row['unified_score']:.2f}\n\n"
            f"Network degree: {row['degree']}\n"
            f"Calls: {row['call_count']}\n"
            f"Repeated contacts: {row['repeated_contacts']}\n"
            f"High-value transactions: {row['high_value_transactions']}\n"
            f"Repeated financial pairs: {row['repeated_financial_pairs']}\n\n"
            "This consolidated score is an analytical prioritization "
            "indicator and requires verification against source evidence.",
            parent=win
        )

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=(0, 8))

    tk.Button(
        toolbar,
        text="REFRESH INTELLIGENCE",
        command=load,
        font=("Segoe UI", 9, "bold"),
        padx=16, pady=7
    ).pack(side="left", padx=(0, 6))

    tk.Button(
        toolbar,
        text="VIEW PERSON",
        command=show_selected,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=7
    ).pack(side="left")

    tk.Label(
        toolbar,
        textvariable=status,
        font=("Segoe UI", 9),
        bg=bg, fg=muted
    ).pack(side="left", padx=12)

    tk.Label(
        outer,
        text=(
            "Unified intelligence combines multiple indicators for review; "
            "it does not establish criminal responsibility."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    load()
    return win


# =========================================================
# STEP 23 — INVESTIGATOR NETWORK VISUALIZATION
# =========================================================

def build_visual_network(person_id=None):
    """Create a compact network representation from the existing graph."""
    try:
        graph = _person_network_graph()
    except Exception:
        graph = {}

    if not graph:
        return {"nodes": [], "edges": []}

    # Normalize graph into undirected adjacency.
    adj = {}
    for node, neighbors in graph.items():
        adj.setdefault(node, set())
        for neighbor in neighbors or set():
            adj.setdefault(neighbor, set())
            adj[node].add(neighbor)
            adj[neighbor].add(node)

    selected = None
    if person_id is not None:
        for node in adj:
            if (
                isinstance(node, tuple)
                and len(node) >= 2
                and str(node[0]).upper() == "PERSON"
                and str(node[1]) == str(person_id)
            ):
                selected = node
                break

    if selected is not None:
        visible = {selected}
        visible.update(adj.get(selected, set()))
    else:
        # Keep the UI responsive for very large prototype graphs.
        visible = set(list(adj.keys())[:180])

    nodes = []
    for node in visible:
        if isinstance(node, tuple) and len(node) >= 2:
            kind = str(node[0]).upper()
            value = node[1]
        else:
            kind = "ENTITY"
            value = node

        try:
            label = get_entity_label(kind, value)
        except Exception:
            label = str(value)

        nodes.append({
            "key": repr(node),
            "node": node,
            "kind": kind,
            "value": str(value),
            "label": str(label),
            "degree": len(adj.get(node, set())),
        })

    visible_set = {n["node"] for n in nodes}
    edges = []
    seen = set()

    for node in visible_set:
        for neighbor in adj.get(node, set()):
            if neighbor not in visible_set:
                continue

            edge_key = frozenset((repr(node), repr(neighbor)))
            if edge_key in seen:
                continue
            seen.add(edge_key)

            edges.append({
                "source": repr(node),
                "target": repr(neighbor),
                "source_node": node,
                "target_node": neighbor,
            })

    return {"nodes": nodes, "edges": edges}


def open_network_visualization():
    """Open a dependency-free interactive-style network viewer."""
    from tkinter import messagebox, simpledialog

    person_id = simpledialog.askstring(
        "Network Visualization",
        "Enter Person ID to focus (leave blank for complete network):",
        parent=root
    )

    if person_id == "":
        person_id = None

    win = tk.Toplevel(root)
    win.title("Investigator Network Visualization")
    win.geometry("1450x850")
    win.minsize(1000, 650)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="INVESTIGATOR NETWORK VISUALIZATION",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text="Explore people, phones, vehicles, locations, organizations and FIR links",
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    content = tk.Frame(outer, bg=bg)
    content.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    canvas_frame = tk.Frame(
        content, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    canvas_frame.pack(side="left", fill="both", expand=True)

    canvas = tk.Canvas(
        canvas_frame,
        bg="white",
        highlightthickness=0
    )
    canvas.pack(fill="both", expand=True, padx=8, pady=8)

    side = tk.Frame(
        content, bg=surface,
        width=310,
        highlightthickness=1,
        highlightbackground=border
    )
    side.pack(side="right", fill="y", padx=(10, 0))
    side.pack_propagate(False)

    tk.Label(
        side,
        text="NETWORK DETAILS",
        font=("Segoe UI", 11, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=16, pady=(16, 10))

    details = tk.StringVar(
        value="Select a node to inspect it."
    )

    tk.Label(
        side,
        textvariable=details,
        justify="left",
        wraplength=270,
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=16)

    status = tk.StringVar(value="Loading network...")
    holder = {"data": None, "positions": {}, "items": {}}

    def kind_short(kind):
        mapping = {
            "PERSON": "PERSON",
            "PHONE": "PHONE",
            "VEHICLE": "VEHICLE",
            "LOCATION": "LOCATION",
            "ORGANIZATION": "ORG",
            "FIR": "FIR",
        }
        return mapping.get(kind, kind)

    def draw():
        data = holder["data"]
        canvas.delete("all")
        holder["positions"].clear()
        holder["items"].clear()

        if not data or not data["nodes"]:
            canvas.create_text(
                400, 250,
                text="No relationship data available",
                font=("Segoe UI", 14),
                fill=text_color
            )
            return

        width = max(canvas.winfo_width(), 800)
        height = max(canvas.winfo_height(), 600)

        cx = width / 2
        cy = height / 2

        selected_key = None
        if person_id is not None:
            for node in data["nodes"]:
                if (
                    node["kind"] == "PERSON"
                    and node["value"] == str(person_id)
                ):
                    selected_key = node["key"]
                    break

        # Simple radial layout; no external graph package required.
        import math

        if selected_key:
            center_nodes = [
                n for n in data["nodes"]
                if n["key"] == selected_key
            ]
            center = center_nodes[0] if center_nodes else data["nodes"][0]
        else:
            center = max(
                data["nodes"],
                key=lambda n: n["degree"]
            )

        ordered = [
            n for n in data["nodes"]
            if n["key"] != center["key"]
        ]

        holder["positions"][center["key"]] = (cx, cy)

        radius = min(width, height) * 0.34
        total = max(len(ordered), 1)

        for i, node in enumerate(ordered):
            angle = (2 * math.pi * i / total) - math.pi / 2
            holder["positions"][node["key"]] = (
                cx + radius * math.cos(angle),
                cy + radius * math.sin(angle)
            )

        # Draw edges first.
        for edge in data["edges"]:
            a = holder["positions"].get(edge["source"])
            b = holder["positions"].get(edge["target"])
            if not a or not b:
                continue

            canvas.create_line(
                a[0], a[1], b[0], b[1],
                fill="#CBD5E1",
                width=1
            )

        # Draw nodes.
        for node in data["nodes"]:
            x, y = holder["positions"][node["key"]]

            if node["kind"] == "PERSON":
                radius_node = 28
            else:
                radius_node = 22

            item = canvas.create_oval(
                x - radius_node, y - radius_node,
                x + radius_node, y + radius_node,
                fill="white",
                outline="#94A3B8",
                width=2
            )

            canvas.create_text(
                x, y,
                text=kind_short(node["kind"]),
                font=("Segoe UI", 8, "bold"),
                fill=text_color
            )

            canvas.create_text(
                x, y + radius_node + 13,
                text=node["label"][:28],
                font=("Segoe UI", 8),
                fill=muted
            )

            holder["items"][item] = node

        canvas.bind("<Button-1>", on_canvas_click)

        status.set(
            f"{len(data['nodes'])} nodes • {len(data['edges'])} relationships"
        )

    def on_canvas_click(event):
        closest = canvas.find_closest(event.x, event.y)
        if not closest:
            return

        node = holder["items"].get(closest[0])
        if not node:
            details.set("Select a node to inspect it.")
            return

        details.set(
            f"Type: {node['kind']}\n"
            f"Name / ID: {node['label']}\n"
            f"Value: {node['value']}\n"
            f"Connections: {node['degree']}"
        )

    def refresh():
        holder["data"] = build_visual_network(person_id)
        draw()

    def reset_view():
        holder["data"] = build_visual_network(None)
        draw()

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=(0, 8))

    tk.Button(
        toolbar,
        text="REFRESH NETWORK",
        command=refresh,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=7
    ).pack(side="left", padx=(0, 6))

    tk.Button(
        toolbar,
        text="SHOW ALL",
        command=reset_view,
        font=("Segoe UI", 9, "bold"),
        padx=14, pady=7
    ).pack(side="left")

    tk.Label(
        toolbar,
        textvariable=status,
        font=("Segoe UI", 9),
        bg=bg, fg=muted
    ).pack(side="left", padx=12)

    tk.Label(
        outer,
        text=(
            "Network visualization is an analytical view of stored relationships. "
            "Connections should be verified against source records."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    canvas.bind("<Configure>", lambda event: draw())

    refresh()
    return win


# =========================================================
# STEP 24 — INTELLIGENCE ALERTS & PRIORITY CENTER
# =========================================================

def generate_intelligence_alerts():
    """
    Generate transparent, rule-based investigative alerts from existing
    graph and ML/financial/CDR summaries. No database schema changes.
    """
    alerts = []

    try:
        ml_result = run_ml_prediction()
    except Exception:
        ml_result = {"rows": []}

    for row in ml_result.get("rows", []):
        score = float(row.get("ml_score", 0))
        if row.get("prediction") == "ANOMALOUS" or score >= 65:
            alerts.append({
                "severity": "HIGH" if score >= 80 else "MEDIUM",
                "category": "ML ANOMALY",
                "entity": row.get("name", row.get("id", "Unknown")),
                "score": score,
                "message": (
                    f"Elevated network anomaly score ({score:.2f}) "
                    "requires source verification."
                ),
            })

    try:
        graph_data = calculate_graph_analytics()
    except Exception:
        graph_data = {"bridges": []}

    for row in graph_data.get("bridges", [])[:25]:
        bridge = float(row.get("betweenness", 0))
        if bridge > 0:
            severity = "HIGH" if bridge >= 0.25 else "MEDIUM"
            alerts.append({
                "severity": severity,
                "category": "NETWORK BRIDGE",
                "entity": row.get("label", "Unknown"),
                "score": round(bridge * 100, 2),
                "message": (
                    f"High intermediary/bridge position in the relationship "
                    f"network (score {bridge:.4f})."
                ),
            })

    # Remove duplicate entity/category alerts while keeping strongest score.
    unique = {}
    for alert in alerts:
        key = (alert["category"], str(alert["entity"]))
        if key not in unique or alert["score"] > unique[key]["score"]:
            unique[key] = alert

    final_alerts = list(unique.values())
    severity_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    final_alerts.sort(
        key=lambda a: (
            severity_rank.get(a["severity"], 9),
            -float(a["score"])
        )
    )

    return final_alerts


def open_intelligence_alerts_center():
    """Open a central alert-review workspace."""
    from tkinter import messagebox

    win = tk.Toplevel(root)
    win.title("Intelligence Alerts")
    win.geometry("1350x800")
    win.minsize(950, 600)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="INTELLIGENCE ALERTS",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text="Prioritized analytical indicators requiring investigator review",
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    summary = tk.Frame(outer, bg=bg)
    summary.pack(fill="x", padx=16, pady=(0, 10))

    cards = {}
    for title, key in [
        ("TOTAL ALERTS", "total"),
        ("HIGH", "high"),
        ("MEDIUM", "medium"),
    ]:
        card = tk.Frame(
            summary, bg=surface,
            highlightthickness=1,
            highlightbackground=border
        )
        card.pack(side="left", fill="x", expand=True, padx=3)

        tk.Label(
            card, text=title,
            font=("Segoe UI", 8, "bold"),
            bg=surface, fg=muted
        ).pack(pady=(10, 2))

        value = tk.StringVar(value="0")
        cards[key] = value

        tk.Label(
            card, textvariable=value,
            font=("Segoe UI", 17, "bold"),
            bg=surface, fg=text_color
        ).pack(pady=(0, 10))

    frame = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    tree = ttk.Treeview(
        frame,
        columns=("severity", "category", "entity", "score", "message"),
        show="headings",
        selectmode="browse"
    )

    for col, heading, width in [
        ("severity", "SEVERITY", 100),
        ("category", "CATEGORY", 170),
        ("entity", "ENTITY", 250),
        ("score", "SCORE", 100),
        ("message", "INDICATOR", 600),
    ]:
        tree.heading(col, text=heading)
        tree.column(col, width=width, anchor="w")

    sy = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    sx = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)

    tree.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=(10, 0))
    sy.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=(10, 0))
    sx.grid(row=1, column=0, sticky="ew", padx=(10, 0), pady=(0, 10))

    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    holder = {"alerts": []}
    status = tk.StringVar(value="Ready.")

    def refresh():
        alerts = generate_intelligence_alerts()
        holder["alerts"] = alerts

        for item in tree.get_children():
            tree.delete(item)

        high = sum(a["severity"] == "HIGH" for a in alerts)
        medium = sum(a["severity"] == "MEDIUM" for a in alerts)

        cards["total"].set(str(len(alerts)))
        cards["high"].set(str(high))
        cards["medium"].set(str(medium))

        for alert in alerts:
            tree.insert(
                "",
                "end",
                values=(
                    alert["severity"],
                    alert["category"],
                    alert["entity"],
                    f"{alert['score']:.2f}",
                    alert["message"],
                )
            )

        status.set(f"{len(alerts)} analytical alert(s) generated")

    def inspect():
        selection = tree.selection()
        if not selection:
            messagebox.showinfo(
                "Select Alert",
                "Select an alert first.",
                parent=win
            )
            return

        index = tree.index(selection[0])
        if index >= len(holder["alerts"]):
            return

        alert = holder["alerts"][index]

        messagebox.showinfo(
            "Alert Details",
            f"Severity: {alert['severity']}\n"
            f"Category: {alert['category']}\n"
            f"Entity: {alert['entity']}\n"
            f"Score: {alert['score']:.2f}\n\n"
            f"{alert['message']}\n\n"
            "This is an analytical lead, not a finding of criminality. "
            "Verify against original records.",
            parent=win
        )

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=(0, 8))

    tk.Button(
        toolbar,
        text="REFRESH ALERTS",
        command=refresh,
        font=("Segoe UI", 9, "bold"),
        padx=15, pady=7
    ).pack(side="left", padx=(0, 6))

    tk.Button(
        toolbar,
        text="INSPECT ALERT",
        command=inspect,
        font=("Segoe UI", 9, "bold"),
        padx=15, pady=7
    ).pack(side="left")

    tk.Label(
        toolbar,
        textvariable=status,
        font=("Segoe UI", 9),
        bg=bg, fg=muted
    ).pack(side="left", padx=12)

    tk.Label(
        outer,
        text=(
            "Alerts are generated from stored analytical indicators and are "
            "intended for review and prioritization, not automatic conclusions."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    refresh()
    return win


# =========================================================
# STEP 25 — INVESTIGATION CASE MANAGEMENT
# =========================================================

def _case_db_connection():
    """Return the application's existing SQLite connection."""
    try:
        return sqlite3.connect(DB_PATH)
    except Exception:
        return sqlite3.connect(
            os.path.join("database", "criminal_intelligence.db")
        )


def initialize_case_management_tables():
    """Create case-management tables if they do not already exist."""
    conn = _case_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS investigation_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_number TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            status TEXT DEFAULT 'OPEN',
            priority TEXT DEFAULT 'MEDIUM',
            lead_investigator TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS case_persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            person_id TEXT NOT NULL,
            role TEXT DEFAULT 'SUBJECT',
            notes TEXT DEFAULT '',
            UNIQUE(case_id, person_id),
            FOREIGN KEY(case_id) REFERENCES investigation_cases(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS case_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            note TEXT NOT NULL,
            author TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(case_id) REFERENCES investigation_cases(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS case_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            evidence_type TEXT DEFAULT 'OTHER',
            reference TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(case_id) REFERENCES investigation_cases(id)
        )
    """)

    conn.commit()
    conn.close()


def create_investigation_case(
    case_number,
    title,
    description="",
    status="OPEN",
    priority="MEDIUM",
    investigator=""
):
    initialize_case_management_tables()

    conn = _case_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO investigation_cases
        (case_number, title, description, status, priority, lead_investigator)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        case_number.strip(),
        title.strip(),
        description.strip(),
        status,
        priority,
        investigator.strip()
    ))

    conn.commit()
    case_id = cur.lastrowid
    conn.close()
    return case_id


def list_investigation_cases():
    initialize_case_management_tables()

    conn = _case_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            id, case_number, title, status, priority,
            lead_investigator, created_at, updated_at
        FROM investigation_cases
        ORDER BY updated_at DESC, id DESC
    """)

    rows = cur.fetchall()
    conn.close()
    return rows


def get_case_details(case_id):
    initialize_case_management_tables()

    conn = _case_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            id, case_number, title, description,
            status, priority, lead_investigator,
            created_at, updated_at
        FROM investigation_cases
        WHERE id = ?
    """, (case_id,))
    case = cur.fetchone()

    cur.execute("""
        SELECT person_id, role, notes
        FROM case_persons
        WHERE case_id = ?
        ORDER BY id DESC
    """, (case_id,))
    persons = cur.fetchall()

    cur.execute("""
        SELECT note, author, created_at
        FROM case_notes
        WHERE case_id = ?
        ORDER BY id DESC
    """, (case_id,))
    notes = cur.fetchall()

    cur.execute("""
        SELECT evidence_type, reference, description, created_at
        FROM case_evidence
        WHERE case_id = ?
        ORDER BY id DESC
    """, (case_id,))
    evidence = cur.fetchall()

    conn.close()

    return {
        "case": case,
        "persons": persons,
        "notes": notes,
        "evidence": evidence
    }


def add_person_to_case(case_id, person_id, role="SUBJECT", notes=""):
    initialize_case_management_tables()

    conn = _case_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO case_persons
        (case_id, person_id, role, notes)
        VALUES (?, ?, ?, ?)
    """, (
        case_id,
        str(person_id).strip(),
        role.strip() or "SUBJECT",
        notes.strip()
    ))

    cur.execute("""
        UPDATE investigation_cases
        SET updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (case_id,))

    conn.commit()
    conn.close()


def add_case_note(case_id, note, author=""):
    if not note.strip():
        return

    initialize_case_management_tables()

    conn = _case_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO case_notes
        (case_id, note, author)
        VALUES (?, ?, ?)
    """, (case_id, note.strip(), author.strip()))

    cur.execute("""
        UPDATE investigation_cases
        SET updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (case_id,))

    conn.commit()
    conn.close()


def add_case_evidence(
    case_id,
    evidence_type,
    reference,
    description=""
):
    if not reference.strip():
        return

    initialize_case_management_tables()

    conn = _case_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO case_evidence
        (case_id, evidence_type, reference, description)
        VALUES (?, ?, ?, ?)
    """, (
        case_id,
        evidence_type.strip() or "OTHER",
        reference.strip(),
        description.strip()
    ))

    cur.execute("""
        UPDATE investigation_cases
        SET updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (case_id,))

    conn.commit()
    conn.close()


def open_case_management():
    """Investigator case workspace."""
    from tkinter import messagebox, simpledialog

    initialize_case_management_tables()

    win = tk.Toplevel(root)
    win.title("Investigation Case Management")
    win.geometry("1450x850")
    win.minsize(1000, 650)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="INVESTIGATION CASE MANAGEMENT",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text="Create cases and organize subjects, evidence and investigation notes",
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    body = tk.Frame(outer, bg=bg)
    body.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    left = tk.Frame(
        body, bg=surface,
        width=520,
        highlightthickness=1,
        highlightbackground=border
    )
    left.pack(side="left", fill="both", padx=(0, 8))
    left.pack_propagate(False)

    right = tk.Frame(
        body, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    right.pack(side="left", fill="both", expand=True)

    tk.Label(
        left,
        text="CASES",
        font=("Segoe UI", 11, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=14, pady=(14, 8))

    case_tree = ttk.Treeview(
        left,
        columns=("number", "title", "status", "priority"),
        show="headings",
        selectmode="browse"
    )

    for col, heading, width in [
        ("number", "CASE", 120),
        ("title", "TITLE", 190),
        ("status", "STATUS", 90),
        ("priority", "PRIORITY", 90),
    ]:
        case_tree.heading(col, text=heading)
        case_tree.column(col, width=width, anchor="w")

    case_tree.pack(fill="both", expand=True, padx=14, pady=(0, 10))

    tk.Label(
        right,
        text="CASE DETAILS",
        font=("Segoe UI", 11, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=16, pady=(14, 8))

    details_text = tk.Text(
        right,
        height=8,
        wrap="word",
        font=("Segoe UI", 9),
        relief="flat",
        bg="white",
        fg=text_color
    )
    details_text.pack(fill="x", padx=16, pady=(0, 10))
    details_text.configure(state="disabled")

    notebook = ttk.Notebook(right)
    notebook.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    persons_tab = tk.Frame(notebook, bg=surface)
    notes_tab = tk.Frame(notebook, bg=surface)
    evidence_tab = tk.Frame(notebook, bg=surface)

    notebook.add(persons_tab, text="Persons")
    notebook.add(notes_tab, text="Notes")
    notebook.add(evidence_tab, text="Evidence")

    persons_tree = ttk.Treeview(
        persons_tab,
        columns=("person", "role", "notes"),
        show="headings"
    )
    for col, heading, width in [
        ("person", "PERSON ID", 180),
        ("role", "ROLE", 120),
        ("notes", "NOTES", 500),
    ]:
        persons_tree.heading(col, text=heading)
        persons_tree.column(col, width=width, anchor="w")
    persons_tree.pack(fill="both", expand=True, padx=10, pady=10)

    notes_tree = ttk.Treeview(
        notes_tab,
        columns=("note", "author", "date"),
        show="headings"
    )
    for col, heading, width in [
        ("note", "NOTE", 600),
        ("author", "AUTHOR", 130),
        ("date", "DATE", 170),
    ]:
        notes_tree.heading(col, text=heading)
        notes_tree.column(col, width=width, anchor="w")
    notes_tree.pack(fill="both", expand=True, padx=10, pady=10)

    evidence_tree = ttk.Treeview(
        evidence_tab,
        columns=("type", "reference", "description", "date"),
        show="headings"
    )
    for col, heading, width in [
        ("type", "TYPE", 120),
        ("reference", "REFERENCE", 230),
        ("description", "DESCRIPTION", 430),
        ("date", "DATE", 170),
    ]:
        evidence_tree.heading(col, text=heading)
        evidence_tree.column(col, width=width, anchor="w")
    evidence_tree.pack(fill="both", expand=True, padx=10, pady=10)

    selected = {"id": None}
    status_var = tk.StringVar(value="Ready.")

    def show_case(case_id):
        selected["id"] = case_id
        data = get_case_details(case_id)

        case = data["case"]
        if not case:
            return

        details_text.configure(state="normal")
        details_text.delete("1.0", "end")
        details_text.insert(
            "end",
            f"Case Number: {case[1]}\n"
            f"Title: {case[2]}\n"
            f"Status: {case[4]}\n"
            f"Priority: {case[5]}\n"
            f"Lead Investigator: {case[6]}\n"
            f"Created: {case[7]}\n"
            f"Updated: {case[8]}\n\n"
            f"{case[3]}"
        )
        details_text.configure(state="disabled")

        for tree in (persons_tree, notes_tree, evidence_tree):
            for item in tree.get_children():
                tree.delete(item)

        for person in data["persons"]:
            persons_tree.insert("", "end", values=person)

        for note in data["notes"]:
            notes_tree.insert("", "end", values=note)

        for evidence in data["evidence"]:
            evidence_tree.insert("", "end", values=evidence)

        status_var.set(
            f"Case {case[1]} • "
            f"{len(data['persons'])} persons • "
            f"{len(data['evidence'])} evidence items"
        )

    def refresh_cases():
        for item in case_tree.get_children():
            case_tree.delete(item)

        for row in list_investigation_cases():
            case_tree.insert(
                "",
                "end",
                iid=str(row[0]),
                values=(row[1], row[2], row[3], row[4])
            )

        status_var.set("Case list refreshed.")

    def selected_case_id():
        selection = case_tree.selection()
        if not selection:
            messagebox.showinfo(
                "Select Case",
                "Select a case first.",
                parent=win
            )
            return None
        return int(selection[0])

    def create_case():
        case_number = simpledialog.askstring(
            "New Case", "Case Number:", parent=win
        )
        if not case_number:
            return

        title = simpledialog.askstring(
            "New Case", "Case Title:", parent=win
        )
        if not title:
            return

        description = simpledialog.askstring(
            "New Case", "Short Description:", parent=win
        ) or ""

        investigator = simpledialog.askstring(
            "New Case", "Lead Investigator:", parent=win
        ) or ""

        try:
            case_id = create_investigation_case(
                case_number,
                title,
                description,
                investigator=investigator
            )
            refresh_cases()
            case_tree.selection_set(str(case_id))
            case_tree.focus(str(case_id))
            show_case(case_id)
            status_var.set("Case created successfully.")
        except sqlite3.IntegrityError:
            messagebox.showerror(
                "Duplicate Case",
                "That case number already exists.",
                parent=win
            )
        except Exception as exc:
            messagebox.showerror(
                "Case Error",
                str(exc),
                parent=win
            )

    def add_person():
        case_id = selected_case_id()
        if case_id is None:
            return

        person = simpledialog.askstring(
            "Add Person",
            "Person ID:",
            parent=win
        )
        if not person:
            return

        role = simpledialog.askstring(
            "Add Person",
            "Role (SUBJECT / WITNESS / ASSOCIATE / OTHER):",
            parent=win
        ) or "SUBJECT"

        notes = simpledialog.askstring(
            "Add Person",
            "Notes:",
            parent=win
        ) or ""

        add_person_to_case(case_id, person, role, notes)
        show_case(case_id)

    def add_note():
        case_id = selected_case_id()
        if case_id is None:
            return

        note = simpledialog.askstring(
            "Case Note", "Investigation Note:", parent=win
        )
        if not note:
            return

        author = simpledialog.askstring(
            "Case Note", "Author:", parent=win
        ) or ""

        add_case_note(case_id, note, author)
        show_case(case_id)

    def add_evidence():
        case_id = selected_case_id()
        if case_id is None:
            return

        evidence_type = simpledialog.askstring(
            "Evidence",
            "Evidence Type (FIR / CDR / FINANCIAL / REPORT / OTHER):",
            parent=win
        ) or "OTHER"

        reference = simpledialog.askstring(
            "Evidence",
            "Evidence Reference:",
            parent=win
        )
        if not reference:
            return

        description = simpledialog.askstring(
            "Evidence",
            "Description:",
            parent=win
        ) or ""

        add_case_evidence(
            case_id,
            evidence_type,
            reference,
            description
        )
        show_case(case_id)

    case_tree.bind(
        "<<TreeviewSelect>>",
        lambda event: (
            show_case(int(case_tree.selection()[0]))
            if case_tree.selection()
            else None
        )
    )

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=(0, 8))

    for label, command in [
        ("NEW CASE", create_case),
        ("ADD PERSON", add_person),
        ("ADD NOTE", add_note),
        ("ADD EVIDENCE", add_evidence),
        ("REFRESH", refresh_cases),
    ]:
        tk.Button(
            toolbar,
            text=label,
            command=command,
            font=("Segoe UI", 9, "bold"),
            padx=13,
            pady=7
        ).pack(side="left", padx=(0, 6))

    tk.Label(
        toolbar,
        textvariable=status_var,
        font=("Segoe UI", 9),
        bg=bg,
        fg=muted
    ).pack(side="left", padx=8)

    tk.Label(
        outer,
        text=(
            "Case management organizes investigative information. "
            "Records and analytical leads should be validated against source evidence."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    refresh_cases()
    return win


# =========================================================
# STEP 26 — INVESTIGATION TIMELINE & EVENT CORRELATION
# =========================================================

def initialize_timeline_table():
    """Create the optional investigation timeline table."""
    initialize_case_management_tables()

    conn = _case_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS case_timeline_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            event_date TEXT NOT NULL,
            event_type TEXT DEFAULT 'OTHER',
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            source_reference TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(case_id) REFERENCES investigation_cases(id)
        )
    """)

    conn.commit()
    conn.close()


def add_timeline_event(
    case_id,
    event_date,
    event_type,
    title,
    description="",
    source_reference=""
):
    initialize_timeline_table()

    if not str(event_date).strip() or not str(title).strip():
        return

    conn = _case_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO case_timeline_events
        (case_id, event_date, event_type, title, description, source_reference)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        case_id,
        str(event_date).strip(),
        str(event_type).strip() or "OTHER",
        str(title).strip(),
        str(description).strip(),
        str(source_reference).strip()
    ))

    cur.execute("""
        UPDATE investigation_cases
        SET updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (case_id,))

    conn.commit()
    conn.close()


def get_case_timeline(case_id):
    initialize_timeline_table()

    conn = _case_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            id, event_date, event_type, title,
            description, source_reference
        FROM case_timeline_events
        WHERE case_id = ?
        ORDER BY event_date ASC, id ASC
    """, (case_id,))

    rows = cur.fetchall()
    conn.close()
    return rows


def _timeline_event_color(event_type):
    """Return a restrained UI color for event categories."""
    palette = {
        "FIR": "#334155",
        "CDR": "#475569",
        "FINANCIAL": "#64748B",
        "SURVEILLANCE": "#0F172A",
        "LOCATION": "#1E293B",
        "MEETING": "#374151",
        "REPORT": "#4B5563",
        "OTHER": "#6B7280",
    }
    return palette.get(str(event_type).upper(), "#6B7280")


def open_investigation_timeline():
    """Open a case timeline and event-correlation workspace."""
    from tkinter import messagebox, simpledialog

    initialize_timeline_table()

    cases = list_investigation_cases()
    if not cases:
        messagebox.showinfo(
            "No Cases",
            "Create an investigation case first.",
            parent=root
        )
        return

    win = tk.Toplevel(root)
    win.title("Investigation Timeline")
    win.geometry("1450x850")
    win.minsize(1000, 650)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer,
        bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="INVESTIGATION TIMELINE",
        font=("Segoe UI", 18, "bold"),
        bg=surface,
        fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text="Correlate dated FIR, CDR, financial, surveillance and report events",
        font=("Segoe UI", 9),
        bg=surface,
        fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    body = tk.Frame(outer, bg=bg)
    body.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    left = tk.Frame(
        body,
        bg=surface,
        width=360,
        highlightthickness=1,
        highlightbackground=border
    )
    left.pack(side="left", fill="y", padx=(0, 8))
    left.pack_propagate(False)

    right = tk.Frame(
        body,
        bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    right.pack(side="left", fill="both", expand=True)

    tk.Label(
        left,
        text="SELECT CASE",
        font=("Segoe UI", 11, "bold"),
        bg=surface,
        fg=text_color
    ).pack(anchor="w", padx=14, pady=(14, 8))

    case_tree = ttk.Treeview(
        left,
        columns=("number", "title", "status"),
        show="headings",
        selectmode="browse"
    )

    for col, heading, width in [
        ("number", "CASE", 105),
        ("title", "TITLE", 165),
        ("status", "STATUS", 80),
    ]:
        case_tree.heading(col, text=heading)
        case_tree.column(col, width=width, anchor="w")

    case_tree.pack(fill="both", expand=True, padx=14, pady=(0, 12))

    for row in cases:
        case_tree.insert(
            "",
            "end",
            iid=str(row[0]),
            values=(row[1], row[2], row[3])
        )

    tk.Label(
        right,
        text="EVENT TIMELINE",
        font=("Segoe UI", 11, "bold"),
        bg=surface,
        fg=text_color
    ).pack(anchor="w", padx=16, pady=(14, 8))

    timeline_tree = ttk.Treeview(
        right,
        columns=(
            "date",
            "type",
            "title",
            "description",
            "source"
        ),
        show="headings",
        selectmode="browse"
    )

    for col, heading, width in [
        ("date", "DATE / TIME", 145),
        ("type", "TYPE", 120),
        ("title", "EVENT", 240),
        ("description", "DESCRIPTION", 430),
        ("source", "SOURCE", 220),
    ]:
        timeline_tree.heading(col, text=heading)
        timeline_tree.column(col, width=width, anchor="w")

    sy = ttk.Scrollbar(
        right,
        orient="vertical",
        command=timeline_tree.yview
    )
    sx = ttk.Scrollbar(
        right,
        orient="horizontal",
        command=timeline_tree.xview
    )
    timeline_tree.configure(
        yscrollcommand=sy.set,
        xscrollcommand=sx.set
    )

    timeline_tree.pack(
        side="left",
        fill="both",
        expand=True,
        padx=(16, 0),
        pady=(0, 10)
    )
    sy.pack(side="right", fill="y", padx=(0, 10), pady=(0, 10))
    sx.pack(side="bottom", fill="x", padx=16, pady=(0, 10))

    selected_case = {"id": None}
    status_var = tk.StringVar(value="Select a case.")

    def load_events(case_id):
        selected_case["id"] = case_id

        for item in timeline_tree.get_children():
            timeline_tree.delete(item)

        rows = get_case_timeline(case_id)

        for row in rows:
            timeline_tree.insert(
                "",
                "end",
                values=(
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5]
                )
            )

        case = get_case_details(case_id).get("case")
        if case:
            status_var.set(
                f"{case[1]} • {len(rows)} timeline event(s)"
            )

    def add_event():
        case_id = selected_case["id"]
        if case_id is None:
            messagebox.showinfo(
                "Select Case",
                "Select a case first.",
                parent=win
            )
            return

        event_date = simpledialog.askstring(
            "Timeline Event",
            "Date / Time (YYYY-MM-DD HH:MM):",
            parent=win
        )
        if not event_date:
            return

        event_type = simpledialog.askstring(
            "Timeline Event",
            "Type (FIR / CDR / FINANCIAL / SURVEILLANCE / LOCATION / REPORT / OTHER):",
            parent=win
        ) or "OTHER"

        title = simpledialog.askstring(
            "Timeline Event",
            "Event Title:",
            parent=win
        )
        if not title:
            return

        description = simpledialog.askstring(
            "Timeline Event",
            "Description:",
            parent=win
        ) or ""

        source = simpledialog.askstring(
            "Timeline Event",
            "Source Reference:",
            parent=win
        ) or ""

        add_timeline_event(
            case_id,
            event_date,
            event_type,
            title,
            description,
            source
        )

        load_events(case_id)

    def inspect_event():
        selection = timeline_tree.selection()
        if not selection:
            messagebox.showinfo(
                "Select Event",
                "Select a timeline event first.",
                parent=win
            )
            return

        index = timeline_tree.index(selection[0])
        rows = get_case_timeline(selected_case["id"])

        if index >= len(rows):
            return

        row = rows[index]

        messagebox.showinfo(
            "Timeline Event",
            f"Date / Time: {row[1]}\n"
            f"Type: {row[2]}\n"
            f"Title: {row[3]}\n\n"
            f"Description:\n{row[4]}\n\n"
            f"Source: {row[5] or 'Not specified'}",
            parent=win
        )

    def clear_selection():
        selected_case["id"] = None
        for item in timeline_tree.get_children():
            timeline_tree.delete(item)
        status_var.set("Select a case.")

    case_tree.bind(
        "<<TreeviewSelect>>",
        lambda event: (
            load_events(int(case_tree.selection()[0]))
            if case_tree.selection()
            else None
        )
    )

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=(0, 8))

    for label, command in [
        ("ADD EVENT", add_event),
        ("INSPECT EVENT", inspect_event),
        ("CLEAR", clear_selection),
    ]:
        tk.Button(
            toolbar,
            text=label,
            command=command,
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=7
        ).pack(side="left", padx=(0, 6))

    tk.Label(
        toolbar,
        textvariable=status_var,
        font=("Segoe UI", 9),
        bg=bg,
        fg=muted
    ).pack(side="left", padx=8)

    tk.Label(
        outer,
        text=(
            "Timeline entries preserve dates and source references for review. "
            "Temporal proximity does not by itself establish causation or responsibility."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg,
        fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    return win


# =========================================================
# FINAL COMPLETION PACK — STEPS 27-32
# NLP • GRAPH ANALYTICS • PATTERNS • RECOMMENDATIONS
# EXPORTS • FINAL INTELLIGENCE CENTER
# =========================================================

def initialize_final_analytics_tables():
    """Create final analytics tables without altering existing tables."""
    initialize_timeline_table()

    conn = _case_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS extracted_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER,
            entity_type TEXT NOT NULL,
            entity_value TEXT NOT NULL,
            source_reference TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS investigator_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER,
            priority TEXT DEFAULT 'MEDIUM',
            recommendation TEXT NOT NULL,
            reason TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def extract_entities_from_text(text_value):
    """
    Lightweight offline NLP/entity extraction.
    Uses deterministic regex/pattern rules so the prototype works
    without downloading an NLP model.
    """
    text_value = str(text_value or "")
    entities = []

    def add(kind, value):
        value = value.strip(" \t\r\n.,;:()[]{}<>\"'")
        if value and len(value) > 1:
            item = (kind, value)
            if item not in entities:
                entities.append(item)

    # Indian-style phone numbers / generic 10-digit numbers.
    for m in re.findall(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)", text_value):
        add("PHONE", m.replace(" ", "").replace("-", ""))

    # Vehicle registrations, tolerant prototype pattern.
    for m in re.findall(
        r"\b[A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,3}[\s-]?\d{3,4}\b",
        text_value.upper()
    ):
        add("VEHICLE", m)

    # FIR / case references.
    for m in re.findall(
        r"\b(?:FIR|CASE|CR|DD|GD)[\s:#-]*[A-Z0-9/-]{2,}\b",
        text_value,
        flags=re.I
    ):
        add("FIR", m)

    # Email addresses.
    for m in re.findall(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        text_value
    ):
        add("EMAIL", m)

    # Dates.
    for m in re.findall(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
        text_value
    ):
        add("DATE", m)

    # Currency amounts.
    for m in re.findall(
        r"(?:₹|INR|Rs\.?)\s*[\d,]+(?:\.\d+)?",
        text_value,
        flags=re.I
    ):
        add("FINANCIAL", m)

    # Explicit entity labels commonly found in FIR/intelligence prose.
    labelled_patterns = {
        "PERSON": r"(?:person|suspect|accused|subject|name)\s*[:=-]\s*([A-Za-z][A-Za-z .'-]{2,60})",
        "LOCATION": r"(?:location|address|place|residence)\s*[:=-]\s*([A-Za-z0-9][A-Za-z0-9 ,./'-]{2,80})",
        "ORGANIZATION": r"(?:organization|organisation|company|gang|agency)\s*[:=-]\s*([A-Za-z0-9 &.'-]{2,80})",
    }

    for kind, pattern in labelled_patterns.items():
        for m in re.findall(pattern, text_value, flags=re.I):
            add(kind, m)

    # Capitalized multi-word names, conservatively limited.
    for m in re.findall(
        r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3}\b",
        text_value
    ):
        if m.lower() not in {
            "New Delhi", "New York", "Indian Police", "Police Station"
        }:
            add("PERSON_CANDIDATE", m)

    return entities


def save_extracted_entities(case_id, entities, source_reference="NLP"):
    initialize_final_analytics_tables()

    conn = _case_db_connection()
    cur = conn.cursor()

    for kind, value in entities:
        cur.execute("""
            INSERT INTO extracted_entities
            (case_id, entity_type, entity_value, source_reference)
            VALUES (?, ?, ?, ?)
        """, (case_id, kind, value, source_reference))

    conn.commit()
    conn.close()


def calculate_advanced_network_metrics():
    """Compute transparent graph metrics from the existing relationship graph."""
    try:
        graph = _person_network_graph()
    except Exception:
        graph = {}

    adj = {}
    for node, neighbors in (graph or {}).items():
        adj.setdefault(node, set())
        for n in neighbors or set():
            adj.setdefault(n, set())
            adj[node].add(n)
            adj[n].add(node)

    if not adj:
        return []

    n_total = len(adj)
    metrics = []

    for node, neighbors in adj.items():
        degree = len(neighbors)

        # Degree centrality.
        degree_centrality = (
            degree / (n_total - 1) if n_total > 1 else 0
        )

        # Local clustering coefficient.
        possible = degree * (degree - 1) / 2
        links = 0
        neighbor_list = list(neighbors)

        for i in range(len(neighbor_list)):
            for j in range(i + 1, len(neighbor_list)):
                if neighbor_list[j] in adj.get(neighbor_list[i], set()):
                    links += 1

        clustering = links / possible if possible else 0

        try:
            kind = str(node[0])
            value = str(node[1])
        except Exception:
            kind = "ENTITY"
            value = str(node)

        try:
            label = get_entity_label(kind, value)
        except Exception:
            label = value

        influence = (
            degree_centrality * 70 +
            clustering * 30
        )

        metrics.append({
            "node": node,
            "kind": kind,
            "value": value,
            "label": str(label),
            "degree": degree,
            "degree_centrality": degree_centrality,
            "clustering": clustering,
            "influence": influence,
        })

    metrics.sort(key=lambda x: x["influence"], reverse=True)
    return metrics


def detect_suspicious_patterns():
    """
    Rule-based pattern detector over stored relationships/timeline.
    These are leads for review, not conclusions.
    """
    findings = []

    try:
        metrics = calculate_advanced_network_metrics()
    except Exception:
        metrics = []

    for m in metrics[:30]:
        if m["degree"] >= 5:
            findings.append({
                "severity": "HIGH" if m["degree"] >= 8 else "MEDIUM",
                "pattern": "HIGH CONNECTIVITY",
                "entity": m["label"],
                "score": round(m["influence"], 2),
                "reason": f"Connected to {m['degree']} entities."
            })

        if m["clustering"] >= 0.70 and m["degree"] >= 3:
            findings.append({
                "severity": "MEDIUM",
                "pattern": "DENSE LOCAL CLUSTER",
                "entity": m["label"],
                "score": round(m["clustering"] * 100, 2),
                "reason": "A high proportion of neighboring entities are interconnected."
            })

    # Timeline burst detection: >=3 events on the same calendar date.
    try:
        cases = list_investigation_cases()
        for case in cases:
            rows = get_case_timeline(case[0])
            dates = {}

            for row in rows:
                date_part = str(row[1]).strip()[:10]
                dates.setdefault(date_part, []).append(row)

            for date_part, events in dates.items():
                if date_part and len(events) >= 3:
                    findings.append({
                        "severity": "MEDIUM",
                        "pattern": "TIMELINE ACTIVITY BURST",
                        "entity": case[1],
                        "score": min(100, len(events) * 20),
                        "reason": (
                            f"{len(events)} recorded events share the date "
                            f"{date_part}."
                        )
                    })
    except Exception:
        pass

    findings.sort(
        key=lambda x: (
            0 if x["severity"] == "HIGH" else 1,
            -x["score"]
        )
    )
    return findings


def generate_investigator_recommendations():
    """Turn analytical indicators into explainable next-step suggestions."""
    recommendations = []

    patterns = detect_suspicious_patterns()

    for p in patterns[:20]:
        if p["pattern"] == "HIGH CONNECTIVITY":
            recommendations.append({
                "priority": p["severity"],
                "recommendation": f"Review source records connected to {p['entity']}.",
                "reason": p["reason"]
            })
        elif p["pattern"] == "DENSE LOCAL CLUSTER":
            recommendations.append({
                "priority": "MEDIUM",
                "recommendation": f"Inspect the surrounding network cluster for {p['entity']}.",
                "reason": p["reason"]
            })
        elif p["pattern"] == "TIMELINE ACTIVITY BURST":
            recommendations.append({
                "priority": "MEDIUM",
                "recommendation": f"Cross-check records around the activity burst for {p['entity']}.",
                "reason": p["reason"]
            })

    try:
        alerts = generate_intelligence_alerts()
        for alert in alerts[:10]:
            recommendations.append({
                "priority": alert["severity"],
                "recommendation": (
                    f"Validate the {alert['category'].lower()} indicator "
                    f"for {alert['entity']}."
                ),
                "reason": alert["message"]
            })
    except Exception:
        pass

    # Deduplicate.
    seen = set()
    output = []
    for r in recommendations:
        key = (r["priority"], r["recommendation"])
        if key not in seen:
            seen.add(key)
            output.append(r)

    return output[:30]


def export_case_report_pdf(case_id):
    """Export a compact case intelligence report using reportlab."""
    from tkinter import filedialog, messagebox

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        )
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception as exc:
        messagebox.showerror(
            "PDF Export",
            "reportlab is required for PDF export.\n\n"
            f"Error: {exc}",
            parent=root
        )
        return

    data = get_case_details(case_id)
    case = data.get("case")

    if not case:
        return

    path = filedialog.asksaveasfilename(
        parent=root,
        title="Export Case Report",
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")]
    )
    if not path:
        return

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("INVESTIGATION CASE REPORT", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"<b>Case:</b> {case[1]} — {case[2]}",
        styles["BodyText"]
    ))
    story.append(Paragraph(
        f"<b>Status:</b> {case[4]} &nbsp;&nbsp; "
        f"<b>Priority:</b> {case[5]}",
        styles["BodyText"]
    ))
    story.append(Paragraph(
        f"<b>Lead Investigator:</b> {case[6] or 'Not specified'}",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(case[3] or "No description.", styles["BodyText"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Persons", styles["Heading2"]))
    person_data = [["Person ID", "Role", "Notes"]]
    person_data.extend([
        [str(p[0]), str(p[1]), str(p[2])]
        for p in data["persons"]
    ])
    story.append(Table(person_data, repeatRows=1))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Evidence", styles["Heading2"]))
    evidence_data = [["Type", "Reference", "Description"]]
    evidence_data.extend([
        [str(e[0]), str(e[1]), str(e[2])]
        for e in data["evidence"]
    ])
    story.append(Table(evidence_data, repeatRows=1))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Timeline", styles["Heading2"]))
    timeline_data = [["Date", "Type", "Event", "Source"]]
    for e in get_case_timeline(case_id):
        timeline_data.append([
            str(e[1]), str(e[2]), str(e[3]), str(e[5])
        ])
    story.append(Table(timeline_data, repeatRows=1))

    for table in story:
        if isinstance(table, Table):
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]))

    SimpleDocTemplate(path, pagesize=A4).build(story)

    messagebox.showinfo(
        "Export Complete",
        f"Case report saved to:\n{path}",
        parent=root
    )


def export_case_report_csv(case_id):
    """Export a flat case intelligence CSV without extra dependencies."""
    from tkinter import filedialog, messagebox
    import csv

    data = get_case_details(case_id)
    case = data.get("case")

    if not case:
        return

    path = filedialog.asksaveasfilename(
        parent=root,
        title="Export Case Data",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if not path:
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(["CASE"])
        writer.writerow([
            "Case Number", "Title", "Status", "Priority",
            "Lead Investigator", "Description"
        ])
        writer.writerow([
            case[1], case[2], case[4], case[5], case[6], case[3]
        ])

        writer.writerow([])
        writer.writerow(["PERSONS"])
        writer.writerow(["Person ID", "Role", "Notes"])
        writer.writerows(data["persons"])

        writer.writerow([])
        writer.writerow(["EVIDENCE"])
        writer.writerow(["Type", "Reference", "Description", "Created"])
        writer.writerows(data["evidence"])

        writer.writerow([])
        writer.writerow(["TIMELINE"])
        writer.writerow(["Date", "Type", "Title", "Description", "Source"])
        writer.writerows([
            [e[1], e[2], e[3], e[4], e[5]]
            for e in get_case_timeline(case_id)
        ])

    messagebox.showinfo(
        "Export Complete",
        f"CSV saved to:\n{path}",
        parent=root
    )


def open_final_intelligence_center():
    """Final integrated analytical workspace."""
    from tkinter import messagebox, simpledialog

    initialize_final_analytics_tables()

    win = tk.Toplevel(root)
    win.title("Investigator Intelligence Center")
    win.geometry("1500x900")
    win.minsize(1050, 680)
    win.resizable(True, True)

    bg = THEME.get("bg", "#F5F7FA")
    surface = THEME.get("surface", "#FFFFFF")
    text_color = THEME.get("text", "#111827")
    muted = THEME.get("muted", "#6B7280")
    border = THEME.get("border", "#E2E8F0")

    outer = tk.Frame(win, bg=bg)
    outer.pack(fill="both", expand=True)

    header = tk.Frame(
        outer, bg=surface,
        highlightthickness=1,
        highlightbackground=border
    )
    header.pack(fill="x", padx=16, pady=(16, 10))

    tk.Label(
        header,
        text="INVESTIGATOR INTELLIGENCE CENTER",
        font=("Segoe UI", 18, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=18, pady=(14, 2))

    tk.Label(
        header,
        text=(
            "NLP extraction • network analytics • suspicious patterns • "
            "recommendations • case reporting"
        ),
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=18, pady=(0, 14))

    body = tk.Frame(outer, bg=bg)
    body.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    left = tk.Frame(
        body, bg=surface, width=390,
        highlightthickness=1, highlightbackground=border
    )
    left.pack(side="left", fill="y", padx=(0, 8))
    left.pack_propagate(False)

    right = tk.Frame(
        body, bg=surface,
        highlightthickness=1, highlightbackground=border
    )
    right.pack(side="left", fill="both", expand=True)

    tk.Label(
        left, text="ANALYSIS INPUT",
        font=("Segoe UI", 11, "bold"),
        bg=surface, fg=text_color
    ).pack(anchor="w", padx=16, pady=(16, 8))

    tk.Label(
        left,
        text="Paste FIR / intelligence report text",
        font=("Segoe UI", 9),
        bg=surface, fg=muted
    ).pack(anchor="w", padx=16)

    source_entry = tk.Entry(
        left, font=("Segoe UI", 9), relief="solid"
    )
    source_entry.pack(fill="x", padx=16, pady=(5, 8))
    source_entry.insert(0, "Manual report")

    text_box = tk.Text(
        left,
        height=18,
        wrap="word",
        font=("Segoe UI", 9),
        relief="solid"
    )
    text_box.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    selected_case = {"id": None}
    status_var = tk.StringVar(value="Ready.")

    def selected_case_or_none():
        if selected_case['id'] is not None:
            return selected_case['id']
        cases = list_investigation_cases()
        return cases[0][0] if cases else None

    def ensure_export_case():
        case_id = selected_case_or_none()
        if case_id is not None:
            return case_id

        # Existing FIRs live in the original crime database. Create a linked
        # investigation-case record so PDF/CSV export can use the case module.
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute('SELECT case_id, case_type, location, date FROM cases ORDER BY rowid DESC LIMIT 1')
            fir = cur.fetchone()
            conn.close()
        except Exception:
            fir = None

        if not fir:
            messagebox.showinfo('Export', 'Create a case or FIR first.', parent=win)
            return None

        try:
            initialize_case_management_tables()
            case_number, case_type, location, date = fir
            existing = list_investigation_cases()
            for row in existing:
                if str(row[1]).upper() == str(case_number).upper():
                    selected_case['id'] = row[0]
                    return row[0]
            new_id = create_investigation_case(
                str(case_number),
                f'{case_type} — {location}',
                f'Imported from FIR {case_number}. Date: {date}',
                'OPEN', 'MEDIUM', ''
            )

            # Carry the FIR's existing subject into the investigation case.
            try:
                src = get_connection()
                src_cur = src.cursor()
                src_cur.execute(
                    'SELECT person_id FROM person_cases WHERE case_id = ?',
                    (str(case_number),)
                )
                fir_people = [r[0] for r in src_cur.fetchall()]
                src.close()

                dst = _case_db_connection()
                dst_cur = dst.cursor()
                for pid in fir_people:
                    dst_cur.execute(
                        'INSERT OR IGNORE INTO case_persons (case_id, person_id, role, notes) VALUES (?, ?, ?, ?)',
                        (new_id, str(pid), 'SUBJECT', f'Linked from FIR {case_number}')
                    )
                dst.commit()
                dst.close()

                initialize_timeline_table()
                add_timeline_event(
                    new_id, str(date), 'FIR',
                    f'FIR {case_number} registered',
                    f'{case_type} at {location}',
                    str(case_number)
                )
            except Exception as link_exc:
                print('FIR-to-case linking warning:', link_exc)

            selected_case['id'] = new_id
            return new_id
        except Exception as exc:
            messagebox.showerror('Export Case', f'Could not prepare case for export.\n\n{exc}', parent=win)
            return None

    def run_nlp():
        raw = text_box.get("1.0", "end").strip()
        if not raw:
            messagebox.showinfo(
                "NLP",
                "Enter report text first.",
                parent=win
            )
            return

        entities = extract_entities_from_text(raw)
        case_id = selected_case_or_none()

        if case_id is not None:
            save_extracted_entities(
                case_id,
                entities,
                source_entry.get().strip() or "Manual report"
            )

        for item in entity_tree.get_children():
            entity_tree.delete(item)

        for kind, value in entities:
            entity_tree.insert("", "end", values=(kind, value))

        status_var.set(f"{len(entities)} entities extracted.")

    tk.Button(
        left, text="EXTRACT ENTITIES",
        command=run_nlp,
        font=("Segoe UI", 9, "bold"),
        padx=12, pady=8
    ).pack(fill="x", padx=16, pady=(0, 8))

    tk.Button(
        left, text="OPEN NETWORK",
        command=open_network_visualization,
        font=("Segoe UI", 9, "bold"),
        padx=12, pady=8
    ).pack(fill="x", padx=16, pady=(0, 8))

    tk.Button(
        left, text="OPEN ALERTS",
        command=open_intelligence_alerts_center,
        font=("Segoe UI", 9, "bold"),
        padx=12, pady=8
    ).pack(fill="x", padx=16, pady=(0, 8))

    notebook = ttk.Notebook(right)
    notebook.pack(fill="both", expand=True, padx=12, pady=12)

    entity_tab = tk.Frame(notebook, bg=surface)
    metrics_tab = tk.Frame(notebook, bg=surface)
    pattern_tab = tk.Frame(notebook, bg=surface)
    rec_tab = tk.Frame(notebook, bg=surface)

    notebook.add(entity_tab, text="NLP Entities")
    notebook.add(metrics_tab, text="Network Metrics")
    notebook.add(pattern_tab, text="Patterns")
    notebook.add(rec_tab, text="Recommendations")

    entity_tree = ttk.Treeview(
        entity_tab,
        columns=("type", "value"),
        show="headings"
    )
    entity_tree.heading("type", text="ENTITY TYPE")
    entity_tree.heading("value", text="VALUE")
    entity_tree.column("type", width=180)
    entity_tree.column("value", width=700)
    entity_tree.pack(fill="both", expand=True, padx=10, pady=10)

    metrics_tree = ttk.Treeview(
        metrics_tab,
        columns=("kind", "entity", "degree", "centrality", "cluster", "influence"),
        show="headings"
    )
    for col, heading, width in [
        ("kind", "TYPE", 100),
        ("entity", "ENTITY", 260),
        ("degree", "DEGREE", 90),
        ("centrality", "CENTRALITY", 110),
        ("cluster", "CLUSTERING", 110),
        ("influence", "INFLUENCE", 100),
    ]:
        metrics_tree.heading(col, text=heading)
        metrics_tree.column(col, width=width)
    metrics_tree.pack(fill="both", expand=True, padx=10, pady=10)

    pattern_tree = ttk.Treeview(
        pattern_tab,
        columns=("severity", "pattern", "entity", "score", "reason"),
        show="headings"
    )
    for col, heading, width in [
        ("severity", "SEVERITY", 100),
        ("pattern", "PATTERN", 190),
        ("entity", "ENTITY", 220),
        ("score", "SCORE", 90),
        ("reason", "REASON", 550),
    ]:
        pattern_tree.heading(col, text=heading)
        pattern_tree.column(col, width=width)
    pattern_tree.pack(fill="both", expand=True, padx=10, pady=10)

    rec_tree = ttk.Treeview(
        rec_tab,
        columns=("priority", "recommendation", "reason"),
        show="headings"
    )
    for col, heading, width in [
        ("priority", "PRIORITY", 100),
        ("recommendation", "RECOMMENDATION", 420),
        ("reason", "REASON", 550),
    ]:
        rec_tree.heading(col, text=heading)
        rec_tree.column(col, width=width)
    rec_tree.pack(fill="both", expand=True, padx=10, pady=10)

    def refresh_analytics():
        for tree in (
            metrics_tree,
            pattern_tree,
            rec_tree
        ):
            for item in tree.get_children():
                tree.delete(item)

        for m in calculate_advanced_network_metrics()[:100]:
            metrics_tree.insert(
                "",
                "end",
                values=(
                    m["kind"],
                    m["label"],
                    m["degree"],
                    f"{m['degree_centrality']:.4f}",
                    f"{m['clustering']:.4f}",
                    f"{m['influence']:.2f}",
                )
            )

        patterns = detect_suspicious_patterns()
        for p in patterns:
            pattern_tree.insert(
                "",
                "end",
                values=(
                    p["severity"],
                    p["pattern"],
                    p["entity"],
                    p["score"],
                    p["reason"]
                )
            )

        for r in generate_investigator_recommendations():
            rec_tree.insert(
                "",
                "end",
                values=(
                    r["priority"],
                    r["recommendation"],
                    r["reason"]
                )
            )

        status_var.set(
            f"Metrics: {len(calculate_advanced_network_metrics())} • "
            f"Patterns: {len(patterns)}"
        )

    def export_report():
        case_id = ensure_export_case()
        if case_id is not None:
            export_case_report_pdf(case_id)

    def export_csv():
        case_id = ensure_export_case()
        if case_id is not None:
            export_case_report_csv(case_id)

    toolbar = tk.Frame(outer, bg=bg)
    toolbar.pack(fill="x", padx=16, pady=(0, 8))

    actions = [
        ('RUN ANALYTICS', refresh_analytics),
        ('PDF REPORT', export_report),
        ('CSV EXPORT', export_csv),
        ('CASE MANAGEMENT', open_case_management),
        ('TIMELINE', open_investigation_timeline),
        ('MAXIMIZE', lambda: toggle_maximize(win)),
        ('CLOSE', lambda: safe_close_window(win)),
    ]
    for index, (label, command) in enumerate(actions, start=1):
        tk.Button(
            toolbar, text=label, command=command,
            font=('Segoe UI', 9, 'bold'), padx=10, pady=7,
            bg='#FFFFFF', fg='#111111', activebackground='#FFFFFF',
            activeforeground='#111111', relief='solid', bd=1,
            highlightthickness=0, cursor='hand2'
        ).grid(row=0, column=index, padx=3, pady=2, sticky='ew')

    tk.Label(
        toolbar, textvariable=status_var, font=('Segoe UI', 9),
        bg=bg, fg=muted
    ).grid(row=1, column=1, columnspan=len(actions), sticky='w', pady=(5, 0))

    enable_window_controls(win)

    tk.Label(
        outer,
        text=(
            "Analytical outputs are decision-support leads. "
            "Verify important findings against original records before action."
        ),
        font=("Segoe UI", 8, "italic"),
        bg=bg, fg=muted
    ).pack(anchor="w", padx=16, pady=(0, 12))

    try:
        enable_window_controls(win)
    except Exception:
        pass

    refresh_analytics()
    return win


# Initialize final tables at startup when the database is available.
try:
    initialize_final_analytics_tables()
except Exception:
    pass

# =========================================================
# TEAM HORIZON — PROFESSIONAL UI THEME
# =========================================================

THEME = {
    "bg": "#F5F7FA",
    "surface": "#FFFFFF",
    "surface2": "#F8FAFC",
    "panel": "#FFFFFF",
    "border": "#E2E8F0",
    "accent": "#374151",
    "accent2": "#4B5563",
    "text": "#111827",
    "muted": "#6B7280",
    "success": "#15803D",
    "warning": "#B45309",
    "danger": "#B91C1C",
}

_ui_logo_cache = None
_ui_animation_jobs = []


def apply_professional_theme(root_window):
    """Apply a consistent dark command-center theme to the existing app."""
    try:
        root_window.configure(bg=THEME["bg"])
    except Exception:
        pass

    style = ttk.Style(root_window)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure(
        "App.TFrame",
        background=THEME["bg"]
    )
    style.configure(
        "Panel.TFrame",
        background=THEME["surface"]
    )
    style.configure(
        "Card.TFrame",
        background=THEME["surface2"],
        relief="flat"
    )
    style.configure(
        "Title.TLabel",
        background=THEME["bg"],
        foreground=THEME["text"],
        font=("Segoe UI", 22, "bold")
    )
    style.configure(
        "Subtitle.TLabel",
        background=THEME["bg"],
        foreground=THEME["muted"],
        font=("Segoe UI", 10)
    )
    style.configure(
        "PanelTitle.TLabel",
        background=THEME["surface"],
        foreground=THEME["text"],
        font=("Segoe UI", 13, "bold")
    )
    style.configure(
        "CardTitle.TLabel",
        background=THEME["surface2"],
        foreground=THEME["muted"],
        font=("Segoe UI", 9, "bold")
    )
    style.configure(
        "CardValue.TLabel",
        background=THEME["surface2"],
        foreground=THEME["text"],
        font=("Segoe UI", 19, "bold")
    )
    style.configure(
        "Accent.TButton",
        font=("Segoe UI", 10, "bold"),
        padding=(16, 10),
        background=THEME["accent"],
        foreground="#001018",
        borderwidth=0
    )
    style.map(
        "Accent.TButton",
        background=[("active", THEME["accent2"]), ("pressed", THEME["accent2"])],
        foreground=[("active", "#001018")]
    )
    style.configure(
        "Secondary.TButton",
        font=("Segoe UI", 10),
        padding=(14, 9),
        background=THEME["surface2"],
        foreground=THEME["text"],
        borderwidth=0
    )
    style.map(
        "Secondary.TButton",
        background=[("active", THEME["border"]), ("pressed", THEME["border"])]
    )
    style.configure(
        "Treeview",
        background=THEME["panel"],
        fieldbackground=THEME["panel"],
        foreground=THEME["text"],
        rowheight=34,
        borderwidth=0,
        font=("Segoe UI", 10)
    )
    style.configure(
        "Treeview.Heading",
        background=THEME["surface2"],
        foreground=THEME["text"],
        font=("Segoe UI", 10, "bold"),
        padding=8
    )
    style.map(
        "Treeview",
        background=[("selected", "#E5E7EB")],
        foreground=[("selected", "white")]
    )


def load_team_horizon_logo(master, max_size=(280, 70)):
    """Load the optional Team Horizon PNG from beside the application."""
    global _ui_logo_cache

    try:
        logo_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "team_horizon_logo.png"
        )
        if not os.path.exists(logo_path):
            return None

        image = Image.open(logo_path).convert("RGBA")
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        _ui_logo_cache = ImageTk.PhotoImage(image, master=master)
        return _ui_logo_cache
    except Exception as exc:
        print("Team Horizon logo warning:", exc)
        return None


def build_window_header(parent, title, subtitle=None):
    """Reusable professional header for every secondary window."""
    header = tk.Frame(
        parent,
        bg=THEME["bg"],
        bd=0,
        highlightthickness=0
    )
    header.pack(fill="x", padx=24, pady=(18, 8))

    left = tk.Frame(header, bg=THEME["bg"])
    left.pack(side="left", fill="x", expand=True)

    tk.Label(
        left,
        text="",
        font=("Segoe UI", 9, "bold"),
        fg=THEME["accent"],
        bg=THEME["bg"]
    ).pack(anchor="w")

    tk.Label(
        left,
        text=title,
        font=("Segoe UI", 20, "bold"),
        fg=THEME["text"],
        bg=THEME["bg"]
    ).pack(anchor="w", pady=(2, 0))

    if subtitle:
        tk.Label(
            left,
            text=subtitle,
            font=("Segoe UI", 9),
            fg=THEME["muted"],
            bg=THEME["bg"]
        ).pack(anchor="w", pady=(3, 0))

    return header


def create_professional_window(title, width=1120, height=720, subtitle=None):
    """Create a consistent top-level window with Team Horizon branding."""
    win = tk.Toplevel(root)
    win.title(title)
    win.geometry(f"{width}x{height}")
    win.minsize(900, 600)
    win.configure(bg=THEME["bg"])
    win.transient(root)
    enable_window_controls(win)

    try:
        win.grab_set()
    except Exception:
        pass

    build_window_header(win, title, subtitle)
    return win


def create_metric_card(parent, title, value="—", icon=""):
    card = tk.Frame(
        parent,
        bg=THEME["surface2"],
        bd=0,
        highlightbackground=THEME["border"],
        highlightthickness=1
    )

    tk.Label(
        card,
        text=f"{icon}  {title}" if icon else title,
        font=("Segoe UI", 9, "bold"),
        fg=THEME["muted"],
        bg=THEME["surface2"]
    ).pack(anchor="w", padx=16, pady=(13, 2))

    value_label = tk.Label(
        card,
        text=str(value),
        font=("Segoe UI", 20, "bold"),
        fg=THEME["text"],
        bg=THEME["surface2"]
    )
    value_label.pack(anchor="w", padx=16, pady=(0, 14))

    return card, value_label


def add_window_footer(parent, status="READY"):
    footer = tk.Frame(parent, bg=THEME["bg"])
    footer.pack(fill="x", padx=24, pady=(5, 16))

    tk.Label(
        footer,
        text=f"● {status}",
        font=("Segoe UI", 9, "bold"),
        fg=THEME["success"],
        bg=THEME["bg"]
    ).pack(side="left")

    tk.Label(
        footer,
        text="",
        font=("Segoe UI", 8),
        fg=THEME["muted"],
        bg=THEME["bg"]
    ).pack(side="right")


def add_hover_effect(button, normal=None, hover=None):
    normal = normal or THEME["surface2"]
    hover = hover or THEME["border"]

    button.configure(
        bg=normal,
        activebackground=hover
    )

    def on_enter(_):
        button.configure(bg=hover)

    def on_leave(_):
        button.configure(bg=normal)

    button.bind("<Enter>", on_enter, add="+")
    button.bind("<Leave>", on_leave, add="+")


def start_header_pulse(label):
    """Subtle live pulse for the command-center status indicator."""
    state = {"on": True}

    def pulse():
        if not label.winfo_exists():
            return

        state["on"] = not state["on"]
        label.configure(
            fg=THEME["accent"] if state["on"] else THEME["muted"]
        )
        job = label.after(900, pulse)
        _ui_animation_jobs.append(job)

    pulse()


def enhance_all_windows():
    """Remove STEP labels from Tk/Toplevel window titles."""
    if root is None:
        return

    def walk(widget):
        try:
            if isinstance(widget, tk.Toplevel):
                current = widget.title()
                current = re.sub(
                    r"\s*[-|:]*\s*STEP\s*\d+(?:\.\d+)*\s*[-|:]*\s*",
                    " ",
                    current,
                    flags=re.IGNORECASE
                ).strip(" -|:")
                if current:
                    widget.title(current)
        except Exception:
            pass

        for child in widget.winfo_children():
            walk(child)

    walk(root)
    root.after(700, enhance_all_windows)


def install_professional_ui():
    """Non-destructive UI enhancement entry point."""
    if root is None:
        return

    apply_professional_theme(root)

    try:
        root.title("Intelligence Command Center")
    except Exception:
        pass

    enhance_all_windows()


# =========================================================
# GLOBAL WINDOW BRANDING DISABLED
# Secondary windows intentionally use task-specific titles only.

search_entry = None


# =========================================================
# DASHBOARD VARIABLES
# =========================================================

person_id_value = None
name_value = None
phone_value = None
vehicle_value = None
address_value = None
organization_value = None
status_value = None
total_cases_value = None


registered_value = None
fir_count_value = None
relationship_value = None


fir_tree = None

history_text = None


# =========================================================
# FIR CARD VARIABLES
# =========================================================

fir_card_id_value = None
fir_card_date_value = None
fir_card_type_value = None
fir_card_location_value = None
fir_card_person_value = None
fir_card_name_value = None
fir_card_status_value = None


# =========================================================
# DATABASE
# =========================================================

def get_connection():

    os.makedirs(
        os.path.dirname(DATABASE_PATH),
        exist_ok=True
    )

    return sqlite3.connect(
        DATABASE_PATH
    )


def database_exists():

    if not os.path.exists(DATABASE_PATH):

        messagebox.showerror(
            "Database Error",
            "Database file not found:\n\n"
            f"{DATABASE_PATH}\n\n"
            "Run database.py first."
        )

        return False

    return True


def ensure_database_schema():

    """Create/migrate the SQLite schema without deleting existing data."""

    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            person_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            phone TEXT,
            vehicle TEXT,
            address TEXT,
            organization TEXT
        )
    """)

    # Safe migration for databases created by the older version.
    cur.execute("PRAGMA table_info(persons)")
    person_columns = {row[1] for row in cur.fetchall()}
    if "address" not in person_columns:
        cur.execute("ALTER TABLE persons ADD COLUMN address TEXT")
    if "organization" not in person_columns:
        cur.execute("ALTER TABLE persons ADD COLUMN organization TEXT")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            case_type TEXT,
            location TEXT,
            date TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS person_cases (
            person_id TEXT,
            case_id TEXT,
            UNIQUE(person_id, case_id),
            FOREIGN KEY(person_id) REFERENCES persons(person_id),
            FOREIGN KEY(case_id) REFERENCES cases(case_id)
        )
    """)

    # ---------------------------------------------------------
    # ENTITY TABLES — # ---------------------------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS phones (
            phone_id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_number TEXT NOT NULL UNIQUE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            vehicle_id INTEGER PRIMARY KEY AUTOINCREMENT,
            registration_number TEXT NOT NULL UNIQUE,
            vehicle_type TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS locations (
            location_id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_name TEXT NOT NULL UNIQUE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS organizations (
            organization_id INTEGER PRIMARY KEY AUTOINCREMENT,
            organization_name TEXT NOT NULL UNIQUE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            relationship_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_type, source_id, relation_type, target_type, target_id)
        )
    """)

    # ---------------------------------------------------------
    # MIGRATE EXISTING PERSON DATA INTO ENTITY GRAPH
    # ---------------------------------------------------------
    cur.execute("SELECT person_id, phone, vehicle, address, organization FROM persons")
    existing_people = cur.fetchall()

    for person_id, phone, vehicle, address, organization in existing_people:
        if phone and str(phone).strip():
            phone = str(phone).strip()
            cur.execute("INSERT OR IGNORE INTO phones(phone_number) VALUES (?)", (phone,))
            cur.execute("SELECT phone_id FROM phones WHERE phone_number = ?", (phone,))
            phone_id = cur.fetchone()[0]
            cur.execute("""INSERT OR IGNORE INTO relationships
                (source_type, source_id, relation_type, target_type, target_id)
                VALUES ('PERSON', ?, 'USES', 'PHONE', ?)""", (person_id, str(phone_id)))

        if vehicle and str(vehicle).strip():
            vehicle = str(vehicle).strip().upper()
            cur.execute("INSERT OR IGNORE INTO vehicles(registration_number) VALUES (?)", (vehicle,))
            cur.execute("SELECT vehicle_id FROM vehicles WHERE registration_number = ?", (vehicle,))
            vehicle_id = cur.fetchone()[0]
            cur.execute("""INSERT OR IGNORE INTO relationships
                (source_type, source_id, relation_type, target_type, target_id)
                VALUES ('PERSON', ?, 'OWNS', 'VEHICLE', ?)""", (person_id, str(vehicle_id)))

        if address and str(address).strip():
            address = str(address).strip()
            cur.execute("INSERT OR IGNORE INTO locations(location_name) VALUES (?)", (address,))
            cur.execute("SELECT location_id FROM locations WHERE location_name = ?", (address,))
            location_id = cur.fetchone()[0]
            cur.execute("""INSERT OR IGNORE INTO relationships
                (source_type, source_id, relation_type, target_type, target_id)
                VALUES ('PERSON', ?, 'LIVES_AT', 'LOCATION', ?)""", (person_id, str(location_id)))

        if organization and str(organization).strip():
            organization = str(organization).strip()
            cur.execute("INSERT OR IGNORE INTO organizations(organization_name) VALUES (?)", (organization,))
            cur.execute("SELECT organization_id FROM organizations WHERE organization_name = ?", (organization,))
            organization_id = cur.fetchone()[0]
            cur.execute("""INSERT OR IGNORE INTO relationships
                (source_type, source_id, relation_type, target_type, target_id)
                VALUES ('PERSON', ?, 'MEMBER_OF', 'ORGANIZATION', ?)""", (person_id, str(organization_id)))

    # Existing FIR links become graph relationships too.
    cur.execute("SELECT person_id, case_id FROM person_cases")
    for person_id, case_id in cur.fetchall():
        cur.execute("""INSERT OR IGNORE INTO relationships
            (source_type, source_id, relation_type, target_type, target_id)
            VALUES ('PERSON', ?, 'INVOLVED_IN', 'FIR', ?)""", (person_id, case_id))

    # Existing FIR locations become LOCATION entities and PERSON -> LOCATION links.
    cur.execute("SELECT case_id, location FROM cases WHERE location IS NOT NULL AND TRIM(location) <> ''")
    for case_id, location in cur.fetchall():
        location = str(location).strip()
        cur.execute("INSERT OR IGNORE INTO locations(location_name) VALUES (?)", (location,))
        cur.execute("SELECT location_id FROM locations WHERE location_name = ?", (location,))
        location_id = cur.fetchone()[0]
        cur.execute("""INSERT OR IGNORE INTO relationships
            (source_type, source_id, relation_type, target_type, target_id)
            VALUES ('FIR', ?, 'OCCURRED_AT', 'LOCATION', ?)""", (case_id, str(location_id)))

    conn.commit()
    conn.close()


# =========================================================
# VALIDATION
# =========================================================

def validate_person_id(person_id):

    return bool(
        PERSON_ID_PATTERN.fullmatch(
            person_id.strip().upper()
        )
    )


def validate_fir_id(fir_id):

    return bool(
        FIR_ID_PATTERN.fullmatch(
            fir_id.strip().upper()
        )
    )


def validate_phone(phone):

    phone = phone.strip()

    return (
        phone == ""
        or bool(
            PHONE_PATTERN.fullmatch(
                phone
            )
        )
    )


def validate_vehicle(vehicle):

    vehicle = vehicle.strip().upper()

    return (
        vehicle == ""
        or bool(
            VEHICLE_PATTERN.fullmatch(
                vehicle
            )
        )
    )


# =========================================================
# DATE FUNCTIONS
# =========================================================

def parse_date(date_text):

    value = date_text.strip()

    for fmt in (
        "%d-%m-%Y",
        "%Y-%m-%d"
    ):

        try:

            return datetime.strptime(
                value,
                fmt
            )

        except ValueError:

            continue

    return None


def validate_date(date_text):

    return parse_date(date_text) is not None


def normalize_date_for_db(date_text):

    parsed = parse_date(
        date_text
    )

    if parsed:

        return parsed.strftime(
            "%Y-%m-%d"
        )

    return ""


def display_date(date_text):

    parsed = parse_date(
        date_text
    )

    if parsed:

        return parsed.strftime(
            "%d-%m-%Y"
        )

    return str(
        date_text or "--"
    )


# =========================================================
# ID GENERATORS
# =========================================================

def next_person_id():

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(
        "SELECT person_id FROM persons"
    )

    rows = cur.fetchall()

    conn.close()

    maximum = 0

    for row in rows:

        value = str(
            row[0]
        ).upper()

        match = re.fullmatch(
            r"P(\d{3})",
            value
        )

        if match:

            maximum = max(
                maximum,
                int(
                    match.group(1)
                )
            )

    return f"P{maximum + 1:03d}"


def next_fir_id():

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(
        "SELECT case_id FROM cases"
    )

    rows = cur.fetchall()

    conn.close()

    maximum = 0

    for row in rows:

        value = str(
            row[0]
        ).upper()

        match = re.fullmatch(
            r"FIR(\d{3})",
            value
        )

        if match:

            maximum = max(
                maximum,
                int(
                    match.group(1)
                )
            )

    return f"FIR{maximum + 1:03d}"


# =========================================================
# INSIGHTFACE
# =========================================================

def load_face_model():

    global face_app

    if face_app is not None:

        return True

    try:

        print(
            "Loading InsightFace model..."
        )

        face_app = FaceAnalysis(
            name=MODEL_NAME
        )

        try:

            face_app.prepare(
                ctx_id=0,
                det_size=(640, 640)
            )

            print(
                "InsightFace loaded using GPU."
            )

        except Exception:

            print(
                "GPU unavailable. Using CPU."
            )

            face_app.prepare(
                ctx_id=-1,
                det_size=(640, 640)
            )

            print(
                "InsightFace loaded using CPU."
            )

        return True

    except Exception as exc:

        messagebox.showerror(
            "Face Model Error",
            "Could not load InsightFace.\n\n"
            + str(exc)
        )

        return False


# =========================================================
# EMBEDDINGS
# =========================================================

def normalize_embedding(embedding):

    arr = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(
        arr
    )

    if norm == 0:

        return arr

    return arr / norm


def get_embedding_from_image(image):

    if not load_face_model():

        return None

    try:

        faces = face_app.get(
            image
        )

        if not faces:

            return None

        # Largest face
        face = max(
            faces,
            key=lambda f: (
                (
                    f.bbox[2] - f.bbox[0]
                )
                *
                (
                    f.bbox[3] - f.bbox[1]
                )
            )
        )

        return normalize_embedding(
            face.embedding
        )

    except Exception as exc:

        print(
            "Embedding error:",
            exc
        )

        return None


def create_embedding_from_photo(
    photo_path
):

    try:

        image = cv2.imread(
            photo_path
        )

        if image is None:

            return None

        return get_embedding_from_image(
            image
        )

    except Exception as exc:

        print(
            "Photo processing error:",
            exc
        )

        return None


def cosine_similarity(a, b):

    a = normalize_embedding(
        a
    )

    b = normalize_embedding(
        b
    )

    if (
        a.size == 0
        or b.size == 0
    ):

        return -1.0

    return float(
        np.dot(
            a,
            b
        )
    )


# =========================================================
# FILE / EMBEDDING DATABASE
# =========================================================

def safe_filename(name):

    result = ""

    for char in name:

        if (
            char.isalnum()
            or char in " _-"
        ):

            result += char

        else:

            result += "_"

    return result.strip()


def load_registered_embeddings():

    """
    Supports:

        faces/Utsav.npy
        faces/Chavvi.npy

    One or multiple embeddings per file.
    """

    database = []

    if not os.path.exists(
        FACES_FOLDER
    ):

        return database

    for filename in os.listdir(
        FACES_FOLDER
    ):

        if not filename.lower().endswith(
            ".npy"
        ):

            continue

        path = os.path.join(
            FACES_FOLDER,
            filename
        )

        try:

            data = np.load(
                path,
                allow_pickle=True
            )

            if data.ndim == 1:

                embeddings = np.asarray(
                    [data],
                    dtype=np.float32
                )

            elif data.ndim == 2:

                embeddings = np.asarray(
                    data,
                    dtype=np.float32
                )

            else:

                continue

            if len(embeddings) == 0:

                continue

            name = os.path.splitext(
                filename
            )[0]

            database.append(
                (
                    name,
                    embeddings
                )
            )

        except Exception as exc:

            print(
                "Could not load:",
                filename,
                exc
            )

    return database


def find_best_match(
    query_embedding
):

    database = load_registered_embeddings()

    if not database:

        return None, 0.0

    best_name = None

    best_score = -1.0

    for name, embeddings in database:

        person_best = -1.0

        for embedding in embeddings:

            score = cosine_similarity(
                query_embedding,
                embedding
            )

            if score > person_best:

                person_best = score

        if person_best > best_score:

            best_score = person_best

            best_name = name

    if best_score >= MATCH_THRESHOLD:

        return (
            best_name,
            best_score
        )

    return (
        None,
        best_score
    )


# =========================================================
# PERSON DATABASE FUNCTIONS
# =========================================================

def get_person_by_name(
    criminal_name
):

    if not database_exists():

        return None

    conn = get_connection()

    cur = conn.cursor()

    cur.execute("""
        SELECT
            person_id,
            name,
            phone,
            vehicle,
            address,
            organization
        FROM persons
        WHERE LOWER(TRIM(name))
            = LOWER(TRIM(?))
        LIMIT 1
    """, (
        criminal_name,
    ))

    person = cur.fetchone()

    if person is None:

        conn.close()

        return None

    person_id, name, phone, vehicle, address, organization = person

    cur.execute("""
        SELECT
            c.case_id,
            c.case_type,
            c.location,
            c.date
        FROM cases c
        INNER JOIN person_cases pc
            ON c.case_id = pc.case_id
        WHERE pc.person_id = ?
        ORDER BY c.date DESC
    """, (
        person_id,
    ))

    cases = cur.fetchall()

    conn.close()

    return {
        "person_id": person_id,
        "name": name,
        "phone": phone,
        "vehicle": vehicle,
        "address": address,
        "organization": organization,
        "cases": cases
    }


def get_person_by_id(
    person_id
):

    if not database_exists():

        return None

    conn = get_connection()

    cur = conn.cursor()

    cur.execute("""
        SELECT
            person_id,
            name,
            phone,
            vehicle,
            address,
            organization
        FROM persons
        WHERE UPPER(person_id)
            = UPPER(?)
        LIMIT 1
    """, (
        person_id,
    ))

    person = cur.fetchone()

    if person is None:

        conn.close()

        return None

    person_id, name, phone, vehicle, address, organization = person

    cur.execute("""
        SELECT
            c.case_id,
            c.case_type,
            c.location,
            c.date
        FROM cases c
        INNER JOIN person_cases pc
            ON c.case_id = pc.case_id
        WHERE pc.person_id = ?
        ORDER BY c.date DESC
    """, (
        person_id,
    ))

    cases = cur.fetchall()

    conn.close()

    return {
        "person_id": person_id,
        "name": name,
        "phone": phone,
        "vehicle": vehicle,
        "address": address,
        "organization": organization,
        "cases": cases
    }


def get_person_from_search(
    text
):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute("""
        SELECT
            person_id,
            name,
            phone,
            vehicle,
            address,
            organization
        FROM persons
        WHERE UPPER(person_id)
            = UPPER(?)
           OR LOWER(name)
            LIKE LOWER(?)
        LIMIT 1
    """, (
        text,
        "%" + text + "%"
    ))

    person = cur.fetchone()

    conn.close()

    if person is None:

        return None

    return get_person_by_id(
        person[0]
    )


# =========================================================
# FIR CARD
# =========================================================

def clear_fir_card():

    if fir_card_id_value is None:

        return

    fir_card_id_value.set("--")

    fir_card_date_value.set("--")

    fir_card_type_value.set("--")

    fir_card_location_value.set("--")

    fir_card_person_value.set("--")

    fir_card_name_value.set("--")

    fir_card_status_value.set("--")


def update_fir_card(
    person
):

    if fir_card_id_value is None:

        return

    cases = (
        person.get(
            "cases",
            []
        )
        if person
        else []
    )

    if not cases:

        clear_fir_card()

        return

    latest = cases[0]

    case_id = latest[0]
    case_type = latest[1]
    location = latest[2]
    case_date = latest[3]

    fir_card_id_value.set(
        str(case_id)
    )

    fir_card_date_value.set(
        display_date(case_date)
    )

    fir_card_type_value.set(
        str(
            case_type or "--"
        )
    )

    fir_card_location_value.set(
        str(
            location or "--"
        )
    )

    fir_card_person_value.set(
        str(
            person.get(
                "person_id",
                "--"
            )
        )
    )

    fir_card_name_value.set(
        str(
            person.get(
                "name",
                "--"
            )
        )
    )

    fir_card_status_value.set(
        "ACTIVE RECORD"
    )


# =========================================================
# ENTITY / RELATIONSHIP ENGINE — # =========================================================

def add_entity_relationship(source_type, source_id, relation_type, target_type, target_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""INSERT OR IGNORE INTO relationships
        (source_type, source_id, relation_type, target_type, target_id)
        VALUES (?, ?, ?, ?, ?)""",
        (source_type.upper(), str(source_id), relation_type.upper(),
         target_type.upper(), str(target_id)))
    conn.commit()
    conn.close()


def get_person_relationships(person_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""SELECT relationship_id, source_type, source_id,
                          relation_type, target_type, target_id
                   FROM relationships
                   WHERE (source_type='PERSON' AND source_id=?)
                      OR (target_type='PERSON' AND target_id=?)
                   ORDER BY relationship_id""",
                (str(person_id), str(person_id)))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_entity_label(entity_type, entity_id):
    entity_type = str(entity_type).upper()
    entity_id = str(entity_id)
    conn = get_connection()
    cur = conn.cursor()
    label = entity_id
    try:
        if entity_type == 'PERSON':
            cur.execute("SELECT name FROM persons WHERE person_id=?", (entity_id,))
            row = cur.fetchone()
            label = f"{row[0]} ({entity_id})" if row else entity_id
        elif entity_type == 'PHONE':
            cur.execute("SELECT phone_number FROM phones WHERE phone_id=?", (entity_id,))
            row = cur.fetchone()
            label = row[0] if row else entity_id
        elif entity_type == 'VEHICLE':
            cur.execute("SELECT registration_number FROM vehicles WHERE vehicle_id=?", (entity_id,))
            row = cur.fetchone()
            label = row[0] if row else entity_id
        elif entity_type == 'LOCATION':
            cur.execute("SELECT location_name FROM locations WHERE location_id=?", (entity_id,))
            row = cur.fetchone()
            label = row[0] if row else entity_id
        elif entity_type == 'ORGANIZATION':
            cur.execute("SELECT organization_name FROM organizations WHERE organization_id=?", (entity_id,))
            row = cur.fetchone()
            label = row[0] if row else entity_id
        elif entity_type == 'FIR':
            cur.execute("SELECT case_type FROM cases WHERE case_id=?", (entity_id,))
            row = cur.fetchone()
            label = f"{entity_id} - {row[0]}" if row else entity_id
    finally:
        conn.close()
    return label


def _normalize_network_filters(relationship_types=None, entity_types=None):
    """Normalize graph filters."""
    if relationship_types is None:
        rel_filter = None
    else:
        rel_filter = {
            str(value).strip().upper()
            for value in relationship_types
            if str(value).strip()
        }
        if not rel_filter or "ALL" in rel_filter:
            rel_filter = None

    if entity_types is None:
        entity_filter = None
    else:
        entity_filter = {
            str(value).strip().upper()
            for value in entity_types
            if str(value).strip()
        }
        if not entity_filter or "ALL" in entity_filter:
            entity_filter = None

    return rel_filter, entity_filter


def get_person_network(person_id, max_depth=1,
                       relationship_types=None, entity_types=None):
    """/4.3 — BFS traversal with optional relationship/entity filters."""
    try:
        max_depth = int(max_depth)
    except (TypeError, ValueError):
        max_depth = 1

    max_depth = max(0, min(max_depth, 20))
    root_person = str(person_id).strip().upper()
    root_node = ('PERSON', root_person)
    rel_filter, entity_filter = _normalize_network_filters(
        relationship_types, entity_types
    )

    # The selected root is always retained, even if the user chooses a
    # different entity-type filter.
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT source_type, source_id, relation_type, target_type, target_id
            FROM relationships
            ORDER BY relationship_id
        """)
        relationship_rows = cur.fetchall()
    finally:
        conn.close()

    adjacency = {}

    for st, sid, rel, tt, tid in relationship_rows:
        st = str(st).upper()
        sid = str(sid)
        rel = str(rel).upper()
        tt = str(tt).upper()
        tid = str(tid)

        if rel_filter is not None and rel not in rel_filter:
            continue

        a = (st, sid)
        b = (tt, tid)

        # Entity filters apply to nodes reached through the graph. The root
        # PERSON is always allowed.
        if entity_filter is not None:
            if a != root_node and st not in entity_filter:
                continue
            if b != root_node and tt not in entity_filter:
                continue

        edge = (st, sid, rel, tt, tid)
        adjacency.setdefault(a, []).append((b, edge))
        adjacency.setdefault(b, []).append((a, edge))

    nodes = {root_node}
    edges = []
    visited = {root_node}
    queue = [(root_node, 0)]
    edge_keys_seen = set()

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        for neighbor, edge in adjacency.get(current, []):
            edge_key = (edge[0], edge[1], edge[2], edge[3], edge[4])
            nodes.add(neighbor)

            if edge_key not in edge_keys_seen:
                edge_keys_seen.add(edge_key)
                edges.append(edge)

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return nodes, edges


def get_person_network_levels(person_id, max_depth=1,
                               relationship_types=None, entity_types=None):
    """/4.3 — BFS levels with optional graph filters."""
    try:
        max_depth = int(max_depth)
    except (TypeError, ValueError):
        max_depth = 1

    max_depth = max(0, min(max_depth, 20))
    root_person = str(person_id).strip().upper()
    root_node = ('PERSON', root_person)

    rel_filter, entity_filter = _normalize_network_filters(
        relationship_types, entity_types
    )

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT source_type, source_id, relation_type, target_type, target_id
            FROM relationships
            ORDER BY relationship_id
        """)
        rows = cur.fetchall()
    finally:
        conn.close()

    adjacency = {}

    for st, sid, rel, tt, tid in rows:
        st = str(st).upper()
        sid = str(sid)
        rel = str(rel).upper()
        tt = str(tt).upper()
        tid = str(tid)

        if rel_filter is not None and rel not in rel_filter:
            continue

        a = (st, sid)
        b = (tt, tid)

        if entity_filter is not None:
            if a != root_node and st not in entity_filter:
                continue
            if b != root_node and tt not in entity_filter:
                continue

        edge = (st, sid, rel, tt, tid)
        adjacency.setdefault(a, []).append((b, edge))
        adjacency.setdefault(b, []).append((a, edge))

    levels = {0: [root_node]}
    depth_map = {root_node: 0}
    queue = [root_node]

    while queue:
        current = queue.pop(0)
        current_depth = depth_map[current]

        if current_depth >= max_depth:
            continue

        for neighbor, _ in adjacency.get(current, []):
            if neighbor in depth_map:
                continue

            depth_map[neighbor] = current_depth + 1
            levels.setdefault(current_depth + 1, []).append(neighbor)
            queue.append(neighbor)

    nodes, edges = get_person_network(
        root_person,
        max_depth,
        relationship_types=relationship_types,
        entity_types=entity_types
    )

    return nodes, edges, levels, depth_map


def create_or_get_phone(cur, phone):
    phone = str(phone).strip()
    if not phone:
        return None
    cur.execute("INSERT OR IGNORE INTO phones(phone_number) VALUES (?)", (phone,))
    cur.execute("SELECT phone_id FROM phones WHERE phone_number=?", (phone,))
    return cur.fetchone()[0]


def create_or_get_vehicle(cur, vehicle):
    vehicle = str(vehicle).strip().upper()
    if not vehicle:
        return None
    cur.execute("INSERT OR IGNORE INTO vehicles(registration_number) VALUES (?)", (vehicle,))
    cur.execute("SELECT vehicle_id FROM vehicles WHERE registration_number=?", (vehicle,))
    return cur.fetchone()[0]


def create_or_get_location(cur, location):
    location = str(location).strip()
    if not location:
        return None
    cur.execute("INSERT OR IGNORE INTO locations(location_name) VALUES (?)", (location,))
    cur.execute("SELECT location_id FROM locations WHERE location_name=?", (location,))
    return cur.fetchone()[0]


def create_or_get_organization(cur, organization):
    organization = str(organization).strip()
    if not organization:
        return None
    cur.execute("INSERT OR IGNORE INTO organizations(organization_name) VALUES (?)", (organization,))
    cur.execute("SELECT organization_id FROM organizations WHERE organization_name=?", (organization,))
    return cur.fetchone()[0]


def sync_person_entities(cur, person_id, phone='', vehicle='', address='', organization=''):
    if phone:
        entity_id = create_or_get_phone(cur, phone)
        cur.execute("""INSERT OR IGNORE INTO relationships
            (source_type, source_id, relation_type, target_type, target_id)
            VALUES ('PERSON', ?, 'USES', 'PHONE', ?)""", (person_id, str(entity_id)))
    if vehicle:
        entity_id = create_or_get_vehicle(cur, vehicle)
        cur.execute("""INSERT OR IGNORE INTO relationships
            (source_type, source_id, relation_type, target_type, target_id)
            VALUES ('PERSON', ?, 'OWNS', 'VEHICLE', ?)""", (person_id, str(entity_id)))
    if address:
        entity_id = create_or_get_location(cur, address)
        cur.execute("""INSERT OR IGNORE INTO relationships
            (source_type, source_id, relation_type, target_type, target_id)
            VALUES ('PERSON', ?, 'LIVES_AT', 'LOCATION', ?)""", (person_id, str(entity_id)))
    if organization:
        entity_id = create_or_get_organization(cur, organization)
        cur.execute("""INSERT OR IGNORE INTO relationships
            (source_type, source_id, relation_type, target_type, target_id)
            VALUES ('PERSON', ?, 'MEMBER_OF', 'ORGANIZATION', ?)""", (person_id, str(entity_id)))


# =========================================================
# AUTOMATIC RELATIONSHIP ENGINE — # =========================================================
#
# Purpose:
#   Automatically connect two PERSON records when they share
#   the same PHONE, VEHICLE, LOCATION, ORGANIZATION or FIR.
#
# Relationship types created:
#   SHARED_PHONE
#   SHARED_VEHICLE
#   SHARED_LOCATION
#   SHARED_ORGANIZATION
#   COMMON_FIR
#
# Important:
#   - Existing PERSON -> ENTITY relationships are kept.
#   - No duplicate relationships are created.
#   - Manual PERSON -> PERSON relationships are not modified.
#   - The engine only creates evidence-based links from data
#     already present in the database.
# =========================================================

AUTO_RELATION_RULES = (
    ("PHONE", "USES", "SHARED_PHONE"),
    ("VEHICLE", "OWNS", "SHARED_VEHICLE"),
    ("LOCATION", "LIVES_AT", "SHARED_LOCATION"),
    ("ORGANIZATION", "MEMBER_OF", "SHARED_ORGANIZATION"),
)


def _insert_person_person_relationship(cur, person_a, relation_type, person_b):
    """Insert a normalized PERSON -> PERSON relationship."""
    person_a = str(person_a).strip().upper()
    person_b = str(person_b).strip().upper()

    if not person_a or not person_b or person_a == person_b:
        return False

    # Keep the relationship deterministic so P001 -> P002 is used
    # instead of sometimes creating P002 -> P001 for the same evidence.
    source_id, target_id = sorted((person_a, person_b))

    cur.execute(
        """
        INSERT OR IGNORE INTO relationships
        (
            source_type,
            source_id,
            relation_type,
            target_type,
            target_id
        )
        VALUES ('PERSON', ?, ?, 'PERSON', ?)
        """,
        (source_id, relation_type.upper(), target_id)
    )

    return cur.rowcount > 0


def _create_shared_entity_links(cur, entity_type, person_relation, auto_relation):
    """
    Find every entity used by more than one person and create
    PERSON -> PERSON automatic relationships.
    """
    cur.execute(
        """
        SELECT
            r.target_id,
            r.source_id
        FROM relationships r
        INNER JOIN persons p
            ON UPPER(p.person_id) = UPPER(r.source_id)
        WHERE r.source_type = 'PERSON'
          AND r.target_type = ?
          AND r.relation_type = ?
        ORDER BY r.target_id, r.source_id
        """,
        (entity_type, person_relation)
    )

    rows = cur.fetchall()

    # entity_id -> list of persons
    grouped = {}

    for entity_id, person_id in rows:
        entity_id = str(entity_id)
        person_id = str(person_id).upper()

        grouped.setdefault(entity_id, set()).add(person_id)

    created = 0
    affected_entities = 0

    for entity_id, person_ids in grouped.items():
        if len(person_ids) < 2:
            continue

        affected_entities += 1
        person_ids = sorted(person_ids)

        for index in range(len(person_ids)):
            for next_index in range(index + 1, len(person_ids)):
                if _insert_person_person_relationship(
                    cur,
                    person_ids[index],
                    auto_relation,
                    person_ids[next_index]
                ):
                    created += 1

    return created, affected_entities


def _create_common_fir_links(cur):
    """
    If a future version allows multiple persons to be linked to the
    same FIR, automatically connect those persons with COMMON_FIR.
    """
    cur.execute(
        """
        SELECT case_id, person_id
        FROM person_cases
        ORDER BY case_id, person_id
        """
    )

    rows = cur.fetchall()
    grouped = {}

    for case_id, person_id in rows:
        grouped.setdefault(str(case_id), set()).add(
            str(person_id).upper()
        )

    created = 0
    affected_firs = 0

    for case_id, person_ids in grouped.items():
        if len(person_ids) < 2:
            continue

        affected_firs += 1
        person_ids = sorted(person_ids)

        for index in range(len(person_ids)):
            for next_index in range(index + 1, len(person_ids)):
                if _insert_person_person_relationship(
                    cur,
                    person_ids[index],
                    "COMMON_FIR",
                    person_ids[next_index]
                ):
                    created += 1

    return created, affected_firs


def run_relationship_engine(connection=None, commit=True):
    """
    master engine.

    Scans the complete relationship graph and automatically creates
    PERSON -> PERSON links wherever two persons share the same entity.

    Returns a dictionary useful for logging/UI:
        {
            "created": total new links,
            "shared_phone": ...,
            "shared_vehicle": ...,
            "shared_location": ...,
            "shared_organization": ...,
            "common_fir": ...
        }

    If an existing SQLite connection is supplied, the caller owns
    that connection. This lets SAVE CRIMINAL / NEW FIR keep all
    changes inside one transaction.
    """
    own_connection = connection is None

    if own_connection:
        connection = get_connection()

    cur = connection.cursor()

    result = {
        "created": 0,
        "shared_phone": 0,
        "shared_vehicle": 0,
        "shared_location": 0,
        "shared_organization": 0,
        "common_fir": 0,
    }

    try:
        for entity_type, person_relation, auto_relation in AUTO_RELATION_RULES:
            created, _ = _create_shared_entity_links(
                cur,
                entity_type,
                person_relation,
                auto_relation
            )

            key = auto_relation.lower()
            result[key] = created
            result["created"] += created

        created, _ = _create_common_fir_links(cur)
        result["common_fir"] = created
        result["created"] += created

        if commit:
            connection.commit()

        return result

    except Exception:
        if own_connection:
            connection.rollback()
        raise

    finally:
        if own_connection:
            connection.close()


def run_relationship_engine_for_person(person_id):
    """
    Convenience function used after saving/updating one person.

    The engine itself scans all records so that adding P002 can
    immediately connect it with P001, P003, etc.
    """
    result = run_relationship_engine()
    return result


def get_automatic_person_relationships(person_id):
    """Return only automatically generated PERSON -> PERSON links."""
    person_id = str(person_id).strip().upper()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            relationship_id,
            source_id,
            relation_type,
            target_id
        FROM relationships
        WHERE source_type = 'PERSON'
          AND target_type = 'PERSON'
          AND relation_type IN
          (
              'SHARED_PHONE',
              'SHARED_VEHICLE',
              'SHARED_LOCATION',
              'SHARED_ORGANIZATION',
              'COMMON_FIR'
          )
          AND (UPPER(source_id) = ? OR UPPER(target_id) = ?)
        ORDER BY relation_type, relationship_id
        """,
        (person_id, person_id)
    )

    rows = cur.fetchall()
    conn.close()
    return rows


def get_person_relationship_summary(person_id):
    """
    Return a small summary for the dashboard / future graph.
    """
    rows = get_automatic_person_relationships(person_id)

    summary = {
        "total": len(rows),
        "SHARED_PHONE": 0,
        "SHARED_VEHICLE": 0,
        "SHARED_LOCATION": 0,
        "SHARED_ORGANIZATION": 0,
        "COMMON_FIR": 0,
    }

    for _, _, relation_type, _ in rows:
        if relation_type in summary:
            summary[relation_type] += 1

    return summary


def print_relationship_engine_result(result):
    """Console-friendly diagnostic output."""
    print("\n" + "=" * 70)
    print("AUTOMATIC RELATIONSHIP ENGINE — ")
    print("=" * 70)
    print("New automatic links :", result.get("created", 0))
    print("Shared phones       :", result.get("shared_phone", 0))
    print("Shared vehicles     :", result.get("shared_vehicle", 0))
    print("Shared locations    :", result.get("shared_location", 0))
    print("Shared organizations:", result.get("shared_organization", 0))
    print("Common FIRs         :", result.get("common_fir", 0))
    print("=" * 70 + "\n")


# =========================================================
# INVESTIGATION GRAPH INTELLIGENCE
# =========================================================

def _load_relationship_graph(relationship_types=None, entity_types=None):
    """Load the relationship table as an undirected investigation graph."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT source_type, source_id, relation_type, target_type, target_id
        FROM relationships
        ORDER BY relationship_id
    """)
    rows = cur.fetchall()
    conn.close()

    rel_set = {str(x).upper() for x in relationship_types} if relationship_types else None
    ent_set = {str(x).upper() for x in entity_types} if entity_types else None
    adjacency = {}

    for st, sid, rel, tt, tid in rows:
        st, sid = str(st).upper(), str(sid)
        tt, tid = str(tt).upper(), str(tid)
        rel = str(rel).upper()
        if rel_set is not None and rel not in rel_set:
            continue
        if ent_set is not None and (st not in ent_set or tt not in ent_set):
            continue
        a = (st, sid)
        b = (tt, tid)
        edge = (st, sid, rel, tt, tid)
        adjacency.setdefault(a, []).append((b, edge))
        adjacency.setdefault(b, []).append((a, edge))
    return adjacency


def find_shortest_investigation_path(source_person, target_person, max_hops=20,
                                     relationship_types=None, entity_types=None):
    """BFS shortest path between two PERSON records."""
    source = ('PERSON', str(source_person).strip().upper())
    target = ('PERSON', str(target_person).strip().upper())
    try:
        max_hops = max(0, min(int(max_hops), 50))
    except (TypeError, ValueError):
        max_hops = 20

    adjacency = _load_relationship_graph(relationship_types, entity_types)
    if source == target:
        return [source], [], 0
    if source not in adjacency or target not in adjacency:
        return None, [], None

    queue = [source]
    parent = {source: None}
    parent_edge = {}
    depth = {source: 0}

    while queue:
        current = queue.pop(0)
        if depth[current] >= max_hops:
            continue
        for neighbor, edge in adjacency.get(current, []):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            parent_edge[neighbor] = edge
            depth[neighbor] = depth[current] + 1
            if neighbor == target:
                path_nodes = []
                path_edges = []
                node = target
                while node is not None:
                    path_nodes.append(node)
                    if node in parent_edge:
                        path_edges.append(parent_edge[node])
                    node = parent[node]
                path_nodes.reverse()
                path_edges.reverse()
                return path_nodes, path_edges, depth[target]
            queue.append(neighbor)
    return None, [], None


def get_common_investigation_links(person_a, person_b, max_hops=2):
    """Return entities/people reachable from both persons within max_hops."""
    def reachable(root):
        root = ('PERSON', str(root).strip().upper())
        adjacency = _load_relationship_graph()
        depth = {root: 0}
        queue = [root]
        while queue:
            cur = queue.pop(0)
            if depth[cur] >= max_hops:
                continue
            for nxt, _ in adjacency.get(cur, []):
                if nxt not in depth:
                    depth[nxt] = depth[cur] + 1
                    queue.append(nxt)
        return depth
    a = reachable(person_a)
    b = reachable(person_b)
    common = set(a) & set(b)
    common.discard(('PERSON', str(person_a).strip().upper()))
    common.discard(('PERSON', str(person_b).strip().upper()))
    return sorted(common, key=lambda n: (n[0], str(n[1])))


def open_investigation_intelligence(person_id=None):
    """UI — shortest path + common-link analysis."""
    if not database_exists():
        return
    if not person_id or person_id == '--':
        text = search_entry.get().strip() if search_entry else ''
        person = get_person_from_search(text) if text else None
        person_id = person['person_id'] if person else None
    if not person_id:
        messagebox.showwarning('Investigation Intelligence', 'Search/select a criminal first.')
        return
    root_id = str(person_id).strip().upper()

    popup = tk.Toplevel(root)
    popup.title(f'Investigation Intelligence - {root_id}')
    popup.geometry('1050x720')
    popup.minsize(900, 600)
    popup.transient(root)

    target_var = tk.StringVar()
    hops_var = tk.IntVar(value=20)
    status_var = tk.StringVar(value='Ready')

    main = ttk.Frame(popup, padding=14)
    main.pack(fill='both', expand=True)
    ttk.Label(main, text='INVESTIGATION GRAPH INTELLIGENCE',
              font=('Segoe UI', 18, 'bold')).pack(anchor='w')
    ttk.Label(main, text=f'ROOT PERSON: {root_id}').pack(anchor='w', pady=(2, 12))

    controls = ttk.LabelFrame(main, text='SHORTEST CONNECTION PATH', padding=10)
    controls.pack(fill='x')
    ttk.Label(controls, text='Target Person ID').grid(row=0, column=0, padx=5, pady=5)
    ttk.Entry(controls, textvariable=target_var, width=20).grid(row=0, column=1, padx=5, pady=5)
    ttk.Label(controls, text='Max hops').grid(row=0, column=2, padx=5, pady=5)
    ttk.Combobox(controls, textvariable=hops_var, values=(3,5,10,20,50),
                 state='readonly', width=7).grid(row=0, column=3, padx=5, pady=5)

    path_box = ttk.LabelFrame(main, text='PATH RESULT', padding=8)
    path_box.pack(fill='both', expand=True, pady=(10, 8))
    path_tree = ttk.Treeview(path_box, columns=('hop','entity','relationship','next'), show='headings')
    for col, title, width in [('hop','HOP',70),('entity','ENTITY',260),
                              ('relationship','RELATIONSHIP',220),('next','NEXT ENTITY',360)]:
        path_tree.heading(col, text=title)
        path_tree.column(col, width=width, anchor='center')
    path_tree.pack(fill='both', expand=True)

    common_box = ttk.LabelFrame(main, text='COMMON LINKS / SHARED CONNECTIONS', padding=8)
    common_box.pack(fill='x')
    common_list = tk.Listbox(common_box, height=4, font=('Consolas', 10))
    common_list.pack(fill='x')

    footer = ttk.Frame(main, padding=(0, 8, 0, 0))
    footer.pack(fill='x')
    ttk.Label(footer, textvariable=status_var).pack(side='left', fill='x', expand=True)

    def analyze():
        target = target_var.get().strip().upper()
        if not validate_person_id(root_id) or not validate_person_id(target):
            messagebox.showwarning('Investigation Intelligence',
                                   'Enter a valid target Person ID, e.g. P002.', parent=popup)
            return
        if not get_person_by_id(target):
            messagebox.showwarning('Investigation Intelligence',
                                   f'Person {target} does not exist in the database.', parent=popup)
            return
        for item in path_tree.get_children():
            path_tree.delete(item)
        common_list.delete(0, tk.END)

        nodes, edges, hops = find_shortest_investigation_path(root_id, target, hops_var.get())
        if nodes is None:
            path_tree.insert('', 'end', values=('-', root_id, 'NO PATH', target))
            status_var.set(f'No connection found within {hops_var.get()} hops')
        else:
            for i in range(len(nodes) - 1):
                a, b = nodes[i], nodes[i + 1]
                rel = edges[i][2] if i < len(edges) else 'CONNECTED'
                path_tree.insert('', 'end', values=(i, get_entity_label(a[0], a[1]),
                                                     rel, get_entity_label(b[0], b[1])))
            status_var.set(f'SHORTEST PATH: {hops} HOPS | {len(nodes)} NODES')

        common = get_common_investigation_links(root_id, target, 2)
        if not common:
            common_list.insert(tk.END, 'No common connection found within 2 hops.')
        else:
            for node in common:
                common_list.insert(tk.END, f'{node[0]:12} {get_entity_label(node[0], node[1])}')

    ttk.Button(footer, text='ANALYZE', command=analyze).pack(side='right', padx=5)
    ttk.Button(footer, text='MAXIMIZE', command=lambda: toggle_maximize(popup)).pack(side='right', padx=5)
    ttk.Button(footer, text='CLOSE', command=lambda: safe_close_window(popup)).pack(side='right', padx=5)
    enable_window_controls(popup)

# =========================================================
# EVIDENCE SCORING & NETWORK RISK ANALYSIS
# =========================================================

STEP45_RELATION_WEIGHTS = {
    'SHARED_PHONE': 5, 'SHARED_VEHICLE': 4, 'COMMON_FIR': 4,
    'SHARED_LOCATION': 3, 'SHARED_ORGANIZATION': 3,
    'COMMUNICATES_WITH': 3, 'PARTNER_OF': 3,
    'ASSOCIATE_OF': 2, 'ASSOCIATED_WITH': 2, 'FAMILY_OF': 2,
    'USES': 2, 'OWNS': 2, 'LIVES_AT': 2, 'MEMBER_OF': 2,
    'INVOLVED_IN': 2, 'OCCURRED_AT': 1,
}

def get_relationship_evidence_score(relation_type):
    return STEP45_RELATION_WEIGHTS.get(str(relation_type).upper(), 1)

def get_relationship_evidence_rows():
    conn = get_connection(); cur = conn.cursor()
    cur.execute("""SELECT relationship_id, source_type, source_id, relation_type, target_type, target_id FROM relationships ORDER BY relationship_id""")
    rows = cur.fetchall(); conn.close()
    return [(rid, str(st).upper(), str(sid), str(rel).upper(), str(tt).upper(), str(tid), get_relationship_evidence_score(rel)) for rid, st, sid, rel, tt, tid in rows]

def calculate_person_evidence_score(person_id):
    person_id = str(person_id).strip().upper(); total = 0; count = 0; breakdown = {}
    for _, st, sid, rel, tt, tid, score in get_relationship_evidence_rows():
        if (st == 'PERSON' and sid.upper() == person_id) or (tt == 'PERSON' and tid.upper() == person_id):
            total += score; count += 1; breakdown[rel] = breakdown.get(rel, 0) + score
    return total, count, breakdown

def find_high_priority_connections(min_score=4):
    try: min_score = max(1, int(min_score))
    except (TypeError, ValueError): min_score = 4
    return [r for r in get_relationship_evidence_rows() if r[6] >= min_score]

def get_person_network_risk_rows(min_score=1):
    conn = get_connection(); cur = conn.cursor(); cur.execute('SELECT person_id, name FROM persons ORDER BY person_id'); people = cur.fetchall(); conn.close()
    rows = []
    for person_id, name in people:
        total, count, breakdown = calculate_person_evidence_score(person_id)
        if total >= min_score: rows.append((str(person_id), str(name or ''), total, count, breakdown))
    rows.sort(key=lambda x: (-x[2], -x[3], x[0])); return rows

def open_evidence_intelligence(person_id=None):
    """transparent rule-based evidence scoring UI."""
    if not database_exists(): return
    if not person_id or person_id == '--':
        text = search_entry.get().strip() if search_entry else ''
        person = get_person_from_search(text) if text else None
        person_id = person['person_id'] if person else None
    if not person_id:
        messagebox.showwarning('Evidence', 'Search/select a criminal first.'); return
    root_id = str(person_id).strip().upper()
    popup = tk.Toplevel(root); popup.title(f'Evidence Intelligence - {root_id}'); popup.geometry('1180x760'); popup.minsize(980,620); popup.transient(root)
    threshold = tk.IntVar(value=3); status = tk.StringVar(value='Ready'); total_v = tk.StringVar(value='0'); links_v = tk.StringVar(value='0')
    main = ttk.Frame(popup, padding=14); main.pack(fill='both', expand=True)
    ttk.Label(main, text='EVIDENCE SCORING & NETWORK TRIAGE', font=('Segoe UI',18,'bold')).pack(anchor='w')
    ttk.Label(main, text=f'ROOT PERSON: {root_id} | Rule-based analytical aid — not a criminality finding.').pack(anchor='w', pady=(2,10))
    controls=ttk.Frame(main); controls.pack(fill='x',pady=(0,8))
    ttk.Label(controls,text='Minimum relationship score').pack(side='left')
    ttk.Combobox(controls,textvariable=threshold,values=(1,2,3,4,5),state='readonly',width=6).pack(side='left',padx=6)
    ttk.Label(controls,text='Person score:').pack(side='left',padx=(18,4)); ttk.Label(controls,textvariable=total_v,font=('Segoe UI',10,'bold')).pack(side='left')
    ttk.Label(controls,text='Links:').pack(side='left',padx=(18,4)); ttk.Label(controls,textvariable=links_v,font=('Segoe UI',10,'bold')).pack(side='left')
    note=ttk.LabelFrame(main,text='SCORING RULES',padding=8); note.pack(fill='x',pady=(0,8))
    ttk.Label(note,text=', '.join(f'{k}={v}' for k,v in STEP45_RELATION_WEIGHTS.items()),wraplength=1080).pack(anchor='w')
    frame=ttk.LabelFrame(main,text='HIGH-PRIORITY CONNECTIONS',padding=8); frame.pack(fill='both',expand=True)
    tree=ttk.Treeview(frame,columns=('score','source','relation','target'),show='headings')
    for col,title,width in [('score','SCORE',80),('source','SOURCE',330),('relation','RELATIONSHIP',220),('target','TARGET',430)]: tree.heading(col,text=title); tree.column(col,width=width,anchor='center')
    tree.pack(fill='both',expand=True)
    def refresh():
        for item in tree.get_children(): tree.delete(item)
        total,count,_=calculate_person_evidence_score(root_id); total_v.set(str(total)); links_v.set(str(count)); shown=0
        for _,st,sid,rel,tt,tid,score in find_high_priority_connections(threshold.get()):
            if (st=='PERSON' and sid.upper()==root_id) or (tt=='PERSON' and tid.upper()==root_id):
                tree.insert('', 'end', values=(score,f'{st}: {get_entity_label(st,sid)}',rel,f'{tt}: {get_entity_label(tt,tid)}')); shown+=1
        status.set(f'ROOT {root_id} | TOTAL SCORE {total} | {shown} links shown | threshold {threshold.get()}')
    threshold.trace_add('write',lambda *args: refresh())
    footer=ttk.Frame(main,padding=(0,8,0,0)); footer.pack(fill='x'); ttk.Label(footer,textvariable=status).pack(side='left')
    ttk.Button(footer,text='REFRESH',command=refresh).pack(side='right',padx=5); ttk.Button(footer,text='CLOSE',command=popup.destroy).pack(side='right',padx=5); refresh()



# =========================================================
# TIMELINE INTELLIGENCE
# =========================================================

def get_step47_timeline(person_id=None, max_depth=5):
    """Build a chronological investigation timeline from existing SQLite data.

    Uses FIR dates, person-FIR links and relationship creation timestamps.
    No new database schema is required.
    """
    root = str(person_id or '').strip().upper()
    if not root:
        return []
    try:
        max_depth = max(0, min(int(max_depth), 20))
    except (TypeError, ValueError):
        max_depth = 5

    nodes, edges, levels, depth_map = get_person_network_levels(root, max_depth)
    relevant_persons = sorted({n[1] for n in nodes if n[0] == 'PERSON'})
    timeline = []
    conn = get_connection()
    cur = conn.cursor()
    try:
        # FIR / case events for every reachable person.
        if relevant_persons:
            placeholders = ','.join('?' for _ in relevant_persons)
            cur.execute(f"""
                SELECT pc.person_id, c.case_id, c.case_type, c.location, c.date
                FROM person_cases pc
                INNER JOIN cases c ON c.case_id = pc.case_id
                WHERE UPPER(pc.person_id) IN ({placeholders})
                ORDER BY c.date, c.case_id
            """, relevant_persons)
            for pid, case_id, case_type, location, case_date in cur.fetchall():
                timeline.append({
                    'date': str(case_date or ''), 'event': 'FIR / CASE',
                    'subject': str(pid), 'detail': f"{case_id} | {case_type or '--'} | {location or '--'}",
                    'level': depth_map.get(('PERSON', str(pid)), 0), 'source': 'FIR'
                })

        # Relationship creation events for reachable graph nodes.
        if nodes:
            cur.execute("""
                SELECT relationship_id, source_type, source_id, relation_type,
                       target_type, target_id, created_at
                FROM relationships
                ORDER BY created_at, relationship_id
            """)
            node_set = set(nodes)
            for rid, st, sid, rel, tt, tid, created_at in cur.fetchall():
                a = (str(st).upper(), str(sid)); b = (str(tt).upper(), str(tid))
                if a not in node_set and b not in node_set:
                    continue
                level = min(depth_map.get(a, 999), depth_map.get(b, 999))
                timeline.append({
                    'date': str(created_at or ''), 'event': 'RELATIONSHIP',
                    'subject': f"{a[0]}:{a[1]}",
                    'detail': f"{rel} → {b[0]}:{b[1]}" if a in node_set else f"{rel} ← {a[0]}:{a[1]}",
                    'level': 0 if level == 999 else level, 'source': 'RELATIONSHIP'
                })
    finally:
        conn.close()

    # Normalize dates enough for stable chronological sorting while retaining original text.
    def sort_key(item):
        value = item.get('date', '').strip()
        return (0, value) if value else (1, '')
    timeline.sort(key=sort_key)
    return timeline


def open_timeline_intelligence(person_id=None):
    """chronological activity and investigation timeline."""
    if not database_exists():
        return
    if not person_id or person_id == '--':
        text = search_entry.get().strip() if search_entry else ''
        person = get_person_from_search(text) if text and text != 'Enter name or Person ID' else None
        person_id = person['person_id'] if person else None
    if not person_id:
        messagebox.showwarning('Timeline Intelligence', 'Search/select a person first.')
        return

    person_id = str(person_id).strip().upper()
    popup = tk.Toplevel(root)
    popup.title(f'Timeline Intelligence — {person_id}')
    popup.geometry('1280x780')
    popup.minsize(1050, 620)
    popup.transient(root)

    main = ttk.Frame(popup, padding=14)
    main.pack(fill='both', expand=True)
    ttk.Label(main, text='INVESTIGATION TIMELINE INTELLIGENCE',
              font=('Segoe UI', 18, 'bold')).pack(anchor='w')
    ttk.Label(main, text=f'ROOT PERSON: {person_id} | Chronology is based only on dates/timestamps stored in the database.')\
        .pack(anchor='w', pady=(2, 10))

    controls = ttk.Frame(main)
    controls.pack(fill='x', pady=(0, 8))
    ttk.Label(controls, text='BFS depth').pack(side='left')
    depth_var = tk.StringVar(value='5')
    depth_box = ttk.Combobox(controls, textvariable=depth_var,
                             values=['1','2','3','4','5','10'], width=7, state='readonly')
    depth_box.pack(side='left', padx=(6, 16))
    event_var = tk.StringVar(value='ALL')
    ttk.Label(controls, text='Event type').pack(side='left')
    event_box = ttk.Combobox(controls, textvariable=event_var,
                             values=['ALL','FIR','RELATIONSHIP'], width=16, state='readonly')
    event_box.pack(side='left', padx=6)
    search_var = tk.StringVar()
    ttk.Label(controls, text='Search').pack(side='left', padx=(16, 4))
    ttk.Entry(controls, textvariable=search_var, width=28).pack(side='left')

    summary = ttk.Frame(main)
    summary.pack(fill='x', pady=(0, 8))
    total_var = tk.StringVar(value='0'); fir_var = tk.StringVar(value='0'); rel_var = tk.StringVar(value='0')
    ttk.Label(summary, text='Events:').pack(side='left'); ttk.Label(summary, textvariable=total_var, font=('Segoe UI', 10, 'bold')).pack(side='left', padx=(4,20))
    ttk.Label(summary, text='FIR events:').pack(side='left'); ttk.Label(summary, textvariable=fir_var, font=('Segoe UI', 10, 'bold')).pack(side='left', padx=(4,20))
    ttk.Label(summary, text='Relationship events:').pack(side='left'); ttk.Label(summary, textvariable=rel_var, font=('Segoe UI', 10, 'bold')).pack(side='left', padx=(4,20))

    table_frame = ttk.Frame(main)
    table_frame.pack(fill='both', expand=True)
    columns = ('date','event','subject','detail','level')
    tree = ttk.Treeview(table_frame, columns=columns, show='headings')
    headings = {'date':'DATE / TIME','event':'EVENT','subject':'SUBJECT','detail':'DETAIL','level':'BFS LEVEL'}
    widths = {'date':155,'event':130,'subject':180,'detail':600,'level':90}
    for col in columns:
        tree.heading(col, text=headings[col]); tree.column(col, width=widths[col], anchor='w')
    ybar = ttk.Scrollbar(table_frame, orient='vertical', command=tree.yview)
    xbar = ttk.Scrollbar(table_frame, orient='horizontal', command=tree.xview)
    tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
    tree.grid(row=0,column=0,sticky='nsew'); ybar.grid(row=0,column=1,sticky='ns'); xbar.grid(row=1,column=0,sticky='ew')
    table_frame.rowconfigure(0,weight=1); table_frame.columnconfigure(0,weight=1)

    status = tk.StringVar(value='Ready')
    def refresh():
        try: depth = int(depth_var.get())
        except Exception: depth = 5
        rows = get_step47_timeline(person_id, depth)
        kind = event_var.get().upper(); needle = search_var.get().strip().lower()
        filtered = []
        for row in rows:
            if kind != 'ALL' and row['source'] != kind: continue
            hay = ' '.join(str(row.get(k,'')) for k in ('date','event','subject','detail')).lower()
            if needle and needle not in hay: continue
            filtered.append(row)
        for iid in tree.get_children(): tree.delete(iid)
        for row in filtered:
            tree.insert('', 'end', values=(row['date'] or '--', row['event'], row['subject'], row['detail'], f"L{row['level']}"))
        total_var.set(str(len(filtered)))
        fir_var.set(str(sum(r['source']=='FIR' for r in filtered)))
        rel_var.set(str(sum(r['source']=='RELATIONSHIP' for r in filtered)))
        status.set(f'ROOT {person_id} | {len(filtered)} timeline events | DEPTH {depth}')

    depth_box.bind('<<ComboboxSelected>>', lambda e: refresh())
    event_box.bind('<<ComboboxSelected>>', lambda e: refresh())
    search_var.trace_add('write', lambda *_: refresh())

    footer = ttk.Frame(main, padding=(0,8,0,0)); footer.pack(fill='x')
    ttk.Label(footer, textvariable=status).pack(side='left')
    ttk.Button(footer, text='CLUSTERS', command=lambda: open_cluster_intelligence(person_id)).pack(side='right', padx=5)
    ttk.Button(footer, text='TIMELINE', command=lambda: open_timeline_intelligence(person_id)).pack(side='right', padx=5)
    ttk.Button(footer, text='NETWORK GRAPH', command=lambda: open_network_graph(person_id)).pack(side='right', padx=5)
    ttk.Button(footer, text='REFRESH', command=refresh).pack(side='right', padx=5)
    ttk.Button(footer, text='CLOSE', command=popup.destroy).pack(side='right', padx=5)
    refresh()


# =========================================================
# SUSPICIOUS CLUSTER & COMMUNITY INTELLIGENCE
# =========================================================

def get_step46_graph_data():
    adjacency = {}
    edge_rows = get_relationship_evidence_rows()
    for _, st, sid, rel, tt, tid, score in edge_rows:
        a=(st,str(sid)); b=(tt,str(tid))
        adjacency.setdefault(a,set()).add(b); adjacency.setdefault(b,set()).add(a)
    return adjacency, edge_rows


# =========================================================
# STEP 4 — KEY INDIVIDUAL / INFLUENCER DETECTION
# =========================================================

def _person_network_graph():
    """
    Build an undirected entity graph from the existing relationships table.
    No external graph library is required.
    """
    adjacency = _load_relationship_graph()
    graph = {}

    for node, neighbors in adjacency.items():
        graph.setdefault(node, set())
        for other, _edge in neighbors:
            graph.setdefault(node, set()).add(other)

    return graph


def _brandes_betweenness(graph):
    """Exact unweighted betweenness centrality using Brandes' algorithm."""
    cb = {v: 0.0 for v in graph}

    for source in graph:
        stack = []
        predecessors = {v: [] for v in graph}
        sigma = {v: 0.0 for v in graph}
        distance = {v: -1 for v in graph}

        sigma[source] = 1.0
        distance[source] = 0
        queue = [source]

        for v in queue:
            stack.append(v)
            for w in graph.get(v, ()):
                if distance[w] < 0:
                    queue.append(w)
                    distance[w] = distance[v] + 1
                if distance[w] == distance[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        dependency = {v: 0.0 for v in graph}

        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                if sigma[w]:
                    dependency[v] += (
                        sigma[v] / sigma[w]
                    ) * (1.0 + dependency[w])
            if w != source:
                cb[w] += dependency[w]

    # Undirected graph normalization.
    n = len(graph)
    if n > 2:
        scale = 1.0 / ((n - 1) * (n - 2))
        cb = {k: v * scale for k, v in cb.items()}

    return cb


def _pagerank(graph, iterations=60, damping=0.85):
    """Small dependency-free PageRank implementation."""
    nodes = list(graph)
    n = len(nodes)
    if not n:
        return {}

    rank = {v: 1.0 / n for v in nodes}

    for _ in range(iterations):
        new_rank = {
            v: (1.0 - damping) / n
            for v in nodes
        }

        dangling = sum(
            rank[v] for v in nodes
            if not graph.get(v)
        )
        dangling_share = damping * dangling / n

        for v in nodes:
            new_rank[v] += dangling_share

        for v in nodes:
            neighbors = graph.get(v, set())
            if not neighbors:
                continue
            contribution = damping * rank[v] / len(neighbors)
            for w in neighbors:
                new_rank[w] += contribution

        rank = new_rank

    return rank


def calculate_network_influencers(limit=10):
    """
    Rank PERSON entities by multiple graph signals.

    Signals:
      - Degree centrality: direct connections
      - Betweenness: intermediary/bridge importance
      - PageRank: influence from important neighbours

    Returns investigator-friendly records with an explainable score.
    """
    graph = _person_network_graph()
    if not graph:
        return []

    degree = {
        node: len(neighbors)
        for node, neighbors in graph.items()
    }

    betweenness = _brandes_betweenness(graph)
    pagerank = _pagerank(graph)

    person_nodes = [
        node for node in graph
        if str(node[0]).upper() == "PERSON"
    ]

    if not person_nodes:
        return []

    max_degree = max((degree[n] for n in person_nodes), default=1) or 1
    max_betweenness = max(
        (betweenness.get(n, 0.0) for n in person_nodes),
        default=0.0
    ) or 1.0
    max_pagerank = max(
        (pagerank.get(n, 0.0) for n in person_nodes),
        default=0.0
    ) or 1.0

    people = {}
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT person_id, name FROM persons")
        people = {
            str(pid).upper(): str(name or "Unknown")
            for pid, name in cur.fetchall()
        }
        conn.close()
    except Exception:
        people = {}

    results = []

    for node in person_nodes:
        pid = str(node[1])
        d = degree.get(node, 0)
        b = betweenness.get(node, 0.0)
        p = pagerank.get(node, 0.0)

        d_score = 100.0 * d / max_degree
        b_score = 100.0 * b / max_betweenness
        p_score = 100.0 * p / max_pagerank

        # Explainable composite influence score.
        score = round(
            (d_score * 0.45) +
            (b_score * 0.35) +
            (p_score * 0.20)
        )

        if b_score >= 70:
            role = "KEY INTERMEDIARY"
        elif d_score >= 70:
            role = "HIGH-CONNECTIVITY NODE"
        elif p_score >= 70:
            role = "NETWORK INFLUENCER"
        else:
            role = "CONNECTED PERSON"

        results.append({
            "person_id": pid,
            "name": people.get(pid.upper(), pid),
            "connections": d,
            "degree_score": round(d_score, 1),
            "betweenness_score": round(b_score, 1),
            "pagerank_score": round(p_score, 1),
            "influence_score": score,
            "role": role,
        })

    results.sort(
        key=lambda x: (
            -x["influence_score"],
            -x["connections"],
            x["name"].lower()
        )
    )

    return results[:max(1, int(limit))]


def get_influencer_summary(limit=5):
    """Human-readable explanation for the investigator dashboard."""
    rows = calculate_network_influencers(limit)
    if not rows:
        return "No person-to-person network data available yet."

    lines = ["KEY INDIVIDUALS / NETWORK INFLUENCE", ""]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"{i}. {row['name']} [{row['person_id']}]"
        )
        lines.append(
            f"   Influence Score : {row['influence_score']}/100"
        )
        lines.append(
            f"   Connections     : {row['connections']}"
        )
        lines.append(
            f"   Role            : {row['role']}"
        )
        lines.append(
            f"   Degree          : {row['degree_score']:.1f}"
            f"   | Bridge: {row['betweenness_score']:.1f}"
            f"   | PageRank: {row['pagerank_score']:.1f}"
        )
        lines.append("")

    return "\n".join(lines)


def get_step46_components():
    adjacency, edge_rows = get_step46_graph_data(); components=[]; visited=set()
    for root_node in sorted(adjacency):
        if root_node in visited: continue
        q=[root_node]; visited.add(root_node); comp=[]
        while q:
            node=q.pop(0); comp.append(node)
            for nxt in sorted(adjacency.get(node,())):
                if nxt not in visited: visited.add(nxt); q.append(nxt)
        components.append(comp)
    components.sort(key=lambda c:(-len(c), str(c[0]) if c else ''))
    return adjacency, edge_rows, components

def get_step46_cluster_rows():
    adjacency, edge_rows, components=get_step46_components(); metrics=[]
    for index, component in enumerate(components,1):
        members=set(component); internal=[]; score=0
        for row in edge_rows:
            _,st,sid,rel,tt,tid,weight=row; a=(st,str(sid)); b=(tt,str(tid))
            if a in members and b in members: internal.append(row); score+=weight
        persons=[n for n in component if n[0]=='PERSON']
        degree_sorted=sorted(component,key=lambda n:(-len(adjacency.get(n,())),n[0],n[1]))
        hub=degree_sorted[0] if degree_sorted else None
        if score>=15 and len(component)>=4: priority='HIGH'
        elif score>=8 or len(component)>=3: priority='MEDIUM'
        else: priority='LOW'
        metrics.append({'cluster':index,'nodes':len(component),'persons':len(persons),'links':len(internal),'score':score,'hub':hub,'members':component,'priority':priority})
    metrics.sort(key=lambda x:(-x['score'],-x['nodes'],-x['links'],x['cluster']))
    for rank,item in enumerate(metrics,1): item['rank']=rank
    return adjacency, edge_rows, metrics

def open_cluster_intelligence(person_id=None):
    if not database_exists(): return
    if not person_id or person_id=='--':
        text=search_entry.get().strip() if search_entry else ''; person=get_person_from_search(text) if text else None; person_id=person['person_id'] if person else None
    root_id=str(person_id).strip().upper() if person_id else ''
    popup=tk.Toplevel(root); popup.title('Cluster & Community Intelligence'); popup.geometry('1250x800'); popup.minsize(1050,650); popup.transient(root)
    main=ttk.Frame(popup,padding=14); main.pack(fill='both',expand=True)
    ttk.Label(main,text='SUSPICIOUS CLUSTER & COMMUNITY INTELLIGENCE',font=('Segoe UI',18,'bold')).pack(anchor='w')
    ttk.Label(main,text='Connected-component and hub analysis. Triage only — not a criminality finding.').pack(anchor='w',pady=(2,10))
    controls=ttk.Frame(main); controls.pack(fill='x',pady=(0,8))
    min_nodes=tk.IntVar(value=1); priority=tk.StringVar(value='ALL'); status=tk.StringVar(value='Ready')
    ttk.Label(controls,text='Minimum nodes').pack(side='left'); ttk.Combobox(controls,textvariable=min_nodes,values=(1,2,3,4,5,10),state='readonly',width=6).pack(side='left',padx=6)
    ttk.Label(controls,text='Priority').pack(side='left',padx=(18,4)); ttk.Combobox(controls,textvariable=priority,values=('ALL','HIGH','MEDIUM','LOW'),state='readonly',width=9).pack(side='left')
    frame=ttk.LabelFrame(main,text='COMMUNITY / CLUSTER RANKING',padding=8); frame.pack(fill='both',expand=True)
    tree=ttk.Treeview(frame,columns=('rank','cluster','priority','nodes','persons','links','score','hub'),show='headings')
    for col,title,width in [('rank','RANK',60),('cluster','CLUSTER',75),('priority','PRIORITY',90),('nodes','NODES',70),('persons','PERSONS',80),('links','LINKS',70),('score','EVIDENCE SCORE',120),('hub','TOP HUB',450)]: tree.heading(col,text=title); tree.column(col,width=width,anchor='center')
    tree.pack(side='left',fill='both',expand=True); sb=ttk.Scrollbar(frame,orient='vertical',command=tree.yview); sb.pack(side='right',fill='y'); tree.configure(yscrollcommand=sb.set)
    selected=tk.StringVar(value='Select a cluster to inspect members'); mf=ttk.LabelFrame(main,textvariable=selected,padding=8); mf.pack(fill='x',pady=(8,0)); txt=tk.Text(mf,height=6,wrap='word',state='disabled'); txt.pack(fill='x')
    current=[]
    def render(event=None):
        sel=tree.selection()
        if not sel:return
        cid=int(tree.item(sel[0],'values')[1]); item=next((x for x in current if x['cluster']==cid),None)
        if not item:return
        selected.set(f"CLUSTER {cid} — {item['priority']} — MEMBERS"); txt.config(state='normal'); txt.delete('1.0',tk.END)
        for node in sorted(item['members'],key=lambda n:(n[0],n[1])): txt.insert(tk.END,f"{node[0]:12} {get_entity_label(node[0],node[1])}\n")
        txt.config(state='disabled')
    def refresh():
        nonlocal current
        for x in tree.get_children(): tree.delete(x)
        _,_,metrics=get_step46_cluster_rows(); current=[m for m in metrics if m['nodes']>=int(min_nodes.get()) and (priority.get()=='ALL' or m['priority']==priority.get())]
        for m in current:
            h=m['hub']; label=get_entity_label(h[0],h[1]) if h else '--'; tree.insert('', 'end',values=(m['rank'],m['cluster'],m['priority'],m['nodes'],m['persons'],m['links'],m['score'],label))
        status.set(f"{len(current)} clusters shown | root {root_id or '--'}")
        selected.set('Select a cluster to inspect members'); txt.config(state='normal'); txt.delete('1.0',tk.END); txt.config(state='disabled')
    tree.bind('<<TreeviewSelect>>',render); min_nodes.trace_add('write',lambda *_:refresh()); priority.trace_add('write',lambda *_:refresh())
    footer=ttk.Frame(main,padding=(0,8,0,0)); footer.pack(fill='x'); ttk.Label(footer,textvariable=status).pack(side='left')
    if root_id: ttk.Button(footer,text='OPEN NETWORK GRAPH',command=lambda:open_network_graph(root_id)).pack(side='right',padx=5)
    if root_id: ttk.Button(footer,text='TIMELINE',command=lambda:open_timeline_intelligence(root_id)).pack(side='right',padx=5)
    ttk.Button(footer,text='REFRESH',command=refresh).pack(side='right',padx=5); ttk.Button(footer,text='CLOSE',command=popup.destroy).pack(side='right',padx=5); refresh()



def _step48_resolve_person(person_id=None):
    """Resolve a dashboard root person without changing the existing search flow."""
    if person_id and str(person_id).strip() not in ('', '--'):
        pid = str(person_id).strip().upper()
        if get_person_by_id(pid):
            return pid
    text = search_entry.get().strip() if search_entry else ''
    if text and text != 'Enter name or Person ID':
        person = get_person_from_search(text)
        if person:
            return str(person['person_id']).strip().upper()
    return None


def open_investigation_dashboard(person_id=None):
    """Unified investigation dashboard with lightweight Tk animations."""
    if not database_exists():
        return
    root_id = _step48_resolve_person(person_id)
    if not root_id:
        messagebox.showwarning('Dashboard', 'Search/select a criminal first.')
        return

    popup = tk.Toplevel(root)
    popup.title(f'Investigation Command Center - {root_id}')
    popup.geometry('1400x860')
    popup.minsize(1150, 720)
    popup.transient(root)

    # ----- state -----
    depth_var = tk.IntVar(value=3)
    status_var = tk.StringVar(value='INITIALIZING INVESTIGATION INTELLIGENCE...')
    person_var = tk.StringVar(value=root_id)
    name_var = tk.StringVar(value='--')
    score_var = tk.StringVar(value='0')
    links_var = tk.StringVar(value='0')
    nodes_var = tk.StringVar(value='0')
    edge_var = tk.StringVar(value='0')
    timeline_var = tk.StringVar(value='0')
    cluster_var = tk.StringVar(value='--')
    hub_var = tk.StringVar(value='--')
    progress_var = tk.DoubleVar(value=0)
    pulse_running = [True]
    pulse_phase = [0]

    main = ttk.Frame(popup, padding=12)
    main.pack(fill='both', expand=True)

    title_row = ttk.Frame(main)
    title_row.pack(fill='x', pady=(0, 8))
    ttk.Label(title_row, text='INVESTIGATION COMMAND CENTER', font=('Segoe UI', 20, 'bold')).pack(side='left')
    ttk.Label(title_row, text=f'  ROOT: {root_id}', font=('Segoe UI', 11, 'bold')).pack(side='left', padx=12)
    ttk.Label(title_row, text='DEPTH').pack(side='right')
    ttk.Combobox(title_row, textvariable=depth_var, values=(1,2,3,4,5,6,7,8,9,10), state='readonly', width=5).pack(side='right', padx=6)

    # Animated status strip.
    anim_canvas = tk.Canvas(main, height=34, highlightthickness=0)
    anim_canvas.pack(fill='x', pady=(0, 10))
    progress = ttk.Progressbar(main, variable=progress_var, maximum=100, mode='determinate')
    progress.pack(fill='x', pady=(0, 10))

    cards = ttk.Frame(main)
    cards.pack(fill='x', pady=(0, 10))
    card_specs = [
        ('PERSON', person_var), ('NETWORK SCORE', score_var), ('LINKS', links_var),
        ('NODES', nodes_var), ('RELATIONSHIPS', edge_var), ('TIMELINE', timeline_var),
        ('CLUSTER', cluster_var), ('TOP HUB', hub_var)
    ]
    for idx, (label, var) in enumerate(card_specs):
        cards.columnconfigure(idx, weight=1)
        box = ttk.LabelFrame(cards, text=label, padding=8)
        box.grid(row=0, column=idx, padx=3, sticky='nsew')
        ttk.Label(box, textvariable=var, font=('Segoe UI', 10, 'bold')).pack(anchor='center')

    body = ttk.Frame(main)
    body.pack(fill='both', expand=True)
    left = ttk.LabelFrame(body, text='INVESTIGATION SNAPSHOT', padding=10)
    left.pack(side='left', fill='both', expand=True, padx=(0, 7))
    right = ttk.LabelFrame(body, text='ACTIVITY & CONNECTIONS', padding=10)
    right.pack(side='right', fill='both', expand=True, padx=(7, 0))

    snapshot = tk.Text(left, height=20, wrap='word', font=('Consolas', 10), relief='flat')
    snapshot.pack(fill='both', expand=True)
    snapshot.insert('end', 'Loading investigation intelligence...')
    snapshot.config(state='disabled')

    tree = ttk.Treeview(right, columns=('type','subject','relation','target'), show='headings')
    for col, text, width in [('type','TYPE',100),('subject','SUBJECT',230),('relation','RELATION',180),('target','TARGET',300)]:
        tree.heading(col, text=text); tree.column(col, width=width, anchor='w')
    tree.pack(fill='both', expand=True)

    def animate_strip():
        if not popup.winfo_exists() or not pulse_running[0]:
            return
        anim_canvas.delete('all')
        w = max(anim_canvas.winfo_width(), 400)
        pulse_phase[0] = (pulse_phase[0] + 1) % 30
        offset = pulse_phase[0] * 8 - 240
        anim_canvas.create_rectangle(0, 0, w, 34, outline='')
        for x in range(offset, w + 60, 70):
            anim_canvas.create_oval(x, 10, x+14, 24, outline='')
        anim_canvas.create_text(14, 17, anchor='w', text='●  LIVE ANALYSIS  |  BFS  |  EVIDENCE  |  TIMELINE  |  CLUSTER')
        popup.after(55, animate_strip)

    def set_snapshot(person, score, count, nodes, edges, timeline_count, cluster_info):
        snapshot.config(state='normal'); snapshot.delete('1.0', tk.END)
        name = person.get('name') or '--'
        snapshot.insert('end', f'PERSON PROFILE\n')
        snapshot.insert('end', f'  ID           : {root_id}\n  NAME         : {name}\n')
        snapshot.insert('end', f'  PHONE        : {person.get("phone") or "--"}\n')
        snapshot.insert('end', f'  VEHICLE      : {person.get("vehicle") or "--"}\n')
        snapshot.insert('end', f'  LOCATION     : {person.get("address") or "--"}\n')
        snapshot.insert('end', f'  ORGANIZATION : {person.get("organization") or "--"}\n\n')
        snapshot.insert('end', 'ANALYTICAL SUMMARY\n')
        snapshot.insert('end', f'  Evidence score : {score}\n  Direct links   : {count}\n')
        snapshot.insert('end', f'  Network nodes  : {nodes}\n  Relationships  : {edges}\n')
        snapshot.insert('end', f'  Timeline events: {timeline_count}\n')
        snapshot.insert('end', f'  Cluster        : {cluster_info or "--"}\n\n')
        snapshot.insert('end', 'NOTE\n  Scores and clusters are analytical triage aids only.\n  They must not be treated as proof of criminality.\n')
        snapshot.config(state='disabled')

    def refresh_dashboard():
        try:
            progress_var.set(10); status_var.set('SCANNING PERSON PROFILE...'); popup.update_idletasks()
            person = get_person_by_id(root_id) or {'person_id': root_id}
            name_var.set(person.get('name') or '--')
            total, count, _ = calculate_person_evidence_score(root_id)
            score_var.set(str(total)); links_var.set(str(count)); progress_var.set(30)

            nodes, edges, levels, depth_map = get_person_network_levels(root_id, depth_var.get())
            nodes_var.set(str(len(nodes))); edge_var.set(str(len(edges))); progress_var.set(55)

            timeline = get_step47_timeline(root_id, depth_var.get())
            timeline_var.set(str(len(timeline))); progress_var.set(75)

            cluster_text = '--'
            hub_text = '--'
            try:
                _, _, metrics = get_step46_cluster_rows()
                for item in metrics:
                    if any(n == ('PERSON', root_id) for n in item.get('members', [])):
                        cluster_text = f"#{item['rank']} {item['priority']}"
                        hub = item.get('hub')
                        if hub:
                            hub_text = get_entity_label(hub[0], hub[1])
                        break
            except Exception:
                pass
            cluster_var.set(cluster_text); hub_var.set(hub_text)

            # Populate recent relationships for the selected person.
            for item in tree.get_children(): tree.delete(item)
            for row in get_person_relationships(root_id)[:80]:
                _, st, sid, rel, tt, tid = row
                tree.insert('', 'end', values=(rel, f'{st}: {get_entity_label(st,sid)}', rel, f'{tt}: {get_entity_label(tt,tid)}'))

            set_snapshot(person, total, count, len(nodes), len(edges), len(timeline), cluster_text)
            progress_var.set(100)
            status_var.set(f'ANALYSIS COMPLETE  |  {root_id}  |  DEPTH {depth_var.get()}  |  {len(nodes)} NODES')
        except Exception as exc:
            progress_var.set(0); status_var.set(f'ANALYSIS ERROR: {exc}')
            messagebox.showerror('Dashboard', str(exc), parent=popup)

    def open_graph(): open_network_graph(root_id)
    def open_timeline(): open_timeline_intelligence(root_id)
    def open_evidence(): open_evidence_intelligence(root_id)
    def open_cluster(): open_cluster_intelligence(root_id)
    def open_path(): open_investigation_intelligence(root_id)

    footer = ttk.Frame(main, padding=(0, 8, 0, 0)); footer.pack(fill='x')
    ttk.Label(footer, textvariable=status_var).pack(side='left')
    for text, command in [('NETWORK GRAPH', open_graph), ('TIMELINE', open_timeline), ('EVIDENCE', open_evidence), ('CLUSTER', open_cluster), ('PATH INTELLIGENCE', open_path)]:
        ttk.Button(footer, text=text, command=command).pack(side='right', padx=3)
    ttk.Button(footer, text='REFRESH', command=refresh_dashboard).pack(side='right', padx=3)
    ttk.Button(footer, text='CLOSE', command=lambda: (pulse_running.__setitem__(0, False), popup.destroy())).pack(side='right', padx=3)

    depth_var.trace_add('write', lambda *_: refresh_dashboard())
    popup.protocol('WM_DELETE_WINDOW', lambda: (pulse_running.__setitem__(0, False), popup.destroy()))
    animate_strip()
    popup.after(120, refresh_dashboard)

def open_network_graph(person_id=None):
    """/4.3 — Multi-level investigation graph with filters."""
    if not database_exists():
        return

    if not person_id or person_id == '--':
        text_value = search_entry.get().strip() if search_entry else ''
        person = (
            get_person_from_search(text_value)
            if text_value and text_value != 'Enter name or Person ID'
            else None
        )
        person_id = person['person_id'] if person else None

    if not person_id:
        messagebox.showwarning(
            'Network Graph',
            'Search/select a criminal first.'
        )
        return

    person_id = str(person_id).strip().upper()

    popup = tk.Toplevel(root)
    popup.title(f'Multi-Level Investigation Network - {person_id}')
    popup.geometry('1350x850')
    popup.minsize(1120, 700)
    popup.transient(root)

    depth_var = tk.IntVar(value=2)
    search_var = tk.StringVar()
    status_var = tk.StringVar(value='Ready')
    nodes_var = tk.StringVar(value='0')
    edges_var = tk.StringVar(value='0')
    levels_var = tk.StringVar(value='0')

    # ---------------------------------------------------------
    # FILTER DEFINITIONS
    # ---------------------------------------------------------
    relationship_options = [
        'USES',
        'OWNS',
        'LIVES_AT',
        'MEMBER_OF',
        'OCCURRED_AT',
        'INVOLVED_IN',
        'SHARED_PHONE',
        'SHARED_VEHICLE',
        'SHARED_LOCATION',
        'SHARED_ORGANIZATION',
        'COMMON_FIR',
        'ASSOCIATED_WITH',
        'COMMUNICATES_WITH',
        'PARTNER_OF',
        'ASSOCIATE_OF',
        'FAMILY_OF',
    ]

    entity_options = [
        'PERSON',
        'PHONE',
        'VEHICLE',
        'LOCATION',
        'ORGANIZATION',
        'FIR',
    ]

    rel_vars = {
        value: tk.BooleanVar(value=True)
        for value in relationship_options
    }
    entity_vars = {
        value: tk.BooleanVar(value=True)
        for value in entity_options
    }

    header = ttk.Frame(popup, padding=10)
    header.pack(fill='x')

    ttk.Label(
        header,
        text='MULTI-LEVEL INVESTIGATION NETWORK',
        font=('Segoe UI', 18, 'bold')
    ).pack(side='left')

    controls = ttk.Frame(header)
    controls.pack(side='right')

    ttk.Label(
        controls,
        text=f'ROOT: {person_id}',
        font=('Segoe UI', 10, 'bold')
    ).pack(side='left', padx=10)

    ttk.Label(controls, text='Depth').pack(side='left')
    depth_box = ttk.Combobox(
        controls,
        textvariable=depth_var,
        values=(1, 2, 3, 4, 5),
        state='readonly',
        width=5
    )
    depth_box.pack(side='left', padx=6)

    ttk.Label(controls, text='Search node').pack(side='left', padx=(12, 4))
    search_box = ttk.Entry(
        controls,
        textvariable=search_var,
        width=22
    )
    search_box.pack(side='left')

    body = ttk.Frame(popup, padding=(10, 0, 10, 5))
    body.pack(fill='both', expand=True)

    # ---------------------------------------------------------
    # LEFT SIDEBAR
    # ---------------------------------------------------------
    info = ttk.Frame(body)
    info.pack(side='left', fill='y', padx=(0, 10))

    summary = ttk.LabelFrame(
        info,
        text='NETWORK SUMMARY',
        padding=10
    )
    summary.pack(fill='x')

    for r, label, var in (
        (0, 'Nodes', nodes_var),
        (1, 'Relationships', edges_var),
        (2, 'Levels', levels_var),
    ):
        ttk.Label(
            summary,
            text=label,
            font=('Segoe UI', 9, 'bold')
        ).grid(row=r, column=0, sticky='w', pady=4)

        ttk.Label(
            summary,
            textvariable=var
        ).grid(row=r, column=1, sticky='e', padx=(20, 0), pady=4)

    # ---------------------------------------------------------
    # RELATIONSHIP FILTER
    # ---------------------------------------------------------
    rel_frame = ttk.LabelFrame(
        info,
        text='RELATIONSHIP FILTER',
        padding=8
    )
    rel_frame.pack(fill='x', pady=(10, 0))

    rel_buttons = ttk.Frame(rel_frame)
    rel_buttons.pack(fill='x')

    def set_all_relationships(value):
        for variable in rel_vars.values():
            variable.set(value)
        build_graph()

    ttk.Button(
        rel_buttons,
        text='ALL',
        width=6,
        command=lambda: set_all_relationships(True)
    ).pack(side='left', padx=2)

    ttk.Button(
        rel_buttons,
        text='NONE',
        width=6,
        command=lambda: set_all_relationships(False)
    ).pack(side='left', padx=2)

    rel_canvas = tk.Canvas(
        rel_frame,
        width=285,
        height=175,
        highlightthickness=0
    )
    rel_scroll = ttk.Scrollbar(
        rel_frame,
        orient='vertical',
        command=rel_canvas.yview
    )
    rel_inner = ttk.Frame(rel_canvas)

    rel_inner.bind(
        '<Configure>',
        lambda e: rel_canvas.configure(
            scrollregion=rel_canvas.bbox('all')
        )
    )

    rel_canvas.create_window(
        (0, 0),
        window=rel_inner,
        anchor='nw'
    )
    rel_canvas.configure(yscrollcommand=rel_scroll.set)

    rel_canvas.pack(side='left', fill='both', expand=True)
    rel_scroll.pack(side='right', fill='y')

    for value, variable in rel_vars.items():
        ttk.Checkbutton(
            rel_inner,
            text=value,
            variable=variable
        ).pack(anchor='w', pady=1)

    # ---------------------------------------------------------
    # ENTITY TYPE FILTER
    # ---------------------------------------------------------
    entity_frame = ttk.LabelFrame(
        info,
        text='ENTITY TYPE FILTER',
        padding=8
    )
    entity_frame.pack(fill='x', pady=(10, 0))

    entity_buttons = ttk.Frame(entity_frame)
    entity_buttons.pack(fill='x')

    def set_all_entities(value):
        for variable in entity_vars.values():
            variable.set(value)
        build_graph()

    ttk.Button(
        entity_buttons,
        text='ALL',
        width=6,
        command=lambda: set_all_entities(True)
    ).pack(side='left', padx=2)

    ttk.Button(
        entity_buttons,
        text='NONE',
        width=6,
        command=lambda: set_all_entities(False)
    ).pack(side='left', padx=2)

    for value, variable in entity_vars.items():
        ttk.Checkbutton(
            entity_frame,
            text=value,
            variable=variable
        ).pack(anchor='w', pady=1)

    # ---------------------------------------------------------
    # LEVEL BREAKDOWN
    # ---------------------------------------------------------
    ttk.Separator(info).pack(fill='x', pady=10)

    ttk.Label(
        info,
        text='LEVEL BREAKDOWN',
        font=('Segoe UI', 10, 'bold')
    ).pack(anchor='w')

    level_text = tk.Text(
        info,
        width=32,
        height=12,
        wrap='word',
        font=('Consolas', 9),
        relief='flat'
    )
    level_text.pack(fill='both', expand=True, pady=6)
    level_text.config(state='disabled')

    ttk.Label(
        info,
        text='L0 = SELECTED PERSON',
        font=('Segoe UI', 9, 'bold')
    ).pack(anchor='w')

    ttk.Label(
        info,
        text='L1-L5 = BFS HOPS'
    ).pack(anchor='w', pady=2)

    # ---------------------------------------------------------
    # GRAPH AREA
    # ---------------------------------------------------------
    graph_frame = ttk.Frame(body)
    graph_frame.pack(side='right', fill='both', expand=True)

    xbar = ttk.Scrollbar(graph_frame, orient='horizontal')
    ybar = ttk.Scrollbar(graph_frame, orient='vertical')

    canvas = tk.Canvas(
        graph_frame,
        bg='white',
        highlightthickness=1,
        xscrollcommand=xbar.set,
        yscrollcommand=ybar.set
    )

    xbar.config(command=canvas.xview)
    ybar.config(command=canvas.yview)

    ybar.pack(side='right', fill='y')
    xbar.pack(side='bottom', fill='x')
    canvas.pack(side='left', fill='both', expand=True)

    state = {
        'nodes': set(),
        'edges': [],
        'levels': {},
        'depth_map': {}
    }

    def selected_relationships():
        return [
            name for name, variable in rel_vars.items()
            if variable.get()
        ]

    def selected_entities():
        return [
            name for name, variable in entity_vars.items()
            if variable.get()
        ]

    def label_for(node):
        et, eid = node
        label = str(get_entity_label(et, eid) or eid)
        return label if len(label) <= 28 else label[:25] + '...'

    def visible_state():
        """Apply the display search without changing BFS data."""
        query = search_var.get().strip().lower()

        if not query:
            return (
                state['nodes'],
                state['edges'],
                state['levels'],
                state['depth_map']
            )

        visible_nodes = set()

        for node in state['nodes']:
            if node == ('PERSON', person_id):
                visible_nodes.add(node)
                continue

            label = str(get_entity_label(node[0], node[1]) or '').lower()
            if query in label or query in str(node[1]).lower():
                visible_nodes.add(node)

        # Keep the root and matching nodes; only draw edges connecting
        # visible nodes.
        visible_edges = [
            edge for edge in state['edges']
            if (edge[0], edge[1]) in visible_nodes
            and (edge[3], edge[4]) in visible_nodes
        ]

        visible_levels = {}
        visible_depth = {}

        for level, group in state['levels'].items():
            filtered = [
                node for node in group
                if node in visible_nodes
            ]
            if filtered:
                visible_levels[level] = filtered
                for node in filtered:
                    visible_depth[node] = state['depth_map'].get(node, level)

        visible_nodes = {
            node
            for group in visible_levels.values()
            for node in group
        }
        visible_nodes.add(('PERSON', person_id))

        return (
            visible_nodes,
            visible_edges,
            visible_levels,
            visible_depth
        )

    def draw_graph(event=None):
        canvas.delete('all')

        nodes, edges, level_map, depth_map = visible_state()

        if not nodes:
            canvas.create_text(
                500,
                300,
                text='NO RELATIONSHIPS FOUND',
                font=('Segoe UI', 16, 'bold')
            )
            canvas.configure(scrollregion=(0, 0, 1100, 700))
            return

        max_level = max(depth_map.values()) if depth_map else 0
        max_count = max(
            (len(value) for value in level_map.values()),
            default=1
        )

        width = max(
            1150,
            canvas.winfo_width(),
            max_count * 210
        )

        height = max(
            700,
            canvas.winfo_height(),
            140 + (max_level + 1) * 175
        )

        positions = {}

        for level in range(max_level + 1):
            group = sorted(
                level_map.get(level, []),
                key=lambda n: (n[0], str(n[1]))
            )

            if not group:
                continue

            y = 85 + level * (
                (height - 150) / max(max_level, 1)
            )

            spacing = width / (len(group) + 1)

            canvas.create_text(
                20,
                y - 58,
                text=f'LEVEL {level}',
                anchor='w',
                font=('Segoe UI', 10, 'bold')
            )

            for i, node in enumerate(group, 1):
                positions[node] = (
                    spacing * i,
                    y
                )

        for st, sid, rel, tt, tid in edges:
            a = (st, sid)
            b = (tt, tid)

            if a not in positions or b not in positions:
                continue

            x1, y1 = positions[a]
            x2, y2 = positions[b]

            canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                width=2
            )

            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2

            canvas.create_text(
                mx,
                my,
                text=str(rel),
                font=('Segoe UI', 7, 'bold'),
                fill='black'
            )

        for node, (x, y) in positions.items():
            et, eid = node
            root_node = node == ('PERSON', person_id)

            radius = (
                55
                if root_node
                else (45 if et == 'PERSON' else 38)
            )

            canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                width=3 if root_node else 2
            )

            canvas.create_text(
                x,
                y - 11,
                text=et,
                font=('Segoe UI', 7, 'bold')
            )

            canvas.create_text(
                x,
                y + 10,
                text=label_for(node),
                width=2 * radius + 20,
                font=('Segoe UI', 8)
            )

            canvas.create_text(
                x,
                y + radius + 12,
                text=f'L{depth_map.get(node, 0)}',
                font=('Segoe UI', 7, 'bold')
            )

        canvas.configure(
            scrollregion=(0, 0, width, height)
        )

    def build_graph(event=None):
        try:
            depth = int(depth_var.get())

            rel_filter = selected_relationships()
            entity_filter = selected_entities()

            nodes, edges, level_map, depth_map = (
                get_person_network_levels(
                    person_id,
                    depth,
                    relationship_types=rel_filter,
                    entity_types=entity_filter
                )
            )

            state.update(
                nodes=nodes,
                edges=edges,
                levels=level_map,
                depth_map=depth_map
            )

            visible_nodes_now, visible_edges_now, _, _ = visible_state()

            nodes_var.set(str(len(visible_nodes_now)))
            edges_var.set(str(len(visible_edges_now)))

            levels_var.set(
                str(
                    max(depth_map.values()) + 1
                    if depth_map
                    else 0
                )
            )

            level_text.config(state='normal')
            level_text.delete('1.0', tk.END)

            for level in sorted(level_map):
                group = level_map[level]

                persons = sum(
                    1 for n in group
                    if n[0] == 'PERSON'
                )

                level_text.insert(
                    tk.END,
                    f'LEVEL {level}\n'
                    f'  Total: {len(group)}\n'
                    f'  Persons: {persons}\n'
                    f'  Entities: {len(group) - persons}\n\n'
                )

            level_text.config(state='disabled')

            filter_count = len(rel_filter)

            entity_count = len(entity_filter)

            status_var.set(
                f'ROOT {person_id} | '
                f'NODES {len(visible_nodes_now)} | '
                f'LINKS {len(visible_edges_now)} | '
                f'DEPTH {depth} | '
                f'REL FILTER {filter_count} | '
                f'ENTITY FILTER {entity_count}'
            )

            draw_graph()

        except Exception as exc:
            messagebox.showerror(
                'Network Graph Error',
                str(exc),
                parent=popup
            )
            status_var.set('Graph build failed')

    # ---------------------------------------------------------
    # FOOTER
    # ---------------------------------------------------------
    footer = ttk.Frame(popup, padding=8)
    footer.pack(fill='x')

    ttk.Label(
        footer,
        textvariable=status_var
    ).pack(side='left')

    ttk.Button(
        footer,
        text='BUILD / REFRESH',
        command=build_graph
    ).pack(side='right', padx=5)

    ttk.Button(
        footer,
        text='CLEAR SEARCH',
        command=lambda: (
            search_var.set(''),
            draw_graph()
        )
    ).pack(side='right', padx=5)

    ttk.Button(
        footer,
        text='RESET VIEW',
        command=lambda: (
            canvas.xview_moveto(0),
            canvas.yview_moveto(0)
        )
    ).pack(side='right', padx=5)

    ttk.Button(
        footer,
        text='INTELLIGENCE',
        command=lambda: open_investigation_intelligence(person_id)
    ).pack(side='right', padx=5)

    ttk.Button(
        footer,
        text='EVIDENCE',
        command=lambda: open_evidence_intelligence(person_id)
    ).pack(side='right', padx=5)

    ttk.Button(footer, text='CLUSTERS', command=lambda: open_cluster_intelligence(person_id)).pack(side='right', padx=5)

    ttk.Button(
        footer,
        text='CLOSE',
        command=popup.destroy
    ).pack(side='right', padx=5)

    depth_box.bind(
        '<<ComboboxSelected>>',
        build_graph
    )

    search_box.bind(
        '<KeyRelease>',
        draw_graph
    )

    canvas.bind(
        '<Configure>',
        draw_graph
    )

    canvas.bind(
        '<MouseWheel>',
        lambda e: canvas.yview_scroll(
            int(-e.delta / 120),
            'units'
        )
    )

    # Rebuild whenever a checkbox changes.
    for variable in list(rel_vars.values()) + list(entity_vars.values()):
        variable.trace_add(
            'write',
            lambda *args: build_graph()
        )

    popup.after(150, build_graph)


def open_entity_relationships():
    if not database_exists():
        return
    current = get_person_from_search(search_entry.get().strip()) if search_entry else None
    popup = tk.Toplevel(root)
    popup.title('Entity & Relationship Engine')
    popup.geometry('1050x680')
    popup.transient(root)

    main = ttk.Frame(popup, padding=15)
    main.pack(fill='both', expand=True)
    ttk.Label(main, text='ENTITY & RELATIONSHIP ENGINE', font=('Segoe UI', 18, 'bold')).pack(anchor='w')
    ttk.Label(main, text='Phone / Vehicle / Location / Organization / FIR / Person relationships').pack(anchor='w', pady=(2, 12))

    form = ttk.LabelFrame(main, text='Add Person to Person Relationship', padding=12)
    form.pack(fill='x', pady=(0, 10))
    form.columnconfigure(1, weight=1)
    form.columnconfigure(3, weight=1)

    p1 = tk.StringVar(value=current['person_id'] if current else '')
    p2 = tk.StringVar()
    rel = tk.StringVar(value='ASSOCIATED_WITH')

    ttk.Label(form, text='Person 1 ID').grid(row=0, column=0, padx=5, pady=5, sticky='w')
    ttk.Entry(form, textvariable=p1).grid(row=0, column=1, padx=5, pady=5, sticky='ew')
    ttk.Label(form, text='Person 2 ID').grid(row=0, column=2, padx=5, pady=5, sticky='w')
    ttk.Entry(form, textvariable=p2).grid(row=0, column=3, padx=5, pady=5, sticky='ew')
    ttk.Label(form, text='Relationship').grid(row=1, column=0, padx=5, pady=5, sticky='w')
    ttk.Combobox(form, textvariable=rel, values=['ASSOCIATED_WITH','COMMUNICATES_WITH','PARTNER_OF','ASSOCIATE_OF','FAMILY_OF'], state='readonly').grid(row=1, column=1, padx=5, pady=5, sticky='ew')

    tree = ttk.Treeview(main, columns=('source','relation','target'), show='headings')
    for col, title, width in [('source','SOURCE',260),('relation','RELATIONSHIP',220),('target','TARGET',420)]:
        tree.heading(col, text=title)
        tree.column(col, width=width, anchor='center')
    tree.pack(fill='both', expand=True)

    def refresh():
        for item in tree.get_children():
            tree.delete(item)
        person_id = p1.get().strip().upper()
        rows = get_person_relationships(person_id) if validate_person_id(person_id) else []
        for _, st, sid, relationship, tt, tid in rows:
            tree.insert('', 'end', values=(f'{st}: {get_entity_label(st, sid)}', relationship, f'{tt}: {get_entity_label(tt, tid)}'))

    def save_relation():
        a, b = p1.get().strip().upper(), p2.get().strip().upper()
        if not validate_person_id(a) or not validate_person_id(b) or a == b:
            messagebox.showwarning('Relationship', 'Enter two different valid Person IDs like P001 and P002.', parent=popup)
            return
        conn = get_connection()
        cur = conn.cursor()
        cur.execute('SELECT 1 FROM persons WHERE person_id=?', (a,))
        ok1 = cur.fetchone()
        cur.execute('SELECT 1 FROM persons WHERE person_id=?', (b,))
        ok2 = cur.fetchone()
        if not ok1 or not ok2:
            conn.close()
            messagebox.showwarning('Relationship', 'Both Person IDs must exist in the database.', parent=popup)
            return
        cur.execute("""INSERT OR IGNORE INTO relationships
            (source_type, source_id, relation_type, target_type, target_id)
            VALUES ('PERSON', ?, ?, 'PERSON', ?)""", (a, rel.get().strip().upper(), b))
        conn.commit()
        conn.close()
        refresh()
        load_statistics()
        messagebox.showinfo('Saved', f'{a} -> {rel.get()} -> {b}', parent=popup)

    ttk.Button(form, text='SAVE RELATIONSHIP', command=save_relation).grid(row=1, column=3, padx=5, pady=5, sticky='e')
    refresh()

    buttons = ttk.Frame(main)
    buttons.pack(fill='x', pady=(10, 0))
    ttk.Button(buttons, text='OPEN NETWORK GRAPH', command=lambda: open_network_graph(p1.get().strip().upper())).pack(side='left')
    ttk.Button(buttons, text='REFRESH', command=refresh).pack(side='left', padx=8)
    ttk.Button(buttons, text='CLOSE', command=popup.destroy).pack(side='right')


# =========================================================
# SHOW PERSON
# =========================================================

def show_person_data(
    person
):

    if person is None:

        return

    person_id = person["person_id"]

    name = person["name"]

    phone = person["phone"]

    vehicle = person["vehicle"]

    cases = person["cases"]

    person_id_value.set(
        str(person_id)
    )

    name_value.set(
        str(name)
    )

    phone_value.set(
        str(phone or "--")
    )

    vehicle_value.set(
        str(vehicle or "--")
    )

    address_value.set(
        str(person.get("address") or "--")
    )

    organization_value.set(
        str(person.get("organization") or "--")
    )

    status_value.set(
        "CRIMINAL FOUND"
    )

    total_cases_value.set(
        str(len(cases))
    )

    update_fir_card(
        person
    )

    # -----------------------------------------------------
    # CLEAR TREE
    # -----------------------------------------------------

    for item in fir_tree.get_children():

        fir_tree.delete(
            item
        )

    # -----------------------------------------------------
    # INSERT FIR HISTORY
    # -----------------------------------------------------

    for case in cases:

        fir_tree.insert(
            "",
            "end",
            values=(
                case[0],
                case[1],
                case[2],
                display_date(
                    case[3]
                )
            )
        )

    # -----------------------------------------------------
    # HISTORY TEXT
    # -----------------------------------------------------

    history_text.config(
        state="normal"
    )

    history_text.delete(
        "1.0",
        tk.END
    )

    history_text.insert(
        tk.END,
        "CRIMINAL HISTORY\n"
    )

    history_text.insert(
        tk.END,
        "=" * 55
        + "\n\n"
    )

    history_text.insert(
        tk.END,
        f"Name: {name}\n"
        f"Person ID: {person_id}\n"
        f"Phone: {phone or '--'}\n"
        f"Vehicle: {vehicle or '--'}\n\n"
    )

    history_text.insert(
        tk.END,
        f"TOTAL FIRs: {len(cases)}\n\n"
    )

    if not cases:

        history_text.insert(
            tk.END,
            "No FIR history found."
        )

    for index, case in enumerate(
        cases,
        start=1
    ):

        history_text.insert(
            tk.END,
            f"{index}. FIR ID: {case[0]}\n"
            f"   Crime Type: {case[1]}\n"
            f"   Location: {case[2]}\n"
            f"   Date: {display_date(case[3])}\n\n"
        )

    history_text.config(
        state="disabled"
    )


# =========================================================
# SEARCH
# =========================================================

def search_person():

    text = search_entry.get().strip()

    if (
        not text
        or text == "Enter name or Person ID"
    ):

        messagebox.showwarning(
            "Search",
            "Enter Person ID or Name."
        )

        return

    if not database_exists():

        return

    try:

        person = get_person_from_search(
            text
        )

        if person is None:

            clear_dashboard()

            messagebox.showinfo(
                "Not Found",
                "No person found."
            )

            return

        show_person_data(
            person
        )

    except Exception as exc:

        messagebox.showerror(
            "Database Error",
            str(exc)
        )


# =========================================================
# CLEAR DASHBOARD
# =========================================================

def clear_dashboard():

    person_id_value.set(
        "--"
    )

    name_value.set(
        "--"
    )

    phone_value.set(
        "--"
    )

    vehicle_value.set(
        "--"
    )

    address_value.set(
        "--"
    )

    organization_value.set(
        "--"
    )

    status_value.set(
        "--"
    )

    total_cases_value.set(
        "0"
    )

    clear_fir_card()

    for item in fir_tree.get_children():

        fir_tree.delete(
            item
        )

    history_text.config(
        state="normal"
    )

    history_text.delete(
        "1.0",
        tk.END
    )

    history_text.insert(
        tk.END,
        "No person selected."
    )

    history_text.config(
        state="disabled"
    )


# =========================================================
# STATISTICS
# =========================================================

def load_statistics():

    if not os.path.exists(
        DATABASE_PATH
    ):

        return

    try:

        conn = get_connection()

        cur = conn.cursor()

        cur.execute(
            "SELECT COUNT(*) FROM persons"
        )

        total_persons = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM cases"
        )

        total_cases = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM relationships"
        )

        total_links = cur.fetchone()[0]

        conn.close()

        registered_value.set(
            str(total_persons)
        )

        fir_count_value.set(
            str(total_cases)
        )

        relationship_value.set(
            str(total_links)
        )

    except Exception as exc:

        print(
            "Statistics error:",
            exc
        )


# =========================================================
# IMAGE PREVIEW
# =========================================================

def preview_photo(
    path,
    label,
    size=(230, 230)
):

    try:

        image = Image.open(
            path
        )

        image.thumbnail(
            size,
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(
            image
        )

        label.configure(
            image=photo,
            text=""
        )

        label.image = photo

    except Exception as exc:

        print(
            "Preview error:",
            exc
        )


# =========================================================
# ADD CRIMINAL
# =========================================================

def open_add_criminal():

    if not database_exists():

        return

    popup = tk.Toplevel(
        root
    )

    popup.title(
        "Add Criminal"
    )

    popup.geometry(
        "1050x760"
    )

    popup.transient(
        root
    )

    popup.grab_set()

    # -----------------------------------------------------
    # VARIABLES
    # -----------------------------------------------------

    person_id_var = tk.StringVar(
        value=next_person_id()
    )

    name_var = tk.StringVar()

    phone_var = tk.StringVar()

    vehicle_var = tk.StringVar()

    address_var = tk.StringVar()

    organization_var = tk.StringVar()

    fir_id_var = tk.StringVar(
        value=next_fir_id()
    )

    case_type_var = tk.StringVar()

    location_var = tk.StringVar()

    date_var = tk.StringVar()

    selected_photos = []

    # -----------------------------------------------------
    # MAIN CONTAINER
    # -----------------------------------------------------

    container = ttk.Frame(
        popup,
        padding=18
    )

    container.pack(
        fill="both",
        expand=True
    )

    ttk.Label(
        container,
        text="ADD CRIMINAL RECORD",
        font=("Segoe UI", 20, "bold")
    ).pack(
        pady=(0, 15)
    )

    body = ttk.Frame(
        container
    )

    body.pack(
        fill="both",
        expand=True
    )

    # -----------------------------------------------------
    # FORM
    # -----------------------------------------------------

    form_frame = ttk.LabelFrame(
        body,
        text="Criminal Information",
        padding=15
    )

    form_frame.pack(
        side="left",
        fill="both",
        expand=True,
        padx=(0, 10)
    )

    form_frame.columnconfigure(
        1,
        weight=1
    )

    def add_row(
        row,
        label,
        variable
    ):

        ttk.Label(
            form_frame,
            text=label,
            font=("Segoe UI", 10, "bold")
        ).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 12),
            pady=8
        )

        entry = ttk.Entry(
            form_frame,
            textvariable=variable
        )

        entry.grid(
            row=row,
            column=1,
            sticky="ew",
            pady=8
        )

        return entry

    add_row(
        0,
        "Person ID *",
        person_id_var
    )

    add_row(
        1,
        "Name *",
        name_var
    )

    add_row(
        2,
        "Phone",
        phone_var
    )

    add_row(
        3,
        "Vehicle",
        vehicle_var
    )

    add_row(
        4,
        "Address",
        address_var
    )

    add_row(
        5,
        "Organization",
        organization_var
    )

    ttk.Label(
        form_frame,
        text="Format: P001, P002, P003..."
    ).grid(
        row=0,
        column=2,
        padx=8,
        sticky="w"
    )

    ttk.Separator(
        form_frame
    ).grid(
        row=6,
        column=0,
        columnspan=3,
        sticky="ew",
        pady=12
    )

    ttk.Label(
        form_frame,
        text="FIRST FIR / CRIME INFORMATION",
        font=("Segoe UI", 11, "bold")
    ).grid(
        row=7,
        column=0,
        columnspan=3,
        sticky="w"
    )

    add_row(
        8,
        "FIR ID *",
        fir_id_var
    )

    add_row(
        7,
        "Crime Type *",
        case_type_var
    )

    add_row(
        8,
        "Location *",
        location_var
    )

    add_row(
        9,
        "Date *",
        date_var
    )

    ttk.Label(
        form_frame,
        text="FIR ID: FIR001 | Date: DD-MM-YYYY"
    ).grid(
        row=12,
        column=1,
        sticky="w"
    )

    # -----------------------------------------------------
    # PHOTO FRAME
    # -----------------------------------------------------

    photo_frame = ttk.LabelFrame(
        body,
        text="Criminal Photos",
        padding=15
    )

    photo_frame.pack(
        side="right",
        fill="both"
    )

    preview_label = ttk.Label(
        photo_frame,
        text="No photo selected",
        anchor="center",
        justify="center"
    )

    preview_label.pack(
        padx=10,
        pady=10
    )

    photo_count_label = ttk.Label(
        photo_frame,
        text="0 photos selected",
        font=("Segoe UI", 10, "bold")
    )

    photo_count_label.pack(
        pady=5
    )

    def select_photos():

        paths = filedialog.askopenfilenames(
            title="Select Criminal Photos",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.webp"
                )
            ],
            parent=popup
        )

        if not paths:

            return

        selected_photos.clear()

        selected_photos.extend(
            paths
        )

        photo_count_label.config(
            text=f"{len(selected_photos)} photos selected"
        )

        preview_photo(
            selected_photos[0],
            preview_label
        )

    ttk.Button(
        photo_frame,
        text="SELECT MULTIPLE PHOTOS",
        command=select_photos
    ).pack(
        fill="x",
        pady=8
    )

    ttk.Label(
        photo_frame,
        text=(
            "Recommended: 3-5 clear photos\n\n"
            "Different angles and expressions\n"
            "can improve matching."
        ),
        justify="left"
    ).pack(
        padx=10,
        pady=10
    )

    # -----------------------------------------------------
    # BUTTON FRAME
    # -----------------------------------------------------

    button_frame = ttk.Frame(
        container
    )

    button_frame.pack(
        fill="x",
        pady=(15, 0)
    )

    # =====================================================
    # SAVE CRIMINAL
    # =====================================================

    def save_criminal():

        person_id = person_id_var.get().strip().upper()

        name = name_var.get().strip()

        phone = phone_var.get().strip()

        vehicle = vehicle_var.get().strip().upper()

        address = address_var.get().strip()

        organization = organization_var.get().strip()

        fir_id = fir_id_var.get().strip().upper()

        case_type = case_type_var.get().strip()

        location = location_var.get().strip()

        date = date_var.get().strip()

        db_date = normalize_date_for_db(
            date
        )

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        if not person_id or not name:

            messagebox.showwarning(
                "Missing Data",
                "Person ID and Name are required.",
                parent=popup
            )

            return

        if not validate_person_id(
            person_id
        ):

            messagebox.showwarning(
                "Invalid Person ID",
                "Person ID format:\n\n"
                "P001\nP002\nP003",
                parent=popup
            )

            return

        if not validate_phone(
            phone
        ):

            messagebox.showwarning(
                "Invalid Phone",
                "Phone exactly 10 digits ka hona chahiye.",
                parent=popup
            )

            return

        if not validate_vehicle(
            vehicle
        ):

            messagebox.showwarning(
                "Invalid Vehicle",
                "Vehicle format example:\nBR01XX1234",
                parent=popup
            )

            return

        if not validate_fir_id(
            fir_id
        ):

            messagebox.showwarning(
                "Invalid FIR ID",
                "FIR format:\n\n"
                "FIR001\nFIR002\nFIR003",
                parent=popup
            )

            return

        if (
            not case_type
            or not location
            or not date
        ):

            messagebox.showwarning(
                "Missing Data",
                "Crime Type, Location and Date are required.",
                parent=popup
            )

            return

        if not validate_date(
            date
        ):

            messagebox.showwarning(
                "Invalid Date",
                "Date format must be DD-MM-YYYY.\n"
                "Example: 28-08-2026",
                parent=popup
            )

            return

        if not selected_photos:

            messagebox.showwarning(
                "Missing Photos",
                "Select at least one criminal photo.",
                parent=popup
            )

            return

        save_button.config(
            state="disabled"
        )

        if not load_face_model():

            save_button.config(
                state="normal"
            )

            return

        # -------------------------------------------------
        # CREATE EMBEDDINGS
        # -------------------------------------------------

        embeddings = []

        failed = []

        for index, photo_path in enumerate(
            selected_photos,
            start=1
        ):

            popup.title(
                f"Processing photo "
                f"{index}/{len(selected_photos)}"
            )

            embedding = create_embedding_from_photo(
                photo_path
            )

            if embedding is not None:

                embeddings.append(
                    embedding
                )

            else:

                failed.append(
                    os.path.basename(
                        photo_path
                    )
                )

            popup.update_idletasks()

        popup.title(
            "Add Criminal"
        )

        if not embeddings:

            save_button.config(
                state="normal"
            )

            messagebox.showerror(
                "No Face",
                "No valid face was detected.",
                parent=popup
            )

            return

        conn = None

        try:

            conn = get_connection()

            cur = conn.cursor()

            # -------------------------------------------------
            # DUPLICATE PERSON ID
            # -------------------------------------------------

            cur.execute(
                """
                SELECT 1
                FROM persons
                WHERE UPPER(person_id)
                    = UPPER(?)
                """,
                (
                    person_id,
                )
            )

            if cur.fetchone():

                messagebox.showwarning(
                    "Duplicate Person ID",
                    f"{person_id} already exists.",
                    parent=popup
                )

                return

            # -------------------------------------------------
            # DUPLICATE NAME
            # -------------------------------------------------

            cur.execute(
                """
                SELECT 1
                FROM persons
                WHERE LOWER(name)
                    = LOWER(?)
                """,
                (
                    name,
                )
            )

            if cur.fetchone():

                messagebox.showwarning(
                    "Duplicate Name",
                    "This name already exists.\n"
                    "Use NEW FIR for existing criminal.",
                    parent=popup
                )

                return

            # -------------------------------------------------
            # DUPLICATE FIR
            # -------------------------------------------------

            cur.execute(
                """
                SELECT 1
                FROM cases
                WHERE UPPER(case_id)
                    = UPPER(?)
                """,
                (
                    fir_id,
                )
            )

            if cur.fetchone():

                messagebox.showwarning(
                    "Duplicate FIR",
                    f"{fir_id} already exists.",
                    parent=popup
                )

                return

            # -------------------------------------------------
            # INSERT PERSON
            # -------------------------------------------------

            cur.execute(
                """
                INSERT INTO persons
                (
                    person_id,
                    name,
                    phone,
                    vehicle,
                    address,
                    organization
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    name,
                    phone,
                    vehicle,
                    address,
                    organization
                )
            )

            # -------------------------------------------------
            # INSERT FIRST FIR
            # -------------------------------------------------

            cur.execute(
                """
                INSERT INTO cases
                (
                    case_id,
                    case_type,
                    location,
                    date
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    fir_id,
                    case_type,
                    location,
                    db_date
                )
            )

            # -------------------------------------------------
            # LINK PERSON + FIR
            # -------------------------------------------------

            cur.execute(
                """
                INSERT INTO person_cases
                (
                    person_id,
                    case_id
                )
                VALUES (?, ?)
                """,
                (
                    person_id,
                    fir_id
                )
            )

            # -------------------------------------------------
            # ENTITY + RELATIONSHIP GRAPH
            # -------------------------------------------------
            sync_person_entities(
                cur,
                person_id,
                phone,
                vehicle,
                address,
                organization
            )

            cur.execute(
                """INSERT OR IGNORE INTO relationships
                (source_type, source_id, relation_type, target_type, target_id)
                VALUES ('PERSON', ?, 'INVOLVED_IN', 'FIR', ?)""",
                (person_id, fir_id)
            )

            location_id = create_or_get_location(cur, location)
            cur.execute(
                """INSERT OR IGNORE INTO relationships
                (source_type, source_id, relation_type, target_type, target_id)
                VALUES ('FIR', ?, 'OCCURRED_AT', 'LOCATION', ?)""",
                (fir_id, str(location_id))
            )

            # -------------------------------------------------
            # AUTOMATIC RELATIONSHIP ENGINE
            # -------------------------------------------------
            relationship_result = run_relationship_engine(
                connection=conn,
                commit=False
            )

            # -------------------------------------------------
            # FOLDERS
            # -------------------------------------------------

            os.makedirs(
                FACES_FOLDER,
                exist_ok=True
            )

            os.makedirs(
                PHOTOS_FOLDER,
                exist_ok=True
            )

            safe_name = safe_filename(
                name
            )

            # -------------------------------------------------
            # SAVE EMBEDDINGS
            # -------------------------------------------------

            embedding_path = os.path.join(
                FACES_FOLDER,
                safe_name + ".npy"
            )

            np.save(
                embedding_path,
                np.asarray(
                    embeddings,
                    dtype=np.float32
                )
            )

            # -------------------------------------------------
            # SAVE PHOTOS
            # -------------------------------------------------

            person_photo_folder = os.path.join(
                PHOTOS_FOLDER,
                safe_name
            )

            os.makedirs(
                person_photo_folder,
                exist_ok=True
            )

            for index, source_path in enumerate(
                selected_photos,
                start=1
            ):

                extension = os.path.splitext(
                    source_path
                )[1].lower()

                destination = os.path.join(
                    person_photo_folder,
                    f"{safe_name}_{index}{extension}"
                )

                shutil.copy2(
                    source_path,
                    destination
                )

            # -------------------------------------------------
            # COMMIT
            # -------------------------------------------------

            conn.commit()

            load_statistics()

            message = (
                "Criminal added successfully.\n\n"
                f"Person ID: {person_id}\n"
                f"Name: {name}\n"
                f"FIR ID: {fir_id}\n"
                f"Valid face photos: {len(embeddings)}"
            )

            if failed:

                message += (
                    "\n\nPhotos without valid face:\n"
                    + "\n".join(
                        failed
                    )
                )

            messagebox.showinfo(
                "Success",
                message,
                parent=popup
            )

            popup.destroy()

            search_entry.delete(
                0,
                tk.END
            )

            search_entry.insert(
                0,
                person_id
            )

            search_person()

        except sqlite3.IntegrityError as exc:

            if conn:

                conn.rollback()

            messagebox.showerror(
                "Database Error",
                str(exc),
                parent=popup
            )

        except Exception as exc:

            if conn:

                conn.rollback()

            messagebox.showerror(
                "Save Error",
                str(exc),
                parent=popup
            )

        finally:

            if conn:

                conn.close()

            save_button.config(
                state="normal"
            )

    # -----------------------------------------------------
    # SAVE CRIMINAL BUTTON
    # -----------------------------------------------------

    save_button = ttk.Button(
        button_frame,
        text="SAVE CRIMINAL",
        command=save_criminal
    )

    save_button.pack(
        side="left",
        padx=5
    )

    ttk.Button(
        button_frame,
        text="CANCEL",
        command=popup.destroy
    ).pack(
        side="left",
        padx=5
    )


# =========================================================
# NEW FIR FOR EXISTING CRIMINAL
# =========================================================

def open_new_fir():

    if not database_exists():

        return

    popup = tk.Toplevel(
        root
    )

    popup.title(
        "Register New FIR"
    )

    popup.geometry(
        "900x720"
    )

    popup.transient(
        root
    )

    popup.grab_set()

    # -----------------------------------------------------
    # VARIABLES
    # -----------------------------------------------------

    fir_id_var = tk.StringVar(
        value=next_fir_id()
    )

    case_type_var = tk.StringVar()

    location_var = tk.StringVar()

    date_var = tk.StringVar()

    matched_id_var = tk.StringVar(
        value="--"
    )

    matched_name_var = tk.StringVar(
        value="--"
    )

    match_score_var = tk.StringVar(
        value="--"
    )

    previous_cases_var = tk.StringVar(
        value="0"
    )

    selected_photo = [
        None
    ]

    matched_person = [
        None
    ]

    # -----------------------------------------------------
    # MAIN
    # -----------------------------------------------------

    popup.minsize(900, 650)
    popup.resizable(True, True)

    outer = ttk.Frame(popup, padding=12)
    outer.pack(fill='both', expand=True)
    outer.rowconfigure(0, weight=1)
    outer.columnconfigure(0, weight=1)

    canvas = tk.Canvas(outer, bg='#FFFFFF', highlightthickness=0, bd=0)
    scroll = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=scroll.set)
    canvas.grid(row=0, column=0, sticky='nsew')
    scroll.grid(row=0, column=1, sticky='ns')

    main = ttk.Frame(canvas, padding=8)
    canvas_window = canvas.create_window((0, 0), window=main, anchor='nw')

    def _resize_form(event=None):
        try:
            canvas.itemconfigure(canvas_window, width=canvas.winfo_width())
            canvas.configure(scrollregion=canvas.bbox('all'))
        except Exception:
            pass
    main.bind('<Configure>', _resize_form)
    canvas.bind('<Configure>', _resize_form)

    ttk.Label(
        main,
        text="REGISTER NEW FIR",
        font=("Segoe UI", 20, "bold")
    ).pack(
        pady=(0, 15)
    )

    # =====================================================
    # FIR INFORMATION
    # =====================================================

    info = ttk.LabelFrame(
        main,
        text="NEW FIR INFORMATION",
        padding=15
    )

    info.pack(
        fill="x",
        pady=8
    )

    info.columnconfigure(
        1,
        weight=1
    )

    def field(
        row,
        label,
        variable
    ):

        ttk.Label(
            info,
            text=label,
            font=("Segoe UI", 10, "bold")
        ).grid(
            row=row,
            column=0,
            sticky="w",
            padx=5,
            pady=8
        )

        ttk.Entry(
            info,
            textvariable=variable
        ).grid(
            row=row,
            column=1,
            sticky="ew",
            padx=5,
            pady=8
        )

    field(
        0,
        "New FIR ID *",
        fir_id_var
    )

    field(
        1,
        "Crime Type *",
        case_type_var
    )

    field(
        2,
        "Location *",
        location_var
    )

    field(
        3,
        "Date *",
        date_var
    )

    ttk.Label(
        info,
        text="FIR format: FIR001 | Date: DD-MM-YYYY"
    ).grid(
        row=4,
        column=1,
        sticky="w"
    )

    # =====================================================
    # IDENTIFY CRIMINAL
    # =====================================================

    photo_frame = ttk.LabelFrame(
        main,
        text="IDENTIFY EXISTING CRIMINAL",
        padding=15
    )

    photo_frame.pack(
        fill="x",
        pady=8
    )

    preview = ttk.Label(
        photo_frame,
        text="No photo selected",
        anchor="center"
    )

    preview.pack(
        pady=10
    )

    # =====================================================
    # MATCH EXISTING CRIMINAL
    # =====================================================

    def match_existing_criminal():

        path = selected_photo[0]

        if not path:

            return

        matched_person[0] = None

        embedding = create_embedding_from_photo(
            path
        )

        if embedding is None:

            matched_id_var.set(
                "--"
            )

            matched_name_var.set(
                "No face detected"
            )

            match_score_var.set(
                "--"
            )

            previous_cases_var.set(
                "0"
            )

            return

        name, score = find_best_match(
            embedding
        )

        if name is None:

            matched_id_var.set(
                "--"
            )

            matched_name_var.set(
                "NO MATCH FOUND"
            )

            match_score_var.set(
                f"{score:.3f}"
            )

            previous_cases_var.set(
                "0"
            )

            return

        person = get_person_by_name(
            name
        )

        if person is None:

            matched_id_var.set(
                "--"
            )

            matched_name_var.set(
                "Database record missing"
            )

            match_score_var.set(
                f"{score:.3f}"
            )

            previous_cases_var.set(
                "0"
            )

            return

        matched_person[0] = person

        matched_id_var.set(
            str(
                person["person_id"]
            )
        )

        matched_name_var.set(
            str(
                person["name"]
            )
        )

        match_score_var.set(
            f"{score:.3f}"
        )

        previous_cases_var.set(
            str(
                len(
                    person["cases"]
                )
            )
        )

    # =====================================================
    # SELECT PHOTO
    # =====================================================

    def select_photo():

        path = filedialog.askopenfilename(
            title="Select Criminal Photo",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.webp"
                )
            ],
            parent=popup
        )

        if not path:

            return

        selected_photo[0] = path

        preview_photo(
            path,
            preview,
            size=(250, 180)
        )

        popup.update_idletasks()

        match_existing_criminal()

    ttk.Button(
        photo_frame,
        text="SELECT CRIMINAL PHOTO",
        command=select_photo
    ).pack(
        pady=8
    )

    # =====================================================
    # MATCH RESULT
    # =====================================================

    match_frame = ttk.LabelFrame(
        main,
        text="FACE MATCH RESULT",
        padding=15
    )

    match_frame.pack(
        fill="x",
        pady=8
    )

    result_rows = [
        (
            "Person ID:",
            matched_id_var
        ),
        (
            "Name:",
            matched_name_var
        ),
        (
            "Similarity:",
            match_score_var
        ),
        (
            "Previous FIRs:",
            previous_cases_var
        )
    ]

    for row, (
        label,
        variable
    ) in enumerate(
        result_rows
    ):

        ttk.Label(
            match_frame,
            text=label,
            font=("Segoe UI", 10, "bold")
        ).grid(
            row=row,
            column=0,
            sticky="w",
            padx=8,
            pady=6
        )

        ttk.Label(
            match_frame,
            textvariable=variable
        ).grid(
            row=row,
            column=1,
            sticky="w",
            padx=15,
            pady=6
        )

    ttk.Label(
        match_frame,
        text=(
            "If matched, this FIR will be linked "
            "to the SAME Person ID. "
            "Previous history will remain."
        ),
        wraplength=700
    ).grid(
        row=4,
        column=0,
        columnspan=2,
        sticky="w",
        padx=8,
        pady=10
    )

    # =====================================================
    # SAVE NEW FIR
    # =====================================================

    def save_new_fir():

        fir_id = fir_id_var.get().strip().upper()

        case_type = case_type_var.get().strip()

        location = location_var.get().strip()

        date = date_var.get().strip()

        person = matched_person[0]

        # -------------------------------------------------
        # VALIDATE FIR ID
        # -------------------------------------------------

        if not validate_fir_id(
            fir_id
        ):

            messagebox.showwarning(
                "Invalid FIR ID",
                "FIR format must be:\n\n"
                "FIR001\nFIR002\nFIR003",
                parent=popup
            )

            return

        # -------------------------------------------------
        # VALIDATE CRIME
        # -------------------------------------------------

        if not case_type:

            messagebox.showwarning(
                "Missing Crime Type",
                "Crime Type is required.",
                parent=popup
            )

            return

        # -------------------------------------------------
        # VALIDATE LOCATION
        # -------------------------------------------------

        if not location:

            messagebox.showwarning(
                "Missing Location",
                "Location is required.",
                parent=popup
            )

            return

        # -------------------------------------------------
        # VALIDATE DATE
        # -------------------------------------------------

        if not date:

            messagebox.showwarning(
                "Missing Date",
                "Date is required.",
                parent=popup
            )

            return

        if not validate_date(
            date
        ):

            messagebox.showwarning(
                "Invalid Date",
                "Date format must be DD-MM-YYYY.\n\n"
                "Example: 28-08-2026",
                parent=popup
            )

            return

        # -------------------------------------------------
        # VALIDATE MATCH
        # -------------------------------------------------

        if person is None:

            messagebox.showwarning(
                "Criminal Not Matched",
                "Select a criminal photo and "
                "match an existing criminal first.",
                parent=popup
            )

            return

        db_date = normalize_date_for_db(
            date
        )

        conn = None

        try:

            conn = get_connection()

            cur = conn.cursor()

            # =================================================
            # CHECK DUPLICATE FIR
            # =================================================

            cur.execute(
                """
                SELECT case_id
                FROM cases
                WHERE UPPER(case_id)
                    = UPPER(?)
                """,
                (
                    fir_id,
                )
            )

            existing_fir = cur.fetchone()

            if existing_fir:

                messagebox.showwarning(
                    "Duplicate FIR",
                    f"{fir_id} already exists in database.",
                    parent=popup
                )

                return

            # =================================================
            # VERIFY PERSON
            # =================================================

            cur.execute(
                """
                SELECT
                    person_id,
                    name
                FROM persons
                WHERE person_id = ?
                """,
                (
                    person["person_id"],
                )
            )

            db_person = cur.fetchone()

            if db_person is None:

                messagebox.showerror(
                    "Person Error",
                    "Selected criminal does not exist "
                    "in the database.",
                    parent=popup
                )

                return

            person_id = db_person[0]

            person_name = db_person[1]

            # =================================================
            # INSERT FIR
            # =================================================

            cur.execute(
                """
                INSERT INTO cases
                (
                    case_id,
                    case_type,
                    location,
                    date
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    fir_id,
                    case_type,
                    location,
                    db_date
                )
            )

            # =================================================
            # LINK PERSON WITH FIR
            # =================================================

            cur.execute(
                """
                INSERT INTO person_cases
                (
                    person_id,
                    case_id
                )
                VALUES (?, ?)
                """,
                (
                    person_id,
                    fir_id
                )
            )

            # =================================================
            # ADD FIR + LOCATION TO NETWORK GRAPH
            # =================================================
            cur.execute(
                """INSERT OR IGNORE INTO relationships
                (source_type, source_id, relation_type, target_type, target_id)
                VALUES ('PERSON', ?, 'INVOLVED_IN', 'FIR', ?)""",
                (person_id, fir_id)
            )

            location_id = create_or_get_location(cur, location)
            cur.execute(
                """INSERT OR IGNORE INTO relationships
                (source_type, source_id, relation_type, target_type, target_id)
                VALUES ('FIR', ?, 'OCCURRED_AT', 'LOCATION', ?)""",
                (fir_id, str(location_id))
            )

            # =================================================
            # AUTOMATIC RELATIONSHIP ENGINE
            # =================================================
            relationship_result = run_relationship_engine(
                connection=conn,
                commit=False
            )

            # =================================================
            # COMMIT
            # =================================================

            conn.commit()

            # =================================================
            # REFRESH STATISTICS
            # =================================================

            load_statistics()

            # =================================================
            # SUCCESS MESSAGE
            # =================================================

            messagebox.showinfo(
                "FIR Saved Successfully",
                "NEW FIR DATABASE ME SAVE HO GAYA.\n\n"
                f"Person ID : {person_id}\n"
                f"Criminal  : {person_name}\n"
                f"FIR ID    : {fir_id}\n"
                f"Crime     : {case_type}\n"
                f"Location  : {location}\n"
                f"Date      : {display_date(db_date)}\n\n"
                "FIR is permanently linked "
                "with this criminal.",
                parent=popup
            )

            # =================================================
            # CLOSE POPUP
            # =================================================

            popup.destroy()

            # =================================================
            # REFRESH MAIN DASHBOARD
            # =================================================

            search_entry.delete(
                0,
                tk.END
            )

            search_entry.insert(
                0,
                person_id
            )

            search_person()

        except sqlite3.IntegrityError as exc:

            if conn:

                conn.rollback()

            messagebox.showerror(
                "Database Error",
                "FIR save nahi ho saka.\n\n"
                + str(exc),
                parent=popup
            )

        except Exception as exc:

            if conn:

                conn.rollback()

            messagebox.showerror(
                "Save FIR Error",
                str(exc),
                parent=popup
            )

        finally:

            if conn:

                conn.close()

    # =====================================================
    # SAVE FIR BUTTON
    # =====================================================

    action_bar = ttk.Frame(outer, padding=(8, 8, 8, 0))
    action_bar.grid(row=1, column=0, columnspan=2, sticky='ew')
    action_bar.columnconfigure(0, weight=1)

    ttk.Button(
        action_bar, text='SAVE FIR TO DATABASE', command=save_new_fir
    ).grid(row=0, column=1, padx=5, pady=4, ipadx=15, ipady=5)
    ttk.Button(
        action_bar, text='MAXIMIZE', command=lambda: toggle_maximize(popup)
    ).grid(row=0, column=2, padx=5, pady=4)
    ttk.Button(
        action_bar, text='CLOSE', command=lambda: safe_close_window(popup)
    ).grid(row=0, column=3, padx=5, pady=4)

    enable_window_controls(popup)


# =========================================================
# DELETE CRIMINAL
# =========================================================

def delete_criminal():

    if not database_exists():

        return

    text = search_entry.get().strip()

    if (
        not text
        or text == "Enter name or Person ID"
    ):

        messagebox.showwarning(
            "Delete",
            "Search a criminal first."
        )

        return

    person = get_person_from_search(
        text
    )

    if person is None:

        messagebox.showinfo(
            "Not Found",
            "Criminal not found."
        )

        return

    person_id = person["person_id"]

    name = person["name"]

    confirm = messagebox.askyesno(
        "Delete Criminal",
        "WARNING!\n\n"
        f"Person ID: {person_id}\n"
        f"Name: {name}\n\n"
        "This will delete:\n"
        "• Criminal database record\n"
        "• Linked FIR relationships\n"
        "• FIR records belonging to this person\n"
        "• Face embeddings\n"
        "• Stored criminal photos\n\n"
        "Continue?",
        parent=root
    )

    if not confirm:

        return

    conn = None

    try:

        conn = get_connection()

        cur = conn.cursor()

        cur.execute(
            """
            SELECT case_id
            FROM person_cases
            WHERE person_id = ?
            """,
            (
                person_id,
            )
        )

        case_ids = [
            row[0]
            for row in cur.fetchall()
        ]

        cur.execute(
            """
            DELETE FROM person_cases
            WHERE person_id = ?
            """,
            (
                person_id,
            )
        )

        for case_id in case_ids:

            cur.execute(
                """
                SELECT COUNT(*)
                FROM person_cases
                WHERE case_id = ?
                """,
                (
                    case_id,
                )
            )

            remaining = cur.fetchone()[0]

            if remaining == 0:

                cur.execute(
                    """
                    DELETE FROM cases
                    WHERE case_id = ?
                    """,
                    (
                        case_id,
                    )
                )

        cur.execute(
            """
            DELETE FROM persons
            WHERE person_id = ?
            """,
            (
                person_id,
            )
        )

        conn.commit()

        conn.close()

        conn = None

        # -------------------------------------------------
        # DELETE EMBEDDING
        # -------------------------------------------------

        safe_name = safe_filename(
            name
        )

        embedding_path = os.path.join(
            FACES_FOLDER,
            safe_name + ".npy"
        )

        if os.path.exists(
            embedding_path
        ):

            try:

                os.remove(
                    embedding_path
                )

            except Exception as exc:

                print(
                    "Embedding delete error:",
                    exc
                )

        # -------------------------------------------------
        # DELETE PHOTOS
        # -------------------------------------------------

        photo_folder = os.path.join(
            PHOTOS_FOLDER,
            safe_name
        )

        if os.path.exists(
            photo_folder
        ):

            try:

                shutil.rmtree(
                    photo_folder
                )

            except Exception as exc:

                print(
                    "Photo folder delete error:",
                    exc
                )

        clear_dashboard()

        load_statistics()

        search_entry.delete(
            0,
            tk.END
        )

        messagebox.showinfo(
            "Deleted",
            f"Criminal {person_id} - {name} "
            "deleted successfully."
        )

    except Exception as exc:

        if conn:

            conn.rollback()

            conn.close()

        messagebox.showerror(
            "Delete Error",
            str(exc)
        )


# =========================================================
# ALL PERSONS
# =========================================================

def show_all_persons():

    if not database_exists():

        return

    try:

        conn = get_connection()

        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                person_id,
                name,
                phone,
                vehicle
            FROM persons
            ORDER BY person_id
            """
        )

        persons = cur.fetchall()

        conn.close()

        popup = tk.Toplevel(
            root
        )

        popup.title(
            "All Registered Criminals"
        )

        popup.geometry(
            "1000x620"
        )

        popup.transient(
            root
        )

        ttk.Label(
            popup,
            text="REGISTERED CRIMINALS",
            font=("Segoe UI", 18, "bold")
        ).pack(
            pady=12
        )

        ttk.Label(
            popup,
            text="Double-click a criminal "
                 "to open complete history."
        ).pack(
            pady=(0, 8)
        )

        frame = ttk.Frame(
            popup,
            padding=10
        )

        frame.pack(
            fill="both",
            expand=True
        )

        columns = (
            "id",
            "name",
            "phone",
            "vehicle"
        )

        tree = ttk.Treeview(
            frame,
            columns=columns,
            show="headings"
        )

        tree.heading(
            "id",
            text="Person ID"
        )

        tree.heading(
            "name",
            text="Name"
        )

        tree.heading(
            "phone",
            text="Phone"
        )

        tree.heading(
            "vehicle",
            text="Vehicle"
        )

        tree.column(
            "id",
            width=110,
            anchor="center"
        )

        tree.column(
            "name",
            width=250
        )

        tree.column(
            "phone",
            width=180
        )

        tree.column(
            "vehicle",
            width=180
        )

        scroll = ttk.Scrollbar(
            frame,
            orient="vertical",
            command=tree.yview
        )

        tree.configure(
            yscrollcommand=scroll.set
        )

        tree.pack(
            side="left",
            fill="both",
            expand=True
        )

        scroll.pack(
            side="right",
            fill="y"
        )

        for person in persons:

            tree.insert(
                "",
                "end",
                values=person
            )

        def open_selected(
            event=None
        ):

            selection = tree.selection()

            if not selection:

                return

            values = tree.item(
                selection[0]
            ).get(
                "values"
            )

            if not values:

                return

            search_entry.delete(
                0,
                tk.END
            )

            search_entry.insert(
                0,
                str(values[0])
            )

            popup.destroy()

            search_person()

        tree.bind(
            "<Double-1>",
            open_selected
        )

    except Exception as exc:

        messagebox.showerror(
            "Database Error",
            str(exc)
        )


# =========================================================
# SCANNER DETAILS
# =========================================================

def clear_scanner_details(
    detail_labels,
    scanner_tree,
    history_box
):

    detail_labels["status"].set(
        "UNKNOWN PERSON"
    )

    detail_labels["person_id"].set(
        "--"
    )

    detail_labels["name"].set(
        "UNKNOWN"
    )

    detail_labels["phone"].set(
        "--"
    )

    detail_labels["vehicle"].set(
        "--"
    )

    detail_labels["score"].set(
        "--"
    )

    detail_labels["cases"].set(
        "0"
    )

    for item in scanner_tree.get_children():

        scanner_tree.delete(
            item
        )

    history_box.config(
        state="normal"
    )

    history_box.delete(
        "1.0",
        tk.END
    )

    history_box.insert(
        tk.END,
        "UNKNOWN PERSON\n\n"
        "No registered criminal matched."
    )

    history_box.config(
        state="disabled"
    )


# =========================================================
# SCANNER UPDATE
# =========================================================

def update_scanner_details(
    detail_labels,
    scanner_tree,
    history_box,
    criminal_name,
    score
):

    if criminal_name is None:

        clear_scanner_details(
            detail_labels,
            scanner_tree,
            history_box
        )

        detail_labels["score"].set(
            f"{score:.3f}"
        )

        return

    person = get_person_by_name(
        criminal_name
    )

    if person is None:

        clear_scanner_details(
            detail_labels,
            scanner_tree,
            history_box
        )

        return

    cases = person["cases"]

    detail_labels["status"].set(
        "CRIMINAL FOUND"
    )

    detail_labels["person_id"].set(
        str(
            person["person_id"]
        )
    )

    detail_labels["name"].set(
        str(
            person["name"]
        )
    )

    detail_labels["phone"].set(
        str(
            person["phone"] or "--"
        )
    )

    detail_labels["vehicle"].set(
        str(
            person["vehicle"] or "--"
        )
    )

    detail_labels["score"].set(
        f"{score:.3f}"
    )

    detail_labels["cases"].set(
        str(
            len(cases)
        )
    )

    for item in scanner_tree.get_children():

        scanner_tree.delete(
            item
        )

    for case in cases:

        scanner_tree.insert(
            "",
            "end",
            values=(
                case[0],
                case[1],
                case[2],
                display_date(
                    case[3]
                )
            )
        )

    history_box.config(
        state="normal"
    )

    history_box.delete(
        "1.0",
        tk.END
    )

    history_box.insert(
        tk.END,
        "CRIMINAL HISTORY\n"
        + "=" * 55
        + "\n\n"
    )

    history_box.insert(
        tk.END,
        f"Name: {person['name']}\n"
        f"Person ID: {person['person_id']}\n"
        f"Phone: {person['phone'] or '--'}\n"
        f"Vehicle: {person['vehicle'] or '--'}\n"
        f"Match Score: {score:.3f}\n\n"
        f"TOTAL FIRs: {len(cases)}\n\n"
    )

    for index, case in enumerate(
        cases,
        start=1
    ):

        history_box.insert(
            tk.END,
            f"{index}. FIR ID: {case[0]}\n"
            f"   Crime Type: {case[1]}\n"
            f"   Location: {case[2]}\n"
            f"   Date: {display_date(case[3])}\n\n"
        )

    history_box.config(
        state="disabled"
    )


# =========================================================
# FACE SCANNER
# =========================================================

def open_scanner():

    global camera

    global camera_running

    if not database_exists():

        return

    if not load_face_model():

        return

    scanner = tk.Toplevel(
        root
    )

    scanner.title(
        "Criminal Face Scanner"
    )

    scanner.geometry(
        "1200x900"
    )

    scanner.minsize(
        1050,
        800
    )

    scanner.transient(
        root
    )

    main = ttk.Frame(
        scanner,
        padding=12
    )

    main.pack(
        fill="both",
        expand=True
    )

    ttk.Label(
        main,
        text="CRIMINAL FACE SCANNER",
        font=("Segoe UI", 20, "bold")
    ).pack(
        pady=(0, 5)
    )

    ttk.Label(
        main,
        text=(
            "Camera → Face Detection → "
            "Embedding → Criminal Database Matching"
        )
    ).pack(
        pady=(0, 10)
    )

    # =====================================================
    # CAMERA
    # =====================================================

    camera_frame = ttk.LabelFrame(
        main,
        text="LIVE CAMERA",
        padding=8
    )

    camera_frame.pack(
        fill="x",
        pady=(0, 8)
    )

    video_label = ttk.Label(
        camera_frame,
        anchor="center"
    )

    video_label.pack(
        fill="both",
        expand=True
    )

    result_var = tk.StringVar(
        value="SCANNING..."
    )

    ttk.Label(
        main,
        textvariable=result_var,
        font=("Segoe UI", 15, "bold")
    ).pack(
        pady=7
    )

    # =====================================================
    # DETAILS
    # =====================================================

    details_frame = ttk.LabelFrame(
        main,
        text="DETECTED CRIMINAL DETAILS",
        padding=12
    )

    details_frame.pack(
        fill="x",
        pady=(0, 8)
    )

    detail_labels = {
        "person_id": tk.StringVar(
            value="--"
        ),
        "name": tk.StringVar(
            value="--"
        ),
        "phone": tk.StringVar(
            value="--"
        ),
        "vehicle": tk.StringVar(
            value="--"
        ),
        "status": tk.StringVar(
            value="--"
        ),
        "score": tk.StringVar(
            value="--"
        ),
        "cases": tk.StringVar(
            value="0"
        )
    }

    labels = [
        (
            "Person ID:",
            "person_id"
        ),
        (
            "Name:",
            "name"
        ),
        (
            "Match Score:",
            "score"
        ),
        (
            "Phone:",
            "phone"
        ),
        (
            "Vehicle:",
            "vehicle"
        ),
        (
            "Status:",
            "status"
        ),
        (
            "Cases Linked:",
            "cases"
        )
    ]

    for index, (
        label_text,
        key
    ) in enumerate(
        labels
    ):

        row = index // 3

        col = (
            index % 3
        ) * 2

        ttk.Label(
            details_frame,
            text=label_text,
            font=("Segoe UI", 10, "bold")
        ).grid(
            row=row,
            column=col,
            sticky="w",
            padx=7,
            pady=5
        )

        ttk.Label(
            details_frame,
            textvariable=detail_labels[key]
        ).grid(
            row=row,
            column=col + 1,
            sticky="w",
            padx=10,
            pady=5
        )

    # =====================================================
    # LOWER
    # =====================================================

    lower = ttk.Frame(
        main
    )

    lower.pack(
        fill="both",
        expand=True
    )

    # =====================================================
    # FIR HISTORY
    # =====================================================

    fir_frame = ttk.LabelFrame(
        lower,
        text="FIR / CASE HISTORY",
        padding=8
    )

    fir_frame.pack(
        side="left",
        fill="both",
        expand=True,
        padx=(0, 5)
    )

    scanner_tree = ttk.Treeview(
        fir_frame,
        columns=(
            "case_id",
            "case_type",
            "location",
            "date"
        ),
        show="headings"
    )

    scanner_tree.heading(
        "case_id",
        text="FIR ID"
    )

    scanner_tree.heading(
        "case_type",
        text="Crime Type"
    )

    scanner_tree.heading(
        "location",
        text="Location"
    )

    scanner_tree.heading(
        "date",
        text="Date"
    )

    scanner_tree.column(
        "case_id",
        width=100
    )

    scanner_tree.column(
        "case_type",
        width=150
    )

    scanner_tree.column(
        "location",
        width=150
    )

    scanner_tree.column(
        "date",
        width=110
    )

    scanner_tree.pack(
        fill="both",
        expand=True
    )

    # =====================================================
    # HISTORY
    # =====================================================

    history_frame = ttk.LabelFrame(
        lower,
        text="COMPLETE CRIMINAL HISTORY",
        padding=8
    )

    history_frame.pack(
        side="right",
        fill="both",
        expand=True,
        padx=(5, 0)
    )

    history_box = tk.Text(
        history_frame,
        wrap="word",
        font=("Consolas", 9)
    )

    history_scroll = ttk.Scrollbar(
        history_frame,
        orient="vertical",
        command=history_box.yview
    )

    history_box.configure(
        yscrollcommand=history_scroll.set
    )

    history_box.pack(
        side="left",
        fill="both",
        expand=True
    )

    history_scroll.pack(
        side="right",
        fill="y"
    )

    history_box.insert(
        tk.END,
        "Waiting for face..."
    )

    history_box.config(
        state="disabled"
    )

    # =====================================================
    # STOP
    # =====================================================

    stop_button = ttk.Button(
        main,
        text="STOP SCAN"
    )

    stop_button.pack(
        pady=8
    )

    # =====================================================
    # CAMERA OPEN
    # =====================================================

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():

        messagebox.showerror(
            "Camera Error",
            "Could not open webcam.",
            parent=scanner
        )

        scanner.destroy()

        return

    camera_running = True

    last_detected_name = None

    # =====================================================
    # STOP CAMERA
    # =====================================================

    def stop_camera():

        global camera_running

        global camera

        camera_running = False

        if camera is not None:

            try:

                camera.release()

            except Exception:

                pass

            camera = None

        try:

            if scanner.winfo_exists():

                scanner.destroy()

        except Exception:

            pass

    stop_button.config(
        command=stop_camera
    )

    # =====================================================
    # CAMERA LOOP
    # =====================================================

    def update_camera():

        nonlocal last_detected_name

        if not camera_running:

            return

        try:

            if not scanner.winfo_exists():

                stop_camera()

                return

        except Exception:

            stop_camera()

            return

        if camera is None:

            return

        ret, frame = camera.read()

        if not ret:

            result_var.set(
                "CAMERA FRAME ERROR"
            )

            scanner.after(
                100,
                update_camera
            )

            return

        display_frame = frame.copy()

        try:

            faces = face_app.get(
                frame
            )

            detected_name = None

            detected_score = 0.0

            for face in faces:

                x1, y1, x2, y2 = map(
                    int,
                    face.bbox
                )

                embedding = normalize_embedding(
                    face.embedding
                )

                name, score = find_best_match(
                    embedding
                )

                if name is not None:

                    detected_name = name

                    detected_score = score

                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        3
                    )

                    cv2.putText(
                        display_frame,
                        f"{name} | {score:.2f}",
                        (
                            x1,
                            max(
                                30,
                                y1 - 10
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2
                    )

                else:

                    detected_score = max(
                        detected_score,
                        score
                    )

                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        display_frame,
                        f"UNKNOWN | {score:.2f}",
                        (
                            x1,
                            max(
                                30,
                                y1 - 10
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2
                    )

            # =================================================
            # CRIMINAL FOUND
            # =================================================

            if detected_name is not None:

                result_var.set(
                    f"CRIMINAL FOUND: "
                    f"{detected_name} | "
                    f"Similarity: "
                    f"{detected_score:.3f}"
                )

                if (
                    detected_name
                    != last_detected_name
                ):

                    update_scanner_details(
                        detail_labels,
                        scanner_tree,
                        history_box,
                        detected_name,
                        detected_score
                    )

                    last_detected_name = (
                        detected_name
                    )

                else:

                    detail_labels[
                        "score"
                    ].set(
                        f"{detected_score:.3f}"
                    )

            # =================================================
            # NO FACE
            # =================================================

            elif len(faces) == 0:

                result_var.set(
                    "NO FACE DETECTED"
                )

                if (
                    last_detected_name
                    is not None
                ):

                    last_detected_name = None

            # =================================================
            # UNKNOWN
            # =================================================

            else:

                result_var.set(
                    "UNKNOWN PERSON"
                )

                if (
                    last_detected_name
                    != "__UNKNOWN__"
                ):

                    update_scanner_details(
                        detail_labels,
                        scanner_tree,
                        history_box,
                        None,
                        detected_score
                    )

                    last_detected_name = (
                        "__UNKNOWN__"
                    )

            # =================================================
            # DISPLAY FRAME
            # =================================================

            frame_rgb = cv2.cvtColor(
                display_frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(
                frame_rgb
            )

            image.thumbnail(
                (1100, 480),
                Image.Resampling.LANCZOS
            )

            photo = ImageTk.PhotoImage(
                image
            )

            video_label.configure(
                image=photo
            )

            video_label.image = photo

        except Exception as exc:

            print(
                "Scanner error:",
                exc
            )

            result_var.set(
                "SCANNING..."
            )

        if camera_running:

            scanner.after(
                40,
                update_camera
            )

    scanner.protocol(
        "WM_DELETE_WINDOW",
        stop_camera
    )

    update_camera()


# =========================================================
# ENTER SEARCH
# =========================================================

def enter_search(event):

    search_person()


# =========================================================
# GUI
# =========================================================

def create_gui():

    global root
    global search_entry

    global person_id_value
    global name_value
    global phone_value
    global vehicle_value
    global address_value
    global organization_value
    global status_value
    global total_cases_value

    global registered_value
    global fir_count_value
    global relationship_value

    global fir_tree
    global history_text

    global fir_card_id_value
    global fir_card_date_value
    global fir_card_type_value
    global fir_card_location_value
    global fir_card_person_value
    global fir_card_name_value
    global fir_card_status_value

    # =====================================================
    # ROOT
    # =====================================================

    root = tk.Tk()
    install_professional_ui()

    root.title(
        "AI-Assisted Investigation Intelligence & Relationship Analysis Command Center"
    )

    root.geometry(
        f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
    )

    root.minsize(
        1050,
        680
    )

    # =====================================================
    # STYLE
    # =====================================================

    style = ttk.Style()

    try:

        style.theme_use(
            "clam"
        )

    except Exception:

        pass

    style.configure(
        "Title.TLabel",
        font=("Segoe UI", 21, "bold")
    )

    style.configure(
        "Subtitle.TLabel",
        font=("Segoe UI", 10)
    )

    style.configure(
        "Section.TLabelframe",
        padding=12
    )

    style.configure(
        "Section.TLabelframe.Label",
        font=("Segoe UI", 11, "bold")
    )

    style.configure(
        "Info.TLabel",
        font=("Segoe UI", 10)
    )

    style.configure(
        "InfoBold.TLabel",
        font=("Segoe UI", 10, "bold")
    )

    # =====================================================
    # VARIABLES
    # =====================================================

    person_id_value = tk.StringVar(
        value="--"
    )

    name_value = tk.StringVar(
        value="--"
    )

    phone_value = tk.StringVar(
        value="--"
    )

    vehicle_value = tk.StringVar(
        value="--"
    )

    address_value = tk.StringVar(
        value="--"
    )

    organization_value = tk.StringVar(
        value="--"
    )

    status_value = tk.StringVar(
        value="--"
    )

    total_cases_value = tk.StringVar(
        value="0"
    )

    registered_value = tk.StringVar(
        value="0"
    )

    fir_count_value = tk.StringVar(
        value="0"
    )

    relationship_value = tk.StringVar(
        value="0"
    )

    # =====================================================
    # MAIN FRAME
    # =====================================================

    main_frame = ttk.Frame(
        root,
        padding=15
    )

    main_frame.pack(
        fill="both",
        expand=True
    )

    # =====================================================
    # HEADER
    # =====================================================

    header = ttk.Frame(
        main_frame
    )

    header.pack(
        fill="x",
        pady=(0, 8)
    )

    ttk.Label(
        header,
        text="CRIMINAL INTELLIGENCE DASHBOARD",
        style="Title.TLabel"
    ).pack(
        anchor="w"
    )

    ttk.Label(
        header,
        text=(
            "Face Recognition → "
            "Person Identification → "
            "FIR Details → "
            "Criminal History"
        ),
        style="Subtitle.TLabel"
    ).pack(
        anchor="w",
        pady=(2, 12)
    )

    # =====================================================
    # SEARCH OPERATIONS
    # =====================================================

    search_frame = ttk.LabelFrame(
        main_frame, text="SEARCH & CRIMINAL OPERATIONS", style="Section.TLabelframe"
    )
    search_frame.pack(fill="x", pady=(0, 12))
    search_top = ttk.Frame(search_frame)
    search_top.pack(fill="x", pady=(0, 7))
    search_top.columnconfigure(0, weight=1)
    search_entry = ttk.Entry(search_top, width=34, font=("Segoe UI", 11))
    search_entry.grid(row=0, column=0, sticky="ew", padx=(2, 8))
    placeholder = "Enter name or Person ID"
    search_entry.insert(0, placeholder)
    def clear_placeholder(event=None):
        if search_entry.get() == placeholder: search_entry.delete(0, tk.END)
    def restore_placeholder(event=None):
        if not search_entry.get().strip(): search_entry.insert(0, placeholder)
    search_entry.bind("<FocusIn>", clear_placeholder)
    search_entry.bind("<FocusOut>", restore_placeholder)
    search_entry.bind("<Return>", enter_search)
    for col, text, cmd in [(1,"SEARCH",search_person),(2,"CLEAR",clear_dashboard),(3,"ADD CRIMINAL",open_add_criminal),(4,"NEW FIR",open_new_fir),(5,"SCAN CRIMINALS",open_scanner),(6,"DELETE CRIMINAL",delete_criminal)]:
        ttk.Button(search_top, text=text, command=cmd).grid(row=0, column=col, padx=3)
    search_bottom = ttk.Frame(search_frame)
    search_bottom.pack(fill="x")
    search_bottom.columnconfigure(0, weight=1)
    actions = [
        (1,"ALL PERSONS",show_all_persons),
        (2,"ENTITIES / RELATIONSHIPS",open_entity_relationships),
        (3,"NETWORK GRAPH",lambda: open_network_graph(person_id_value.get())),
        (4,"CLUSTER INTELLIGENCE",lambda: open_cluster_intelligence(person_id_value.get())),
        (5,"INVESTIGATION CENTER",lambda: open_investigation_dashboard(person_id_value.get())),
        (6,"ADVANCED SEARCH",open_global_investigation_search),
        (7,"ANALYTICS",open_investigation_analytics),
        (8,"PRIORITY",lambda: open_investigation_priority(person_id_value.get())),
    ]
    for col, text, cmd in actions:
        ttk.Button(search_bottom, text=text, command=cmd).grid(row=0, column=col, padx=3)

    # =====================================================
    # STATISTICS
    # =====================================================

    stats_frame = ttk.Frame(
        main_frame
    )

    stats_frame.pack(
        fill="x",
        pady=(0, 12)
    )

    def create_stat(
        parent,
        title,
        variable
    ):

        frame = ttk.LabelFrame(
            parent,
            text=title,
            padding=10
        )

        frame.pack(
            side="left",
            fill="x",
            expand=True,
            padx=5
        )

        ttk.Label(
            frame,
            textvariable=variable,
            font=("Segoe UI", 18, "bold")
        ).pack()

    create_stat(
        stats_frame,
        "REGISTERED PERSONS",
        registered_value
    )

    create_stat(
        stats_frame,
        "TOTAL FIRs",
        fir_count_value
    )

    create_stat(
        stats_frame,
        "TOTAL RELATIONSHIPS",
        relationship_value
    )

    # =====================================================
    # CONTENT
    # =====================================================

    content = ttk.Frame(
        main_frame
    )

    content.pack(
        fill="both",
        expand=True
    )

    # =====================================================
    # LEFT PANEL
    # =====================================================

    left_panel = ttk.Frame(
        content
    )

    left_panel.pack(
        side="left",
        fill="both",
        expand=True,
        padx=(0, 7)
    )

    # =====================================================
    # PERSON INFORMATION
    # =====================================================

    person_frame = ttk.LabelFrame(
        left_panel,
        text="Person Information",
        style="Section.TLabelframe"
    )

    person_frame.pack(
        fill="x",
        pady=(0, 10)
    )

    def add_info_row(
        parent,
        row,
        label,
        variable
    ):

        ttk.Label(
            parent,
            text=label,
            style="InfoBold.TLabel"
        ).grid(
            row=row,
            column=0,
            sticky="w",
            padx=5,
            pady=6
        )

        ttk.Label(
            parent,
            textvariable=variable,
            style="Info.TLabel"
        ).grid(
            row=row,
            column=1,
            sticky="w",
            padx=20,
            pady=6
        )

    add_info_row(
        person_frame,
        0,
        "Person ID:",
        person_id_value
    )

    add_info_row(
        person_frame,
        1,
        "Name:",
        name_value
    )

    add_info_row(
        person_frame,
        2,
        "Phone:",
        phone_value
    )

    add_info_row(
        person_frame,
        3,
        "Vehicle:",
        vehicle_value
    )

    add_info_row(
        person_frame,
        4,
        "Address:",
        address_value
    )

    add_info_row(
        person_frame,
        5,
        "Organization:",
        organization_value
    )

    add_info_row(
        person_frame,
        6,
        "Status:",
        status_value
    )

    add_info_row(
        person_frame,
        7,
        "Cases Linked:",
        total_cases_value
    )

    # =====================================================
    # FIR INTELLIGENCE CARD
    # =====================================================

    fir_card = ttk.LabelFrame(
        left_panel,
        text="FIR INTELLIGENCE CARD",
        style="Section.TLabelframe"
    )

    fir_card.pack(
        fill="x",
        pady=(0, 10)
    )

    fir_card.columnconfigure(
        1,
        weight=1
    )

    fir_card.columnconfigure(
        3,
        weight=1
    )

    fir_card_id_value = tk.StringVar(
        value="--"
    )

    fir_card_date_value = tk.StringVar(
        value="--"
    )

    fir_card_type_value = tk.StringVar(
        value="--"
    )

    fir_card_location_value = tk.StringVar(
        value="--"
    )

    fir_card_person_value = tk.StringVar(
        value="--"
    )

    fir_card_name_value = tk.StringVar(
        value="--"
    )

    fir_card_status_value = tk.StringVar(
        value="--"
    )

    card_fields = [
        (
            "FIR ID",
            fir_card_id_value,
            0,
            0
        ),
        (
            "Date",
            fir_card_date_value,
            0,
            2
        ),
        (
            "Crime Type",
            fir_card_type_value,
            1,
            0
        ),
        (
            "Location",
            fir_card_location_value,
            1,
            2
        ),
        (
            "Person ID",
            fir_card_person_value,
            2,
            0
        ),
        (
            "Name",
            fir_card_name_value,
            2,
            2
        ),
        (
            "Record Status",
            fir_card_status_value,
            3,
            0
        )
    ]

    for (
        label_text,
        variable,
        row,
        col
    ) in card_fields:

        ttk.Label(
            fir_card,
            text=f"{label_text}:",
            style="InfoBold.TLabel"
        ).grid(
            row=row,
            column=col,
            sticky="w",
            padx=(5, 8),
            pady=5
        )

        ttk.Label(
            fir_card,
            textvariable=variable,
            style="Info.TLabel"
        ).grid(
            row=row,
            column=col + 1,
            sticky="w",
            padx=(0, 15),
            pady=5
        )

    ttk.Label(
        fir_card,
        text=(
            "Format: FIR001 • "
            "Date: DD-MM-YYYY • "
            "Stored internally as YYYY-MM-DD"
        ),
        font=("Segoe UI", 8)
    ).grid(
        row=4,
        column=0,
        columnspan=4,
        sticky="w",
        padx=5,
        pady=(3, 0)
    )

    # =====================================================
    # FIR TABLE
    # =====================================================

    fir_frame = ttk.LabelFrame(
        left_panel,
        text="FIR Details",
        style="Section.TLabelframe"
    )

    fir_frame.pack(
        fill="both",
        expand=True
    )

    fir_tree = ttk.Treeview(
        fir_frame,
        columns=(
            "case_id",
            "case_type",
            "location",
            "date"
        ),
        show="headings"
    )

    fir_tree.heading(
        "case_id",
        text="FIR ID"
    )

    fir_tree.heading(
        "case_type",
        text="Crime Type"
    )

    fir_tree.heading(
        "location",
        text="Location"
    )

    fir_tree.heading(
        "date",
        text="Date"
    )

    fir_tree.column(
        "case_id",
        width=100,
        anchor="center"
    )

    fir_tree.column(
        "case_type",
        width=150,
        anchor="center"
    )

    fir_tree.column(
        "location",
        width=130,
        anchor="center"
    )

    fir_tree.column(
        "date",
        width=120,
        anchor="center"
    )

    fir_scroll = ttk.Scrollbar(
        fir_frame,
        orient="vertical",
        command=fir_tree.yview
    )

    fir_tree.configure(
        yscrollcommand=fir_scroll.set
    )

    fir_tree.pack(
        side="left",
        fill="both",
        expand=True
    )

    fir_scroll.pack(
        side="right",
        fill="y"
    )

    # =====================================================
    # RIGHT PANEL
    # =====================================================

    right_panel = ttk.Frame(
        content
    )

    right_panel.pack(
        side="right",
        fill="both",
        expand=True,
        padx=(7, 0)
    )

    history_frame = ttk.LabelFrame(
        right_panel,
        text="Complete Criminal History",
        style="Section.TLabelframe"
    )

    history_frame.pack(
        fill="both",
        expand=True
    )

    history_text = tk.Text(
        history_frame,
        wrap="word",
        font=("Consolas", 10),
        padx=10,
        pady=10
    )

    history_scroll = ttk.Scrollbar(
        history_frame,
        orient="vertical",
        command=history_text.yview
    )

    history_text.configure(
        yscrollcommand=history_scroll.set
    )

    history_text.pack(
        side="left",
        fill="both",
        expand=True
    )

    history_scroll.pack(
        side="right",
        fill="y"
    )

    history_text.insert(
        tk.END,
        "No person selected.\n\n"
        "Available operations:\n\n"
        "• Search Criminal\n"
        "• Add Criminal\n"
        "• New FIR\n"
        "• Scan Criminals\n"
        "• Delete Criminal\n"
        "• All Persons\n\n"
        "Fixed formats:\n"
        "Person ID: P001\n"
        "FIR ID: FIR001\n"
        "Date: DD-MM-YYYY\n"
        "Phone: 10 digits\n"
        "Vehicle: BR01XX1234"
    )

    history_text.config(
        state="disabled"
    )

    # =====================================================
    # FOOTER
    # =====================================================

    footer = ttk.Label(
        main_frame,
        text=(
            "Criminal Intelligence System | "
            "SQLite | InsightFace / ArcFace"
        ),
        font=("Segoe UI", 9)
    )

    footer.pack(
        pady=(10, 0)
    )

    # =====================================================
    # INTERACTION POLISH
    # =====================================================
    def polish_buttons(parent):
        for child in parent.winfo_children():
            if isinstance(child, ttk.Button):
                child.configure(style="Action.TButton")
                child.bind("<Enter>", lambda e, w=child: w.configure(cursor="hand2"))
                child.bind("<Leave>", lambda e, w=child: w.configure(cursor=""))
            if child.winfo_children(): polish_buttons(child)
    style.configure("Action.TButton", padding=(10, 7), font=("Segoe UI", 9, "bold"))
    polish_buttons(root)
    pulse_state = {"on": False}
    def pulse_footer():
        if not root.winfo_exists(): return
        pulse_state["on"] = not pulse_state["on"]
        footer.configure(text=(
            "● SYSTEM ONLINE  |  Criminal Intelligence System  |  SQLite | InsightFace / ArcFace"
            if pulse_state["on"] else
            "○ SYSTEM ONLINE  |  Criminal Intelligence System  |  SQLite | InsightFace / ArcFace"))
        root.after(900, pulse_footer)
    root.after(900, pulse_footer)

    # =====================================================
    # INITIAL STATISTICS
    # =====================================================

    load_statistics()

    # =====================================================
    # CLOSE
    # =====================================================

    root.protocol(
        "WM_DELETE_WINDOW",
        root.destroy
    )


# =========================================================
# MAIN
# =========================================================

# =========================================================
# FINAL UI/EDIT ENHANCEMENT — GRAPH / SCAN / DATA WINDOWS
# =========================================================

def _ui_table_window(parent, title, columns, rows, edit_callback=None, delete_callback=None):
    win = tk.Toplevel(parent)
    win.title(title)
    win.geometry("1180x700")
    win.minsize(900, 520)
    win.transient(parent)

    outer = ttk.Frame(win, padding=12)
    outer.pack(fill="both", expand=True)

    top = ttk.Frame(outer)
    top.pack(fill="x", pady=(0, 8))
    ttk.Label(top, text=title, font=("Segoe UI", 17, "bold")).pack(side="left")

    search_var = tk.StringVar()
    ttk.Label(top, text="Search:").pack(side="left", padx=(30, 5))
    search = ttk.Entry(top, textvariable=search_var, width=32)
    search.pack(side="left")

    tree_frame = ttk.Frame(outer)
    tree_frame.pack(fill="both", expand=True)

    tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")
    ybar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    xbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
    tree.grid(row=0, column=0, sticky="nsew")
    ybar.grid(row=0, column=1, sticky="ns")
    xbar.grid(row=1, column=0, sticky="ew")
    tree_frame.rowconfigure(0, weight=1)
    tree_frame.columnconfigure(0, weight=1)

    for c in columns:
        tree.heading(c, text=c.replace("_", " ").title())
        tree.column(c, width=max(120, min(260, len(c) * 12)), anchor="w")

    data = list(rows)

    def refill(*_):
        needle = search_var.get().strip().lower()
        tree.delete(*tree.get_children())
        for idx, row in enumerate(data):
            values = tuple("" if v is None else str(v) for v in row)
            if not needle or needle in " | ".join(values).lower():
                tree.insert("", "end", iid=str(idx), values=values)

    refill()
    search_var.trace_add("write", refill)

    buttons = ttk.Frame(outer)
    buttons.pack(fill="x", pady=(8, 0))

    def selected_row():
        sel = tree.selection()
        if not sel:
            messagebox.showwarning("Selection", "Please select a row first.", parent=win)
            return None
        return data[int(sel[0])]

    if edit_callback:
        ttk.Button(buttons, text="✏ EDIT", command=lambda: edit_callback(selected_row(), win, refill)).pack(side="left", padx=4)
    if delete_callback:
        ttk.Button(buttons, text="🗑 DELETE", command=lambda: delete_callback(selected_row(), win, refill)).pack(side="left", padx=4)
    ttk.Button(buttons, text="REFRESH", command=refill).pack(side="right", padx=4)
    ttk.Button(buttons, text="CLOSE", command=win.destroy).pack(side="right", padx=4)
    return win


def _edit_person_row(row, parent, refresh):
    """Edit a PERSON record in SQLite and migrate stable-ID references safely."""
    if not row:
        return
    old_person_id = str(row[0]).strip().upper()
    vals = list(row[1:6])

    win = tk.Toplevel(parent)
    win.title(f"Edit Person — {old_person_id}")
    win.geometry("640x520")
    win.minsize(580, 470)
    win.transient(parent)
    win.grab_set()

    frm = ttk.Frame(win, padding=18)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(1, weight=1)

    ttk.Label(frm, text=f"EDIT PERSON / ENTITY  •  {old_person_id}",
              font=("Segoe UI", 15, "bold")).grid(row=0, column=0, columnspan=2,
                                                   sticky="w", pady=(0, 14))

    labels = ["Person ID", "Name", "Phone", "Vehicle", "Address / Location", "Organization"]
    initial = [old_person_id] + vals
    vars_ = []
    for i, (lab, val) in enumerate(zip(labels, initial), start=1):
        ttk.Label(frm, text=lab, font=("Segoe UI", 10, "bold")).grid(
            row=i, column=0, sticky="w", pady=7, padx=(0, 12))
        v = tk.StringVar(value="" if val is None else str(val))
        ttk.Entry(frm, textvariable=v, width=48).grid(
            row=i, column=1, sticky="ew", pady=7)
        vars_.append(v)

    status = tk.StringVar(value="Person ID must match P001 format.")
    ttk.Label(frm, textvariable=status).grid(
        row=7, column=0, columnspan=2, sticky="w", pady=(10, 6))

    def save():
        new_id, name, phone, vehicle, address, organization = (v.get().strip() for v in vars_)
        new_id = new_id.upper()
        if not validate_person_id(new_id):
            messagebox.showerror("Invalid Person ID", "Person ID must use format P001.", parent=win)
            return
        if not name:
            messagebox.showerror("Invalid Data", "Name is required.", parent=win)
            return
        if not validate_phone(phone):
            messagebox.showerror("Invalid Phone", "Phone must contain exactly 10 digits.", parent=win)
            return
        if not validate_vehicle(vehicle):
            messagebox.showerror("Invalid Vehicle", "Use vehicle format such as BR01AB1234.", parent=win)
            return

        conn = None
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM persons WHERE UPPER(person_id)=UPPER(?) AND UPPER(person_id)<>UPPER(?)",
                        (new_id, old_person_id))
            if cur.fetchone():
                raise ValueError(f"Person ID {new_id} already exists.")

            cur.execute("""
                UPDATE persons
                SET person_id=?, name=?, phone=?, vehicle=?, address=?, organization=?
                WHERE UPPER(person_id)=UPPER(?)
            """, (new_id, name, phone, vehicle, address, organization, old_person_id))
            if cur.rowcount != 1:
                raise ValueError(f"Person {old_person_id} was not found.")

            # Preserve all historical graph records while moving PERSON references
            # when the stable identifier itself is changed.
            if new_id != old_person_id:
                cur.execute("UPDATE person_cases SET person_id=? WHERE UPPER(person_id)=UPPER(?)",
                            (new_id, old_person_id))
                cur.execute("""
                    UPDATE relationships SET source_id=?
                    WHERE source_type='PERSON' AND UPPER(source_id)=UPPER(?)
                """, (new_id, old_person_id))
                cur.execute("""
                    UPDATE relationships SET target_id=?
                    WHERE target_type='PERSON' AND UPPER(target_id)=UPPER(?)
                """, (new_id, old_person_id))

            # Add/synchronize current entity links; old links remain as historical records.
            sync_person_entities(cur, new_id, phone, vehicle, address, organization)
            conn.commit()
            conn.close()
            conn = None

            # Face embeddings are filename-keyed in the existing pipeline. Keep that
            # pipeline working when the display name changes.
            old_safe = safe_filename(str(row[1] or ""))
            new_safe = safe_filename(name)
            if old_safe and new_safe and old_safe != new_safe:
                old_emb = os.path.join(FACES_FOLDER, old_safe + ".npy")
                new_emb = os.path.join(FACES_FOLDER, new_safe + ".npy")
                if os.path.exists(old_emb) and not os.path.exists(new_emb):
                    try: os.replace(old_emb, new_emb)
                    except OSError as exc: print("Embedding rename warning:", exc)
                old_photo = os.path.join(PHOTOS_FOLDER, old_safe)
                new_photo = os.path.join(PHOTOS_FOLDER, new_safe)
                if os.path.isdir(old_photo) and not os.path.exists(new_photo):
                    try: os.replace(old_photo, new_photo)
                    except OSError as exc: print("Photo folder rename warning:", exc)

            try:
                run_relationship_engine_for_person(new_id)
            except Exception as rel_exc:
                print("Relationship refresh after edit:", rel_exc)

            messagebox.showinfo(
                "Saved",
                f"{new_id} was updated in SQLite.\n\n"
                "Historical relationships were preserved; current entity links "
                "were synchronized.",
                parent=win
            )
            win.destroy()
            refresh()
            try: load_statistics()
            except Exception: pass
        except sqlite3.IntegrityError as exc:
            if conn:
                conn.rollback(); conn.close()
            messagebox.showerror("Database Constraint", str(exc), parent=win)
        except Exception as exc:
            if conn:
                conn.rollback(); conn.close()
            messagebox.showerror("Update Error", str(exc), parent=win)

    btns = ttk.Frame(frm)
    btns.grid(row=8, column=0, columnspan=2, sticky="e", pady=(14, 0))
    ttk.Button(btns, text="SAVE CHANGES", command=save).pack(side="right", padx=4)
    ttk.Button(btns, text="CANCEL", command=win.destroy).pack(side="right", padx=4)
    win.bind("<Escape>", lambda e: win.destroy())

def open_entity_data_manager():
    """Cleaner data window with edit support for persons/entities."""
    if not database_exists():
        return
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT person_id, name, phone, vehicle, address, organization FROM persons ORDER BY person_id")
        rows = cur.fetchall()
    finally:
        conn.close()
    _ui_table_window(
        root,
        "ENTITY DATA MANAGER",
        ("person_id", "name", "phone", "vehicle", "address", "organization"),
        rows,
        edit_callback=_edit_person_row
    )


def _edit_relationship_row(row, parent, refresh):
    if not row:
        return
    # relationship_id, source_type, source_id, relation_type, target_type, target_id
    rid, st, sid, rel, tt, tid = row
    win = tk.Toplevel(parent)
    win.title(f"Edit Relationship — #{rid}")
    win.geometry("620x420")
    win.transient(parent)

    frm = ttk.Frame(win, padding=18)
    frm.pack(fill="both", expand=True)

    fields = [
        ("Source Type", st), ("Source ID", sid), ("Relationship", rel),
        ("Target Type", tt), ("Target ID", tid)
    ]
    vars_ = []
    for i, (lab, val) in enumerate(fields):
        ttk.Label(frm, text=lab).grid(row=i, column=0, sticky="w", pady=8)
        v = tk.StringVar(value=str(val))
        ttk.Entry(frm, textvariable=v, width=45).grid(row=i, column=1, sticky="ew", pady=8)
        vars_.append(v)
    frm.columnconfigure(1, weight=1)

    def save():
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("""
                UPDATE relationships
                SET source_type=?, source_id=?, relation_type=?, target_type=?, target_id=?
                WHERE relationship_id=?
            """, tuple(v.get().strip().upper() for v in vars_) + (rid,))
            conn.commit()
            conn.close()
            messagebox.showinfo("Saved", "Relationship updated successfully.", parent=win)
            win.destroy()
            refresh()
        except Exception as exc:
            try:
                conn.close()
            except Exception:
                pass
            messagebox.showerror("Update Error", str(exc), parent=win)

    ttk.Button(frm, text="SAVE CHANGES", command=save).grid(row=6, column=1, sticky="e", pady=18)


def _delete_relationship_row(row, parent, refresh):
    if not row:
        return
    rid = row[0]
    if not messagebox.askyesno("Delete Relationship",
                               f"Delete relationship #{rid}?", parent=parent):
        return
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM relationships WHERE relationship_id=?", (rid,))
        conn.commit()
        conn.close()
        refresh()
    except Exception as exc:
        messagebox.showerror("Delete Error", str(exc), parent=parent)


def open_relationship_data_manager():
    """Dedicated relationship table with EDIT/DELETE controls."""
    if not database_exists():
        return
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT relationship_id, source_type, source_id,
                   relation_type, target_type, target_id
            FROM relationships
            ORDER BY relationship_id DESC
        """)
        rows = cur.fetchall()
    finally:
        conn.close()
    _ui_table_window(
        root,
        "RELATIONSHIP DATA MANAGER",
        ("relationship_id", "source_type", "source_id", "relation_type", "target_type", "target_id"),
        rows,
        edit_callback=_edit_relationship_row,
        delete_callback=_delete_relationship_row
    )


def open_scan_data_window():
    """Cleaner scan/recognition records window using available person data."""
    if not database_exists():
        return
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT person_id, name, phone, vehicle, address, organization
            FROM persons
            ORDER BY name COLLATE NOCASE
        """)
        rows = cur.fetchall()
    finally:
        conn.close()
    _ui_table_window(
        root,
        "SCAN / RECOGNITION DATA",
        ("person_id", "name", "phone", "vehicle", "address", "organization"),
        rows,
        edit_callback=_edit_person_row
    )


def open_clean_graph_window(person_id=None):
    """Clean graph launcher that reuses the existing graph engine."""
    if not person_id:
        text = search_entry.get().strip() if search_entry else ""
        person = get_person_from_search(text) if text else None
        person_id = person["person_id"] if person else None
    if not person_id:
        messagebox.showwarning("Graph", "Search/select a person first.", parent=root)
        return
    # Existing graph engine already supports depth and graph rendering.
    open_network_graph(str(person_id).strip().upper())


def add_ui_enhancement_buttons():
    """Add compact managers without replacing existing controls."""
    if not root or not root.winfo_exists():
        return
    # Avoid duplicate installation.
    if getattr(root, "_final_ui_buttons_added", False):
        return

    panel = ttk.LabelFrame(root, text="DATA & INVESTIGATION TOOLS", padding=8)
    panel.pack(fill="x", padx=12, pady=(0, 8))
    ttk.Button(panel, text="🕸 GRAPH", command=open_clean_graph_window).pack(side="left", padx=4)
    ttk.Button(panel, text="📷 SCAN DATA", command=open_scan_data_window).pack(side="left", padx=4)
    ttk.Button(panel, text="👤 EDIT ENTITIES", command=open_entity_data_manager).pack(side="left", padx=4)
    ttk.Button(panel, text="🔗 EDIT RELATIONSHIPS", command=open_relationship_data_manager).pack(side="left", padx=4)

    # Lightweight pulse animation; no extra package required.
    pulse = ttk.Label(panel, text="● SYSTEM READY")
    pulse.pack(side="right", padx=8)
    state = {"on": False}
    def animate():
        if not panel.winfo_exists():
            return
        state["on"] = not state["on"]
        pulse.configure(text="● LIVE SYSTEM" if state["on"] else "○ LIVE SYSTEM")
        panel.after(700, animate)
    animate()
    root._final_ui_buttons_added = True


# =========================================================
# .x — COMMAND CENTER EXTENSIONS
# =========================================================

def _safe_table_exists(cur, table_name):
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None


def get_database_counts():
    """Return live counts from SQLite. Missing optional tables are reported as zero."""
    tables = ("persons", "cases", "phones", "vehicles", "locations",
              "organizations", "relationships")
    conn = get_connection()
    cur = conn.cursor()
    counts = {}
    try:
        for table in tables:
            if _safe_table_exists(cur, table):
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = int(cur.fetchone()[0])
            else:
                counts[table] = 0
    finally:
        conn.close()
    return counts


def get_network_statistics():
    counts = get_database_counts()
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT source_type, source_id, relation_type, target_type, target_id
            FROM relationships ORDER BY relationship_id
        """)
        rows = cur.fetchall()
    finally:
        conn.close()

    nodes = set()
    degree = {}
    rel_counts = {}
    for st, sid, rel, tt, tid in rows:
        a=(str(st).upper(), str(sid)); b=(str(tt).upper(), str(tid))
        nodes.update((a,b))
        degree[a] = degree.get(a,0)+1
        degree[b] = degree.get(b,0)+1
        rel = str(rel).upper()
        rel_counts[rel] = rel_counts.get(rel,0)+1

    connected = sum(1 for n,d in degree.items() if d > 0)
    components = get_step46_components()[2] if rows else []
    return {
        "counts": counts, "nodes": len(nodes), "connected": connected,
        "components": len(components), "relationships_by_type": rel_counts,
        "max_degree": max(degree.values(), default=0),
        "top_nodes": sorted(degree.items(), key=lambda x:(-x[1],x[0][0],x[0][1]))[:10],
    }


def calculate_investigation_priority(person_id):
    """Explainable triage indicator; never a criminality/guilt classifier."""
    pid = str(person_id).strip().upper()
    direct = get_person_relationships(pid)
    total_score, _, breakdown = calculate_person_evidence_score(pid)

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM person_cases WHERE UPPER(person_id)=?", (pid,))
        cases = int(cur.fetchone()[0])
        cur.execute("""
            SELECT COUNT(DISTINCT r.target_id)
            FROM relationships r
            WHERE r.source_type='PERSON' AND UPPER(r.source_id)=?
        """, (pid,))
        linked = int(cur.fetchone()[0])
    finally:
        conn.close()

    nodes, edges, _, _ = get_person_network_levels(pid, 3)
    cross_case = 0
    for row in direct:
        if str(row[3]).upper() == "COMMON_FIR":
            cross_case += 1

    # Conservative, explainable thresholds based only on graph connectivity.
    indicator = direct.__len__() + cases + linked + cross_case
    if indicator >= 12 or total_score >= 20:
        level = "HIGH"
    elif indicator >= 6 or total_score >= 8:
        level = "MEDIUM"
    else:
        level = "LOW"

    reasons = [
        f"{len(direct)} direct relationships",
        f"{cases} connected cases",
        f"{linked} directly linked entities",
        f"{len(nodes)} reachable network nodes (depth 3)",
    ]
    if cross_case:
        reasons.append(f"{cross_case} cross-case relationship(s)")
    if breakdown:
        strongest = max(breakdown.items(), key=lambda x:x[1])
        reasons.append(f"highest weighted relationship type: {strongest[0]} ({strongest[1]})")

    return {"level": level, "reasons": reasons, "score": total_score,
            "direct": len(direct), "cases": cases, "linked": linked,
            "nodes": len(nodes), "edges": len(edges)}


def _resolve_entity(entity_type, entity_id):
    et = str(entity_type).upper()
    eid = str(entity_id)
    conn = get_connection()
    cur = conn.cursor()
    try:
        if et == "PERSON":
            cur.execute("SELECT person_id, name FROM persons WHERE UPPER(person_id)=UPPER(?)", (eid,))
            row=cur.fetchone()
            return row[0], row[1] if row else eid
        if et == "FIR":
            cur.execute("SELECT case_id, case_type, date, location FROM cases WHERE case_id=?", (eid,))
            row=cur.fetchone()
            return (row[0], f"{row[0]} — {row[1] or '--'}") if row else (eid,eid)
        table_map = {
            "PHONE": ("phones","phone_id","phone_number"),
            "VEHICLE": ("vehicles","vehicle_id","registration_number"),
            "LOCATION": ("locations","location_id","location_name"),
            "ORGANIZATION": ("organizations","organization_id","organization_name"),
        }
        if et in table_map:
            table, key, label = table_map[et]
            cur.execute(f"SELECT {key}, {label} FROM {table} WHERE {key}=?", (eid,))
            row=cur.fetchone()
            return (str(row[0]), str(row[1])) if row else (eid,eid)
    finally:
        conn.close()
    return eid, eid


def get_entity_timeline(entity_type, entity_id, max_depth=0):
    """Generic timeline for PERSON/FIR/PHONE/VEHICLE/LOCATION/ORGANIZATION."""
    et=str(entity_type).upper(); eid=str(entity_id)
    if et=="PERSON":
        return get_step47_timeline(eid, max_depth or 5)

    root=(et,eid)
    nodes={root}; edges=[]
    if max_depth > 0:
        try:
            # Generic BFS over the same relationships table.
            conn=get_connection(); cur=conn.cursor()
            cur.execute("SELECT source_type,source_id,relation_type,target_type,target_id,created_at FROM relationships ORDER BY relationship_id")
            rows=cur.fetchall(); conn.close()
            adj={}
            for st,sid,rel,tt,tid,created_at in rows:
                a=(str(st).upper(),str(sid)); b=(str(tt).upper(),str(tid))
                adj.setdefault(a,[]).append((b,(st,sid,rel,tt,tid,created_at)))
                adj.setdefault(b,[]).append((a,(st,sid,rel,tt,tid,created_at)))
            depth={root:0}; q=[root]
            while q:
                cur_node=q.pop(0)
                if depth[cur_node]>=max_depth: continue
                for nxt,edge in adj.get(cur_node,[]):
                    if nxt not in depth:
                        depth[nxt]=depth[cur_node]+1; q.append(nxt)
                    nodes.add(nxt); edges.append((edge,depth.get(cur_node,0)))
        except Exception:
            pass

    events=[]
    conn=get_connection(); cur=conn.cursor()
    try:
        # Direct FIR/case events for FIR and entities linked to FIR.
        if et=="FIR":
            cur.execute("SELECT case_id,case_type,location,date FROM cases WHERE case_id=?", (eid,))
            for case_id,case_type,location,date in cur.fetchall():
                events.append({"date":str(date or ""), "event":"FIR / CASE",
                               "subject":case_id,
                               "detail":f"{case_type or '--'} | {location or '--'}",
                               "level":0, "source":"FIR"})
        cur.execute("""
            SELECT relationship_id,source_type,source_id,relation_type,target_type,target_id,created_at
            FROM relationships ORDER BY created_at,relationship_id
        """)
        node_set=set(nodes)
        for rid,st,sid,rel,tt,tid,created_at in cur.fetchall():
            a=(str(st).upper(),str(sid)); b=(str(tt).upper(),str(tid))
            if a not in node_set and b not in node_set: continue
            events.append({"date":str(created_at or ""), "event":"RELATIONSHIP",
                           "subject":f"{a[0]}:{a[1]}",
                           "detail":f"{rel} → {b[0]}:{b[1]}" if a in node_set else f"{rel} ← {a[0]}:{a[1]}",
                           "level":0, "source":"RELATIONSHIP",
                           "record_id":rid})
    finally:
        conn.close()

    events.sort(key=lambda x:(str(x.get("date","")), str(x.get("event",""))))
    return events


def _show_entity_timeline(entity_type, entity_id, parent=None):
    et=str(entity_type).upper(); eid=str(entity_id)
    label=get_entity_label(et,eid)
    popup=tk.Toplevel(parent or root); popup.title(f"Investigation Timeline — {label}")
    popup.geometry("1250x760"); popup.minsize(980,600); popup.transient(parent or root)
    main=ttk.Frame(popup,padding=14); main.pack(fill="both",expand=True)
    ttk.Label(main,text="INVESTIGATION TIMELINE",font=("Segoe UI",18,"bold")).pack(anchor="w")
    ttk.Label(main,text=f"{et}: {label}  |  Only database-supported events are shown.").pack(anchor="w",pady=(2,10))

    controls=ttk.Frame(main); controls.pack(fill="x",pady=(0,8))
    order=tk.StringVar(value="NEWEST FIRST")
    ttk.Label(controls,text="Order").pack(side="left")
    ttk.Combobox(controls,textvariable=order,values=("NEWEST FIRST","OLDEST FIRST"),state="readonly",width=16).pack(side="left",padx=6)
    search=tk.StringVar(); ttk.Label(controls,text="Search").pack(side="left",padx=(18,4))
    ttk.Entry(controls,textvariable=search,width=32).pack(side="left")
    count=tk.StringVar(value="0"); ttk.Label(controls,text="Events:").pack(side="right")
    ttk.Label(controls,textvariable=count,font=("Segoe UI",10,"bold")).pack(side="right",padx=5)

    frame=ttk.Frame(main); frame.pack(fill="both",expand=True)
    cols=("date","event","description","level","source")
    tree=ttk.Treeview(frame,columns=cols,show="headings")
    for c,t,w in [("date","DATE / TIME",170),("event","EVENT TYPE",130),("description","EVENT DESCRIPTION",650),("level","LEVEL",70),("source","SOURCE RECORD",180)]:
        tree.heading(c,text=t); tree.column(c,width=w,anchor="w")
    y=ttk.Scrollbar(frame,orient="vertical",command=tree.yview); x=ttk.Scrollbar(frame,orient="horizontal",command=tree.xview)
    tree.configure(yscrollcommand=y.set,xscrollcommand=x.set)
    tree.grid(row=0,column=0,sticky="nsew"); y.grid(row=0,column=1,sticky="ns"); x.grid(row=1,column=0,sticky="ew")
    frame.rowconfigure(0,weight=1); frame.columnconfigure(0,weight=1)

    def refresh():
        rows=get_entity_timeline(et,eid,2 if et!="PERSON" else 5)
        needle=search.get().strip().lower()
        if needle:
            rows=[r for r in rows if needle in " ".join(str(r.get(k,"")) for k in r).lower()]
        rows=sorted(rows,key=lambda r:str(r.get("date","")),reverse=(order.get()=="NEWEST FIRST"))
        for i in tree.get_children(): tree.delete(i)
        for r in rows:
            tree.insert("", "end", values=(r.get("date") or "--",r.get("event"),
                        r.get("detail",""),f"L{r.get('level',0)}",r.get("source","--")))
        count.set(str(len(rows)))
    search.trace_add("write",lambda *_:refresh()); order.trace_add("write",lambda *_:refresh())
    foot=ttk.Frame(main,padding=(0,8,0,0)); foot.pack(fill="x")
    ttk.Button(foot,text="REFRESH",command=refresh).pack(side="right",padx=4)
    ttk.Button(foot,text="CLOSE",command=popup.destroy).pack(side="right",padx=4)
    refresh()


def _global_search_rows(query="", entity_type="ALL", case_id="", relation_type="ALL",
                        date_from="", date_to=""):
    q=str(query).strip().lower()
    et=str(entity_type).upper()
    results=[]
    conn=get_connection(); cur=conn.cursor()
    try:
        if et in ("ALL","PERSON"):
            cur.execute("SELECT person_id,name,phone,vehicle,address,organization FROM persons ORDER BY name COLLATE NOCASE")
            for pid,name,phone,vehicle,address,org in cur.fetchall():
                hay=" ".join(map(lambda x:str(x or ""),[pid,name,phone,vehicle,address,org])).lower()
                if q and q not in hay: continue
                results.append(("PERSON",str(pid),str(name or pid),str(phone or ""),str(address or ""),str(org or "")))
        table_map={"PHONE":("phones","phone_id","phone_number"),"VEHICLE":("vehicles","vehicle_id","registration_number"),
                   "LOCATION":("locations","location_id","location_name"),"ORGANIZATION":("organizations","organization_id","organization_name")}
        for typ,(table,key,label) in table_map.items():
            if et not in ("ALL",typ): continue
            cur.execute(f"SELECT {key},{label} FROM {table} ORDER BY {label} COLLATE NOCASE")
            for eid,name in cur.fetchall():
                if q and q not in f"{eid} {name}".lower(): continue
                results.append((typ,str(eid),str(name),"","",""))
        if et in ("ALL","FIR"):
            cur.execute("SELECT case_id,case_type,location,date FROM cases ORDER BY date DESC")
            for cid,ctype,loc,date in cur.fetchall():
                hay=f"{cid} {ctype or ''} {loc or ''} {date or ''}".lower()
                if q and q not in hay: continue
                d=parse_date(str(date or "")); 
                if date_from and (not d or d < parse_date(date_from)): continue
                if date_to and (not d or d > parse_date(date_to)): continue
                results.append(("FIR",str(cid),f"{cid} — {ctype or '--'}",str(date or ""),str(loc or ""),""))
        # Relationship/case filters are applied after entity discovery.
        if case_id or (relation_type and relation_type!="ALL"):
            allowed=set()
            cur.execute("SELECT person_id,case_id FROM person_cases")
            person_case={(str(p).upper(),str(c).upper()) for p,c in cur.fetchall()}
            cur.execute("SELECT source_type,source_id,relation_type,target_type,target_id FROM relationships")
            rels=cur.fetchall()
            for typ,eid,*_ in results:
                ok=True
                if case_id:
                    target=str(case_id).upper()
                    if typ=="PERSON":
                        ok=any(p==eid.upper() and c==target for p,c in person_case)
                    elif typ=="FIR":
                        ok=eid.upper()==target
                    else:
                        ok=False
                if ok and relation_type and relation_type!="ALL":
                    ok=any((str(st).upper()==typ and str(sid)==eid and str(rel).upper()==relation_type) or
                           (str(tt).upper()==typ and str(tid)==eid and str(rel).upper()==relation_type)
                           for st,sid,rel,tt,tid in rels)
                if ok: allowed.add((typ,eid))
            results=[r for r in results if (r[0],r[1]) in allowed]
    finally:
        conn.close()
    return results


def open_global_investigation_search():
    if not database_exists(): return
    popup=tk.Toplevel(root); popup.title("Advanced Investigation Search")
    popup.geometry("1320x780"); popup.minsize(1050,620); popup.transient(root)
    main=ttk.Frame(popup,padding=14); main.pack(fill="both",expand=True)
    ttk.Label(main,text="ADVANCED INVESTIGATION SEARCH",font=("Segoe UI",18,"bold")).pack(anchor="w")
    ttk.Label(main,text="Searches actual SQLite entity, FIR and relationship records.").pack(anchor="w",pady=(2,10))
    controls=ttk.Frame(main); controls.pack(fill="x",pady=(0,8))
    q=tk.StringVar(); ttk.Label(controls,text="Query").pack(side="left"); ttk.Entry(controls,textvariable=q,width=30).pack(side="left",padx=5)
    typ=tk.StringVar(value="ALL"); ttk.Label(controls,text="Entity type").pack(side="left",padx=(15,4))
    ttk.Combobox(controls,textvariable=typ,values=("ALL","PERSON","FIR","PHONE","VEHICLE","LOCATION","ORGANIZATION"),state="readonly",width=16).pack(side="left")
    case=tk.StringVar(); ttk.Label(controls,text="FIR / Case").pack(side="left",padx=(15,4)); ttk.Entry(controls,textvariable=case,width=14).pack(side="left")
    rel=tk.StringVar(value="ALL"); ttk.Label(controls,text="Relationship").pack(side="left",padx=(15,4))
    ttk.Combobox(controls,textvariable=rel,values=("ALL","USES","OWNS","LIVES_AT","MEMBER_OF","OCCURRED_AT","INVOLVED_IN","SHARED_PHONE","SHARED_VEHICLE","SHARED_LOCATION","SHARED_ORGANIZATION","COMMON_FIR","ASSOCIATED_WITH","COMMUNICATES_WITH","PARTNER_OF","ASSOCIATE_OF","FAMILY_OF"),state="readonly",width=20).pack(side="left")
    df=tk.StringVar(); dt=tk.StringVar()
    ttk.Label(controls,text="From").pack(side="left",padx=(15,4)); ttk.Entry(controls,textvariable=df,width=11).pack(side="left")
    ttk.Label(controls,text="To").pack(side="left",padx=(8,4)); ttk.Entry(controls,textvariable=dt,width=11).pack(side="left")
    frame=ttk.Frame(main); frame.pack(fill="both",expand=True,pady=(5,0))
    cols=("type","id","label","detail1","detail2","detail3")
    tree=ttk.Treeview(frame,columns=cols,show="headings")
    for c,t,w in [("type","TYPE",100),("id","ID",130),("label","NAME / LABEL",330),("detail1","PHONE / DATE",150),("detail2","LOCATION / VEHICLE",230),("detail3","ORGANIZATION",230)]:
        tree.heading(c,text=t); tree.column(c,width=w,anchor="w")
    y=ttk.Scrollbar(frame,orient="vertical",command=tree.yview); tree.configure(yscrollcommand=y.set)
    tree.grid(row=0,column=0,sticky="nsew"); y.grid(row=0,column=1,sticky="ns"); frame.rowconfigure(0,weight=1); frame.columnconfigure(0,weight=1)
    status=tk.StringVar(value="0 results")
    def refresh():
        for i in tree.get_children(): tree.delete(i)
        try:
            if df.get().strip() and not validate_date(df.get()): raise ValueError("From date must be DD-MM-YYYY or YYYY-MM-DD.")
            if dt.get().strip() and not validate_date(dt.get()): raise ValueError("To date must be DD-MM-YYYY or YYYY-MM-DD.")
            rows=_global_search_rows(q.get(),typ.get(),case.get(),rel.get(),df.get(),dt.get())
            for r in rows: tree.insert("", "end", values=r)
            status.set(f"{len(rows)} results")
        except Exception as exc: status.set(str(exc))
    for v in (q,typ,case,rel,df,dt): v.trace_add("write",lambda *_:refresh())
    def open_selected(_=None):
        sel=tree.selection()
        if not sel:return
        vals=tree.item(sel[0],"values"); et,eid=vals[0],vals[1]
        if et=="PERSON":
            show_person_data(get_person_by_id(eid))
        elif et=="FIR":
            open_case_workspace(eid)
        else:
            _show_entity_timeline(et,eid,popup)
    tree.bind("<Double-1>",open_selected)
    foot=ttk.Frame(main,padding=(0,8,0,0)); foot.pack(fill="x")
    ttk.Label(foot,textvariable=status).pack(side="left")
    ttk.Button(foot,text="OPEN SELECTED",command=open_selected).pack(side="right",padx=4)
    ttk.Button(foot,text="REFRESH",command=refresh).pack(side="right",padx=4)
    ttk.Button(foot,text="CLOSE",command=popup.destroy).pack(side="right",padx=4)
    refresh()


def open_investigation_analytics():
    if not database_exists(): return
    popup=tk.Toplevel(root); popup.title("Investigation Analytics")
    popup.geometry("1280x820"); popup.minsize(1050,650); popup.transient(root)
    main=ttk.Frame(popup,padding=14); main.pack(fill="both",expand=True)
    ttk.Label(main,text="INVESTIGATION ANALYTICS",font=("Segoe UI",18,"bold")).pack(anchor="w")
    ttk.Label(main,text="Live statistics derived from SQLite and the relationship graph.").pack(anchor="w",pady=(2,10))
    cards=ttk.Frame(main); cards.pack(fill="x",pady=(0,10))
    vars_={}
    for i,(key,title) in enumerate([("persons","PERSONS"),("cases","FIRs"),("phones","PHONES"),("vehicles","VEHICLES"),
                                    ("locations","LOCATIONS"),("organizations","ORGANIZATIONS"),("relationships","RELATIONSHIPS")]):
        vars_[key]=tk.StringVar(value="0"); cards.columnconfigure(i%4,weight=1)
        box=ttk.LabelFrame(cards,text=title,padding=10); box.grid(row=i//4,column=i%4,padx=4,pady=4,sticky="nsew")
        ttk.Label(box,textvariable=vars_[key],font=("Segoe UI",16,"bold")).pack()
    body=ttk.Frame(main); body.pack(fill="both",expand=True)
    left=ttk.LabelFrame(body,text="RELATIONSHIPS BY TYPE",padding=8); left.pack(side="left",fill="both",expand=True,padx=(0,6))
    chart=tk.Canvas(left,height=300,highlightthickness=0); chart.pack(fill="both",expand=True)
    right=ttk.LabelFrame(body,text="NETWORK STATISTICS / MOST CONNECTED",padding=8); right.pack(side="right",fill="both",expand=True,padx=(6,0))
    txt=tk.Text(right,wrap="word",state="disabled",font=("Consolas",10)); txt.pack(fill="both",expand=True)
    def refresh():
        s=get_network_statistics(); c=s["counts"]
        for k in vars_: vars_[k].set(str(c.get(k,0)))
        txt.config(state="normal"); txt.delete("1.0",tk.END)
        txt.insert(tk.END,f"Connected entities : {s['connected']}\n")
        txt.insert(tk.END,f"Graph nodes        : {s['nodes']}\n")
        txt.insert(tk.END,f"Connected components: {s['components']}\n")
        txt.insert(tk.END,f"Max graph degree   : {s['max_degree']}\n\n")
        txt.insert(tk.END,"MOST CONNECTED ENTITIES\n")
        for node,deg in s["top_nodes"]:
            txt.insert(tk.END,f"• {node[0]}  {get_entity_label(node[0],node[1])}  | degree {deg}\n")

        txt.insert(tk.END,"\n" + "="*54 + "\n")
        txt.insert(tk.END, get_influencer_summary(5) + "\n")
        txt.config(state="disabled")
        chart.delete("all")
        items=sorted(s["relationships_by_type"].items(),key=lambda x:-x[1])[:10]
        if not items: chart.create_text(10,20,anchor="w",text="No relationship data available."); return
        maxv=max(v for _,v in items) or 1; w=max(chart.winfo_width(),500); rowh=28
        for i,(label,val) in enumerate(items):
            y=18+i*rowh; chart.create_text(5,y,anchor="w",text=label)
            chart.create_rectangle(180,y-8,180+(w-250)*(val/maxv),y+8)
            chart.create_text(190+(w-250)*(val/maxv),y,anchor="w",text=str(val))
    foot=ttk.Frame(main,padding=(0,8,0,0)); foot.pack(fill="x")
    ttk.Button(foot,text="REFRESH",command=refresh).pack(side="right",padx=4); ttk.Button(foot,text="CLOSE",command=popup.destroy).pack(side="right",padx=4)
    popup.bind("<Configure>",lambda e: refresh() if e.widget==popup else None)
    refresh()


def open_investigation_priority(person_id=None):
    if not database_exists(): return
    if not person_id or person_id=="--":
        text=search_entry.get().strip() if search_entry else ""
        person=get_person_from_search(text) if text else None
        person_id=person["person_id"] if person else None
    if not person_id:
        messagebox.showwarning("Investigation Priority","Search/select a person first.",parent=root); return
    pid=str(person_id).strip().upper()
    data=calculate_investigation_priority(pid)
    popup=tk.Toplevel(root); popup.title(f"Investigation Priority — {pid}")
    popup.geometry("700x560"); popup.minsize(600,450); popup.transient(root)
    main=ttk.Frame(popup,padding=18); main.pack(fill="both",expand=True)
    ttk.Label(main,text="INVESTIGATION PRIORITY INDICATOR",font=("Segoe UI",18,"bold")).pack(anchor="w")
    ttk.Label(main,text=f"PERSON: {get_entity_label('PERSON',pid)}").pack(anchor="w",pady=(3,14))
    badge=ttk.LabelFrame(main,text="INDICATOR",padding=18); badge.pack(fill="x")
    ttk.Label(badge,text=data["level"],font=("Segoe UI",26,"bold")).pack()
    ttk.Label(badge,text=f"Analytical score: {data['score']}").pack()
    reasons=ttk.LabelFrame(main,text="EXPLAINABLE REASONS",padding=12); reasons.pack(fill="both",expand=True,pady=12)
    txt=tk.Text(reasons,wrap="word",state="disabled",font=("Consolas",10)); txt.pack(fill="both",expand=True)
    txt.config(state="normal")
    for r in data["reasons"]: txt.insert(tk.END,f"• {r}\n")
    txt.insert(tk.END,"\nThis is an investigation-support triage indicator only. It does not label a person criminal, guilty, or dangerous.")
    txt.config(state="disabled")
    foot=ttk.Frame(main); foot.pack(fill="x")
    ttk.Button(foot,text="OPEN INVESTIGATION PANEL",command=lambda:open_investigation_intelligence(pid)).pack(side="left")
    ttk.Button(foot,text="TIMELINE",command=lambda:_show_entity_timeline("PERSON",pid,popup)).pack(side="left",padx=5)
    ttk.Button(foot,text="CLOSE",command=popup.destroy).pack(side="right")


def open_case_workspace(case_id=None):
    if not database_exists(): return
    cid=str(case_id or "").strip().upper()
    if not cid:
        text=search_entry.get().strip() if search_entry else ""
        if text.upper().startswith("FIR"): cid=text.upper()
    conn=get_connection(); cur=conn.cursor()
    try:
        cur.execute("SELECT case_id,case_type,location,date FROM cases WHERE UPPER(case_id)=UPPER(?)",(cid,))
        case=cur.fetchone()
        if not case:
            messagebox.showwarning("Case Workspace","FIR/case not found.",parent=root); return
        cur.execute("""SELECT p.person_id,p.name,p.phone,p.vehicle,p.address,p.organization
                       FROM persons p INNER JOIN person_cases pc ON p.person_id=pc.person_id
                       WHERE pc.case_id=? ORDER BY p.name COLLATE NOCASE""",(case[0],))
        persons=cur.fetchall()
        cur.execute("""SELECT relationship_id,source_type,source_id,relation_type,target_type,target_id
                       FROM relationships
                       WHERE (source_type='FIR' AND source_id=?) OR (target_type='FIR' AND target_id=?)
                       ORDER BY relationship_id""",(case[0],case[0]))
        rels=cur.fetchall()
    finally: conn.close()

    popup=tk.Toplevel(root); popup.title(f"Case Workspace — {case[0]}")
    popup.geometry("1250x800"); popup.minsize(1000,650); popup.transient(root)
    main=ttk.Frame(popup,padding=14); main.pack(fill="both",expand=True)
    ttk.Label(main,text="CASE INVESTIGATION WORKSPACE",font=("Segoe UI",18,"bold")).pack(anchor="w")
    info=ttk.LabelFrame(main,text="FIR",padding=10); info.pack(fill="x",pady=(8,10))
    ttk.Label(info,text=f"ID: {case[0]}    TYPE: {case[1] or '--'}    LOCATION: {case[2] or '--'}    DATE: {display_date(case[3])}",font=("Segoe UI",11,"bold")).pack(anchor="w")
    body=ttk.Frame(main); body.pack(fill="both",expand=True)
    lf=ttk.LabelFrame(body,text=f"PERSONS ({len(persons)})",padding=8); lf.pack(side="left",fill="both",expand=True,padx=(0,6))
    cols=("id","name","phone","vehicle","location","organization")
    pt=ttk.Treeview(lf,columns=cols,show="headings")
    for c,t,w in [("id","ID",90),("name","NAME",180),("phone","PHONE",120),("vehicle","VEHICLE",130),("location","LOCATION",180),("organization","ORGANIZATION",180)]:
        pt.heading(c,text=t); pt.column(c,width=w,anchor="w")
    pt.pack(fill="both",expand=True)
    for r in persons: pt.insert("", "end",values=r)
    rf=ttk.LabelFrame(body,text=f"FIR RELATIONSHIPS ({len(rels)})",padding=8); rf.pack(side="right",fill="both",expand=True,padx=(6,0))
    rt=ttk.Treeview(rf,columns=("source","relation","target"),show="headings")
    for c,t,w in [("source","SOURCE",250),("relation","RELATION",180),("target","TARGET",300)]: rt.heading(c,text=t); rt.column(c,width=w,anchor="w")
    rt.pack(fill="both",expand=True)
    for rid,st,sid,rel,tt,tid in rels: rt.insert("", "end",values=(f"{st}:{sid}",rel,f"{tt}:{tid}"))
    def open_person():
        sel=pt.selection()
        if sel: show_person_data(get_person_by_id(pt.item(sel[0],"values")[0]))
    pt.bind("<Double-1>",lambda e:open_person())
    foot=ttk.Frame(main,padding=(0,8,0,0)); foot.pack(fill="x")
    ttk.Button(foot,text="VIEW FIR",command=lambda: open_new_fir(existing_case_id=case[0]) if "existing_case_id" in open_new_fir.__code__.co_varnames else messagebox.showinfo("FIR",str(case),parent=popup)).pack(side="left")
    ttk.Button(foot,text="VIEW ENTITIES",command=lambda: open_entity_relationships()).pack(side="left",padx=4)
    ttk.Button(foot,text="VIEW RELATIONSHIPS",command=lambda: open_relationship_data_manager()).pack(side="left",padx=4)
    ttk.Button(foot,text="OPEN GRAPH",command=lambda: open_network_graph(persons[0][0]) if persons else messagebox.showinfo("Graph","No linked person.",parent=popup)).pack(side="left",padx=4)
    ttk.Button(foot,text="OPEN TIMELINE",command=lambda: _show_entity_timeline("FIR",case[0],popup)).pack(side="left",padx=4)
    ttk.Button(foot,text="ANALYZE NETWORK",command=lambda: open_investigation_analytics()).pack(side="left",padx=4)
    ttk.Button(foot,text="CLOSE",command=popup.destroy).pack(side="right")


def open_evidence_traceability(person_id=None):
    """Display actual relationship records and their available source fields."""
    if not database_exists(): return
    pid=str(person_id or "").strip().upper()
    if not pid:
        text=search_entry.get().strip() if search_entry else ""; p=get_person_from_search(text) if text else None; pid=p["person_id"] if p else ""
    if not pid:
        messagebox.showwarning("Traceability","Search/select a person first.",parent=root); return
    conn=get_connection(); cur=conn.cursor()
    try:
        cur.execute("""
            SELECT relationship_id,source_type,source_id,relation_type,target_type,target_id,created_at
            FROM relationships
            WHERE (source_type='PERSON' AND UPPER(source_id)=?)
               OR (target_type='PERSON' AND UPPER(target_id)=?)
            ORDER BY created_at DESC, relationship_id DESC
        """,(pid,pid))
        rows=cur.fetchall()
        cur.execute("""SELECT pc.case_id,c.case_type,c.date,c.location
                       FROM person_cases pc INNER JOIN cases c ON c.case_id=pc.case_id
                       WHERE UPPER(pc.person_id)=? ORDER BY c.date DESC""",(pid,))
        cases=cur.fetchall()
    finally: conn.close()
    popup=tk.Toplevel(root); popup.title(f"Source Traceability — {pid}")
    popup.geometry("1250x760"); popup.minsize(1000,600); popup.transient(root)
    main=ttk.Frame(popup,padding=14); main.pack(fill="both",expand=True)
    ttk.Label(main,text="EVIDENCE / SOURCE TRACEABILITY",font=("Segoe UI",18,"bold")).pack(anchor="w")
    ttk.Label(main,text="Only source information actually present in SQLite is displayed.").pack(anchor="w",pady=(2,10))
    casebox=ttk.LabelFrame(main,text="LINKED FIR / CASE RECORDS",padding=8); casebox.pack(fill="x",pady=(0,8))
    ctree=ttk.Treeview(casebox,columns=("id","type","date","location"),show="headings",height=4)
    for c,t,w in [("id","FIR / CASE ID",130),("type","TYPE",220),("date","DATE",130),("location","LOCATION",400)]: ctree.heading(c,text=t); ctree.column(c,width=w,anchor="w")
    ctree.pack(fill="x")
    for r in cases: ctree.insert("", "end",values=(r[0],r[1] or "--",display_date(r[2]),r[3] or "--"))
    box=ttk.LabelFrame(main,text="RELATIONSHIP SOURCE RECORDS",padding=8); box.pack(fill="both",expand=True)
    tree=ttk.Treeview(box,columns=("id","source","rel","target","created"),show="headings")
    for c,t,w in [("id","RECORD ID",80),("source","SOURCE",260),("rel","RELATIONSHIP",180),("target","TARGET",300),("created","DATE / TIME",180)]: tree.heading(c,text=t); tree.column(c,width=w,anchor="w")
    tree.pack(fill="both",expand=True)
    for rid,st,sid,rel,tt,tid,created in rows:
        tree.insert("", "end",values=(rid,f"{st}:{sid}",rel,f"{tt}:{tid}",created or "--"))
    foot=ttk.Frame(main,padding=(0,8,0,0)); foot.pack(fill="x")
    ttk.Label(foot,text=f"{len(rows)} relationship records | {len(cases)} linked cases").pack(side="left")
    ttk.Button(foot,text="TIMELINE",command=lambda:_show_entity_timeline("PERSON",pid,popup)).pack(side="right",padx=4)
    ttk.Button(foot,text="CLOSE",command=popup.destroy).pack(side="right",padx=4)


def open_history_for_person(person_id=None):
    if not person_id or person_id=="--":
        text=search_entry.get().strip() if search_entry else ""
        p=get_person_from_search(text) if text else None
        person_id=p["person_id"] if p else None
    if person_id: open_investigation_intelligence(person_id)


def install_command_center_controls():
    """Add new controls without replacing the original GUI."""
    if not root or getattr(root,"_advanced_controls_added",False): return
    host = None
    for w in root.winfo_children():
        if isinstance(w, ttk.Frame):
            # create_gui's main frame is the first large frame; using the last
            # packed frame is safer for this existing architecture.
            host=w
            break
    if host is None: return
    bar=ttk.LabelFrame(host,text="INVESTIGATION COMMAND CENTER",padding=8)
    bar.pack(fill="x",pady=(0,8),before=host.winfo_children()[-1] if host.winfo_children() else None)
    buttons=[
        ("⌕ GLOBAL SEARCH",open_global_investigation_search),
        ("📊 ANALYTICS",open_investigation_analytics),
        ("⚠ PRIORITY",lambda:open_investigation_priority(person_id_value.get())),
        ("🕒 TIMELINE",lambda:_show_entity_timeline("PERSON",person_id_value.get()) if person_id_value.get()!="--" else open_timeline_intelligence()),
        ("🔎 TRACEABILITY",lambda:open_evidence_traceability(person_id_value.get())),
        ("📁 CASE WORKSPACE",lambda:open_case_workspace(fir_card_id_value.get()) if fir_card_id_value.get()!="--" else messagebox.showwarning("Case Workspace","Select a person with a linked FIR first.",parent=root)),
    ]
    for text,cmd in buttons: ttk.Button(bar,text=text,command=cmd).pack(side="left",padx=3)
    clock=tk.StringVar(value="--:--:--")
    status=tk.StringVar(value="SYSTEM: ONLINE  |  DATABASE: CONNECTED  |  FACE ENGINE: READY  |  GRAPH ENGINE: READY")
    right=ttk.Frame(bar); right.pack(side="right",fill="x",expand=True)
    ttk.Label(right,textvariable=status,font=("Segoe UI",8,"bold")).pack(side="right",padx=8)
    ttk.Label(right,textvariable=clock,font=("Consolas",10,"bold")).pack(side="right")
    def tick():
        if not root.winfo_exists(): return
        clock.set(datetime.now().strftime("%H:%M:%S  %d-%m-%Y"))
        try:
            get_connection().close(); db_ok=True
        except Exception: db_ok=False
        face_state="READY" if face_app is not None else "AVAILABLE"
        status.set(f"SYSTEM: ONLINE  |  DATABASE: {'CONNECTED' if db_ok else 'ERROR'}  |  FACE ENGINE: {face_state}  |  GRAPH ENGINE: READY")
        root.after(1000,tick)
    root.after(0,tick)
    root._advanced_controls_added=True


def main():

    print(
        "=" * 70
    )

    print(
        "CRIMINAL INTELLIGENCE DASHBOARD"
    )

    print(
        "=" * 70
    )

    print(
        "Database:",
        DATABASE_PATH
    )

    try:

        ensure_database_schema()

        print(
            "Database ready."
        )

        # : build automatic relationships from existing data.
        try:
            startup_relationship_result = run_relationship_engine()
            print_relationship_engine_result(
                startup_relationship_result
            )
        except Exception as relationship_exc:
            print(
                "Relationship engine initialization error:",
                relationship_exc
            )

    except Exception as exc:

        print(
            "Database initialization error:",
            exc
        )

    print(
        "=" * 70
    )

    create_gui()

    try:
        add_ui_enhancement_buttons()
        install_command_center_controls()
    except Exception as ui_exc:
        print("UI enhancement initialization error:", ui_exc)

    root.mainloop()


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()
