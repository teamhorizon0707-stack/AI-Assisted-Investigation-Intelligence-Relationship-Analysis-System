# Criminal Intelligence and Investigation Command Center

> AI-assisted investigation support platform for organizing case records and supporting investigative analysis through NLP-based entity extraction, face recognition, relationship/network analysis, anomaly detection, intelligence alerts, and report generation.

## 📌 Project Overview

The **Criminal Intelligence and Investigation Command Center** is a desktop-based investigative intelligence prototype designed to help investigators organize criminal case information and identify meaningful relationships, patterns, and anomalies across investigation data.

Traditional investigation workflows may require investigators to manually examine person records, FIR/case details, phone numbers, vehicles, addresses, locations, organizations, reports, and photographs. When this information is distributed across different records, identifying hidden relationships and unusual patterns can become difficult and time-consuming.

This project provides a unified investigation interface where structured and unstructured information can be analyzed using:

- Natural Language Processing (NLP)
- Face Recognition
- Graph-Based Relationship Analysis
- Machine Learning
- Anomaly Detection
- Intelligence Alerts
- Investigation Visualization
- PDF Report Generation

> **Important:** This is an investigation-support prototype. It is not intended to automatically determine criminal guilt, replace investigators, or make autonomous law-enforcement decisions.

---

## 🎯 Objectives

1. Provide a centralized interface for investigation records.
2. Reduce manual effort in connecting related investigation entities.
3. Extract useful entities from investigation/FIR text.
4. Identify relationships between persons, cases, phones, vehicles, locations, and organizations.
5. Represent investigation relationships as a graph.
6. Detect unusual or potentially important records using anomaly detection.
7. Generate intelligence alerts based on analytical indicators.
8. Support face-based identity matching from authorized investigation images.
9. Provide investigators with a visual intelligence dashboard.
10. Generate investigation reports for documentation and review.

---

## 🚨 Problem Statement

Modern investigations can involve large and heterogeneous datasets.

A single investigation may contain:

```text
Person Records
      ↓
FIR / Case Records
      ↓
Phone Numbers
      ↓
Vehicles
      ↓
Locations
      ↓
Organizations
      ↓
Photographs
      ↓
Investigation Reports
```

Manually connecting these entities can lead to:

- Slow investigation workflows
- Missed relationships
- Difficulty identifying recurring entities
- Difficulty detecting unusual patterns
- Fragmented information
- Increased cognitive workload for investigators

The proposed system addresses this by bringing investigation entities into a unified analytical environment.

---

## 💡 Proposed Solution

The system follows an integrated intelligence pipeline:

```text
Investigation Data
       │
       ▼
┌───────────────────────┐
│ Data Management       │
│ Persons / FIR / Cases │
└───────────┬───────────┘
            │
            ▼
┌────────────────────────────┐
│ AI-Assisted Analysis       │
├────────────────────────────┤
│ NLP Entity Extraction      │
│ Face Recognition           │
│ Relationship Analysis      │
│ Graph Construction         │
│ Anomaly Detection          │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ Intelligence Layer         │
│ Network Indicators         │
│ Anomaly Scores             │
│ Intelligence Alerts        │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ Investigator Dashboard     │
│ Graphs / Alerts / Reports  │
└────────────────────────────┘
```

The final decision remains with the authorized human investigator.

---

# ✨ Key Features

## 1. Investigation Record Management

The system provides a centralized interface for managing:

- Persons
- Cases
- FIRs
- Phone numbers
- Vehicles
- Locations
- Organizations
- Investigation information

This creates a structured foundation for further analysis.

## 2. NLP-Based Entity Extraction

The system can process investigation/FIR/report text and extract useful entities.

Example:

```text
The suspect Rahul Kumar was seen using mobile number
9876543210 near Patna Railway Station and travelling
in vehicle BR01AB1234.
```

Potential entities:

```text
PERSON      → Rahul Kumar
PHONE       → 9876543210
LOCATION    → Patna Railway Station
VEHICLE     → BR01AB1234
```

### NLP Pipeline

```text
FIR / Investigation Text
          ↓
     Text Processing
          ↓
Pattern-Based Entity Extraction
          ↓
 Person / Phone / Vehicle /
 Location / Organization
          ↓
      Case Association
```

The current prototype uses a conservative pattern-based extraction approach rather than claiming a fully trained domain-specific NLP model.

## 3. Face Recognition

The system provides face-based matching using the **InsightFace FaceAnalysis** pipeline.

```text
InsightFace
    ↓
Face Detection
    ↓
Face Embedding
    ↓
Similarity Comparison
    ↓
Potential Match
```

Configured recognition model:

```text
buffalo_l
```

Current application threshold:

```text
MATCH_THRESHOLD = 0.45
```

Face recognition results should be treated as **investigative leads**, not final proof of identity.

## 4. Graph-Based Relationship Analysis

Investigation entities can be represented as nodes in a graph.

Typical relationships are based on shared:

- Phone numbers
- Vehicles
- Locations/addresses
- Organizations
- FIR/case associations

Example:

```text
             Phone
               │
               │
Person ─────── Case
  │              │
  │              │
Vehicle        FIR
  │
Location
```

This allows investigators to explore connections that may not be immediately visible in tabular records.

## 5. Network Intelligence Features

The system derives person-level graph features such as:

```text
Degree
Phone Links
Vehicle Links
Location Links
Organization Links
FIR Links
Network Score
```

These features provide a numerical representation of an entity's connectivity.

> Network connectivity alone does not establish criminal involvement.

## 6. Machine Learning-Based Anomaly Detection

The system uses **Isolation Forest** for unsupervised anomaly detection when sufficient records are available.

### Feature Pipeline

```text
Investigation Records
        ↓
Relationship Graph
        ↓
Graph Features
        ├── Degree
        ├── Phone Links
        ├── Vehicle Links
        ├── Location Links
        ├── Organization Links
        └── FIR Links
        ↓
Isolation Forest
        ↓
Anomaly Score
        ↓
Intelligence Indicator
```

Prototype configuration:

```text
Algorithm: Isolation Forest
Estimators: 150
Random State: 42
Contamination: auto
```

When scikit-learn is unavailable, the application provides a transparent percentile-based fallback.

## 7. Intelligence Alerts

The system combines analytical indicators to generate investigation alerts.

Alert generation considers factors such as:

- Machine learning anomaly score
- Network/graph bridge indicators
- Entity connectivity
- Investigation context

Example:

```text
HIGH
Potentially significant anomalous network pattern detected.

MEDIUM
Entity shows unusual relationship characteristics.

LOW
Additional review may be useful.
```

High-priority alerts are visually emphasized in the interface.

The purpose of alerts is to help investigators decide **where to look first**, not to automatically determine conclusions.

## 8. Intelligence Center

The Intelligence Center provides a consolidated analytical view of:

- Network information
- Anomaly indicators
- Investigation statistics
- Alerts
- Relationship analysis
- Analytical summaries

## 9. Relationship Graph Visualization

The application can visualize relationships between:

```text
Person
Case
FIR
Phone
Vehicle
Location
Organization
```

The graph can help investigators identify:

- Highly connected entities
- Shared resources
- Common locations
- Repeated associations
- Potential bridge entities
- Investigation clusters

## 10. Investigation Clustering

Clusters can represent groups of entities with stronger internal relationships.

```text
        Person A
        /      \
       /        \
   Phone X     Case 1
      │          │
      │          │
   Person B ─ Vehicle Y
       \
        \
       Location Z
```

## 11. SQLite Database

The application uses **SQLite** for local data storage.

Database location:

```text
database/criminal_intelligence.db
```

Runtime directories:

```text
database/
faces/
criminal_photos/
```

## 12. PDF Report Generation

The system supports investigation report export using **ReportLab**.

```text
Investigation Data
       ↓
Analysis
       ↓
Alerts / Relationships / Findings
       ↓
Report Generation
       ↓
PDF Report
```

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │ Investigation Input  │
                         └──────────┬───────────┘
                                    ↓
                         ┌──────────────────────┐
                         │ SQLite Data Layer    │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  ↓                 ↓                 ↓
             ┌─────────┐      ┌──────────┐      ┌───────────┐
             │   NLP   │      │   Face   │      │Relations  │
             │Analysis │      │Recognition│     │/Graph     │
             └────┬────┘      └─────┬────┘      └─────┬─────┘
                  │                 │                 │
                  └─────────────────┼─────────────────┘
                                    ↓
                         ┌──────────────────────┐
                         │ Feature Engineering  │
                         └──────────┬───────────┘
                                    ↓
                         ┌──────────────────────┐
                         │ Anomaly Detection    │
                         │   Isolation Forest   │
                         └──────────┬───────────┘
                                    ↓
                         ┌──────────────────────┐
                         │ Intelligence Layer   │
                         │ Alerts + Priorities  │
                         └──────────┬───────────┘
                                    ↓
                         ┌──────────────────────┐
                         │ Investigator UI      │
                         │ Graph + Dashboard    │
                         │ Reports              │
                         └──────────────────────┘
```

---

# 🧩 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Tkinter / ttk | Desktop user interface |
| SQLite | Local database |
| NumPy | Numerical processing |
| OpenCV | Image processing |
| Pillow | Image handling |
| InsightFace | Face detection and recognition |
| ONNX Runtime | InsightFace model runtime |
| Scikit-learn | Machine learning and anomaly detection |
| Graph Analysis | Relationship and network intelligence |
| ReportLab | PDF report generation |

---

# 📁 Project Structure

```text
criminal-intelligence-command-center/
│
├── app/
│   └── main.py
│
├── docs/
│   ├── TECHNICAL_OVERVIEW.md
│   └── DEMO_CHECKLIST.md
│
├── database/
│   └── .gitkeep
│
├── faces/
│   └── .gitkeep
│
├── criminal_photos/
│   └── .gitkeep
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Important

Runtime and sensitive data are intentionally excluded from version control.

Do not upload:

- Real criminal records
- Real biometric images
- Personal phone numbers
- Private addresses
- Credentials
- API keys
- Production databases

---

# ⚙️ Installation

## Requirements

Recommended:

```text
Windows / Linux / macOS
Python 3.10+
```

Check Python:

```bash
python --version
```

On Windows:

```bash
py --version
```

## 1. Clone the Repository

```bash
git clone https://github.com/teamhorizon0707-stack/criminal-intelligence-command-center.git
cd criminal-intelligence-command-center
```

## 2. Install Dependencies

Windows:

```bash
py -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m pip install -r requirements.txt
```

## 3. Run the Application

From the repository root:

```bash
py app/main.py
```

or:

```bash
python app/main.py
```

---

# 🧪 Example Investigation Workflow

### Step 1 — Add Investigation Records

Use authorized demonstration records containing:

```text
Person
Case
Phone
Vehicle
Location
Organization
```

### Step 2 — Add Case/FIR Information

Example:

```text
The suspect was reported near Patna Railway Station
while travelling in vehicle BR01AB1234.
```

### Step 3 — Run NLP Analysis

The system identifies relevant entities.

### Step 4 — Generate Relationships

Shared attributes are used to identify relationships.

### Step 5 — Open Relationship Graph

Explore connected entities and investigation clusters.

### Step 6 — Run Intelligence Analysis

The system generates graph-based features and applies anomaly detection.

Example:

```text
Entity       Anomaly Score
--------------------------
Person A          82
Person B          31
Person C          47
```

### Step 7 — Review Intelligence Alerts

High-priority indicators are shown prominently.

### Step 8 — Face Recognition

An authorized reference image can be compared against available investigation images.

Result:

```text
Potential Match
```

This is not definitive identity proof.

### Step 9 — Generate Investigation Report

Relevant information and analytical results can be exported to PDF.

---

# 🔬 Machine Learning Methodology

The anomaly detection component follows an unsupervised approach.

The system converts relationship information into numerical features:

```text
X =
[
    degree,
    phone_links,
    vehicle_links,
    location_links,
    organization_links,
    fir_links,
    network_score
]
```

These features are passed to Isolation Forest.

The prototype normalizes anomaly-related scores to a 0–100 range for easier interpretation in the UI.

The application uses an anomaly decision threshold of approximately:

```text
65
```

for intelligence prioritization.

This score is an investigative indicator and should not be interpreted as a probability of criminality.

---

# 🧠 Why Isolation Forest?

Isolation Forest was selected because the prototype focuses on identifying unusual patterns without requiring a large manually labelled dataset.

Advantages:

- Unsupervised operation
- Suitable for anomaly detection
- Relatively lightweight
- Works with numerical feature representations
- Does not require predefined anomaly labels

---

# 🔗 Graph Intelligence Methodology

The relationship graph is constructed from shared investigation attributes.

Example:

```text
Person A ── Phone X ── Person B
Person A ── Vehicle Y ── Person C
Person B ── Location Z ── Person C
```

Graph-derived features are then used as inputs to anomaly detection.

```text
Relationship Intelligence
          +
Machine Learning
          =
Prioritized Investigation Indicators
```

---

# 🔐 Privacy and Responsible Use

The prototype should use:

- Authorized datasets
- Synthetic demonstration records
- Public aggregate datasets where appropriate
- Local storage for prototype data
- No credentials inside source code
- No real biometric data in the public repository

Responsible workflow:

```text
AI Output
   ↓
Investigative Lead
   ↓
Human Verification
   ↓
Investigation Decision
```

The final decision remains with authorized human personnel.

---

# ⚠️ Limitations

### Dataset Dependency

Results depend heavily on data quality and completeness.

### NLP Limitations

The current NLP component uses conservative pattern-based extraction and may not correctly understand every linguistic variation or complex contextual relationship.

### Face Recognition Limitations

Performance can be affected by lighting, image quality, pose, occlusion, camera quality, age differences, and similar-looking individuals.

### Anomaly Detection Limitations

Isolation Forest identifies statistical irregularities. An anomalous record is not necessarily a criminal record.

### Graph Interpretation

Highly connected entities may occur naturally in legitimate circumstances. Graph connectivity should not be interpreted as proof of wrongdoing.

### Prototype Environment

Production deployment would require additional authentication, authorization, audit logging, encryption, secure storage, access controls, retention policies, model validation, and legal/regulatory compliance.

---

# 🚀 Future Scope

## Advanced NLP

- Transformer-based NLP
- Named Entity Recognition
- Relation Extraction
- Multilingual NLP
- Hindi/English investigation text processing

## Advanced Graph Analytics

- Community detection
- Centrality analysis
- Link prediction
- Temporal graph analysis
- Shortest-path investigation
- Dynamic network visualization

## Improved Machine Learning

- Autoencoder-based anomaly detection
- Graph Neural Networks
- Supervised classification
- Temporal anomaly detection
- Explainable AI
- Model comparison and benchmarking

## Secure Multi-User Architecture

```text
User Authentication
        ↓
Role-Based Access Control
        ↓
Secure API
        ↓
Central Database
        ↓
Audit Logging
```

## Secure Cloud Deployment

Potential future technologies:

- PostgreSQL
- REST APIs
- Secure authentication
- Encrypted storage
- Centralized audit logs
- Role-based access control

## Mobile and Web Interfaces

The dashboard could be extended to:

- Web applications
- Mobile applications
- Secure investigator portals
- Real-time alert systems

---

# 📊 Evaluation Strategy

For a research-grade implementation, each component should be evaluated independently.

### NLP

```text
Precision
Recall
F1-score
```

### Face Recognition

```text
True Acceptance Rate
False Acceptance Rate
False Rejection Rate
```

### Anomaly Detection

Depending on labelled evaluation data:

```text
Precision
Recall
F1-score
ROC-AUC
PR-AUC
```

### Graph Analysis

Possible measurements:

```text
Graph construction time
Node count
Edge count
Query time
Relationship discovery
```

The current hackathon prototype does not claim these metrics unless they are actually measured on an appropriate evaluation dataset.

---

# 🏆 Hackathon Value

The project demonstrates how multiple AI-assisted techniques can be integrated into a single investigation workflow.

Instead of isolated modules:

```text
NLP
Face Recognition
Graph Analysis
Machine Learning
```

the project connects them into:

```text
Data
 ↓
Entity Extraction
 ↓
Relationship Discovery
 ↓
Graph Construction
 ↓
Feature Engineering
 ↓
Anomaly Detection
 ↓
Intelligence Alerts
 ↓
Human Investigation
```

This integrated workflow is the central value of the prototype.

---

# 🎥 Recommended Hackathon Demo

A short demonstration can follow:

```text
1. Launch Command Center
        ↓
2. Show investigation records
        ↓
3. Add FIR/report text
        ↓
4. Run NLP extraction
        ↓
5. Show extracted entities
        ↓
6. Generate relationship graph
        ↓
7. Show connected entities
        ↓
8. Run anomaly analysis
        ↓
9. Show intelligence alert
        ↓
10. Demonstrate face matching
        ↓
11. Generate PDF report
```

---

# 🖥️ Application Modules

| Module | Function |
|---|---|
| Dashboard | Central investigation overview |
| Person Management | Manage person-related records |
| Case/FIR Management | Manage investigation cases |
| NLP Analysis | Extract entities from text |
| Face Recognition | Compare authorized face images |
| Relationship Analysis | Identify shared attributes |
| Graph Visualization | Display investigation networks |
| Intelligence Center | Consolidated analytical view |
| Anomaly Detection | Identify unusual patterns |
| Intelligence Alerts | Prioritize analytical indicators |
| Report Generation | Export investigation information |

---

# 📌 Example End-to-End Scenario

Consider a demonstration dataset:

```text
Person A
 ├── Phone: X
 ├── Vehicle: Y
 └── Location: Z

Person B
 ├── Phone: X
 └── Location: Z

Person C
 ├── Vehicle: Y
 └── Organization: O
```

The system can identify:

```text
Person A ↔ Person B
       │
       │ shared phone
       │
Person A ↔ Person C
       │
       │ shared vehicle
       │
Person B ↔ Person A
       │
       │ shared location
```

These relationships form a graph. Graph features can then be analyzed by the anomaly detection component.

If an unusual connectivity pattern is detected, the system may generate an intelligence alert. The investigator can inspect the underlying records before drawing any conclusion.

---

# 🔄 Overall Workflow

```text
┌─────────────────────┐
│ Investigation Data  │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Data Organization   │
└──────────┬──────────┘
           │
           ├──────────────────┐
           ↓                  ↓
┌─────────────────┐   ┌─────────────────┐
│ NLP Extraction  │   │ Face Recognition│
└────────┬────────┘   └────────┬────────┘
         │                     │
         └──────────┬──────────┘
                    ↓
          ┌────────────────────┐
          │ Relationship Graph │
          └──────────┬─────────┘
                     ↓
          ┌────────────────────┐
          │ Feature Engineering│
          └──────────┬─────────┘
                     ↓
          ┌────────────────────┐
          │ Anomaly Detection  │
          └──────────┬─────────┘
                     ↓
          ┌────────────────────┐
          │ Intelligence Alerts│
          └──────────┬─────────┘
                     ↓
          ┌────────────────────┐
          │ Human Investigation│
          └────────────────────┘
```

---

# 📚 Research and Development Context

This project can serve as a prototype for research into the integration of:

- Artificial Intelligence
- Criminal intelligence
- Investigation support systems
- Natural Language Processing
- Computer Vision
- Graph analytics
- Unsupervised machine learning
- Human-in-the-loop decision support

The architecture demonstrates how heterogeneous investigation information can be transformed into structured intelligence indicators.

---

# 🛡️ Responsible AI Statement

This project is designed as an **AI-assisted decision-support prototype**.

It does not:

- Automatically declare a person guilty
- Automatically make arrests
- Automatically determine criminal intent
- Treat anomaly scores as crime probabilities
- Treat face similarity as definitive identity proof

AI-generated results are intended to support human review.

All real-world deployment would require appropriate legal, ethical, security, privacy, and institutional safeguards.

---

# 👥 Project Team

**Team Horizon**

### Project

**Criminal Intelligence and Investigation Command Center**

### Core Areas

- Artificial Intelligence
- Machine Learning
- Natural Language Processing
- Computer Vision
- Graph Analytics
- Anomaly Detection
- Investigation Intelligence

---

# ⭐ Key Takeaway

> **Convert scattered investigation information into connected, explainable intelligence that helps investigators identify relationships, unusual patterns, and priority leads faster — while keeping the human investigator in control.**

---

## 🚀 Quick Start

```bash
git clone https://github.com/teamhorizon0707-stack/criminal-intelligence-command-center.git
cd criminal-intelligence-command-center
py -m pip install -r requirements.txt
py app/main.py
```

---

**Built as an AI-assisted investigation intelligence prototype for hackathon and research demonstration.**
