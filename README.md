# Criminal Intelligence and Investigation Command Center

AI-assisted investigation prototype for organizing case records and supporting investigative analysis through **NLP-based entity extraction, face recognition, relationship/network analysis, anomaly detection, intelligence alerts, and report export**.

> **Hackathon Prototype:** This repository is intended for demonstration and evaluation. It is not a production law-enforcement system and must not be used to make automated decisions about guilt or criminal responsibility.

## Key Features

- Criminal/person and FIR record management
- Entity extraction from investigation/FIR text
- Relationship discovery using shared attributes and case links
- Network visualization and graph-based analytical metrics
- Unsupervised anomaly detection using Isolation Forest when scikit-learn is available
- Priority/intelligence alerts with high-severity indicators
- Face detection/recognition using InsightFace (`buffalo_l`) embeddings
- Camera-based face recognition workflow
- Case intelligence dashboard and analytics
- PDF/CSV export support
- SQLite local database for the prototype

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application |
| Tkinter | Desktop user interface |
| SQLite | Local case/person database |
| OpenCV | Camera and image processing |
| InsightFace / ArcFace | Face embeddings and recognition |
| NumPy | Numerical processing |
| Scikit-learn | Isolation Forest anomaly detection |
| ReportLab | PDF report generation |

## Project Structure

```text
criminal-intelligence-command-center/
├── app/
│   └── main.py
├── database/
│   └── .gitkeep
├── faces/
│   └── .gitkeep
├── criminal_photos/
│   └── .gitkeep
├── docs/
│   └── TECHNICAL_OVERVIEW.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd criminal-intelligence-command-center
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app/main.py
```

The application creates its SQLite database and runtime folders when required. Face data, photos, local databases and generated artifacts are intentionally excluded from Git through `.gitignore`.

## Face Recognition Note

The application uses InsightFace with the `buffalo_l` model. The first model initialization may require model files to be downloaded/configured by the InsightFace runtime. A working camera is required for live recognition.

## Machine Learning Note

The anomaly component builds person-level network features and uses Isolation Forest when enough records and scikit-learn are available. The output is an investigative analytical indicator, not a proof of criminal activity.

## Data and Privacy

Do not commit real personal information, biometric data, confidential FIR records, credentials, or operational law-enforcement data to this public repository. Use synthetic/demo or properly authorised data for the hackathon demonstration.

## Responsible Use

The prototype is designed to **assist investigators by organizing and surfacing information for human review**. Analytical scores, graph relationships, and face-recognition matches require verification against source records and should not be treated as autonomous determinations.
