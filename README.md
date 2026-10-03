# AI-Assisted Investigation Intelligence & Relationship Analysis System

AI-Assisted Investigation Intelligence & Relationship Analysis System for organizing case records and supporting investigative analysis through **NLP-based entity extraction, face recognition, relationship/network analysis, anomaly detection, intelligence alerts, and report export**.

## Key Features

- Criminal/person and FIR record management
- Entity extraction from FIRs, CDRs, transactions, reports and intelligence records
- Relationship discovery using shared attributes and case links
- Network visualization and graph-based analytics
- Unsupervised anomaly detection using scikit-learn
- Priority/intelligence alerts with high-intense indicators
- Face detection/recognition using InsightFace embeddings
- Camera-based face recognition workflow
- Case intelligence dashboard and analytics
- PDF/CSV export support
- SQLite local database for the prototype

## Technology Stack

| **Technology** | **Purpose** |

Python |	Core Programming Language

SQLite |	Database

OpenCV |	Image Processing

NLP	Text | Analysis

Scikit-learn |	Machine Learning

NetworkX |	Graph Analysis

Pandas & NumPy |	Data Processing

Joblib |	Model Persistence

ReportLab | Report Generation

## Project Structure

```text
criminal-intelligence-command-center/
├── app/
├── database/
├── faces/
├── criminal_photos/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/teamhorizon0707-stack/AI-Assisted-Investigation-Intelligence-Relationship-Analysis-System
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

The application uses InsightFace model. The first model initialization may require model files to be downloaded/configured by the InsightFace runtime. A working camera is required for live recognition.

## Machine Learning Note

The anomaly component builds person-level network features and using scikit-learn. The output is an investigative analytical indicator, not a proof of criminal activity.

## Responsible Use

The prototype is designed to **assist investigators by organizing and surfacing information for human review**. Analytical scores, graph relationships, and face-recognition matches require verification against source records and should not be treated as autonomous determinations.
