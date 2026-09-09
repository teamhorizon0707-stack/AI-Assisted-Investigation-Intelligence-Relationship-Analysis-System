# Technical Overview

## Processing Pipeline

```text
Person / FIR / Investigation Data
              |
              v
     Data & Entity Processing
              |
      +-------+--------+
      |       |        |
      v       v        v
     NLP    Face     Relationships
      |    InsightFace     |
      |       |            |
      +-------+------------+
              |
              v
       Graph / Network Analysis
              |
              v
      Feature Construction
              |
              v
       Isolation Forest ML
              |
              v
      Intelligence / Priority
              |
              v
       Investigator Review
```

## NLP

The application extracts structured entities from investigation text and can save extracted entities against a case. The implementation is intentionally conservative and is intended to support human verification.

## Face Recognition

The application loads InsightFace's `buffalo_l` model, generates face embeddings and compares stored embeddings using cosine similarity. The current prototype uses a configurable matching threshold in the application.

## Relationship Analysis

The system models people and their links through shared information such as phone, vehicle, address/location, organization and FIR/case associations. These links are visualized for investigative exploration.

## Machine Learning

The anomaly pipeline constructs numerical features such as network degree, shared-attribute links and a combined network score. When sufficient records are available, `IsolationForest` is used as an unsupervised anomaly detector. Scores are used for prioritisation and require source verification.

## Alerts

The intelligence alert center combines analytical indicators such as anomaly scores and network bridge position, de-duplicates alerts and prioritizes higher-severity items for investigator review.
