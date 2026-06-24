# Analisis-Dinamico-de-Ciberataques

# Dynamic Cyberattack Analysis

Detection and representation of multi-stage cyberattacks through the integration of supervised Machine Learning and graph-based event correlation.

This project proposes an approach for identifying attack tactics from network security events and representing their relationships using Neo4j.

The Machine Learning module was designed and executed in a Google Colab environment for data preprocessing, model training, validation, and prediction generation. The generated results are later integrated into Neo4j to build a graph-based representation of attack phases aligned with the MITRE ATT&CK framework.

## Modules

* machine_learning → Event preprocessing, training and prediction generation (Google Colab)
* neo4j → Graph storage, event correlation and attack path analysis

## Dataset

UWF-ZeekData24

## Technologies

* Python
* Google Colab
* XGBoost
* Neo4j
* Pandas
* Cypher

