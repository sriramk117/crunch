# Crunch

Crunch is a python package that generates data visualizations on large image datasets to enhance the pre-training process. The objective of Crunch is to merge **explainable AI** with **beautiful data visualization** into a **painless python library**.

<img width="2570" height="1444" alt="image" src="https://github.com/user-attachments/assets/e3a881c2-a482-411c-8c8e-aee76debe2d6" />

## How does Crunch work?
Crunch works by extracting embeddings (feature extraction is customizable), reducing the dimensionality of those embeddings, and applying classic clustering algorithms (e.g. <a href="https://www.datacamp.com/tutorial/k-means-clustering-python">k-means clustering</a>, <a href="https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html">HDBSCAN</a>, etc.) to create interpretable graph visualizations.
Here is a high-level diagram detailing the pipeline and underlying infrastructure behind Crunch:

<img width="2570" height="1444" alt="image" src="https://github.com/user-attachments/assets/d2e420cb-9665-4d2e-a334-eecbf0d78c54" />
