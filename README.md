# Multimodal Forecasting of Influenza with Self-Supervised and Attention-Based Fusion

**Author:** Lacey Dinh  

This repository contains my personal deep learning project on epidemic forecasting. The goal is to explore how multimodal signals—time-series surveillance data, vaccination and weather reports, and spatial influenza heatmaps—can be combined to build accurate, interpretable models for forecasting influenza outbreaks.

---

## Motivation
Influenza epidemics strain healthcare systems annually, leading to high morbidity, mortality, and economic disruption. Accurate and timely forecasts enable better resource allocation, preventive interventions, and preparedness.  
Traditional statistical models often fall short in capturing complex temporal and spatial dependencies. This project demonstrates how modern deep learning methods—RNNs, Transformers, CNN autoencoders, and cross-modal attention fusion—can advance epidemic forecasting.

---

## Datasets

The project integrates multiple public datasets:

1. **CDC Influenza Surveillance Reports**  
   - Weekly outpatient ILI counts, lab-confirmed flu cases, and patient statistics.  
   - [CDC Influenza Data](https://data.chhs.ca.gov/dataset/influenza-surveillance)

2. **Influenza Vaccination Rates**  
   - County- and hospital-level vaccination percentages.  
   - [Vaccination Dataset](https://catalog.data.gov/dataset/health-care-personnel-influenza-vaccination-026e8)

3. **NOAA Climate Data (2009–2020)**  
   - Average weekly temperatures and climate variables.  
   - [NOAA Climate Data](https://www.ncdc.noaa.gov/cdo-web/datasets)

4. **CDC Weekly Influenza Heatmaps (2022–2023)**  
   - State-level images visualizing flu intensity progression.  
   - [CDC Weekly Flu Maps](https://www.cdc.gov/fluview/surveillance/usmap.html)

---

## Methods

### 1. Data Exploration & Preprocessing
- Statistical summaries and visualizations of surveillance data.
- Handling missing values (imputation, forward/backward fill).
- PCA and correlation analysis.
- Custom PyTorch Datasets for both tabular and image modalities.

### 2. Baseline Models
- **Feedforward Neural Network (FFNN)**:  
  Simple 3-layer network trained on 4-week tabular sequences.  
- **LSTM (RNN)**:  
  Sequence-aware model for temporal forecasting.  

### 3. Advanced Architectures
- **Transformer Encoder**:  
  Captures long-range dependencies with attention, interpretable via attention weights.  
- **Self-Supervised Convolutional Autoencoder (CAE)**:  
  Learns spatial flu representations from CDC heatmaps without labels.  

### 4. Multimodal Fusion
- **Concatenation Fusion**: Combines temporal and spatial embeddings.  
- **Cross-Attention Fusion**: Temporal Transformer embeddings attend to CNN spatial features, mimicking epidemiological reasoning.  

---

## Results

| Model              | Input Type | RMSE (↓) | Notes |
|--------------------|------------|----------|-------|
| FFNN               | Tabular    | 0.0485   | Simple baseline |
| LSTM (RNN)         | Tabular    | 0.0479   | Captures short-term trends |
| Transformer Encoder | Tabular   | 0.0089   | Best temporal model |
| CNN (CAE Encoder)  | Images     | 0.18     | Extracts spatial signals |
| Fusion (Concat)    | Multi      | 0.076    | Improves over single modalities |
| Fusion (Cross-Attn)| Multi      | 0.070    | Best overall, interpretable |

**Key Insights:**
- Transformer attention highlights mid-sequence weeks as critical to predictions.  
- CAE filters localize geographic flu hotspots (e.g., Northeast, Texas).  
- Cross-attention fusion aligns temporal spikes with spatial outbreak zones.  

---

## Tech Stack
- **Languages/Frameworks**: Python, PyTorch, PyTorch Lightning  
- **Tools**: TensorBoard, scikit-learn, NumPy, pandas, matplotlib, seaborn  
- **Methods**: Self-supervised learning, attention mechanisms, multimodal fusion  

---

## Future Work
- Extend to sequence-to-sequence multi-week forecasts.  
- Incorporate behavioral/mobility data (Google Flu Trends, Facebook movement).  
- Explore graph neural networks (spatiotemporal GNNs on states/mobility).  
- Apply to COVID-19 or other epidemic datasets for generalization.  

---

## Visual Highlights
- Transformer attention maps → interpret influential weeks.  
- CAE reconstructions → capture intensity regions in flu heatmaps.  
- Cross-attention visualizations → reveal how time and space interact in predictions.  

---

## Repository Structure


---

## References
- Vaswani et al. (2017). *Attention is All You Need*.  
- Azizi et al. (2021). *Big self-supervised models advance medical image classification*.  
- Rajkomar et al. (2018). *Scalable and accurate deep learning with electronic health records*.  

---

## Acknowledgments
This project was inspired by real-world influenza forecasting challenges and integrates public datasets from CDC, NOAA, and CHHS.

---

