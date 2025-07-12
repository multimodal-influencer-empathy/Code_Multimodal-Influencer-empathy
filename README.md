
## Empathy and Sales in Influencer Marketing

## Overview
This repository contains the code and data used in the research paper "Empathy and Sales in Influencer Marketing". 

## Installation
Before running the scripts, ensure you have the following dependencies installed:
- Python 3.x
- PyTorch
- Pandas
- Numpy
- Pickle
- Statsmodels
- Scikit-learn

You can install these packages using pip:
```bash
pip install torch pandas statsmodels numpy scikit-learn
```
--

## Dataset
Due to GitHub's storage limitations, only the test data is displayed in this repository. The test data has been uploaded to Google Drive. You can access the dataset using the following link: [Google Drive Dataset](https://drive.google.com/drive/folders/1d97Ox0in0WNW5miQZZ-zCo5xwq7QEivM?usp=drive_link).


### Multimodal Features 
#### Vocal Features
- **Tool Used**: Covarep.
- **Features**: 74 dimensions.


#### Visual Features
- **Tool Used**: OpenFace 2.0.
- **Features**: 35 dimensions.


#### Verbal Features
- **Tool Used**: BERT.
- **Features**: 768 dimensions.




## Usage
To run the analysis, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/multimodal-influencer-empathy/Multimodal-Influencer-empathy.git.
   ```
2. Navigate to the cloned directory.
3. Run the main script:
   ```bash
   python A1_GMFN_run_evaluation_pretrained_model.py
   python A2_GMFN_run_evaluation_pretrained_model.py
   python A3_GMFN_run_evaluation_pretrained_model.py
   python A4_GMFN_run_evaluation_pretrained_model.py
   python A5_GMFN_run_evaluation_pretrained_model.py
   python A6_GMFN_run_evaluation_pretrained_model.py
   python A7_GMFN_run_evaluation_pretrained_model.py
   python A8_GMFN_noM_run_evaluation_pretrained_model.py
   python A9_GMFN_noG_run_evaluation_pretrained_model.py
   python A10_GMFN_noW_run_evaluation_pretrained_model.py
   python A11_EF_LSTM_run_evaluation_pretrained_model.py
   python A12_TFN_run_evaluation_pretrained_model.py
   python A13_LMF_run_evaluation_pretrained_model.py
   python A14_MFN_run_evaluation_pretrained_model.py
   ```

## Code Structure
- `14 main models`: The main script that orchestrates the data loading,  pretrained model loading, and evaluation.

- `pretrained_model`: Contains 14 pretrained models.

## Model Description
This codebase implements 14 multimodal models for evaluating influencer empathy, each tailored to different input modality combinations. The core architecture is based on GMFN (Graph Memory Fusion Network), with extended configurations for:

- Unimodal settings (Text / Audio / Image),

- Bimodal combinations (e.g., Text + Audio),

- Trimodal fusion (Text + Audio + Image),

- Ablation studies to investigate the effect of individual modules within GMFN (e.g., temporal reasoning, cross-modal interaction, dynamic fusion graph).

### 1. Unimodal Models (3 Models)

| ID  | File Name                                      | Description                          |
|-----|-----------------------------------------------|--------------------------------------|
| A1  | `A1_GMFN_run_evaluation_pretrained_model.py`  | GMFN using **Text only**             |
| A2  | `A2_GMFN_run_evaluation_pretrained_model.py`  | GMFN using **Audio only**            |
| A3  | `A3_GMFN_run_evaluation_pretrained_model.py`  | GMFN using **Image only**            |

Serve as unimodal baselines to evaluate the individual contribution of each modality.

 

### 2. Bimodal Models (3 Models)

| ID  | File Name                                      | Description                                |
|-----|-----------------------------------------------|--------------------------------------------|
| A4  | `A4_GMFN_run_evaluation_pretrained_model.py`  | GMFN using **Text + Audio**                |
| A5  | `A5_GMFN_run_evaluation_pretrained_model.py`  | GMFN using **Text + Image**                |
| A6  | `A6_GMFN_run_evaluation_pretrained_model.py`  | GMFN using **Audio + Image**               |

Compare performance across different modality combinations and investigate multimodal complementarity.



### 3. Trimodal Models and Ablation Studies (8 Models)

| ID   | File Name                                         | Description |
|------|--------------------------------------------------|-------------|
| A7   | `A7_GMFN_run_evaluation_pretrained_model.py`     | GMFN with **Text + Audio + Image** (full fusion) |
| A8   | `A8_GMFN_noM_run_evaluation_pretrained_model.py` | GMFN **without Cross-modal Interaction** module |
| A9   | `A9_GMFN_noG_run_evaluation_pretrained_model.py` | GMFN **without Dynamic Fusion Graph (DFG)** |
| A10  | `A10_GMFN_noW_run_evaluation_pretrained_model.py`| GMFN **without temporal features**, using only static mean features |
| A11  | `A11_EF_LSTM_run_evaluation_pretrained_model.py` | Baseline: Early Fusion LSTM with all three modalities |
| A12  | `A12_TFN_run_evaluation_pretrained_model.py`     | Baseline: TFN (Tensor Fusion Network) |
| A13  | `A13_LMF_run_evaluation_pretrained_model.py`     | Baseline: LMF (Low-rank Multimodal Fusion) |
| A14  | `A14_MFN_run_evaluation_pretrained_model.py`     | Baseline: MFN (Memory Fusion Network) |

 Perform comprehensive ablation studies to verify the effectiveness of GMFN components and compare against mainstream multimodal fusion baselines.
Each model is based on pretrained models and demonstrates the prediction results for different data modalities . The models are specifically tailored for analyzing the influencer emapthy using multimodal data.



## Evaluation Metrics
The code includes metrics for evaluating the model performance:
- Accuracy
- F1 Score
- Mean Absolute Error (MAE)
- Correlation Coefficient
- Loss






