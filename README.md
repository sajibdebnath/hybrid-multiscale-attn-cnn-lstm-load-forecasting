# ⚡  Hybrid Multi-Scale Deep Learning Enhanced Electricity Load Forecasting Using Attention-Based Convolutional Neural Network and LSTM Model

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**🏆 R² = 0.9677 | MAPE = 2.53%**

*Deep learning models for electricity demand forecasting using ERCOT data and weather features*

</div>

> NOTE: A supplemental file with detailed numeric tables and demo instructions is available: `README_EXTRA.md` (see project root).

---

## 🎯 Overview

Deep learning models for electricity demand forecasting using ERCOT data (2018-2024) and weather features. The hybrid CNN-LSTM-Attention model achieves 96.77% accuracy by combining convolutional layers, LSTM networks, and attention mechanisms.

## 📊 Dataset

- **Load Data**: ERCOT electricity demand (2018-2024, hourly)
- **Weather Data**: 3 ASOS stations (temperature, humidity, wind, pressure)
- **Features**: Temperature, weather variables, temporal patterns, US holidays
- **Final Dataset**: `Final_dataset_ERCOT_v2.csv`


## 📈 Results

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | sMAPE (%) | PSNR (dB) | R² |
|-------|----------:|----------:|---------:|----------:|----------:|----:|
| LSTM | 2393.4662 | 3298.2063 | 3.8416 | 3.9473 | 28.2431 | 0.9042 |
| CNN-LSTM | 1984.1900 | 2654.4977 | 3.5111 | 3.4465 | 30.1290 | 0.9380 |
| Attention-LSTM | 2193.6967 | 2878.8981 | 3.9219 | 3.9617 | 29.4241 | 0.9270 |
| **Attention-based CNN-LSTM (Proposed / Hybrid)** | **1430.5544** | **1915.1668** | **2.5278** | **2.5063** | **32.9645** | **0.9677** |


### Model Size & Performance

| Model | Params (M) | Train Time (s/epoch) | Inference Time (ms) | Peak Mem (MB, GPU) |
|-------:|----------:|--------------------:|--------------------:|-------------------:|
| LSTM | 0.0003 | 2.132 | 71.19 | 84.7 |
| CNN-LSTM | 0.0602 | 5.034 | 66.26 | 148.9 |
| Attention-LSTM | 0.0088 | 6.995 | 80.74 | 172.2 |
| Proposed Hybrid (Attention-based CNN-LSTM) | 0.3614 | 6.327 | 66.60 | 172.8 |


## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Analysis**: Jupyter Notebook

## 🚀 Getting Started

### Installation
```bash
# Clone repository
git clone https://github.com/sajibdebnath/hybrid-multiscale-attn-cnn-lstm-load-forecasting.git
cd hybrid-multiscale-attn-cnn-lstm-load-forecasting

# Install dependencies
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn holidays

# Run main notebook
jupyter notebook "Notebook/updated_EROCT.ipynb"
```


### Workflow
1. **Data Cleaning**: `Data cleaning/ERCOT_data_cleaning.ipynb`
2. **Data Analysis**: `Data Preprocessing/data_analytic_and_preprocessing.ipynb`
3. **Model Training**: `Notebook/updated_EROCT.ipynb`

## 📁 Project Structure

```
hybrid-multiscale-attn-cnn-lstm-load-forecasting/
├── Data/                               # Dataset files
│   ├── Final_dataset_ERCOT_v2.csv     # Main processed dataset
│   ├── Load data/                      # Raw ERCOT load data (2018-2024)
│   │   ├── Native_Load_2018.xlsx
│   │   ├── Native_Load_2019.xlsx
│   │   ├── Native_Load_2020.xlsx
│   │   ├── Native_Load_2021.xlsx
│   │   ├── Native_Load_2022.xlsx
│   │   ├── Native_Load_2023.xlsx
│   │   └── Native_Load_2024.xlsx
│   └── Weather data/                   # Meteorological data from ASOS stations
│       ├── asos_hourly_BKS.csv        # Blackland Army Airfield
│       ├── asos_hourly_JDD.csv        # Laredo International Airport
│       ├── asos_hourly_TME.csv        # Houston Executive Airport
│       └── asos_weather_data.csv      # Combined weather dataset
├── Correlation Analysis/               # Feature correlation analysis
│   └── Correlation Analysis.ipynb
├── Data cleaning/                      # Data cleaning and integration
│   └── ERCOT_data_cleaning.ipynb     # Load and weather data integration
├── Data Preprocessing/                 # EDA and feature engineering
│   └── data_analytic_and_preprocessing.ipynb
├── Notebook/                          # Main model training
│   └── updated_EROCT.ipynb           # All model implementations and evaluation
└── README.md                          # Project documentation
```

## 🔬 Model Architectures

### 1. **LSTM Baseline** - Simple LSTM with dropout regularization
### 2. **CNN-LSTM** - Conv1D feature extraction + LSTM temporal modeling
### 3. **Attention-LSTM** - LSTM with attention mechanism
### 4. **Hybrid CNN-LSTM-Attention** - Combined architecture (Best: 96.77% R²)

## 🔬 Key Features

- **Data Integration**: 7 years ERCOT load data + 3 weather stations
- **Feature Engineering**: Temporal patterns, weather variables, US holidays
- **Model Architecture**: Hybrid CNN-LSTM-Attention with multi-head attention
- **Reproducible Results**: Fixed seeds (SEED=42) for consistent outcomes

## � License

MIT License - See [LICENSE](LICENSE) for details.


## 🏗️ Model Architectures

### 1. � **LSTM **
- Simple LSTM with 2 layers (4→2 units)
- Dropout for regularization
- Basic temporal modeling

### 2. 🔸 **CNN-LSTM**
- Conv1D for feature extraction
- Bidirectional LSTM for temporal dependencies
- Batch normalization and dropout

### 3. 🔶 **Attention-LSTM**
- LSTM with custom attention mechanism
- Focus on important temporal patterns
- L2 regularization

### 4. � ** Attention-based CNN-LSTM** (Proposed)
- Conv1D feature extraction layers
- Bidirectional LSTM for sequence modeling
- Multi-head attention (4 heads)
- Residual connections + layer normalization
- Superior performance with 96.77% accuracy


## �🔮 Future Research Directions

### Model Enhancements
- [ ] **Multi-step Forecasting**: Extend to predict 24/48/72 hours ahead
- [ ] **Transformer Architecture**: Implement pure transformer models
- [ ] **Ensemble Methods**: Combine multiple models for improved robustness
- [ ] **Transfer Learning**: Apply pre-trained models across different regions

### System Integration
- [ ] **Real-time Prediction**: Deploy models for live forecasting
- [ ] **Renewable Integration**: Include solar/wind generation data
- [ ] **Grid Stability Analysis**: Incorporate frequency and voltage data
- [ ] **Demand Response**: Model impact of pricing on demand patterns

### Advanced Analytics
- [ ] **Uncertainty Quantification**: Bayesian neural networks for confidence intervals
- [ ] **Model Interpretability**: SHAP/LIME analysis for feature importance
- [ ] **Extreme Weather Events**: Enhanced modeling for climate extremes
- [ ] **Cross-Regional Validation**: Test models across different power grids

## 📚 References & Data Sources

### Data Sources
- **ERCOT**: Electric Reliability Council of Texas - Native Load Data
- **ASOS**: Automated Surface Observing System - Weather Data
  - BKS: Blackland Army Airfield Weather Station
  - JDD: Laredo International Airport Weather Station
  - TME: Houston Executive Airport Weather Station

### Technical References
- **Deep Learning**: TensorFlow/Keras framework for neural networks
- **Time Series**: Sequence modeling with LSTM and attention mechanisms
- **CNN Features**: 1D Convolutional layers for temporal pattern extraction
- **Attention Mechanisms**: Multi-head attention for temporal focus
- **Regularization**: Batch normalization, dropout, and L2 regularization techniques

### Research Methodology
- **Reproducible Research**: Fixed random seeds and deterministic operations
- **Cross-Validation**: Time series split for temporal validation
- **Performance Metrics**: Industry-standard evaluation (R², MAE, RMSE, MAPE)
- **Feature Engineering**: Domain knowledge integration for power systems



## 📄 License

<div align="center">

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)

**📖 Educational & Research Use**

This project is released under the MIT License for research and educational purposes.
Please ensure proper attribution when using this code or data in your work.

[View License](LICENSE) • [Citation Guidelines](#-contact--citation)

</div>

---

### 🌐 **Connect With Us**

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-sajibdebnath-black?style=flat-square&logo=github)](https://github.com/sajibdebnath)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/sajibdebnath)
[![Email](https://img.shields.io/badge/Email-Research-red?style=flat-square&logo=gmail)](mailto:research@example.com)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Profile-green?style=flat-square&logo=researchgate)](https://researchgate.net/profile/Sajib-Debnath)

**🔗 Repository**: [hybrid-multiscale-attn-cnn-lstm-load-forecasting](https://github.com/sajibdebnath/hybrid-multiscale-attn-cnn-lstm-load-forecasting)

</div>

---

<div align="center">




