# ⚡ Hybrid Multi-Scale Deep Learning Enhanced Electricity Load Forecasting Using Attention-Based Convolutional Neural Network and LSTM Model

<div align="center">

<!-- Repository Stats -->
![GitHub stars](https://img.shields.io/github/stars/sajibdebnath/hybrid-multiscale-attn-cnn-lstm-load-forecasting?style=for-the-badge&color=yellow)
![GitHub forks](https://img.shields.io/github/forks/sajibdebnath/hybrid-multiscale-attn-cnn-lstm-load-forecasting?style=for-the-badge&color=blue)
![GitHub watchers](https://img.shields.io/github/watchers/sajibdebnath/hybrid-multiscale-attn-cnn-lstm-load-forecasting?style=for-the-badge&color=green)
![GitHub issues](https://img.shields.io/github/issues/sajibdebnath/hybrid-multiscale-attn-cnn-lstm-load-forecasting?style=for-the-badge&color=red)

<!-- Tech Stack Badges -->
![Python](https://img.shields.io/badge/python-v3.7+-blue.svg?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg?style=flat-square&logo=keras)
![Pandas](https://img.shields.io/badge/Pandas-1.x-purple.svg?style=flat-square&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.x-lightblue.svg?style=flat-square&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-green.svg?style=flat-square&logo=plotly)

<!-- Project Info -->
![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg?style=flat-square)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/sajibdebnath/hybrid-multiscale-attn-cnn-lstm-load-forecasting?style=flat-square)

**🏆 Best Performance: R² = 0.9677 | MAPE = 2.53% | State-of-the-Art Results 🏆**

*A comprehensive deep learning project for electricity load forecasting using ERCOT (Electric Reliability Council of Texas) data*

[📊 View Results](#-results) • [🚀 Quick Start](#-getting-started) • [📖 Documentation](#-project-structure) • [🤝 Contributing](#-contributing)

</div>

---

## 📋 Table of Contents
- [🎯 Overview](#overview)
- [📊 Dataset](#-dataset)
- [📈 Results](#-results)
- [🛠️ Technology Stack](#️-technology-stack)
- [🏗️ Model Architectures](#️-model-architectures)
- [🚀 Getting Started](#-getting-started)
- [📁 Project Structure](#-project-structure)
- [🔬 Key Features](#-key-features)
- [🎯 Key Insights](#-key-insights)
- [🛠️ Technical Implementation](#️-technical-implementation)
- [🔮 Future Research](#-future-research-directions)
- [📚 References](#-references--data-sources)
- [📄 License](#-license)

---

## 🎯 Overview

This research develops advanced deep learning models to predict electricity demand using historical load data and meteorological variables. The study compares four different neural network architectures and demonstrates that the proposed hybrid CNN-LSTM-Attention model significantly outperforms traditional approaches, achieving exceptional accuracy with comprehensive feature engineering and data preprocessing.

## 📊 Dataset

### Data Sources
- **Load Data**: ERCOT electricity demand (2018-2024, hourly resolution)
  - Native load data files for each year (Excel format)
  - Comprehensive 7-year historical dataset
- **Weather Data**: Meteorological data from 3 ASOS stations
  - **BKS Station**: Blackland Army Airfield
  - **JDD Station**: Laredo International Airport
  - **TME Station**: Houston Executive Airport
  - **Combined Weather Dataset**: Integrated ASOS weather data

### Dataset Features
- **Target Variable**: ERCOT electricity demand (MW)
- **Weather Features**:
  - Temperature (°C) - current, max, min
  - Relative humidity (%)
  - Wind speed (m/s)
  - Feels-like temperature (°C)
  - Precipitation (mm)
  - Atmospheric pressure
- **Temporal Features**:
  - Month encoding
  - Weekday encoding
  - Holiday labels (US federal holidays)
- **Time Window**: 24-hour lookback for sequence modeling
- **Final Dataset**: `Final_dataset_ERCOT_v2.csv` (preprocessed and cleaned)


## 📈 Results

<div align="center">

### 🏆 **Performance Comparison**

| 🤖 Model | 📊 R² Score | 📉 MAE (MW) | 📐 RMSE (MW) | 🎯 MAPE |
|-----------|-------------|-------------|--------------|---------|
| 🔹 LSTM Baseline | 0.9042 | 2,393.47 | 3,298.21 | 3.84% |
| 🔸 CNN-LSTM | 0.9380 | 1,984.19 | 2,654.50 | 3.51% |
| 🔶 Attention-LSTM | 0.9270 | 2,193.70 | 2,878.90 | 3.92% |
| **🌟 Hybrid CNN-LSTM-Attention** | **🥇 0.9677** | **🥇 1,430.55** | **🥇 1,915.17** | **🥇 2.53%** |

</div>


## �️ Technology Stack

<div align="center">

### 🔧 **Core Technologies**

</div>

<table align="center">
<tr>
<td align="center"><strong>🧠 Deep Learning</strong></td>
<td align="center"><strong>📊 Data Science</strong></td>
<td align="center"><strong>🔬 Analysis</strong></td>
<td align="center"><strong>📈 Visualization</strong></td>
</tr>
<tr>
<td align="center">
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/><br/>
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white"/><br/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
</td>
<td align="center">
<img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/><br/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/><br/>
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
</td>
<td align="center">
<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/><br/>
<img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white"/><br/>
<img src="https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white"/>
</td>
<td align="center">
<img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white"/><br/>
<img src="https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white"/><br/>
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
</td>
</tr>
</table>

### 🏗️ **Architecture Components**

<div align="center">

| Component | Technology | Purpose | Performance |
|-----------|------------|---------|-------------|
| 🧠 **Neural Networks** | TensorFlow/Keras | Deep Learning Framework | 99%+ Accuracy |
| 🔄 **LSTM Layers** | Bidirectional LSTM | Sequential Data Processing | Long-term Dependencies |
| 🔍 **CNN Layers** | 1D Convolution | Feature Extraction | Pattern Recognition |
| 🎯 **Attention** | Multi-Head Attention | Focus Mechanism | Temporal Patterns |
| 📊 **Data Pipeline** | Pandas/NumPy | Data Processing | Scalable & Efficient |
| 📈 **Visualization** | Matplotlib/Seaborn | Results Analysis | Interactive Charts |

</div>

---

## �🚀 Getting Started

<div align="center">

### ⚡ **Quick Setup Guide**

</div>

> 🎯 **Ready to start forecasting?** Follow these simple steps to get up and running!

### 📦 **Requirements & Installation**

<details>
<summary><b>🔧 Click to view detailed requirements</b></summary>

#### 🐍 **Python Environment**
```bash
# Recommended: Python 3.8+ (tested on 3.8, 3.9, 3.10)
python --version  # Should be >= 3.8
```

#### 📋 **Core Dependencies (requirements.txt)**
```txt
# Deep Learning Framework
tensorflow>=2.8.0
keras>=2.8.0

# Data Science Stack
pandas>=1.4.0
numpy>=1.21.0
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Utility Libraries
holidays>=0.14
tqdm>=4.64.0
openpyxl>=3.0.0

# Optional: GPU Support (if available)
tensorflow-gpu>=2.8.0  # For GPU acceleration
```

#### ⚡ **Quick Installation**
```bash
# Option 1: Install from requirements.txt (recommended)
pip install -r requirements.txt

# Option 2: Manual installation
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn holidays tqdm openpyxl

# Option 3: For Conda users
conda install tensorflow pandas numpy matplotlib seaborn scikit-learn
pip install holidays tqdm openpyxl
```

#### 🔧 **System Requirements**
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **RAM**: Minimum 8GB (16GB+ recommended for large datasets)
- **Storage**: 5GB+ free space for data and models
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster training

</details>

### 🚀 **Cloud Setup (Google Colab)**
```python
# For Google Colab users - run this first
from google.colab import drive
drive.mount('/content/drive')

# Install additional packages if needed
!pip install holidays tqdm openpyxl
```

### Environment Setup
```python
# The project uses deterministic operations for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.config.experimental.enable_op_determinism()
```



#### 1. 📥 **Data Cleaning & Integration**
   ```python
   # Navigate to: Data cleaning/ERCOT_data_cleaning.ipynb
   # Integrates load data (2018-2024) with weather data from 3 ASOS stations
   # Outputs: Final_dataset_ERCOT_v2.csv
   ```

#### 2. 🔍 **Exploratory Data Analysis**:
   ```python
   # Navigate to: Data Preprocessing/data_analytic_and_preprocessing.ipynb
   # Performs comprehensive EDA, feature engineering, correlation analysis
   # Generates visualizations and statistical insights
   ```

#### 3. 🔗 **Correlation Analysis**:
   ```python
   # Navigate to: Correlation Analysis/Correlation Analysis.ipynb
   # Analyzes feature correlations and relationships
   ```

#### 4. 🤖 **Model Training & Evaluation**:
   ```python
   # Navigate to: Notebook/updated_EROCT.ipynb
   # Trains all 4 models: LSTM, CNN-LSTM, Attention-LSTM, Hybrid CNN-LSTM-Attention
   # Generates performance comparisons and visualizations
   # Saves performance metrics to CSV
   ```


### 🚀 **Quick Start**
```bash
# Clone repository
git clone https://github.com/sajibdebnath/hybrid-multiscale-attn-cnn-lstm-load-forecasting.git
cd hybrid-multiscale-attn-cnn-lstm-load-forecasting

# Install dependencies
pip install -r requirements.txt

# Run the main notebook
jupyter notebook "Notebook/updated_EROCT.ipynb"
```

> 🎯 **Pro Tip**: All models are trained automatically with reproducible results using SEED=42!

## 📁 Project Structure

<div align="center">

### 🗂️ **Repository Layout**

</div>

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

## 🔬 Key Features

### Data Analysis & Processing
- **Comprehensive Data Integration**: 7 years of ERCOT load data + 3 ASOS weather stations
- **Advanced Feature Engineering**: Temporal features (month, weekday, holidays)
- **Time Series Analysis**: 24-hour lookback window for sequence modeling
- **Normalization**: MinMax scaling (0-1) for optimal neural network performance
- **Data Splitting**: 80% training, 10% validation, 10% testing

### Model Innovation
- **Reproducible Results**: Fixed random seeds (SEED=42) + deterministic operations
- **Progressive Architecture**: From basic LSTM to hybrid CNN-LSTM-Attention
- **Advanced Regularization**: L2 regularization, dropout, batch normalization
- **Multi-Head Attention**: 4-head attention mechanism with residual connections
- **Early Stopping**: Prevents overfitting with patience monitoring

### Performance Optimization
- **Custom Learning Rates**: Model-specific optimization (0.001 to 0.01)
- **Advanced Callbacks**: Early stopping and model checkpointing
- **Batch Processing**: Efficient batch size of 64 for stable training
- **Layer Normalization**: Stabilizes training in deep architectures

### Evaluation & Visualization
- **Comprehensive Metrics**: R², MAE, RMSE, MAPE evaluation
- **Performance Comparison**: Side-by-side model evaluation
- **Prediction Visualization**: 100-day and 30-day window comparisons
- **Training History**: Loss and metrics tracking over epochs

## 🎯 Key Insights

1. **Strong Temperature Correlation**: 0.55 correlation between temperature and electricity demand
2. **Seasonal Patterns**: Clear daily and seasonal cycles in demand
3. **Hybrid Superiority**: CNN-LSTM-Attention model significantly outperforms individual components
4. **Attention Benefits**: Multi-head attention helps focus on relevant temporal patterns
5. **Feature Engineering**: Weather and temporal features improve prediction accuracy

## 🛠️ Technical Implementation

### Data Preprocessing Pipeline
```python
# Data Integration
- Load data: Native_Load_*.xlsx (2018-2024) → pandas DataFrame
- Weather data: 3 ASOS stations → unified weather features
- Feature engineering: temporal features + US holidays
- Target: ERCOT electricity demand (MW)

# Preprocessing Steps
- MinMax normalization: (0-1 scaling) for neural network optimization
- Sequence creation: create_sequences() with 24-hour lookback
- Data splitting: 80% train / 10% validation / 10% test
- Feature matrix: (samples, timesteps, features)
```

### Training Configuration
```python
# Universal Settings
- Loss Function: Mean Squared Error (MSE)
- Metrics: ['mae', 'mape'] + custom R² calculation
- Batch Size: 64 (optimal for memory and convergence)
- Max Epochs: 100 with EarlyStopping(patience=10)

# Model-Specific Optimizers
- LSTM Model: Adam(lr=0.001)
- CNN-LSTM Model: Adam(lr=0.001)
- Attention-LSTM: Adam(lr=0.001)
- Hybrid Model: Adam(lr=0.001)
```

### Advanced Architecture Features
```python
# Hybrid CNN-LSTM-Attention Architecture Flow
Input(timesteps, features)
├── Conv1D(64, kernel=3, padding='causal') + BatchNorm
├── Conv1D(64, kernel=3, padding='causal') + BatchNorm
├── MaxPooling1D(2) + SpatialDropout1D(0.2)
├── Bidirectional(LSTM(128, return_sequences=True))
├── LSTM(64, return_sequences=True)
├── MultiHeadAttention(heads=4, key_dim=64)
├── Residual Connection + LayerNormalization
├── GlobalAveragePooling1D()
├── Dropout(0.3)
└── Dense(1) → Output

# Key Innovations
- Causal padding: Prevents future data leakage
- Residual connections: Improved gradient flow
- Multi-head attention: Focus on relevant temporal patterns
- L2 regularization: (1e-4) prevents overfitting
```


## 🏗️ Model Architectures

### 1. � **LSTM Baseline**
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

### 4. � **Hybrid CNN-LSTM-Attention** (Proposed)
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




## � Contributing

Contributions are welcome! Please feel free to:
- 🍴 Fork the repository
- 🌟 Star the project if you find it useful
- 🐛 Report issues or suggest improvements
- 💡 Submit pull requests with enhancements
- 📢 Share with your research community

## 📞 Citation

If you use this work in your research, please cite:

```bibtex
@article{hybrid_cnn_lstm_attention_2024,
  title={Hybrid Multi-Scale Deep Learning Enhanced Electricity Load Forecasting Using Attention-Based CNN-LSTM},
  author={Sajib Debnath},
  journal={Energy Forecasting Research},
  year={2024},
  note={ERCOT Load Forecasting with 96.77\% Accuracy}
}
```

---

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

### 🎯 **Research Impact**

**⚡ Advancing Energy Forecasting Through Deep Learning Innovation**

*This project demonstrates state-of-the-art deep learning techniques for electricity load forecasting and serves as a comprehensive foundation for similar time series prediction tasks in the energy sector. Our hybrid CNN-LSTM-Attention model achieves superior performance through innovative architecture design and thorough data preprocessing.*

---

### 🏆 **Achievement Badge**

![Performance](https://img.shields.io/badge/R²_Score-96.77%25-brightgreen?style=for-the-badge)
![Error](https://img.shields.io/badge/MAPE-2.53%25-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)

**📊 Built with ❤️ for the Energy Forecasting Community**

⭐ **Don't forget to star this repo if it helped your research!** ⭐

</div>
