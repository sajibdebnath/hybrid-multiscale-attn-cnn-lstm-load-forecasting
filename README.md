# Hybrid Multiscale Attention CNN-LSTM Load Forecasting

A comprehensive deep learning project for electricity load forecasting using ERCOT (Electric Reliability Council of Texas) data. This project implements and compares multiple neural network architectures, with a focus on a novel hybrid model combining CNN, LSTM, and attention mechanisms.

## 🎯 Project Overview

This project aims to predict electricity demand using historical load data and meteorological variables. The proposed hybrid model achieves state-of-the-art performance with an R² of 0.9677 and MAPE of 2.53%.

## 📊 Dataset

### Data Sources
- **Load Data**: ERCOT electricity demand (2018-2024, hourly resolution)
  - **Data source**: https://www.ercot.com/gridinfo/load/load_hist
- **Weather Data**: Meteorological data from 3 ASOS stations (BKS, JDD, TME)
  - **Data source**: https://mesonet.agron.iastate.edu/request/download.phtml?network=TX_ASOS#
    - Temperature (°C)
    - Relative humidity (%)
    - Wind speed (m/s)
    - Feels-like temperature (°C)
    - Precipitation (mm)
    - etc.,

### Features
- **Target Variable**: ERCOT electricity demand (MW)
- **Weather Features**: Current and daily max/min temperatures
- **Temporal Features**: Month, weekday, and holiday labels
- **Time Window**: 24-hour lookback for sequence modeling

## 🏗️ Model Architectures

### 1. LSTM Model
- Basic LSTM with 2 layers (4→2 units)
- Dropout for regularization
- Simple baseline model

### 2. CNN-LSTM Model
- Conv1D layers for feature extraction
- Bidirectional LSTM for temporal modeling
- Batch normalization and spatial dropout

### 3. Attention-based LSTM
- LSTM layers with custom attention mechanism
- Focus on important temporal patterns
- L2 regularization

### 4. Hybrid CNN-LSTM-Attention (Proposed)
- **Conv1D layers**: Extract local patterns
- **Bidirectional LSTM**: Model temporal dependencies
- **Multi-head attention**: Focus on relevant features
- **Layer normalization**: Stabilize training
- **Residual connections**: Improve gradient flow

## 📈 Results

| Model | R² | MAE | RMSE | MAPE |
|-------|----|----|----|----|
| LSTM Model | 0.9042 | 2393.47 | 3298.21 | 3.84% |
| CNN-LSTM Model | 0.9380 | 1984.19 | 2654.50 | 3.51% |
| Attention-based LSTM | 0.9270 | 2193.70 | 2878.90 | 3.92% |
| **Hybrid CNN-LSTM-Attention** | **0.9677** | **1430.55** | **1915.17** | **2.53%** |

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn holidays
```

### Data Structure
```
Data/
├── Final_dataset_ERCOT_v2.csv          # Processed dataset
├── Load data/                          # Raw ERCOT load data
│   ├── Native_Load_2018.xlsx
│   ├── Native_Load_2019.xlsx
│   └── ...
└── Weather data/                       # Meteorological data
    ├── asos_hourly_BKS.csv
    ├── asos_hourly_JDD.csv
    └── asos_hourly_TME.csv
```

### Usage

1. **Data Preprocessing**:
   ```python
   # Run ERCOT_data_cleaning.ipynb for data integration
   # Run data_analytic_and_preprocessing.ipynb for EDA
   ```

2. **Model Training**:
   ```python
   # Run Notebook/updated_EROCT.ipynb for model training
   # All models will be trained and compared automatically
   ```

## 📁 Project Structure

```
├── Data/                               # Dataset files
│   ├── Final_dataset_ERCOT_v2.csv     # Main processed dataset
│   ├── Load data/                     # Raw load data (2018-2024)
│   └── Weather data/                  # Meteorological data
├── Notebook/                          # Main training notebook
│   └── updated_EROCT.ipynb           # Model training and evaluation
├── data_analytic_and_preprocessing.ipynb  # EDA and preprocessing
├── ERCOT_data_cleaning.ipynb         # Data cleaning pipeline
└── README.md                         # This file
```

## 🔬 Key Features

### Data Analysis
- **Time Series Analysis**: FFT analysis revealing daily and seasonal patterns
- **Correlation Analysis**: Pearson and Spearman correlations between features
- **Extreme Event Detection**: Hampel filter for outlier identification
- **Statistical Analysis**: Daily, weekly, monthly aggregations

### Model Features
- **Reproducible Results**: Fixed random seeds for consistent outputs
- **Early Stopping**: Prevents overfitting with patience=10
- **Regularization**: L2 regularization and dropout layers
- **Attention Mechanism**: Custom attention layer for temporal focus

### Visualization
- Temperature-demand relationship plots
- Seasonal pattern analysis
- Model performance comparisons
- Training history visualization

## 🎯 Key Insights

1. **Strong Temperature Correlation**: 0.55 correlation between temperature and electricity demand
2. **Seasonal Patterns**: Clear daily and seasonal cycles in demand
3. **Hybrid Superiority**: CNN-LSTM-Attention model significantly outperforms individual components
4. **Attention Benefits**: Multi-head attention helps focus on relevant temporal patterns
5. **Feature Engineering**: Weather and temporal features improve prediction accuracy

## 🛠️ Technical Details

### Data Preprocessing
- MinMax normalization (0-1 scaling)
- 24-hour lookback window for sequence creation
- Train/Validation/Test split (80%/10%/10%)
- Feature engineering with temporal and weather variables

### Training Configuration
- **Optimizer**: Adam with different learning rates per model
- **Loss Function**: Mean Squared Error
- **Metrics**: MAE, MAPE, R²
- **Batch Size**: 64
- **Epochs**: 100 (with early stopping)

### Model Architecture Details
```python
# Hybrid CNN-LSTM-Attention Model
Conv1D(64, 3) → BatchNorm → Conv1D(64, 3) → BatchNorm
→ MaxPooling1D(2) → SpatialDropout1D(0.2)
→ Bidirectional(LSTM(128)) → LSTM(64)
→ MultiHeadAttention(4 heads) → LayerNorm
→ GlobalAveragePooling1D → Dropout(0.3) → Dense(1)
```

## 📊 Performance Analysis

The hybrid model achieves:
- **97% accuracy** (R² = 0.9677)
- **Low error rates**: MAE = 1430.55 MW, MAPE = 2.53%
- **Robust predictions**: RMSE = 1915.17 MW
- **Superior performance** compared to baseline models

## 🔮 Future Work

- [ ] Multi-step ahead forecasting
- [ ] Real-time prediction system
- [ ] Integration with renewable energy sources
- [ ] Uncertainty quantification
- [ ] Model interpretability analysis

## 📚 References

- ERCOT (Electric Reliability Council of Texas) data
- ASOS (Automated Surface Observing System) weather data
- Deep learning architectures for time series forecasting
- Attention mechanisms in neural networks

## 👥 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is for research and educational purposes. Please ensure proper attribution when using this code or data.

---

**Note**: This project demonstrates advanced deep learning techniques for electricity load forecasting and can serve as a foundation for similar time series prediction tasks in the energy sector.

