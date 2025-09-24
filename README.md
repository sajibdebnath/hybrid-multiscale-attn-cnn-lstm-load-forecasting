# ⚡ Hybrid CNN-LSTM-Attention Electricity Load Forecasting

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**🏆 R² = 0.9677 | MAPE = 2.53%**

*Deep learning models for electricity demand forecasting using ERCOT data and weather features*

</div>

---

## 🎯 Overview

Deep learning models for electricity demand forecasting using ERCOT data (2018-2024) and weather features. The hybrid CNN-LSTM-Attention model achieves 96.77% accuracy by combining convolutional layers, LSTM networks, and attention mechanisms.

## 📊 Dataset

- **Load Data**: ERCOT electricity demand (2018-2024, hourly)
- **Weather Data**: 3 ASOS stations (temperature, humidity, wind, pressure)
- **Features**: Temperature, weather variables, temporal patterns, US holidays
- **Final Dataset**: `Final_dataset_ERCOT_v2.csv`


## 📈 Results

| Model | R² Score | MAPE | MAE (MW) | RMSE (MW) |
|-------|----------|------|----------|-----------|
| LSTM | 0.9042 | 3.84% | 2,393.47 | 3,298.21 |
| CNN-LSTM | 0.9380 | 3.51% | 1,984.19 | 2,654.50 |
| Attention-LSTM | 0.9270 | 3.92% | 2,193.70 | 2,878.90 |
| **Hybrid CNN-LSTM-Attention** | **0.9677** | **2.53%** | **1,430.55** | **1,915.17** |


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
├── Data/                               # Dataset files
│   ├── Final_dataset_ERCOT_v2.csv     # Processed dataset
│   ├── Load data/                      # ERCOT load data (2018-2024)
│   └── Weather data/                   # ASOS weather stations data
├── Correlation Analysis/               # Feature analysis
├── Data cleaning/                      # Data integration
├── Data Preprocessing/                 # EDA and feature engineering
└── Notebook/                          # Model training and evaluation
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
