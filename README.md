# Power Prediction System

A comprehensive deep learning system for predicting power consumption using time series data. The system implements and compares three different neural network architectures: LSTM, Transformer, and CNN-Transformer hybrid models.

## Features

- **Multiple Model Architectures**: LSTM, Transformer, and CNN-Transformer hybrid models
- **Comprehensive Data Processing**: Automated feature engineering and data preprocessing
- **Time Series Analysis**: Support for both short-term (90 days) and long-term (365 days) predictions
- **Robust Evaluation**: Multiple experiments with statistical analysis of results
- **Export Results**: Automatic generation of Excel reports with evaluation metrics

## Requirements

```
numpy
pandas
torch
scikit-learn
matplotlib
openpyxl
```

## Dataset

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

## Usage

1. Prepare your data files:
   - `train.csv`: Training dataset
   - `test.csv`: Testing dataset

2. Update the file paths in the script:
```python
train_path = "path/to/your/train.csv"
test_path = "path/to/your/test.csv"
```

3. Run the prediction system:
```bash
python power_prediction_system.py
```

## Data Format

The system expects CSV files with the following columns:
- `Global_active_power`: Target variable for prediction
- `Global_reactive_power`: Reactive power measurement
- `Voltage`: Voltage measurement
- `Global_intensity`: Current intensity
- `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`: Sub-metering measurements
- `RR`, `NBJRR1`, `NBJRR5`, `NBJRR10`, `NBJBROU`: Weather-related features
- `DateTime`: Timestamp (optional, for time-based features)

## Model Architectures

### LSTM Model
- Bidirectional LSTM layers with dropout
- Suitable for capturing long-term dependencies
- Good baseline performance

### Transformer Model
- Multi-head self-attention mechanism
- Positional encoding for sequence understanding
- Excellent for capturing complex patterns

### CNN-Transformer Hybrid
- Combines CNN feature extraction with Transformer attention
- CNN layers for local pattern detection
- Transformer layers for global sequence modeling
- Best overall performance

## Output

The system generates:
1. **Console Output**: Real-time training progress and evaluation metrics
2. **Excel Report** (`evaluation_results.xlsx`): Detailed results for both short-term and long-term predictions
3. **Model Checkpoints**: Best performing models saved automatically

## Evaluation Metrics

- **MSE (Mean Squared Error)**: Primary loss metric
- **MAE (Mean Absolute Error)**: Absolute prediction error
- **STD (Standard Deviation)**: Prediction uncertainty measure

## Configuration

Key parameters can be adjusted:
- `sequence_length`: Input sequence length (default: 60)
- `num_epochs`: Training epochs (default: 30 for quick experiments)
- `batch_size`: Dynamically adjusted based on dataset size
- `learning_rate`: Optimized with OneCycleLR scheduler
