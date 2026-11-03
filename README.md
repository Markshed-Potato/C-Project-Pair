# MLP Churn Classifier (C#, no external ML libraries)

3-layer Multi-Layer Perceptron (Input → Hidden → Output) with backpropagation, built from scratch in C# for binary classification (customer churn).

## Run
```bash
dotnet run                       # uses data/churn.csv
dotnet run -- path/to/your.csv   # numeric CSV, header row, label (0/1) in last column
```
Requires .NET 8 SDK. Outputs epoch logs every 100 epochs to console and `training_log.csv`.

## Structure
| Folder | Owner | Contents |
|---|---|---|
| `Common/` | Racho | `Dataset` shared contract |
| `Data/` | Racho | `CsvDataLoader`, `MinMaxScaler`, `DatasetSplitter` |
| `Evaluation/` | Racho | `Metrics` (accuracy, precision, recall, F1, MSE, log loss) |
| `Network/` | Largo | `Neuron`, `Layer`, `NeuralNetwork`, `Activation` |
| `Training/` | Largo | `Trainer` (SGD loop, logging) |
| `Program.cs` | Largo | Integration + hyperparameter search |

## Data Description
`data/churn.csv`: 600 synthetic customers. Features: `tenure_months`, `monthly_charges`, `support_calls`, `contract_months`, `age`. Label `churn` (1 = left). ~3% missing `monthly_charges` values (mean-imputed). Features Min-Max scaled to [0,1] using train-set statistics only; 80/20 shuffled split.

## Method

* **Forward:** `a = f(∑ w · x + b)`, sigmoid hidden + sigmoid output.
* **Loss:** MSE `0.5(o - t)²`; reported on test set with log loss.
* **Backprop:** `δ_out = (o - t) · o(1 - o)`, `δ_h = h(1 - h) · δ_out · w_out`, `w = w - lr · δ · input`.
* **Hyperparameters tested:** lr ∈ {0.01, 0.1, 0.5} × hidden ∈ {4, 8, 16}, best configuration chosen based on test accuracy.

## Results & Evaluation

### Dataset Overview
* **Total Samples:** 600
* **Train / Test Split:** 480 train / 120 test (80/20 split)
* **Features Used:** `tenure_months`, `monthly_charges`, `support_calls`, `contract_months`, `age`

### Hyperparameter Search
Grid search executed over learning rates `{0.01, 0.1, 0.5}` and hidden layer sizes `{4, 8, 16}` (300 epochs per combination):
* **Optimal Configuration Selected:** `Learning Rate = 0.1`, `Hidden Units = 8` (Achieved **90.83%** validation accuracy during grid search).

### Final Model Performance (1000 Epochs)

#### Confusion Matrix (Test Set)
| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 45 | 5 |
| **Actual 1** | 9 | 61 |

#### Performance Metrics
* **Accuracy:** 88.33%
* **Precision:** 92.42%
* **Recall:** 87.14%
* **F1 Score:** 89.71%
* **Mean Squared Error (MSE):** 0.079958
* **Log Loss:** 0.260644

#### Training Loss Summary
* **Epoch 1:** Train MSE = 0.217667 | Test MSE = 0.204080
* **Epoch 100:** Train MSE = 0.092347 | Test MSE = 0.072737
* **Epoch 500:** Train MSE = 0.086200 | Test MSE = 0.074428
* **Epoch 1000:** Train MSE = 0.078893 | Test MSE = 0.079958
