using MlpChurn.Data;
using MlpChurn.Evaluation;
using MlpChurn.Network;
using MlpChurn.Training;

// ---- 1. Load, split, scale (Person A) ----
string csvPath = args.Length > 0 ? args[0] : Path.Combine(AppContext.BaseDirectory, "data", "churn.csv");
var raw = CsvDataLoader.Load(csvPath);
var (trainRaw, testRaw) = DatasetSplitter.Split(raw, testRatio: 0.2, seed: 42);

var scaler = new MinMaxScaler();
scaler.Fit(trainRaw.Features); // fit on train only -> no data leakage
var train = new MlpChurn.Common.Dataset(scaler.Transform(trainRaw.Features), trainRaw.Labels, raw.FeatureNames);
var test = new MlpChurn.Common.Dataset(scaler.Transform(testRaw.Features), testRaw.Labels, raw.FeatureNames);

Console.WriteLine($"Samples: {raw.Count} | Train: {train.Count} | Test: {test.Count} | Features: {raw.FeatureCount}");
Console.WriteLine($"Features: {string.Join(", ", raw.FeatureNames)}\n");

// ---- 2. Hyperparameter experiments (Person B) ----
Console.WriteLine("=== Hyperparameter Search (quick, 300 epochs) ===");
double bestAcc = -1; double bestLr = 0.1; int bestHidden = 8;
foreach (double lr in new[] { 0.01, 0.1, 0.5 })
foreach (int hidden in new[] { 4, 8, 16 })
{
    var n = new NeuralNetwork(train.FeatureCount, hidden);
    new Trainer(n, new TrainingConfig(300, lr)).Train(train, test, verbose: false);
    var r = Metrics.Evaluate(test.Labels, n.PredictBatch(test.Features));
    Console.WriteLine($"lr={lr,-5} hidden={hidden,-3} -> acc={r.Accuracy:P2} mse={r.Mse:F4}");
    if (r.Accuracy > bestAcc) { bestAcc = r.Accuracy; bestLr = lr; bestHidden = hidden; }
}
Console.WriteLine($"Best: lr={bestLr}, hidden={bestHidden}\n");

// ---- 3. Final training with best config ----
Console.WriteLine("=== Final Training ===");
var net = new NeuralNetwork(train.FeatureCount, bestHidden);
new Trainer(net, new TrainingConfig(Epochs: 1000, LearningRate: bestLr, LogEvery: 100))
    .Train(train, test, logPath: "training_log.csv");

// ---- 4. Evaluation (Person A) ----
Console.WriteLine("\n=== Test Set Evaluation ===");
Console.WriteLine(Metrics.Evaluate(test.Labels, net.PredictBatch(test.Features)));
Console.WriteLine("\nLog saved to training_log.csv");
