using MlpChurn.Common;
using MlpChurn.Evaluation;
using MlpChurn.Network;

namespace MlpChurn.Training;

public sealed record TrainingConfig(int Epochs = 1000, double LearningRate = 0.1, int LogEvery = 100, int Seed = 7);

/// <summary>Main training loop with console + CSV logging.</summary>
public sealed class Trainer
{
    private readonly NeuralNetwork _net;
    private readonly TrainingConfig _cfg;

    public Trainer(NeuralNetwork net, TrainingConfig cfg)
    {
        _net = net;
        _cfg = cfg;
    }

    /// <param name="logPath">Optional CSV path for epoch,train_loss,test_loss.</param>
    /// <param name="verbose">Print epoch logs to console.</param>
    public double Train(Dataset train, Dataset? test = null, string? logPath = null, bool verbose = true)
    {
        var rng = new Random(_cfg.Seed);
        var order = Enumerable.Range(0, train.Count).ToArray();
        var log = new List<string> { "epoch,train_mse,test_mse" };
        double lastLoss = 0;

        for (int epoch = 1; epoch <= _cfg.Epochs; epoch++)
        {
            // shuffle each epoch for stable SGD
            for (int i = order.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }

            double totalSe = 0;
            foreach (int i in order)
                totalSe += _net.TrainSample(train.Features[i], train.Labels[i], _cfg.LearningRate);
            lastLoss = totalSe / train.Count;

            if (epoch == 1 || epoch % _cfg.LogEvery == 0)
            {
                double testLoss = test is null ? double.NaN : Metrics.Mse(test.Labels, _net.PredictBatch(test.Features));
                if (verbose)
                    Console.WriteLine($"Epoch {epoch,5}/{_cfg.Epochs} | Train MSE: {lastLoss:F6} | Test MSE: {testLoss:F6}");
                log.Add($"{epoch},{lastLoss:F6},{testLoss:F6}");
            }
        }

        if (logPath is not null) File.WriteAllLines(logPath, log);
        return lastLoss;
    }
}
