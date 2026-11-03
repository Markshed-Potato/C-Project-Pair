namespace MlpChurn.Common;

/// <summary>
/// Shared contract between the data module (Person A) and the engine (Person B).
/// Features: [sample][feature]. Labels: 0.0 or 1.0 per sample.
/// </summary>
public sealed class Dataset
{
    public double[][] Features { get; }
    public double[] Labels { get; }
    public string[] FeatureNames { get; }
    public int Count => Features.Length;
    public int FeatureCount => Features.Length == 0 ? 0 : Features[0].Length;

    public Dataset(double[][] features, double[] labels, string[] featureNames)
    {
        if (features.Length != labels.Length)
            throw new ArgumentException("Features and labels must have the same length.");
        Features = features;
        Labels = labels;
        FeatureNames = featureNames;
    }
}
