using MlpChurn.Common;

namespace MlpChurn.Data;

/// <summary>Shuffled train/test split.</summary>
public static class DatasetSplitter
{
    public static (Dataset Train, Dataset Test) Split(Dataset data, double testRatio = 0.2, int seed = 42)
    {
        var rng = new Random(seed);
        int[] idx = Enumerable.Range(0, data.Count).OrderBy(_ => rng.Next()).ToArray();
        int testCount = (int)Math.Round(data.Count * testRatio);

        Dataset Build(IEnumerable<int> ids)
        {
            var list = ids.ToArray();
            return new Dataset(
                list.Select(i => data.Features[i]).ToArray(),
                list.Select(i => data.Labels[i]).ToArray(),
                data.FeatureNames);
        }

        return (Build(idx.Skip(testCount)), Build(idx.Take(testCount)));
    }
}
