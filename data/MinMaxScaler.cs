namespace MlpChurn.Data;

/// <summary>Min-Max scaling to [0,1]. Fit on TRAIN only, then transform train and test.</summary>
public sealed class MinMaxScaler
{
    private double[] _min = Array.Empty<double>();
    private double[] _max = Array.Empty<double>();

    public void Fit(double[][] data)
    {
        int f = data[0].Length;
        _min = new double[f];
        _max = new double[f];
        for (int j = 0; j < f; j++)
        {
            _min[j] = data.Min(r => r[j]);
            _max[j] = data.Max(r => r[j]);
        }
    }

    public double[][] Transform(double[][] data)
    {
        if (_min.Length == 0) throw new InvalidOperationException("Call Fit() first.");
        return data.Select(row =>
        {
            var scaled = new double[row.Length];
            for (int j = 0; j < row.Length; j++)
            {
                double range = _max[j] - _min[j];
                scaled[j] = range == 0 ? 0.0 : Math.Clamp((row[j] - _min[j]) / range, 0.0, 1.0);
            }
            return scaled;
        }).ToArray();
    }
}
