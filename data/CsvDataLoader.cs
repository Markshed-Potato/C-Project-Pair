using System.Globalization;
using MlpChurn.Common;

namespace MlpChurn.Data;

/// <summary>Parses a numeric CSV (header row, label in the last column by default).</summary>
public static class CsvDataLoader
{
    public static Dataset Load(string path, bool hasHeader = true, int labelColumn = -1)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"CSV file not found: {path}");

        var lines = File.ReadAllLines(path).Where(l => !string.IsNullOrWhiteSpace(l)).ToList();
        if (lines.Count == 0) throw new InvalidDataException("CSV is empty.");

        string[] header = lines[0].Split(',').Select(h => h.Trim()).ToArray();
        int columns = header.Length;
        int labelIdx = labelColumn < 0 ? columns - 1 : labelColumn;
        if (hasHeader) lines.RemoveAt(0);

        var featureNames = header.Where((_, i) => i != labelIdx).ToArray();
        var rows = new List<double[]>();
        var labels = new List<double>();

        foreach (var line in lines)
        {
            var parts = line.Split(',');
            if (parts.Length != columns) continue; // skip malformed row

            if (!TryParse(parts[labelIdx], out double label)) continue; // unusable without label

            var row = new double[columns - 1];
            int c = 0;
            for (int i = 0; i < columns; i++)
            {
                if (i == labelIdx) continue;
                row[c++] = TryParse(parts[i], out double v) ? v : double.NaN; // missing -> NaN
            }
            rows.Add(row);
            labels.Add(label);
        }

        ImputeMissingWithMean(rows);
        return new Dataset(rows.ToArray(), labels.ToArray(), featureNames);
    }

    private static bool TryParse(string s, out double value) =>
        double.TryParse(s.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out value);

    /// <summary>Replaces NaN values with the column mean.</summary>
    private static void ImputeMissingWithMean(List<double[]> rows)
    {
        if (rows.Count == 0) return;
        int f = rows[0].Length;
        for (int j = 0; j < f; j++)
        {
            var valid = rows.Select(r => r[j]).Where(v => !double.IsNaN(v)).ToList();
            double mean = valid.Count > 0 ? valid.Average() : 0.0;
            foreach (var r in rows)
                if (double.IsNaN(r[j])) r[j] = mean;
        }
    }
}
