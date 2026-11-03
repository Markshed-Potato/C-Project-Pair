namespace MlpChurn.Evaluation;

public sealed record ClassificationReport(
    int TP, int FP, int TN, int FN,
    double Accuracy, double Precision, double Recall, double F1,
    double Mse, double CrossEntropy)
{
    public override string ToString() =>
        $"Confusion Matrix:\n" +
        $"              Pred 0   Pred 1\n" +
        $"  Actual 0    {TN,6}   {FP,6}\n" +
        $"  Actual 1    {FN,6}   {TP,6}\n" +
        $"Accuracy : {Accuracy:P2}\n" +
        $"Precision: {Precision:P2}\n" +
        $"Recall   : {Recall:P2}\n" +
        $"F1 Score : {F1:P2}\n" +
        $"MSE      : {Mse:F6}\n" +
        $"Log Loss : {CrossEntropy:F6}";
}

public static class Metrics
{
    public static ClassificationReport Evaluate(double[] actual, double[] predictedProb, double threshold = 0.5)
    {
        int tp = 0, fp = 0, tn = 0, fn = 0;
        double se = 0, ce = 0;
        const double eps = 1e-12;

        for (int i = 0; i < actual.Length; i++)
        {
            int pred = predictedProb[i] >= threshold ? 1 : 0;
            int act = (int)actual[i];
            if (pred == 1 && act == 1) tp++;
            else if (pred == 1 && act == 0) fp++;
            else if (pred == 0 && act == 0) tn++;
            else fn++;

            double p = Math.Clamp(predictedProb[i], eps, 1 - eps);
            se += Math.Pow(actual[i] - predictedProb[i], 2);
            ce += -(actual[i] * Math.Log(p) + (1 - actual[i]) * Math.Log(1 - p));
        }

        int n = actual.Length;
        double acc = n == 0 ? 0 : (double)(tp + tn) / n;
        double prec = tp + fp == 0 ? 0 : (double)tp / (tp + fp);
        double rec = tp + fn == 0 ? 0 : (double)tp / (tp + fn);
        double f1 = prec + rec == 0 ? 0 : 2 * prec * rec / (prec + rec);
        return new ClassificationReport(tp, fp, tn, fn, acc, prec, rec, f1, se / n, ce / n);
    }

    public static double Mse(double[] actual, double[] predicted) =>
        actual.Zip(predicted, (a, p) => (a - p) * (a - p)).Average();
}
