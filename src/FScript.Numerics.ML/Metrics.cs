namespace FScript.Numerics.ML;

public static class Metrics
{
    private static (double[] Actual, double[] Predicted) Pair(NDArray actual, NDArray predicted)
    {
        ArgumentNullException.ThrowIfNull(actual);
        ArgumentNullException.ThrowIfNull(predicted);
        if (actual.Rank != 1 || predicted.Rank != 1 || actual.Size == 0 || actual.Size != predicted.Size)
            throw new ArgumentException("Metrics require nonempty vectors of equal length.");
        var a = actual.ToArray();
        var b = predicted.ToArray();
        if (a.Concat(b).Any(x => !double.IsFinite(x)))
            throw new ArgumentException("Metrics require finite values.");
        return (a, b);
    }
    public static double Mse(NDArray actual, NDArray predicted)
    {
        var (a, b) = Pair(actual, predicted);
        return a.Select((v, i) => (v - b[i]) * (v - b[i])).Average();
    }
    public static double Mae(NDArray actual, NDArray predicted)
    {
        var (a, b) = Pair(actual, predicted);
        return a.Select((v, i) => Math.Abs(v - b[i])).Average();
    }
    public static double R2(NDArray actual, NDArray predicted)
    {
        var (a, b) = Pair(actual, predicted);
        if (a.Length < 2)
            throw new ArgumentException("R² requires at least two samples.");
        double mean = a.Average(), total = a.Sum(v => (v - mean) * (v - mean)), residual = a.Select((v, i) => (v - b[i]) * (v - b[i])).Sum();
        return total == 0 ? (residual == 0 ? 1 : 0) : 1 - residual / total;
    }
    public static double Accuracy(NDArray actual, NDArray predicted)
    {
        var (a, b) = Pair(actual, predicted);
        Labels(a);
        Labels(b);
        return a.Where((v, i) => v == b[i]).Count() / (double)a.Length;
    }
    private static void Labels(double[] values)
    {
        if (values.Any(x => !double.IsInteger(x) || x < 0 || x > int.MaxValue))
            throw new ArgumentException("Class labels must be nonnegative int32 values.");
    }
    /// <summary>Rows are actual classes; columns are predicted. Explicit classCount retains absent classes.</summary>
    public static NDArray ConfusionMatrix(NDArray actual, NDArray predicted, int classCount)
    {
        var (a, b) = Pair(actual, predicted);
        Labels(a);
        Labels(b);
        if (classCount <= 0 || a.Concat(b).Any(x => x >= classCount))
            throw new ArgumentException("Labels must be below the positive class count.");
        var counts = new long[new Shape(classCount, classCount).Size];
        for (int i = 0; i < a.Length; i++)
        {
            int position = checked((int)a[i] * classCount + (int)b[i]);
            counts[position] = checked(counts[position] + 1);
        }
        return NDArray.FromInt64(counts, new Shape(classCount, classCount));
    }
}
