using FScript.Numerics.Statistics;
namespace FScript.Numerics.ML;

internal static class Data
{
    internal static double[][] Matrix(NDArray array, bool nonempty = true)
    {
        ArgumentNullException.ThrowIfNull(array);
        if (array.Rank != 2 || array.DType is not (DType.Float32 or DType.Float64) || (nonempty && (array.Shape[0] == 0 || array.Shape[1] == 0)))
            throw new ArgumentException("Expected a nonempty samples-by-features float32/float64 matrix.");
        var rows = array.ToRows();
        if (rows.Any(row => row.Any(x => !double.IsFinite(x))))
            throw new ArgumentException("ML inputs must be finite; missing values are not supported.");
        return rows;
    }
    internal static double[] Target(NDArray target, int samples, bool binary = false)
    {
        ArgumentNullException.ThrowIfNull(target);
        if (target.Rank != 1 || target.Size != samples || target.DType is not (DType.Float32 or DType.Float64 or DType.Int32 or DType.Int64 or DType.Bool))
            throw new ArgumentException("Targets must be a vector matching the sample count.");
        var values = target.ToArray();
        if (values.Any(x => !double.IsFinite(x) || (binary && x != 0 && x != 1)))
            throw new ArgumentException(binary ? "Binary targets must contain only 0 and 1." : "Targets must be finite.");
        return values;
    }
    internal static void Features(double[][] rows, int count)
    {
        if (rows.Any(row => row.Length != count))
            throw new ArgumentException($"Model requires {count} features.");
    }
    internal static void Tolerance(double tolerance, int iterations)
    {
        if (!double.IsFinite(tolerance) || tolerance <= 0 || iterations <= 0)
            throw new ArgumentException("Tolerance and iteration limit must be positive and finite.");
    }
}

public sealed record SplitResult(NDArray TrainFeatures, NDArray TestFeatures, NDArray TrainTargets, NDArray TestTargets);
public static class Preprocessing
{
    public static SplitResult TrainTestSplit(NDArray features, NDArray targets, double testFraction, long seed)
    {
        var x = Data.Matrix(features);
        Data.Target(targets, x.Length);
        if (!double.IsFinite(testFraction) || testFraction <= 0 || testFraction >= 1 || x.Length < 2)
            throw new ArgumentException("Split requires at least two samples and a test fraction strictly between zero and one.");
        int test = (int)Math.Ceiling(x.Length * testFraction);
        if (test >= x.Length)
            throw new ArgumentException("Split leaves no training samples.");
        var indices = Enumerable.Range(0, x.Length).ToArray();
        var rng = new SeededRandom(seed);
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = rng.NextInt(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
        NDArray Rows(int start, int count) => NDArray.FromRows(indices.Skip(start).Take(count).Select(i => x[i]).ToArray()).Astype(features.DType);
        NDArray Targets(int start, int count)
        {
            var selected = indices.Skip(start).Take(count).ToArray();
            return targets.DType switch
            {
                DType.Int64 => NDArray.FromInt64(selected.Select(i => targets.GetInt64(i)).ToArray()),
                DType.Int32 => NDArray.FromInt32(selected.Select(i => checked((int)targets.GetInt64(i))).ToArray()),
                DType.Bool => NDArray.FromBooleans(selected.Select(i => targets.GetBoolean(i)).ToArray()),
                _ => NDArray.FromArray(selected.Select(i => targets.Get(i)).ToArray(), targets.DType)
            };
        }
        return new(Rows(test, x.Length - test), Rows(0, test), Targets(test, x.Length - test), Targets(0, test));
    }
}

public sealed class StandardScaler
{
    private readonly double[] mean, scale;
    private StandardScaler(double[] mean, double[] scale)
    {
        this.mean = mean;
        this.scale = scale;
    }
    public int FeatureCount => mean.Length;
    public NDArray Mean => NDArray.FromArray(mean);
    public NDArray Scale => NDArray.FromArray(scale);
    public static StandardScaler Fit(NDArray features)
    {
        var x = Data.Matrix(features);
        int p = x[0].Length;
        var mean = new double[p];
        var scale = new double[p];
        for (int j = 0; j < p; j++)
        {
            mean[j] = x.Sum(row => row[j]) / x.Length;
            scale[j] = Math.Sqrt(x.Sum(row => (row[j] - mean[j]) * (row[j] - mean[j])) / x.Length);
            if (scale[j] == 0)
                scale[j] = 1;
        }
        if (mean.Any(v => !double.IsFinite(v)) || scale.Any(v => !double.IsFinite(v)))
            throw new ArithmeticException("Scaler statistics overflowed; rescale inputs first.");
        return new(mean, scale);
    }
    public NDArray Transform(NDArray features)
    {
        var x = Data.Matrix(features, false);
        Data.Features(x, FeatureCount);
        if (features.Shape[1] != FeatureCount)
            throw new ArgumentException("Feature count mismatch.");
        return NDArray.FromArray(x.SelectMany(row => row.Select((v, j) => (v - mean[j]) / scale[j])).ToArray(), features.Shape);
    }
    public static (StandardScaler Model, NDArray Values) FitTransform(NDArray features)
    {
        var model = Fit(features);
        return (model, model.Transform(features));
    }
}
public sealed class MinMaxScaler
{
    private readonly double[] minimum, range;
    private MinMaxScaler(double[] minimum, double[] range)
    {
        this.minimum = minimum;
        this.range = range;
    }
    public int FeatureCount => minimum.Length;
    public NDArray Minimum => NDArray.FromArray(minimum);
    public NDArray Range => NDArray.FromArray(range);
    public static MinMaxScaler Fit(NDArray features)
    {
        var x = Data.Matrix(features);
        int p = x[0].Length;
        var min = new double[p];
        var range = new double[p];
        for (int j = 0; j < p; j++)
        {
            min[j] = x.Min(row => row[j]);
            range[j] = x.Max(row => row[j]) - min[j];
            if (range[j] == 0)
                range[j] = 1;
        }
        if (range.Any(v => !double.IsFinite(v)))
            throw new ArithmeticException("Scaler range overflowed; rescale inputs first.");
        return new(min, range);
    }
    public NDArray Transform(NDArray features)
    {
        var x = Data.Matrix(features, false);
        if (features.Shape[1] != FeatureCount)
            throw new ArgumentException("Feature count mismatch.");
        return NDArray.FromArray(x.SelectMany(row => row.Select((v, j) => (v - minimum[j]) / range[j])).ToArray(), features.Shape);
    }
    public static (MinMaxScaler Model, NDArray Values) FitTransform(NDArray features)
    {
        var model = Fit(features);
        return (model, model.Transform(features));
    }
}
