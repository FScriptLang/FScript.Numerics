using FScript.Numerics.Lapack;
using FScript.Numerics.Statistics;
namespace FScript.Numerics.ML;

public sealed class KMeans
{
    private readonly double[][] centers;
    private KMeans(double[][] centers, int iterations, bool converged, double inertia)
    {
        this.centers = centers;
        Iterations = iterations;
        Converged = converged;
        Inertia = inertia;
    }
    public int FeatureCount => centers[0].Length;
    public int ClusterCount => centers.Length;
    public int Iterations
    {
        get;
    }
    public bool Converged
    {
        get;
    }
    public double Inertia
    {
        get;
    }
    public NDArray Centers => NDArray.FromRows(centers);
    private static double Distance(double[] a, double[] b)
    {
        double sum = 0;
        for (int j = 0; j < a.Length; j++)
        {
            double d = a[j] - b[j];
            sum += d * d;
        }
        return sum;
    }
    private static int Nearest(double[] row, double[][] centers)
    {
        int best = 0;
        double distance = Distance(row, centers[0]);
        for (int k = 1; k < centers.Length; k++)
        {
            double candidate = Distance(row, centers[k]);
            if (candidate < distance)
            {
                best = k;
                distance = candidate;
            }
        }
        return best;
    }
    /// <summary>Seeded k-means++ initialization, Lloyd iterations, empty clusters retain their previous centers.</summary>
    public static KMeans Fit(NDArray features, int clusters, long seed, int maxIterations = 300, double tolerance = 1e-6)
    {
        var x = Data.Matrix(features);
        Data.Tolerance(tolerance, maxIterations);
        if (clusters <= 0 || clusters > x.Length)
            throw new ArgumentOutOfRangeException(nameof(clusters));
        var rng = new SeededRandom(seed);
        var centers = new double[clusters][];
        centers[0] = (double[])x[rng.NextInt(x.Length)].Clone();
        for (int k = 1; k < clusters; k++)
        {
            var distances = x.Select(row => centers.Take(k).Min(c => Distance(row, c))).ToArray();
            double total = distances.Sum();
            if (!double.IsFinite(total))
                throw new ArithmeticException("K-means distances overflowed; scale features.");
            double target = rng.NextDouble() * total;
            int chosen = 0;
            while (chosen < x.Length - 1 && (target -= distances[chosen]) > 0)
                chosen++;
            centers[k] = (double[])x[chosen].Clone();
        }
        int iteration = 0;
        bool converged = false;
        for (; iteration < maxIterations; iteration++)
        {
            var sums = Enumerable.Range(0, clusters).Select(_ => new double[x[0].Length]).ToArray();
            var counts = new int[clusters];
            foreach (var row in x)
            {
                int c = Nearest(row, centers);
                counts[c]++;
                for (int j = 0; j < row.Length; j++)
                    sums[c][j] += row[j];
            }
            double shift = 0;
            for (int k = 0; k < clusters; k++)
            {
                if (counts[k] == 0)
                    continue;
                for (int j = 0; j < sums[k].Length; j++)
                    sums[k][j] /= counts[k];
                shift = Math.Max(shift, Distance(centers[k], sums[k]));
                centers[k] = sums[k];
            }
            if (!double.IsFinite(shift))
                throw new ArithmeticException("K-means distances overflowed; scale features.");
            if (Math.Sqrt(shift) <= tolerance)
            {
                converged = true;
                iteration++;
                break;
            }
        }
        double inertia = x.Sum(row => Distance(row, centers[Nearest(row, centers)]));
        if (!double.IsFinite(inertia))
            throw new ArithmeticException("K-means inertia overflowed; scale features.");
        return new(centers, iteration, converged, inertia);
    }
    public NDArray Predict(NDArray features)
    {
        var x = Data.Matrix(features, false);
        if (features.Shape[1] != FeatureCount)
            throw new ArgumentException("Feature count mismatch.");
        return NDArray.FromInt64(x.Select(row => (long)Nearest(row, centers)).ToArray());
    }
}

public sealed class Pca
{
    private readonly double[] mean, variance;
    private readonly double[][] components;
    private Pca(double[] mean, double[][] components, double[] variance)
    {
        this.mean = mean;
        this.components = components;
        this.variance = variance;
    }
    public int FeatureCount => mean.Length;
    public int ComponentCount => components.Length;
    public NDArray Mean => NDArray.FromArray(mean);
    public NDArray Components => NDArray.FromRows(components);
    public NDArray ExplainedVariance => NDArray.FromArray(variance);
    public static Pca Fit(NDArray features, int componentCount)
    {
        var x = Data.Matrix(features);
        int n = x.Length, p = x[0].Length;
        if (n < 2 || componentCount <= 0 || componentCount > Math.Min(n, p))
            throw new ArgumentException("PCA requires at least two samples and 1..min(samples,features) components.");
        var mean = Enumerable.Range(0, p).Select(j => x.Sum(row => row[j]) / n).ToArray();
        var centered = NDArray.FromRows(x.Select(row => row.Select((v, j) => v - mean[j]).ToArray()).ToArray());
        var svd = Decompositions.Svd(centered);
        var components = new double[componentCount][];
        var variance = new double[componentCount];
        for (int k = 0; k < componentCount; k++)
        {
            components[k] = Enumerable.Range(0, p).Select(j => svd.VT.Get(k, j)).ToArray();
            int largest = Enumerable.Range(0, p).MaxBy(j => Math.Abs(components[k][j]));
            if (components[k][largest] < 0)
                for (int j = 0; j < p; j++)
                    components[k][j] = -components[k][j];
            variance[k] = Math.Pow(svd.SingularValues.Get(k), 2) / (n - 1);
        }
        if (variance.Any(v => !double.IsFinite(v)))
            throw new ArithmeticException("PCA variance overflowed; scale features.");
        return new(mean, components, variance);
    }
    public NDArray Transform(NDArray features)
    {
        var x = Data.Matrix(features, false);
        if (features.Shape[1] != FeatureCount)
            throw new ArgumentException("Feature count mismatch.");
        var output = new double[new Shape(x.Length, ComponentCount).Size];
        for (int i = 0; i < x.Length; i++)
            for (int k = 0; k < ComponentCount; k++)
                for (int j = 0; j < FeatureCount; j++)
                    output[i * ComponentCount + k] += (x[i][j] - mean[j]) * components[k][j];
        return NDArray.FromArray(output, new Shape(x.Length, ComponentCount));
    }
    public static (Pca Model, NDArray Values) FitTransform(NDArray features, int components)
    {
        var model = Fit(features, components);
        return (model, model.Transform(features));
    }
}
