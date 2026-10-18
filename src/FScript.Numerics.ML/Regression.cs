using FScript.Numerics.Lapack;
namespace FScript.Numerics.ML;

public sealed class LinearRegression
{
    private readonly double[] coefficients;
    private LinearRegression(double[] coefficients, double intercept)
    {
        this.coefficients = coefficients;
        Intercept = intercept;
    }
    public int FeatureCount => coefficients.Length;
    public double Intercept
    {
        get;
    }
    public NDArray Coefficients => NDArray.FromArray(coefficients);
    /// <summary>Minimum-norm SVD fit. Ridge penalizes coefficients, never the intercept.</summary>
    public static LinearRegression Fit(NDArray features, NDArray targets, bool fitIntercept = true, double ridge = 0)
    {
        var x = Data.Matrix(features);
        var y = Data.Target(targets, x.Length);
        if (!double.IsFinite(ridge) || ridge < 0)
            throw new ArgumentOutOfRangeException(nameof(ridge));
        int n = x.Length, p = x[0].Length, extra = fitIntercept ? 1 : 0;
        var design = NDArray.Zeros(new Shape(checked(n + (ridge > 0 ? p : 0)), checked(p + extra)));
        var response = NDArray.Zeros(new Shape(design.Shape[0]));
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
                design.Set(x[i][j], i, j);
            if (fitIntercept)
                design.Set(1, i, p);
            response.Set(y[i], i);
        }
        if (ridge > 0)
            for (int j = 0; j < p; j++)
                design.Set(Math.Sqrt(ridge), n + j, j);
        var solution = Decompositions.LeastSquares(design, response).ToArray();
        if (solution.Any(v => !double.IsFinite(v)))
            throw new ArithmeticException("Regression coefficients overflowed; scale inputs.");
        return new(solution.Take(p).ToArray(), fitIntercept ? solution[p] : 0);
    }
    public NDArray Predict(NDArray features)
    {
        var x = Data.Matrix(features, false);
        if (features.Shape[1] != FeatureCount)
            throw new ArgumentException("Feature count mismatch.");
        return NDArray.FromArray(x.Select(row => row.Select((v, j) => v * coefficients[j]).Sum() + Intercept).ToArray());
    }
}
public sealed class LogisticRegression
{
    private readonly double[] coefficients;
    private LogisticRegression(double[] coefficients, double intercept, int iterations, bool converged)
    {
        this.coefficients = coefficients;
        Intercept = intercept;
        Iterations = iterations;
        Converged = converged;
    }
    public int FeatureCount => coefficients.Length;
    public double Intercept
    {
        get;
    }
    public int Iterations
    {
        get;
    }
    public bool Converged
    {
        get;
    }
    public NDArray Coefficients => NDArray.FromArray(coefficients);
    private static double Sigmoid(double z) => z >= 0 ? 1 / (1 + Math.Exp(-z)) : Math.Exp(z) / (1 + Math.Exp(z));
    /// <summary>Full-batch gradient descent on mean binary cross entropy plus ridge/2 * ||w||².</summary>
    public static LogisticRegression Fit(NDArray features, NDArray targets, double learningRate = 0.1, double ridge = 0, int maxIterations = 1000, double tolerance = 1e-8)
    {
        var x = Data.Matrix(features);
        var y = Data.Target(targets, x.Length, true);
        Data.Tolerance(tolerance, maxIterations);
        if (!double.IsFinite(learningRate) || learningRate <= 0 || !double.IsFinite(ridge) || ridge < 0)
            throw new ArgumentException("Learning rate must be positive and ridge nonnegative, both finite.");
        if (!y.Contains(0) || !y.Contains(1))
            throw new ArgumentException("Logistic fitting requires both classes.");
        int p = x[0].Length;
        var weights = new double[p];
        double intercept = 0;
        bool converged = false;
        int iteration = 0;
        for (; iteration < maxIterations; iteration++)
        {
            var gradient = new double[p];
            double bias = 0;
            for (int i = 0; i < x.Length; i++)
            {
                double z = intercept;
                for (int j = 0; j < p; j++)
                    z += weights[j] * x[i][j];
                double error = Sigmoid(z) - y[i];
                bias += error;
                for (int j = 0; j < p; j++)
                    gradient[j] += error * x[i][j];
            }
            bias /= x.Length;
            double norm = bias * bias;
            for (int j = 0; j < p; j++)
            {
                gradient[j] = gradient[j] / x.Length + ridge * weights[j];
                norm += gradient[j] * gradient[j];
                weights[j] -= learningRate * gradient[j];
            }
            intercept -= learningRate * bias;
            if (!double.IsFinite(norm) || !double.IsFinite(intercept) || weights.Any(v => !double.IsFinite(v)))
                throw new ArithmeticException("Logistic optimization diverged; scale features or lower the learning rate.");
            if (Math.Sqrt(norm) <= tolerance)
            {
                converged = true;
                iteration++;
                break;
            }
        }
        return new(weights, intercept, iteration, converged);
    }
    public NDArray PredictProbability(NDArray features)
    {
        var x = Data.Matrix(features, false);
        if (features.Shape[1] != FeatureCount)
            throw new ArgumentException("Feature count mismatch.");
        return NDArray.FromArray(x.Select(row => Sigmoid(row.Select((v, j) => v * coefficients[j]).Sum() + Intercept)).ToArray());
    }
    public NDArray Predict(NDArray features) => NDArray.FromInt64(PredictProbability(features).ToArray().Select(v => v >= 0.5 ? 1L : 0L).ToArray());
}
