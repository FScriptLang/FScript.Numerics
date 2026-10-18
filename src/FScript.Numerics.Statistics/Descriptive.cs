namespace FScript.Numerics.Statistics;

public sealed record HistogramResult(NDArray Counts, NDArray Edges);

/// <summary>Descriptive statistics. Quantiles use linear interpolation at (n-1)*p; NaNs propagate.</summary>
public static class Descriptive
{
    public static double Quantile(NDArray array, double probability)
    {
        ArgumentNullException.ThrowIfNull(array);
        if (!double.IsFinite(probability) || probability < 0 || probability > 1)
            throw new ArgumentOutOfRangeException(nameof(probability));
        var values = array.ToArray();
        if (values.Length == 0)
            throw new ArgumentException("Quantiles require at least one value.");
        if (values.Any(double.IsNaN))
            return double.NaN;
        Array.Sort(values);
        double position = (values.Length - 1) * probability;
        int index = (int)position;
        double fraction = position - index;
        if (fraction == 0)
            return values[index];
        if (values[index] == values[index + 1])
            return values[index];
        return (1 - fraction) * values[index] + fraction * values[index + 1];
    }
    public static double Median(NDArray array) => Quantile(array, 0.5);
    public static NDArray QuantileAxis(NDArray array, int axis, double probability, bool keepDimensions = false)
    {
        ArgumentNullException.ThrowIfNull(array);
        if ((uint)axis >= (uint)array.Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));
        if (!double.IsFinite(probability) || probability < 0 || probability > 1)
            throw new ArgumentOutOfRangeException(nameof(probability));
        var dimensions = array.Shape.ToArray().ToList();
        if (keepDimensions)
            dimensions[axis] = 1;
        else
            dimensions.RemoveAt(axis);
        var output = NDArray.Zeros(new Shape(dimensions.ToArray()));
        var input = new int[array.Rank];
        var target = new int[output.Rank];
        var values = new double[array.Shape[axis]];
        for (int n = 0; n < output.Size; n++)
        {
            for (int d = 0, j = 0; d < array.Rank; d++)
                if (d != axis)
                    input[d] = target[keepDimensions ? d : j++];
            for (int k = 0; k < values.Length; k++)
            {
                input[axis] = k;
                values[k] = array.Get(input);
            }
            output.Set(Quantile(NDArray.FromArray(values), probability), target);
            for (int d = target.Length - 1; d >= 0; d--)
            {
                if (++target[d] < output.Shape[d])
                    break;
                target[d] = 0;
            }
        }
        return output;
    }
    /// <summary>Columns are variables, rows are observations; denominator is observations - degreesOfFreedom.</summary>
    public static NDArray Covariance(NDArray observations, int degreesOfFreedom = 1)
    {
        ArgumentNullException.ThrowIfNull(observations);
        if (observations.Rank != 2 || degreesOfFreedom < 0 || observations.Shape[0] <= degreesOfFreedom)
            throw new ArgumentException("Covariance requires a samples-by-features matrix with more rows than degrees of freedom.");
        int n = observations.Shape[0], p = observations.Shape[1];
        var mean = observations.MeanAxis(0);
        var output = NDArray.Zeros(new Shape(p, p));
        for (int i = 0; i < p; i++)
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++)
                    sum += (observations.Get(k, i) - mean.Get(i)) * (observations.Get(k, j) - mean.Get(j));
                double value = sum / (n - degreesOfFreedom);
                output.Set(value, i, j);
                output.Set(value, j, i);
            }
        return output;
    }
    public static NDArray Correlation(NDArray observations)
    {
        var covariance = Covariance(observations);
        int p = covariance.Shape[0];
        var result = NDArray.Zeros(covariance.Shape);
        for (int i = 0; i < p; i++)
            for (int j = 0; j < p; j++)
                result.Set(covariance.Get(i, j) / (Math.Sqrt(covariance.Get(i, i)) * Math.Sqrt(covariance.Get(j, j))), i, j);
        return result;
    }
    /// <summary>Equal-width bins, left-closed/right-open except for the inclusive final edge. Values outside bounds are ignored.</summary>
    public static HistogramResult Histogram(NDArray array, int bins, double minimum, double maximum)
    {
        ArgumentNullException.ThrowIfNull(array);
        if (bins <= 0 || !double.IsFinite(minimum) || !double.IsFinite(maximum) || minimum >= maximum)
            throw new ArgumentException("Histogram requires positive bins and finite increasing bounds.");
        var counts = new long[bins];
        foreach (double x in array.ToArray())
        {
            if (!double.IsFinite(x))
                throw new ArgumentException("Histogram inputs must be finite.");
            if (x < minimum || x > maximum)
                continue;
            double position = maximum - minimum is double width && double.IsFinite(width) ? (x - minimum) / width : (x / 2 - minimum / 2) / (maximum / 2 - minimum / 2);
            int index = x == maximum ? bins - 1 : Math.Min(bins - 1, (int)(position * bins));
            counts[index] = checked(counts[index] + 1);
        }
        return new(NDArray.FromInt64(counts), NDArray.Linspace(minimum, maximum, checked(bins + 1)));
    }
}
