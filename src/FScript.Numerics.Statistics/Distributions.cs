namespace FScript.Numerics.Statistics;

/// <summary>SplitMix64 with an explicit seed; instances are private to an operation, with no global RNG.</summary>
public sealed class SeededRandom
{
    private ulong state;
    public SeededRandom(long seed)
    {
        state = unchecked((ulong)seed);
    }
    public ulong NextUInt64()
    {
        ulong z = unchecked(state += 0x9E3779B97F4A7C15UL);
        z = unchecked((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL);
        z = unchecked((z ^ (z >> 27)) * 0x94D049BB133111EBUL);
        return z ^ (z >> 31);
    }
    public double NextDouble() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);
    public int NextInt(int exclusiveMaximum)
    {
        if (exclusiveMaximum <= 0)
            throw new ArgumentOutOfRangeException(nameof(exclusiveMaximum));
        ulong bound = (ulong)exclusiveMaximum, threshold = unchecked(0UL - bound) % bound, value;
        do
        {
            value = NextUInt64();
        } while (value < threshold);
        return (int)(value % bound);
    }
}

public static class Distributions
{
    private static void NormalParameters(double mean, double std)
    {
        if (!double.IsFinite(mean) || !double.IsFinite(std) || std <= 0)
            throw new ArgumentException("Normal mean must be finite and standard deviation finite and positive.");
    }
    private static void UniformParameters(double lower, double upper)
    {
        if (!double.IsFinite(lower) || !double.IsFinite(upper) || lower >= upper)
            throw new ArgumentException("Uniform bounds must be finite and increasing.");
    }
    private static void BinomialParameters(int trials, double probability)
    {
        if (trials < 0 || !double.IsFinite(probability) || probability < 0 || probability > 1)
            throw new ArgumentException("Binomial requires nonnegative trials and probability in [0,1].");
    }
    private static void PoissonParameters(double rate)
    {
        if (!double.IsFinite(rate) || rate < 0 || rate > 1e9)
            throw new ArgumentException("Poisson rate must be finite in [0, 1e9].");
    }
    private static double Standardize(double x, double mean, double std)
    {
        if (double.IsInfinity(x))
            return x;
        double difference = x - mean;
        return double.IsFinite(difference) ? difference / std : x / std - mean / std;
    }
    public static double NormalPdf(double x, double mean, double std)
    {
        NormalParameters(mean, std);
        double z = Standardize(x, mean, std);
        return (Math.Exp(-z * z / 2) / std) / Math.Sqrt(2 * Math.PI);
    }
    // Complementary error function approximation, maximum absolute error about 1.5e-7.
    private static double Erfc(double x)
    {
        double z = Math.Abs(x), t = 1 / (1 + 0.5 * z);
        double result = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? result : 2 - result;
    }
    public static double NormalCdf(double x, double mean, double std)
    {
        NormalParameters(mean, std);
        if (x == mean)
            return 0.5;
        return 0.5 * Erfc(-Standardize(x, mean, std) / Math.Sqrt(2));
    }
    public static double UniformPdf(double x, double lower, double upper)
    {
        UniformParameters(lower, upper);
        if (double.IsNaN(x))
            return double.NaN;
        return x >= lower && x <= upper ? (double.IsFinite(upper - lower) ? 1 / (upper - lower) : 0.5 / (upper / 2 - lower / 2)) : 0;
    }
    public static double UniformCdf(double x, double lower, double upper)
    {
        UniformParameters(lower, upper);
        return x <= lower ? 0 : x >= upper ? 1 : double.IsFinite(upper - lower) ? (x - lower) / (upper - lower) : (x / 2 - lower / 2) / (upper / 2 - lower / 2);
    }
    private static double LogGamma(double z)
    {
        double[] c = [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
        z -= 1;
        double x = 0.99999999999980993;
        for (int i = 0; i < c.Length; i++)
            x += c[i] / (z + i + 1);
        double t = z + c.Length - 0.5;
        return 0.9189385332046727 + (z + 0.5) * Math.Log(t) - t + Math.Log(x);
    }
    public static double BinomialPmf(int successes, int trials, double probability)
    {
        BinomialParameters(trials, probability);
        if (successes < 0 || successes > trials)
            return 0;
        if (probability == 0)
            return successes == 0 ? 1 : 0;
        if (probability == 1)
            return successes == trials ? 1 : 0;
        return Math.Exp(LogGamma(trials + 1.0) - LogGamma(successes + 1.0) - LogGamma(trials - successes + 1.0) + successes * Math.Log(probability) + (trials - successes) * Math.Log(1 - probability));
    }
    // Regularized incomplete beta, continued fraction (modified Lentz).
    private static double BetaFraction(double a, double b, double x)
    {
        const double tiny = 1e-300;
        double qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
        if (Math.Abs(d) < tiny)
            d = tiny;
        d = 1 / d;
        double h = d;
        for (int m = 1; m <= 10000; m++)
        {
            double m2 = 2.0 * m, aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1 + aa * d;
            if (Math.Abs(d) < tiny)
                d = tiny;
            c = 1 + aa / c;
            if (Math.Abs(c) < tiny)
                c = tiny;
            d = 1 / d;
            h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1 + aa * d;
            if (Math.Abs(d) < tiny)
                d = tiny;
            c = 1 + aa / c;
            if (Math.Abs(c) < tiny)
                c = tiny;
            d = 1 / d;
            double delta = d * c;
            h *= delta;
            if (Math.Abs(delta - 1) < 3e-14)
                return h;
        }
        throw new InvalidOperationException("Incomplete beta did not converge.");
    }
    private static double Beta(double x, double a, double b)
    {
        if (x <= 0)
            return 0;
        if (x >= 1)
            return 1;
        double factor = Math.Exp(LogGamma(a + b) - LogGamma(a) - LogGamma(b) + a * Math.Log(x) + b * Math.Log(1 - x));
        return Math.Clamp(x < (a + 1) / (a + b + 2) ? factor * BetaFraction(a, b, x) / a : 1 - factor * BetaFraction(b, a, 1 - x) / b, 0, 1);
    }
    public static double BinomialCdf(int successes, int trials, double probability)
    {
        BinomialParameters(trials, probability);
        if (successes < 0)
            return 0;
        if (successes >= trials)
            return 1;
        if (probability == 0)
            return 1;
        if (probability == 1)
            return 0;
        return Beta(1 - probability, trials - successes, successes + 1.0);
    }
    public static double PoissonPmf(int count, double rate)
    {
        PoissonParameters(rate);
        if (count < 0)
            return 0;
        if (rate == 0)
            return count == 0 ? 1 : 0;
        return Math.Exp(count * Math.Log(rate) - rate - LogGamma(count + 1.0));
    }
    public static double PoissonCdf(int count, double rate)
    {
        PoissonParameters(rate);
        if (count < 0)
            return 0;
        if (rate == 0)
            return 1;
        double a = count + 1.0, x = rate, log = a * Math.Log(x) - x - LogGamma(a);
        if (x < a + 1)
        {
            double sum = 1 / a, term = sum;
            for (int n = 1; n <= 1000000; n++)
            {
                term *= x / (a + n);
                sum += term;
                if (Math.Abs(term) < Math.Abs(sum) * 3e-14)
                    return Math.Clamp(1 - sum * Math.Exp(log), 0, 1);
            }
        }
        else
        {
            const double tiny = 1e-300;
            double b = x + 1 - a, c = 1 / tiny, d = 1 / b, h = d;
            for (int n = 1; n <= 1000000; n++)
            {
                double an = -n * (n - a);
                b += 2;
                d = an * d + b;
                if (Math.Abs(d) < tiny)
                    d = tiny;
                c = b + an / c;
                if (Math.Abs(c) < tiny)
                    c = tiny;
                d = 1 / d;
                double delta = d * c;
                h *= delta;
                if (Math.Abs(delta - 1) < 3e-14)
                    return Math.Clamp(Math.Exp(log) * h, 0, 1);
            }
        }
        throw new InvalidOperationException("Poisson CDF did not converge.");
    }
    private static NDArray Samples(Shape shape, long seed, Func<SeededRandom, double> next)
    {
        ArgumentNullException.ThrowIfNull(shape);
        var rng = new SeededRandom(seed);
        var values = new double[shape.Size];
        for (int i = 0; i < values.Length; i++)
            values[i] = next(rng);
        return NDArray.FromArray(values, shape);
    }
    public static NDArray Normal(Shape shape, double mean, double std, long seed)
    {
        NormalParameters(mean, std);
        return Samples(shape, seed, r => mean + std * Math.Sqrt(-2 * Math.Log(1 - r.NextDouble())) * Math.Cos(2 * Math.PI * r.NextDouble()));
    }
    public static NDArray Uniform(Shape shape, double lower, double upper, long seed)
    {
        UniformParameters(lower, upper);
        return Samples(shape, seed, r => { double u = r.NextDouble(); return (1 - u) * lower + u * upper; });
    }
    public static NDArray Binomial(Shape shape, int trials, double probability, long seed)
    {
        BinomialParameters(trials, probability);
        ArgumentNullException.ThrowIfNull(shape);
        var rng = new SeededRandom(seed);
        var values = new long[shape.Size];
        if (probability is 0 or 1)
        {
            if (probability == 1)
                Array.Fill(values, (long)trials);
            return NDArray.FromInt64(values, shape);
        }
        // Inverse CDF via binary search avoids work proportional to the trial count.
        for (int i = 0; i < values.Length; i++)
        {
            double u = rng.NextDouble();
            int low = 0, high = trials;
            while (low < high)
            {
                int mid = low + (high - low) / 2;
                if (BinomialCdf(mid, trials, probability) >= u)
                    high = mid;
                else
                    low = mid + 1;
            }
            values[i] = low;
        }
        return NDArray.FromInt64(values, shape);
    }
    public static NDArray Poisson(Shape shape, double rate, long seed)
    {
        PoissonParameters(rate);
        ArgumentNullException.ThrowIfNull(shape);
        var rng = new SeededRandom(seed);
        var values = new long[shape.Size];
        for (int i = 0; i < values.Length; i++)
        {
            double u = rng.NextDouble();
            int low = 0, high = checked((int)Math.Ceiling(rate + 16 * Math.Sqrt(rate) + 64));
            while (low < high)
            {
                int mid = low + (high - low) / 2;
                if (PoissonCdf(mid, rate) >= u)
                    high = mid;
                else
                    low = mid + 1;
            }
            values[i] = low;
        }
        return NDArray.FromInt64(values, shape);
    }
}
