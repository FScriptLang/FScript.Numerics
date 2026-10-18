using FScript.Numerics.Blas;
using FScript.Numerics.Lapack;
using FScript.Numerics.ML;
using FScript.Numerics.Statistics;
using NUnit.Framework;
namespace FScript.Numerics.Tests;

public class EdgeCaseTests
{
    [Test]
    public void Comparisons_do_not_round_away_wide_integer_differences()
    {
        var a = NDArray.FromInt64([9007199254740993, long.MaxValue, long.MinValue]);
        var b = NDArray.FromInt64([9007199254740992, long.MaxValue - 1, long.MinValue + 1]);
        Assert.That(a.Greater(b).ToBooleanArray(), Is.EqualTo(new[] { true, true, false }));
        Assert.That(a.Less(b).ToBooleanArray(), Is.EqualTo(new[] { false, false, true }));
        long nearFloatMidpoint = (1L << 62) + (1L << 38) + 1;
        float roundedUp = MathF.BitIncrement((float)(1L << 62));
        Assert.That(NDArray.FromInt64([nearFloatMidpoint]).Astype(DType.Float32).ToFloat32Array()[0], Is.EqualTo(roundedUp));
        Assert.That(NDArray.FromInt64([nearFloatMidpoint]).ToFloat32Array()[0], Is.EqualTo(roundedUp));
        var boundary = NDArray.FromArray([9007199254740992.0, 9223372036854775808.0, -9223372036854775808.0]);
        Assert.That(a.Greater(boundary).ToBooleanArray(), Is.EqualTo(new[] { true, false, false }));
        Assert.That(a.Less(boundary).ToBooleanArray(), Is.EqualTo(new[] { false, true, false }));
        Assert.That(NDArray.FromInt64([1]).Less(NDArray.FromArray([1.5])).GetBoolean(0), Is.True);
        Assert.That(a.AllClose(b, 0.5, 0), Is.False);
        Assert.That(a.AllClose(b, 1, 0), Is.True);
        Assert.That(NDArray.FromInt64([1]).Less(NDArray.FromArray([double.NaN])).GetBoolean(0), Is.False);
    }
    [Test]
    public void Typed_views_keep_precision_and_readonly_flags_across_all_shape_operations()
    {
        var a = NDArray.FromInt64([9007199254740993, 2, 3, 4, 5, 6], new Shape(2, 3));
        var transposed = a.Transpose();
        Assert.That(transposed.ToInt64Array(), Is.EqualTo(new long[] { 9007199254740993, 4, 2, 5, 3, 6 }));
        Assert.That(a.AsFortranContiguous().ToInt64Array(), Is.EqualTo(a.ToInt64Array()));
        var view = a.Slice(0, 1, 2).Reshape(new Shape(3));
        view.SetInt64(long.MaxValue, 0);
        Assert.That(a.GetInt64(1, 0), Is.EqualTo(long.MaxValue));
        var broadcast = NDArray.FromInt32([7]).BroadcastTo(new Shape(2, 3));
        foreach (var v in new[] { broadcast.Transpose(), broadcast.Slice(0, 0, 1), broadcast.ExpandDims(0).Squeeze(), broadcast.Permute(1, 0) })
            Assert.That(v.IsReadOnly, Is.True);
        Assert.That(NDArray.Where(NDArray.FromBooleans([true, false]), NDArray.FromInt64([9007199254740993]), NDArray.FromInt64([2])).ToInt64Array(), Is.EqualTo(new long[] { 9007199254740993, 2 }));
        Assert.That(NDArray.Concatenate([a, a], 0).GetInt64(2, 0), Is.EqualTo(9007199254740993));
    }
    [TestCase(DType.Float32, 3e-5)]
    [TestCase(DType.Float64, 1e-10)]
    public void Wide_and_strided_decompositions_and_underdetermined_solutions_are_consistent(DType dtype, double tolerance)
    {
        var a = NDArray.FromRows([1, 0, 2], [0, 1, 3]).Astype(dtype);
        var qr = Decompositions.Qr(a);
        Assert.That(LinearAlgebra.Matmul(qr.Q, qr.R).AllClose(a, tolerance, tolerance), Is.True);
        var strided = NDArray.FromRows([1, 99, 0, 99, 2], [0, 99, 1, 99, 3]).Astype(dtype).Slice(1, 0, 5, 2);
        Assert.That(strided.ContentEquals(a), Is.True);
        var y = NDArray.FromArray([1, 2], dtype);
        var solution = Decompositions.LeastSquares(strided, y);
        Assert.That(LinearAlgebra.Matvec(a, solution).AllClose(y, tolerance, tolerance), Is.True);
        // Minimum norm is orthogonal to the nullspace [-2,-3,1].
        Assert.That(solution.Get(0) * -2 + solution.Get(1) * -3 + solution.Get(2), Is.EqualTo(0).Within(tolerance));
        var rhs = NDArray.FromRows([1, 2], [3, 4]).Astype(dtype);
        var identity = NDArray.Identity(2, dtype);
        Assert.That(Decompositions.Solve(identity, rhs).AllClose(rhs, tolerance, tolerance), Is.True);
        Assert.Throws<ArgumentException>(() => Decompositions.Solve(identity, rhs.Astype(dtype == DType.Float32 ? DType.Float64 : DType.Float32)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Decompositions.LeastSquares(a, y, -1));
        Assert.That(Decompositions.Qr(NDArray.Zeros(new Shape(3, 0), dtype)).Q.Shape, Is.EqualTo(new Shape(3, 0)));
    }
    [TestCase(DType.Float64)]
    [TestCase(DType.Float32)]
    public void Blas_layout_adapter_handles_all_transpose_packing_and_offset_combinations(DType dtype)
    {
        var left = NDArray.FromRows([1, 2, 3], [4, 5, 6]).Astype(dtype);
        var right = NDArray.FromRows([7, 8], [9, 10], [11, 12]).Astype(dtype);
        var lefts = new[] { left, left.AsFortranContiguous(), NDArray.FromRows([1, 99, 2, 99, 3], [4, 99, 5, 99, 6]).Astype(dtype).Slice(1, 0, 5, 2), NDArray.Concatenate([left, left], 0).Slice(0, 1, 3) };
        var rights = new[] { right, right.AsFortranContiguous(), NDArray.FromRows([7, 99, 8], [9, 99, 10], [11, 99, 12]).Astype(dtype).Slice(1, 0, 3, 2) };
        foreach (var a in lefts)
            foreach (var b in rights)
            {
                var beforeA = a.Copy();
                var beforeB = b.Copy();
                var result = LinearAlgebra.Matmul(a, b);
                Assert.That(result.AllClose(a.Matmul(b), 1e-5, 1e-5), Is.True);
                result.Set(99, 0, 0);
                Assert.That(a.ContentEquals(beforeA), Is.True);
                Assert.That(b.ContentEquals(beforeB), Is.True);
            }
        var vector = NDArray.FromArray([99, 1, 99, 2, 99, 3, 99], dtype).Slice(0, 1, 7, 2);
        foreach (var a in lefts)
            Assert.That(LinearAlgebra.Matvec(a, vector).AllClose(a.Matvec(vector), 1e-5, 1e-5), Is.True);
        Assert.That(LinearAlgebra.Dot(vector, vector), Is.EqualTo(14));
        Assert.That(LinearAlgebra.Norm(vector), Is.EqualTo(Math.Sqrt(14)).Within(1e-6));
        var mixed = LinearAlgebra.Matmul(left.AsFortranContiguous(), right.Astype(dtype == DType.Float32 ? DType.Float64 : DType.Float32).AsFortranContiguous());
        Assert.That(mixed.DType, Is.EqualTo(DType.Float64));
        Assert.That(mixed.AllClose(left.Matmul(right), 1e-5, 1e-5), Is.True);
    }
    [Test]
    public void Degenerate_distributions_and_extreme_finite_bounds_are_defined()
    {
        Assert.That(Distributions.Binomial(new Shape(4), int.MaxValue, 1, 0).ToInt64Array(), Is.All.EqualTo((long)int.MaxValue));
        Assert.That(Distributions.BinomialCdf(-1, 3, 0.5), Is.Zero);
        Assert.That(Distributions.BinomialCdf(3, 3, 0.5), Is.EqualTo(1));
        Assert.That(Distributions.PoissonCdf(-1, 0), Is.Zero);
        // Convergence at the documented upper rate requires more terms than moderate-rate cases.
        Assert.That(Distributions.PoissonCdf(1000000000, 1e9), Is.InRange(0.499, 0.501));
        Assert.That(Distributions.PoissonPmf(0, 0), Is.EqualTo(1));
        Assert.That(Distributions.NormalCdf(double.PositiveInfinity, 0, 1), Is.EqualTo(1));
        Assert.That(Distributions.NormalCdf(double.NegativeInfinity, 0, 1), Is.Zero);
        Assert.That(Distributions.NormalCdf(double.MaxValue, 0, double.MaxValue), Is.EqualTo(0.8413447460685429).Within(1e-7));
        Assert.That(Distributions.UniformCdf(0, -double.MaxValue, double.MaxValue), Is.EqualTo(0.5));
        Assert.That(Distributions.UniformPdf(0, -double.MaxValue, double.MaxValue), Is.GreaterThan(0));
        Assert.That(Descriptive.Histogram(NDArray.FromArray([-double.MaxValue, 0, double.MaxValue]), 2, -double.MaxValue, double.MaxValue).Counts.ToInt64Array(), Is.EqualTo(new long[] { 1, 2 }));
        Assert.That(Descriptive.Quantile(NDArray.FromArray([double.PositiveInfinity, double.PositiveInfinity]), 0.5), Is.EqualTo(double.PositiveInfinity));
        Assert.That(Descriptive.Quantile(NDArray.FromArray([double.NaN, 1]), 0.5), Is.NaN);
        Assert.Throws<ArgumentException>(() => Distributions.Poisson(new Shape(1), double.NaN, 0));
        Assert.Throws<ArgumentException>(() => Descriptive.Histogram(NDArray.FromArray([double.NaN]), 2, 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Descriptive.Quantile(NDArray.Ones(new Shape(2)), 1.1));
    }
    [Test]
    public void Model_validation_convergence_and_empty_prediction_shapes_are_explicit()
    {
        var x = NDArray.FromRows([1, 2], [3, 4]);
        var y = NDArray.FromArray([1, 2]);
        Assert.Throws<ArgumentException>(() => Preprocessing.TrainTestSplit(x, y, 0.99, 42));
        Assert.Throws<ArgumentException>(() => Preprocessing.TrainTestSplit(x, NDArray.Ones(new Shape(3)), 0.5, 42));
        Assert.Throws<ArgumentException>(() => LogisticRegression.Fit(x, NDArray.Ones(new Shape(2))));
        Assert.Throws<ArgumentException>(() => LogisticRegression.Fit(x, y));
        Assert.That(LogisticRegression.Fit(x, NDArray.FromInt64([0, 1]), maxIterations: 1).Converged, Is.False);
        Assert.Throws<ArgumentOutOfRangeException>(() => KMeans.Fit(x, 3, 42));
        Assert.Throws<ArgumentException>(() => KMeans.Fit(x, 1, 42, maxIterations: 0));
        Assert.Throws<ArgumentException>(() => Pca.Fit(x, 3));
        Assert.Throws<ArgumentException>(() => LinearRegression.Fit(x, NDArray.FromArray([1, double.PositiveInfinity])));
        Assert.That(StandardScaler.Fit(x).Transform(NDArray.Zeros(new Shape(0, 2))).Shape, Is.EqualTo(new Shape(0, 2)));
        Assert.That(KMeans.Fit(x, 1, 42).Predict(NDArray.Zeros(new Shape(0, 2))).Size, Is.Zero);
        Assert.That(Metrics.R2(NDArray.Ones(new Shape(2)), NDArray.Zeros(new Shape(2))), Is.Zero);
        Assert.Throws<ArgumentException>(() => Metrics.Mse(NDArray.Zeros(new Shape(0)), NDArray.Zeros(new Shape(0))));
        Assert.Throws<ArgumentException>(() => Metrics.ConfusionMatrix(NDArray.FromInt64([2]), NDArray.FromInt64([0]), 2));
    }
}
