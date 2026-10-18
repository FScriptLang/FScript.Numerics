using FScript.Numerics.Blas;
using FScript.Numerics.Lapack;
using FScript.Numerics.Statistics;
using FScript.Numerics.ML;
using NUnit.Framework;
namespace FScript.Numerics.Tests;

public class ExpansionTests
{
    private static void Close(NDArray actual, NDArray expected, double tolerance = 1e-10) => Assert.That(actual.AllClose(expected, tolerance, tolerance), Is.True, $"{string.Join(',', actual.ToArray())} != {string.Join(',', expected.ToArray())}");
    [Test]
    public void Dtypes_preserve_wide_integers_and_check_overflow()
    {
        long wide = 9007199254740993;
        var a = NDArray.FromInt64([wide, wide + 1]);
        Assert.That(a.Copy().ToInt64Array(), Is.EqualTo(new[] { wide, wide + 1 }));
        Assert.That(a.GetInt64(0), Is.EqualTo(wide));
        Assert.That(a.Add(NDArray.FromInt64([1])).ToInt64Array(), Is.EqualTo(new[] { wide + 1, wide + 2 }));
        Assert.That(a.ContentEquals(NDArray.FromInt64([wide - 1, wide + 1])), Is.False);
        Assert.That(a.ContentEquals(NDArray.FromArray([(double)wide, (double)(wide + 1)])), Is.False);
        Assert.Throws<OverflowException>(() => NDArray.FromInt64([long.MaxValue]).Add(NDArray.FromInt64([1])));
        Assert.Throws<OverflowException>(() => NDArray.FromInt32([int.MaxValue]).Add(NDArray.FromInt32([1])));
        Assert.Throws<OverflowException>(() => NDArray.FromInt64([long.MinValue]).Negate());
        Assert.Throws<OverflowException>(() => NDArray.FromArray([double.NaN]).Astype(DType.Int32));
        Assert.That(NDArray.FromArray([1.9, -1.9]).Astype(DType.Int32).ToInt64Array(), Is.EqualTo(new long[] { 1, -1 }));
        Assert.That(NDArray.FromArray([0, 2]).Astype(DType.Bool).ToBooleanArray(), Is.EqualTo(new[] { false, true }));
        Assert.That(NDArray.FromFloat32([1, 2]).Add(0.5).DType, Is.EqualTo(DType.Float32));
    }
    [Test]
    public void Broadcasting_is_readonly_but_source_mutations_remain_visible()
    {
        var source = NDArray.FromArray([1, 2, 3]);
        var view = source.BroadcastTo(new Shape(2, 3));
        Assert.That(view.Strides, Is.EqualTo(new[] { 0, 1 }));
        Assert.Throws<InvalidOperationException>(() => view.Set(9, 0, 0));
        source.Set(4, 0);
        Assert.That(view.Get(1, 0), Is.EqualTo(4));
        var copy = view.Copy();
        copy.Set(100, 0, 0);
        Assert.That(source.Get(0), Is.EqualTo(4));
        Close(NDArray.Ones(new Shape(2, 1, 3)).Add(NDArray.Full(new Shape(1, 4, 1), 2)), NDArray.Full(new Shape(2, 4, 3), 3));
        Assert.That(NDArray.Zeros(new Shape(0, 3)).Add(source).Shape, Is.EqualTo(new Shape(0, 3)));
        Assert.That(NDArray.Scalar(2).BroadcastTo(new Shape(0)).Size, Is.Zero);
        Assert.Throws<ArgumentException>(() => source.BroadcastTo(new Shape(2, 2)));
        for (int m = 0; m < 5; m++)
            for (int n = 0; n < 5; n++)
            {
                var a = NDArray.Full(new Shape(m, 1), 2);
                var b = NDArray.Full(new Shape(n), 3);
                var c = a.Add(b);
                Assert.That(c.Shape, Is.EqualTo(new Shape(m, n)));
                Assert.That(c.ToArray(), Is.All.EqualTo(5));
            }
    }
    [Test]
    public void Layout_composition_and_axis_reductions_work_on_views()
    {
        var a = NDArray.FromRows([1, 2, 3], [4, 5, 6]);
        var f = a.AsFortranContiguous();
        Assert.That(f.IsFortranContiguous, Is.True);
        Assert.That(f.IsContiguous, Is.False);
        Close(f, a);
        Close(a.Transpose().SumAxis(1), NDArray.FromArray([5, 7, 9]));
        Close(a.MeanAxis(0, true), NDArray.FromRows([2.5, 3.5, 4.5]));
        Close(a.VarianceAxis(1, false, 1), NDArray.FromArray([1, 1]));
        Assert.That(NDArray.FromInt64([9007199254740993, 1]).SumAxis(0).GetInt64(), Is.EqualTo(9007199254740994));
        Assert.Throws<OverflowException>(() => NDArray.FromInt64([long.MaxValue, 1]).SumAxis(0));
        Close(NDArray.Concatenate([a, a], 1), NDArray.FromRows([1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]));
        Assert.That(NDArray.Stack([a, a], 1).Shape, Is.EqualTo(new Shape(2, 2, 3)));
        Close(NDArray.Where(a.Greater(NDArray.Scalar(3)), a, NDArray.Scalar(-1)), NDArray.FromRows([-1, -1, -1], [4, 5, 6]));
        var flat = a.Transpose().Flatten();
        flat.Set(100, 0);
        Assert.That(a.Get(0, 0), Is.EqualTo(1));
        Assert.That(a.ExpandDims(1).Squeeze().Shape, Is.EqualTo(a.Shape));
        Assert.That(NDArray.Zeros(new Shape(2, 0)).SumAxis(1).ToArray(), Is.EqualTo(new[] { 0d, 0d }));
        Assert.That(NDArray.Zeros(new Shape(2, 0)).AllAxis(1).ToBooleanArray(), Is.EqualTo(new[] { true, true }));
        Assert.Throws<InvalidOperationException>(() => NDArray.Zeros(new Shape(2, 0)).MeanAxis(1));
        Assert.That(NDArray.FromArray([1, double.NaN]).Variance(), Is.NaN);
    }
    [TestCase(DType.Float64, 1e-10)]
    [TestCase(DType.Float32, 2e-5)]
    public void Native_blas_and_lapack_reconstruct_known_answers_without_mutating_inputs(DType dtype, double tolerance)
    {
        TestContext.WriteLine(LinearAlgebra.BackendConfiguration);
        var a = NDArray.FromRows([3, 1], [1, 2]).Astype(dtype);
        var b = NDArray.FromArray([9, 8], dtype);
        var original = a.Copy();
        Close(Decompositions.Solve(a, b), NDArray.FromArray([2, 3]), tolerance);
        Close(Decompositions.Lu(a).Solve(b), NDArray.FromArray([2, 3]), tolerance);
        var factor = Decompositions.Lu(a);
        factor.Factors.Set(100, 0, 0);
        Close(factor.Solve(b), NDArray.FromArray([2, 3]), tolerance);
        var rectangular = NDArray.FromRows([1, 2], [3, 4], [5, 7]).Astype(dtype).AsFortranContiguous();
        Close(LinearAlgebra.Matmul(rectangular.Transpose(), rectangular), rectangular.Transpose().Matmul(rectangular), tolerance);
        Close(LinearAlgebra.Matvec(rectangular, b), rectangular.Matvec(b), tolerance);
        Assert.That(LinearAlgebra.Dot(b, b), Is.EqualTo(145).Within(tolerance));
        Assert.That(LinearAlgebra.Norm(b), Is.EqualTo(Math.Sqrt(145)).Within(tolerance));
        var qr = Decompositions.Qr(rectangular);
        Close(LinearAlgebra.Matmul(qr.Q, qr.R), rectangular, tolerance);
        Close(LinearAlgebra.Matmul(qr.Q.Transpose(), qr.Q), NDArray.Identity(2), tolerance);
        var chol = Decompositions.Cholesky(a);
        Close(LinearAlgebra.Matmul(chol, chol.Transpose()), a, tolerance);
        var svd = Decompositions.Svd(rectangular);
        var diagonal = NDArray.Zeros(new Shape(2, 2), dtype);
        for (int i = 0; i < 2; i++)
            diagonal.Set(svd.SingularValues.Get(i), i, i);
        Close(LinearAlgebra.Matmul(LinearAlgebra.Matmul(svd.U, diagonal), svd.VT), rectangular, tolerance);
        var eigen = Decompositions.SymmetricEigen(a);
        var diag = NDArray.Zeros(new Shape(2, 2), dtype);
        for (int i = 0; i < 2; i++)
            diag.Set(eigen.Values.Get(i), i, i);
        Close(LinearAlgebra.Matmul(LinearAlgebra.Matmul(eigen.Vectors, diag), eigen.Vectors.Transpose()), a, tolerance);
        var rankOne = NDArray.FromRows([1, 1], [2, 2]).Astype(dtype);
        Close(Decompositions.LeastSquares(rankOne, NDArray.FromArray([2, 4], dtype)), NDArray.FromArray([1, 1]), tolerance);
        Assert.That(a.ContentEquals(original), Is.True);
        Assert.That(b.ToArray(), Is.EqualTo(new[] { 9d, 8d }));
    }
    [Test]
    public void Native_validation_handles_singular_empty_and_concurrent_calls()
    {
        Assert.Throws<InvalidOperationException>(() => Decompositions.Solve(NDArray.FromRows([1, 1], [1, 1]), NDArray.Ones(new Shape(2))));
        Assert.Throws<InvalidOperationException>(() => Decompositions.Cholesky(NDArray.FromRows([1, 2], [2, 1])));
        Assert.Throws<ArgumentException>(() => Decompositions.SymmetricEigen(NDArray.FromRows([1, 2], [0, 1])));
        Assert.Throws<ArgumentException>(() => Decompositions.Svd(NDArray.FromRows([double.NaN])));
        Assert.Throws<ArgumentException>(() => LinearAlgebra.Dot(NDArray.FromInt64([1]), NDArray.FromInt64([1])));
        Assert.That(LinearAlgebra.Matmul(NDArray.Zeros(new Shape(2, 0)), NDArray.Zeros(new Shape(0, 3))).ToArray(), Is.All.Zero);
        Assert.That(Decompositions.Svd(NDArray.Zeros(new Shape(0, 3))).VT.Shape, Is.EqualTo(new Shape(0, 3)));
        Parallel.For(0, 20, _ => Close(Decompositions.Solve(NDArray.Identity(3), NDArray.Ones(new Shape(3))), NDArray.Ones(new Shape(3))));
    }
    [Test]
    public void Descriptive_and_distribution_results_match_independent_references()
    {
        var a = NDArray.FromRows([1, 2], [3, 4], [5, 6]);
        Assert.That(Descriptive.Median(a), Is.EqualTo(3.5));
        Assert.That(Descriptive.Quantile(a, 0.25), Is.EqualTo(2.25));
        Close(Descriptive.QuantileAxis(a, 0, 0.5), NDArray.FromArray([3, 4]));
        Close(Descriptive.Covariance(a), NDArray.Full(new Shape(2, 2), 4));
        Close(Descriptive.Correlation(a), NDArray.Ones(new Shape(2, 2)));
        Assert.That(Descriptive.Correlation(NDArray.FromRows([1, 1], [1, 2])).Get(0, 0), Is.NaN);
        Assert.That(Descriptive.Histogram(NDArray.FromArray([0, 0.5, 1, 1.5, 2]), 2, 0, 2).Counts.ToInt64Array(), Is.EqualTo(new long[] { 2, 3 }));
        Assert.That(Distributions.NormalPdf(0, 0, 1), Is.EqualTo(0.3989422804014327).Within(1e-14));
        Assert.That(Distributions.NormalCdf(1.96, 0, 1), Is.EqualTo(0.9750021048517795).Within(1e-7));
        Assert.That(Distributions.NormalCdf(10, 0, 1), Is.EqualTo(1).Within(1e-12));
        Assert.That(Distributions.NormalCdf(-10, 0, 1), Is.GreaterThan(0));
        Assert.That(Distributions.BinomialPmf(3, 10, 0.5), Is.EqualTo(0.1171875).Within(1e-12));
        Assert.That(Distributions.BinomialCdf(3, 10, 0.5), Is.EqualTo(0.171875).Within(1e-12));
        Assert.That(Distributions.PoissonPmf(3, 2), Is.EqualTo(0.1804470443154836).Within(1e-12));
        Assert.That(Distributions.PoissonCdf(3, 2), Is.EqualTo(0.857123460498547).Within(1e-12));
        Assert.That(Distributions.PoissonCdf(1000, 1000), Is.EqualTo(0.508409367168506).Within(1e-10));
        Assert.That(Distributions.UniformCdf(2, 0, 4), Is.EqualTo(0.5));
        var sample = Distributions.Normal(new Shape(10000), 3, 2, 42);
        Assert.That(sample.ContentEquals(Distributions.Normal(sample.Shape, 3, 2, 42)), Is.True);
        Assert.That(sample.Mean(), Is.EqualTo(3).Within(0.06));
        Assert.That(sample.Std(), Is.EqualTo(2).Within(0.06));
        Assert.That(Distributions.Binomial(new Shape(2000), 10000, 0.3, 42).Mean(), Is.EqualTo(3000).Within(5));
        Assert.That(Distributions.Poisson(new Shape(2000), 1000, 42).Mean(), Is.EqualTo(1000).Within(3));
        Assert.Throws<ArgumentException>(() => Distributions.Normal(new Shape(1), 0, 0, 1));
        Assert.Throws<ArgumentException>(() => Descriptive.Quantile(NDArray.Zeros(new Shape(0)), 0.5));
    }
    [Test]
    public void Fitted_models_are_independent_and_predict_known_answers()
    {
        var x = NDArray.FromRows([0, 5], [1, 5], [2, 5], [3, 5]);
        var y = NDArray.FromArray([1, 3, 5, 7]);
        var original = x.Copy();
        var scaler = StandardScaler.Fit(x);
        Close(scaler.Transform(x).MeanAxis(0), NDArray.FromArray([0, 0]));
        scaler.Mean.Set(100, 0);
        Close(scaler.Transform(x).MeanAxis(0), NDArray.FromArray([0, 0]));
        Close(MinMaxScaler.Fit(x).Transform(x), NDArray.FromRows([0, 0], [1.0 / 3, 0], [2.0 / 3, 0], [1, 0]));
        var model = LinearRegression.Fit(x, y);
        Close(model.Predict(x), y);
        model.Coefficients.Set(100, 0);
        Close(model.Predict(x), y);
        var ridge = LinearRegression.Fit(x, y, ridge: 10);
        Assert.That(Math.Abs(ridge.Coefficients.Get(0)), Is.LessThan(Math.Abs(model.Coefficients.Get(0))));
        var split = Preprocessing.TrainTestSplit(x, y, 0.25, 42);
        Assert.That(split.TestFeatures.Shape[0], Is.EqualTo(1));
        Assert.That(split.TrainFeatures.Shape[0], Is.EqualTo(3));
        Close(split.TestFeatures, Preprocessing.TrainTestSplit(x, y, 0.25, 42).TestFeatures);
        var logisticX = NDArray.FromRows([-2], [-1], [1], [2]);
        var labels = NDArray.FromInt64([0, 0, 1, 1]);
        var logistic = LogisticRegression.Fit(logisticX, labels, ridge: 0.1, maxIterations: 5000, tolerance: 1e-6);
        Assert.That(logistic.Converged, Is.True);
        Assert.That(Metrics.Accuracy(labels, logistic.Predict(logisticX)), Is.EqualTo(1));
        var clusters = NDArray.FromRows([-2, -2], [-1, -1], [9, 9], [10, 10]);
        var km = KMeans.Fit(clusters, 2, 42);
        Assert.That(km.Converged, Is.True);
        Assert.That(km.Inertia, Is.EqualTo(2).Within(1e-12));
        Assert.That(km.Predict(clusters).ContentEquals(KMeans.Fit(clusters, 2, 42).Predict(clusters)), Is.True);
        km.Centers.Set(100, 0, 0);
        Assert.That(km.Inertia, Is.EqualTo(2));
        var pca = Pca.Fit(NDArray.FromRows([1, 2], [2, 4], [3, 6]), 1);
        Assert.That(pca.ExplainedVariance.Get(0), Is.EqualTo(5).Within(1e-10));
        Close(pca.Transform(NDArray.FromRows([2, 4])), NDArray.FromRows([0]));
        Assert.That(Metrics.Mse(y, y), Is.Zero);
        Assert.That(Metrics.R2(y, y), Is.EqualTo(1));
        Assert.That(Metrics.Mae(y, y.Add(2)), Is.EqualTo(2));
        Assert.That(Metrics.ConfusionMatrix(labels, NDArray.FromInt64([0, 1, 1, 1]), 2).ToInt64Array(), Is.EqualTo(new long[] { 1, 1, 0, 2 }));
        Assert.That(x.ContentEquals(original), Is.True);
        Assert.Throws<ArgumentException>(() => StandardScaler.Fit(NDArray.FromRows([double.NaN])));
        Assert.Throws<ArgumentException>(() => model.Predict(NDArray.Ones(new Shape(2, 3))));
    }
}
