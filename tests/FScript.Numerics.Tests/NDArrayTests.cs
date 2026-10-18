using System.Runtime.CompilerServices;
using NUnit.Framework;

namespace FScript.Numerics.Tests;

[TestFixture]
public sealed class NDArrayTests
{
    private static NDArray Matrix() => NDArray.FromRows([1, 2, 3], [4, 5, 6]);
    private static void Values(NDArray actual, params double[] expected) => Assert.That(actual.ToArray(), Is.EqualTo(expected).Within(1e-12));

    [Test]
    public void Construction_copies_inputs_and_metadata()
    {
        int[] dimensions = [2, 3]; var shape = new Shape(dimensions); dimensions[0] = 99;
        double[] values = [1, 2, 3, 4, 5, 6]; var a = NDArray.FromArray(values, shape); values[0] = 100;
        Assert.Multiple(() =>
        {
            Assert.That(a.Shape.ToArray(), Is.EqualTo(new[] { 2, 3 }));
            Assert.That(a.Strides, Is.EqualTo(new[] { 3, 1 }));
            Assert.That(a.Rank, Is.EqualTo(2)); Assert.That(a.Size, Is.EqualTo(6));
            Assert.That(a.DType, Is.EqualTo(DType.Float64)); Assert.That(a.Get(0, 0), Is.EqualTo(1));
            Assert.That(a.Shape.Equals(new Shape(2, 3)), Is.True);
            Assert.That(a.Shape.GetHashCode(), Is.EqualTo(new Shape(2, 3).GetHashCode()));
        });
        double[] row = [7, 8]; var fromRows = NDArray.FromRows(row); row[0] = 99; Values(fromRows, 7, 8);
        Assert.Throws<NotSupportedException>(() => ((IList<int>)a.Strides)[0] = 42);
        var copy = a.Shape.ToArray(); copy[0] = 42; Assert.That(a.Shape[0], Is.EqualTo(2));
    }
    [Test]
    public void Scalar_vector_rows_and_empty_dimensions_have_defined_shapes()
    {
        var scalar = NDArray.Scalar(42); Assert.That(scalar.Rank, Is.Zero); Assert.That(scalar.Size, Is.EqualTo(1)); Assert.That(scalar.Get(), Is.EqualTo(42));
        scalar.Set(43); Assert.That(scalar.Get(), Is.EqualTo(43));
        Assert.That(NDArray.FromArray([1, 2]).Shape.ToArray(), Is.EqualTo(new[] { 2 }));
        Assert.That(NDArray.FromRows().Shape.ToArray(), Is.EqualTo(new[] { 0, 0 }));
        Assert.That(NDArray.FromRows([], []).Shape.ToArray(), Is.EqualTo(new[] { 2, 0 }));
        var empty = NDArray.Zeros(new Shape(2, 0, 3)); Assert.That(empty.Size, Is.Zero); Assert.That(empty.IsContiguous, Is.True);
        Assert.That(empty.Reshape(new Shape(0, 4)).Size, Is.Zero); Values(NDArray.Zeros(new Shape()), 0);
        Assert.Throws<ArgumentException>(() => NDArray.FromRows([1], [2, 3]));
        Assert.Throws<ArgumentNullException>(() => NDArray.FromRows(null!));
        Assert.Throws<ArgumentException>(() => NDArray.FromRows(new double[][] { null! }));
        Assert.Throws<ArgumentException>(() => NDArray.FromArray(new double[2], new Shape(3)));
    }
    [Test]
    public void Invalid_shapes_layouts_and_indices_fail_before_access()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Shape(-1));
        Assert.Throws<OverflowException>(() => new Shape(int.MaxValue, int.MaxValue));
        Assert.Throws<OverflowException>(() => NDArray.Zeros(new Shape(0, int.MaxValue, int.MaxValue)));
        Assert.Throws<ArgumentException>(() => NDArray.CreateView([1, 2], new Shape(2, 2), [2, 1], 0));
        Assert.Throws<ArgumentException>(() => NDArray.CreateView([1, 2], new Shape(2), [-1], 0));
        var a = Matrix();
        Assert.Throws<ArgumentException>(() => a.Get(0)); Assert.Throws<ArgumentOutOfRangeException>(() => a.Get(-1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => a.Set(1, 0, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => NDArray.Zeros(new Shape(0)).Get(0));
        Assert.Throws<ArgumentException>(() => a.Permute(0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => a.Slice(2, 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => a.Slice(0, 0, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => a.Slice(0, 1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => a.Slice(0, 0, 1, 0));
    }
    [Test]
    public void Views_share_storage_and_use_logical_order()
    {
        var a = Matrix(); var transpose = a.Transpose();
        Assert.That(transpose.Shape.ToArray(), Is.EqualTo(new[] { 3, 2 }));
        Assert.That(transpose.Strides, Is.EqualTo(new[] { 1, 3 }));
        Values(transpose, 1, 4, 2, 5, 3, 6); Assert.That(transpose.SharesStorageWith(a), Is.True);
        transpose.Set(40, 0, 1); Assert.That(a.Get(1, 0), Is.EqualTo(40));
        var slice = a.Slice(1, 0, 3, 2); Values(slice, 1, 3, 40, 6);
        slice.Set(60, 1, 1); Assert.That(a.Get(1, 2), Is.EqualTo(60));
        Assert.That(slice.SharesStorageWith(a), Is.True);
        var rank3 = NDArray.Arange(0, 12, 1).Reshape(new Shape(2, 2, 3)).Permute(2, 0, 1);
        Assert.That(rank3.Get(2, 1, 1), Is.EqualTo(11));
        var strided = NDArray.CreateView([0, 1, 2, 3, 4, 5, 6], new Shape(2, 2), [3, 1], 1);
        Values(strided, 1, 2, 4, 5);
    }
    [Test]
    public void Copies_and_contiguous_conversion_distinguish_sharing()
    {
        var a = Matrix(); Assert.That(a.AsContiguous(), Is.SameAs(a));
        var transposed = a.Transpose(); var contiguous = transposed.AsContiguous();
        Assert.That(contiguous.IsContiguous, Is.True); Assert.That(contiguous.SharesStorageWith(a), Is.False); Values(contiguous, 1, 4, 2, 5, 3, 6);
        var copy = a.Copy(); Assert.That(copy.SharesStorageWith(a), Is.False); a.Set(100, 0, 0); Assert.That(copy.Get(0, 0), Is.EqualTo(1));
        Assert.That(NDArray.Zeros(new Shape(0)).Copy().IsContiguous, Is.True);
        Assert.That(transposed.TryGetContiguousSpan(out _), Is.False);
        Assert.That(copy.TryGetContiguousSpan(out var span), Is.True); Assert.That(span[0], Is.EqualTo(1));
    }
    [Test]
    public void Reshape_accepts_offsets_and_singleton_layouts_but_never_silently_copies()
    {
        var a = Matrix(); var row = a.Slice(0, 1, 2); Assert.That(row.Offset, Is.EqualTo(3));
        var vector = row.Reshape(new Shape(3)); vector.Set(50, 1); Assert.That(a.Get(1, 1), Is.EqualTo(50));
        Assert.Throws<InvalidOperationException>(() => a.Transpose().Reshape(new Shape(6)));
        Assert.Throws<ArgumentException>(() => a.Reshape(new Shape(5)));
        var singleton = NDArray.FromRows([1, 2, 3]).Transpose(); Assert.That(singleton.IsContiguous, Is.True); Values(singleton.Reshape(new Shape(3)), 1, 2, 3);
        var empty = a.Slice(0, 2, 2).Slice(1, 3, 3); Values(empty); Assert.That(empty.Copy().Size, Is.Zero);
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static NDArray MakeSurvivingView() => Matrix().Transpose().Slice(0, 1, 3);
    [Test]
    public void View_keeps_storage_alive_without_the_original_wrapper()
    {
        var view = MakeSurvivingView(); GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect(); Values(view, 2, 5, 3, 6);
    }
    [Test]
    public void Factories_handle_direction_rounding_and_nonfinite_ranges()
    {
        Values(NDArray.Ones(new Shape(2)), 1, 1); Values(NDArray.Full(new Shape(2), 3), 3, 3);
        Values(NDArray.Identity(2), 1, 0, 0, 1); Assert.That(NDArray.Identity(0).Size, Is.Zero);
        Values(NDArray.Arange(0, 5, 2), 0, 2, 4); Values(NDArray.Arange(5, 0, -2), 5, 3, 1);
        Values(NDArray.Arange(0.1, 0.4, 0.1), 0.1, 0.2, 0.3); Values(NDArray.Arange(0, 1, -1));
        Values(NDArray.Arange(0, double.Epsilon, double.MaxValue), 0);
        Values(NDArray.Arange(-double.MaxValue, double.MaxValue, double.MaxValue), -double.MaxValue, 0);
        Assert.Throws<ArgumentException>(() => NDArray.Arange(0, 1, 0));
        Assert.Throws<ArgumentException>(() => NDArray.Arange(0, double.PositiveInfinity, 1));
        Assert.Throws<OverflowException>(() => NDArray.Arange(0, 1, double.Epsilon));
    }
    [TestCase(false)] [TestCase(true)]
    public void Arithmetic_and_reductions_work_for_contiguous_and_strided_operands(bool strided)
    {
        var a = strided ? NDArray.FromRows([1, 3], [2, 4]).Transpose() : NDArray.FromRows([1, 2], [3, 4]);
        var b = NDArray.Full(new Shape(2, 2), 2);
        Values(a.Add(b), 3, 4, 5, 6); Values(a.Subtract(b), -1, 0, 1, 2); Values(a.Multiply(b), 2, 4, 6, 8); Values(a.Divide(b), 0.5, 1, 1.5, 2);
        Values(a.Add(2), 3, 4, 5, 6); Values(a.Subtract(2), -1, 0, 1, 2); Values(a.Multiply(2), 2, 4, 6, 8); Values(a.Divide(2), 0.5, 1, 1.5, 2); Values(a.Negate(), -1, -2, -3, -4);
        Values(a, 1, 2, 3, 4); Assert.That(a.Sum(), Is.EqualTo(10).Within(1e-12)); Assert.That(a.Mean(), Is.EqualTo(2.5)); Assert.That(a.Min(), Is.EqualTo(1)); Assert.That(a.Max(), Is.EqualTo(4));
        Assert.Throws<ArgumentException>(() => a.Add(NDArray.Scalar(2)));
    }
    [Test]
    public void Linear_algebra_matches_hand_calculated_answers_and_zero_inner_dimensions()
    {
        var a = Matrix(); var b = NDArray.FromRows([7, 8], [9, 10], [11, 12]);
        Values(a.Matmul(b), 58, 64, 139, 154); Values(a.Matvec(NDArray.FromArray([1, 2, 3])), 14, 32);
        Values(a.Transpose().Matmul(a), 17, 22, 27, 22, 29, 36, 27, 36, 45);
        Assert.That(a.Slice(0, 0, 1).Reshape(new Shape(3)).Dot(NDArray.FromArray([4, 5, 6])), Is.EqualTo(32));
        Assert.That(NDArray.FromArray([1, 9, 2, 9, 3]).Slice(0, 0, 5, 2).Dot(NDArray.FromArray([4, 5, 6])), Is.EqualTo(32));
        Values(NDArray.Zeros(new Shape(2, 0)).Matmul(NDArray.Zeros(new Shape(0, 3))), 0, 0, 0, 0, 0, 0);
        Values(NDArray.Zeros(new Shape(2, 0)).Matvec(NDArray.FromArray([])), 0, 0);
        Assert.That(NDArray.Zeros(new Shape(int.MaxValue, 0)).Matmul(NDArray.Zeros(new Shape(0, 0))).Size, Is.Zero);
        Assert.That(NDArray.FromArray([]).Dot(NDArray.FromArray([])), Is.Zero);
        Assert.Throws<ArgumentException>(() => a.Matmul(a)); Assert.Throws<ArgumentException>(() => a.Dot(a)); Assert.Throws<ArgumentException>(() => a.Matvec(NDArray.FromArray([1])));
    }
    [Test]
    public void Empty_reductions_and_ieee_values_have_explicit_behavior()
    {
        var empty = NDArray.FromArray([]); Assert.That(empty.Sum(), Is.Zero);
        Assert.Throws<InvalidOperationException>(() => empty.Mean()); Assert.Throws<InvalidOperationException>(() => empty.Min()); Assert.Throws<InvalidOperationException>(() => empty.Max());
        foreach (var a in new[] { NDArray.FromArray([1, double.NaN, 2]), NDArray.FromRows([1, double.NaN], [2, 3]).Transpose() })
        { Assert.That(a.Sum(), Is.NaN); Assert.That(a.Min(), Is.NaN); Assert.That(a.Max(), Is.NaN); }
        Assert.That(NDArray.FromArray([1, 0]).Divide(0).ToArray()[0], Is.EqualTo(double.PositiveInfinity));
        Assert.That(NDArray.FromArray([1, 0]).Divide(0).ToArray()[1], Is.NaN);
    }
    [Test]
    public void Numerical_comparisons_are_shape_sensitive_and_handle_special_values()
    {
        var a = NDArray.FromArray([1, 2]); var b = NDArray.FromArray([1.00001, 2]);
        Assert.That(a.ContentEquals(b), Is.False); Assert.That(a.AllClose(b, 0, 0.0001), Is.True);
        Assert.That(b.AllClose(a, 0, 0.0001), Is.True); Assert.That(a.AllClose(b, 0, 0), Is.False);
        Assert.That(a.ContentEquals(a.Copy()), Is.True);
        Assert.That(NDArray.FromRows([1, 3], [2, 4]).Transpose().ContentEquals(NDArray.FromRows([1, 2], [3, 4])), Is.True);
        Assert.That(NDArray.FromArray([]).AllClose(NDArray.FromArray([]), 0, 0), Is.True);
        Assert.That(a.AllClose(a.Reshape(new Shape(1, 2)), 0, 0), Is.False); Assert.That(a.ContentEquals(a.Reshape(new Shape(1, 2))), Is.False);
        Assert.That(NDArray.FromArray([0.0]).ContentEquals(NDArray.FromArray([-0.0])), Is.True);
        var nan = NDArray.Scalar(double.NaN); Assert.That(nan.ContentEquals(nan), Is.False); Assert.That(nan.AllClose(nan, 1, 1), Is.False);
        var inf = NDArray.Scalar(double.PositiveInfinity); Assert.That(inf.AllClose(inf.Copy(), 0, 0), Is.True);
        Assert.That(inf.AllClose(NDArray.Scalar(1), 1, 1), Is.False);
        Assert.That(NDArray.Scalar(double.MaxValue).AllClose(NDArray.Scalar(-double.MaxValue), 0, 1), Is.False);
        Assert.Throws<ArgumentOutOfRangeException>(() => a.AllClose(b, -1, 0)); Assert.Throws<ArgumentOutOfRangeException>(() => a.AllClose(b, 0, double.NaN));
    }
}
