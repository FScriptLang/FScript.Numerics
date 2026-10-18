using System.Numerics.Tensors;

namespace FScript.Numerics;

public sealed partial class NDArray
{
    private enum Binary { Add, Subtract, Multiply, Divide }
    private static double Calculate(double a, double b, Binary operation) => operation switch
    {
        Binary.Add => a + b, Binary.Subtract => a - b, Binary.Multiply => a * b, Binary.Divide => a / b,
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };
    private NDArray Combine(NDArray other, Binary operation)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (!Shape.Equals(other.Shape)) throw new ArgumentException("Array shapes must match.", nameof(other));
        var result = Zeros(Shape);
        if (Size == 0) return result;
        if (TryGetContiguousSpan(out var a) && other.TryGetContiguousSpan(out var b))
        {
            switch (operation)
            {
                case Binary.Add: TensorPrimitives.Add<double>(a, b, result.WritableSpan); break;
                case Binary.Subtract: TensorPrimitives.Subtract<double>(a, b, result.WritableSpan); break;
                case Binary.Multiply: TensorPrimitives.Multiply<double>(a, b, result.WritableSpan); break;
                case Binary.Divide: TensorPrimitives.Divide<double>(a, b, result.WritableSpan); break;
            }
        }
        else
        {
            using var left = Offsets().GetEnumerator(); using var right = other.Offsets().GetEnumerator();
            int i = 0;
            while (left.MoveNext() && right.MoveNext()) result.storage[i++] = Calculate(storage[left.Current], other.storage[right.Current], operation);
        }
        return result;
    }
    private NDArray Combine(double scalar, Binary operation)
    {
        var result = Zeros(Shape);
        if (Size == 0) return result;
        if (TryGetContiguousSpan(out var a))
        {
            switch (operation)
            {
                case Binary.Add: TensorPrimitives.Add<double>(a, scalar, result.WritableSpan); break;
                case Binary.Subtract: TensorPrimitives.Subtract<double>(a, scalar, result.WritableSpan); break;
                case Binary.Multiply: TensorPrimitives.Multiply<double>(a, scalar, result.WritableSpan); break;
                case Binary.Divide: TensorPrimitives.Divide<double>(a, scalar, result.WritableSpan); break;
            }
        }
        else { int i = 0; foreach (int offset in Offsets()) result.storage[i++] = Calculate(storage[offset], scalar, operation); }
        return result;
    }
    public NDArray Add(NDArray other) => Combine(other, Binary.Add);
    public NDArray Subtract(NDArray other) => Combine(other, Binary.Subtract);
    public NDArray Multiply(NDArray other) => Combine(other, Binary.Multiply);
    public NDArray Divide(NDArray other) => Combine(other, Binary.Divide);
    public NDArray Add(double scalar) => Combine(scalar, Binary.Add);
    public NDArray Subtract(double scalar) => Combine(scalar, Binary.Subtract);
    public NDArray Multiply(double scalar) => Combine(scalar, Binary.Multiply);
    public NDArray Divide(double scalar) => Combine(scalar, Binary.Divide);
    public NDArray Negate()
    {
        var result = Zeros(Shape);
        if (Size == 0) return result;
        if (TryGetContiguousSpan(out var span)) TensorPrimitives.Negate<double>(span, result.WritableSpan);
        else { int i = 0; foreach (int offset in Offsets()) result.storage[i++] = -storage[offset]; }
        return result;
    }
    public double Sum()
    {
        if (Size == 0) return 0;
        if (TryGetContiguousSpan(out var span)) return TensorPrimitives.Sum<double>(span);
        double sum = 0; foreach (int offset in Offsets()) sum += storage[offset]; return sum;
    }
    private void RequireElements() { if (Size == 0) throw new InvalidOperationException("Reduction requires a nonempty array."); }
    public double Mean() { RequireElements(); return Sum() / Size; }
    public double Min()
    {
        RequireElements();
        if (TryGetContiguousSpan(out var span)) return TensorPrimitives.Min<double>(span);
        double minimum = double.PositiveInfinity; foreach (int offset in Offsets()) minimum = Math.Min(minimum, storage[offset]); return minimum;
    }
    public double Max()
    {
        RequireElements();
        if (TryGetContiguousSpan(out var span)) return TensorPrimitives.Max<double>(span);
        double maximum = double.NegativeInfinity; foreach (int offset in Offsets()) maximum = Math.Max(maximum, storage[offset]); return maximum;
    }
    public double Dot(NDArray other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (Rank != 1 || other.Rank != 1 || Size != other.Size) throw new ArgumentException("Dot requires vectors of equal length.", nameof(other));
        if (Size == 0) return 0;
        if (TryGetContiguousSpan(out var a) && other.TryGetContiguousSpan(out var b)) return TensorPrimitives.Dot<double>(a, b);
        double sum = 0; for (int k = 0; k < Size; k++) sum += storage[Offset + k * strides[0]] * other.storage[other.Offset + k * other.strides[0]]; return sum;
    }
    public NDArray Matvec(NDArray vector)
    {
        ArgumentNullException.ThrowIfNull(vector);
        if (Rank != 2 || vector.Rank != 1 || Shape[1] != vector.Size) throw new ArgumentException("Matvec requires a matrix and a vector matching its columns.", nameof(vector));
        var result = Zeros(new Shape(Shape[0]));
        if (result.Size == 0 || Shape[1] == 0) return result;
        for (int i = 0; i < Shape[0]; i++)
            for (int k = 0; k < Shape[1]; k++)
                result.storage[i] += storage[Offset + i * strides[0] + k * strides[1]] * vector.storage[vector.Offset + k * vector.strides[0]];
        return result;
    }
    public NDArray Matmul(NDArray other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (Rank != 2 || other.Rank != 2 || Shape[1] != other.Shape[0]) throw new ArgumentException("Matmul requires matrices with matching inner dimensions.", nameof(other));
        var result = Zeros(new Shape(Shape[0], other.Shape[1]));
        if (result.Size == 0 || Shape[1] == 0) return result;
        for (int i = 0; i < Shape[0]; i++)
            for (int j = 0; j < other.Shape[1]; j++)
                for (int k = 0; k < Shape[1]; k++)
                    result.storage[i * other.Shape[1] + j] += storage[Offset + i * strides[0] + k * strides[1]] * other.storage[other.Offset + k * other.strides[0] + j * other.strides[1]];
        return result;
    }
    public bool ContentEquals(NDArray other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (!Shape.Equals(other.Shape)) return false;
        using var left = Offsets().GetEnumerator(); using var right = other.Offsets().GetEnumerator();
        while (left.MoveNext() && right.MoveNext()) if (storage[left.Current] != other.storage[right.Current]) return false;
        return true;
    }
    public bool AllClose(NDArray other, double absoluteTolerance, double relativeTolerance)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (!double.IsFinite(absoluteTolerance) || absoluteTolerance < 0 || !double.IsFinite(relativeTolerance) || relativeTolerance < 0)
            throw new ArgumentOutOfRangeException(nameof(absoluteTolerance), "Tolerances must be finite and nonnegative.");
        if (!Shape.Equals(other.Shape)) return false;
        using var left = Offsets().GetEnumerator(); using var right = other.Offsets().GetEnumerator();
        while (left.MoveNext() && right.MoveNext())
        {
            double a = storage[left.Current], b = other.storage[right.Current];
            if (a == b) continue;
            if (!double.IsFinite(a) || !double.IsFinite(b)) return false;
            double scale = Math.Max(Math.Abs(a), Math.Abs(b));
            double difference = Math.Abs(a - b), tolerance = absoluteTolerance + relativeTolerance * scale;
            if (double.IsFinite(difference) && double.IsFinite(tolerance)) { if (difference > tolerance) return false; }
            else if (Math.Abs(a / scale - b / scale) > absoluteTolerance / scale + relativeTolerance) return false;
        }
        return true;
    }
}
