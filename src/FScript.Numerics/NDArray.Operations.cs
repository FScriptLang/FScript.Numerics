using System.Numerics.Tensors;

namespace FScript.Numerics;

public sealed partial class NDArray
{
    private enum Binary
    {
        Add, Subtract, Multiply, Divide, Power, Minimum, Maximum
    }
    private static DType Promote(DType a, DType b)
    {
        if (a == DType.Bool || b == DType.Bool)
        {
            if (a == b)
                return DType.Bool;
            throw new ArgumentException("Boolean and numerical arrays cannot be implicitly mixed.");
        }
        if (a == DType.Float64 || b == DType.Float64)
            return DType.Float64;
        if (a == DType.Float32 || b == DType.Float32)
            return a == b ? DType.Float32 : DType.Float64;
        return a == DType.Int64 || b == DType.Int64 ? DType.Int64 : DType.Int32;
    }
    private static Shape Broadcasting(Shape left, Shape right)
    {
        var dimensions = new int[Math.Max(left.Rank, right.Rank)];
        for (int i = 1; i <= dimensions.Length; i++)
        {
            int a = i <= left.Rank ? left[left.Rank - i] : 1, b = i <= right.Rank ? right[right.Rank - i] : 1;
            if (a != b && a != 1 && b != 1)
                throw new ArgumentException($"Cannot broadcast shapes {left} and {right}.");
            dimensions[^i] = a == 1 ? b : a;
        }
        return new Shape(dimensions);
    }
    public NDArray BroadcastTo(Shape shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        if (shape.Rank < Rank || !Broadcasting(Shape, shape).Equals(shape))
            throw new ArgumentException($"Cannot broadcast shape {Shape} to {shape}.");
        var layout = new int[shape.Rank];
        for (int i = 1; i <= Rank; i++)
            layout[^i] = Shape[Rank - i] == 1 ? 0 : strides[Rank - i];
        return new(storage, shape, layout, Offset, true);
    }
    public NDArray ExpandDims(int axis)
    {
        if ((uint)axis > (uint)Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));
        var dimensions = Shape.ToArray().ToList();
        dimensions.Insert(axis, 1);
        var layout = strides.ToList();
        layout.Insert(axis, 0);
        return new(storage, new Shape(dimensions.ToArray()), layout.ToArray(), Offset, IsReadOnly);
    }
    public NDArray Squeeze()
    {
        var axes = Enumerable.Range(0, Rank).Where(d => Shape[d] != 1).ToArray();
        return new(storage, new Shape(axes.Select(d => Shape[d]).ToArray()), axes.Select(d => strides[d]).ToArray(), Offset, IsReadOnly);
    }
    public NDArray Flatten() => Copy().Reshape(new Shape(Size));
    public NDArray Astype(DType dtype)
    {
        var result = Zeros(Shape, dtype);
        int i = 0;
        foreach (int offset in Offsets())
        {
            if (DType is DType.Int32 or DType.Int64)
                result.WriteInteger(i++, ReadInteger(offset));
            else
                result.Write(i++, Read(offset));
        }
        return result;
    }
    private static double Calculate(double a, double b, Binary operation) => operation switch
    {
        Binary.Add => a + b,
        Binary.Subtract => a - b,
        Binary.Multiply => a * b,
        Binary.Divide => a / b,
        Binary.Power => Math.Pow(a, b),
        Binary.Minimum => Math.Min(a, b),
        Binary.Maximum => Math.Max(a, b),
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };
    private NDArray Combine(NDArray other, Binary operation)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (DType == DType.Bool || other.DType == DType.Bool)
            throw new ArgumentException("Arithmetic requires numerical arrays.");
        var shape = Broadcasting(Shape, other.Shape);
        var dtype = Promote(DType, other.DType);
        if (operation is Binary.Divide or Binary.Power && dtype is DType.Int32 or DType.Int64)
            dtype = DType.Float64;
        var result = Zeros(shape, dtype);
        if (result.Size == 0)
            return result;
        if (Shape.Equals(other.Shape) && dtype == DType.Float64 && TryGetContiguousSpan(out var a) && other.TryGetContiguousSpan(out var b))
        {
            switch (operation)
            {
                case Binary.Add:
                    TensorPrimitives.Add<double>(a, b, result.WritableSpan);
                    return result;
                case Binary.Subtract:
                    TensorPrimitives.Subtract<double>(a, b, result.WritableSpan);
                    return result;
                case Binary.Multiply:
                    TensorPrimitives.Multiply<double>(a, b, result.WritableSpan);
                    return result;
                case Binary.Divide:
                    TensorPrimitives.Divide<double>(a, b, result.WritableSpan);
                    return result;
            }
        }
        using var left = BroadcastTo(shape).Offsets().GetEnumerator();
        using var right = other.BroadcastTo(shape).Offsets().GetEnumerator();
        int i = 0;
        while (left.MoveNext() && right.MoveNext())
        {
            if (dtype is DType.Int32 or DType.Int64)
            {
                long x = ReadInteger(left.Current), y = other.ReadInteger(right.Current);
                result.WriteInteger(i++, operation switch
                {
                    Binary.Add => checked(x + y),
                    Binary.Subtract => checked(x - y),
                    Binary.Multiply => checked(x * y),
                    Binary.Minimum => Math.Min(x, y),
                    Binary.Maximum => Math.Max(x, y),
                    _ => throw new InvalidOperationException()
                });
            }
            else
                result.Write(i++, Calculate(Read(left.Current), other.Read(right.Current), operation));
        }
        return result;
    }
    private NDArray Combine(double scalar, Binary operation) => Combine(Scalar(scalar, DType == DType.Float32 ? DType.Float32 : DType.Float64), operation);
    public NDArray Add(NDArray other) => Combine(other, Binary.Add);
    public NDArray Subtract(NDArray other) => Combine(other, Binary.Subtract);
    public NDArray Multiply(NDArray other) => Combine(other, Binary.Multiply);
    public NDArray Divide(NDArray other) => Combine(other, Binary.Divide);
    public NDArray Power(NDArray other) => Combine(other, Binary.Power);
    public NDArray Minimum(NDArray other) => Combine(other, Binary.Minimum);
    public NDArray Maximum(NDArray other) => Combine(other, Binary.Maximum);
    public NDArray Add(double scalar) => Combine(scalar, Binary.Add);
    public NDArray Subtract(double scalar) => Combine(scalar, Binary.Subtract);
    public NDArray Multiply(double scalar) => Combine(scalar, Binary.Multiply);
    public NDArray Divide(double scalar) => Combine(scalar, Binary.Divide);
    public NDArray Power(double scalar) => Combine(scalar, Binary.Power);
    private NDArray Unary(Func<double, double> operation, bool floating = false)
    {
        if (DType == DType.Bool)
            throw new InvalidOperationException("A numerical array is required.");
        var dtype = floating && DType is DType.Int32 or DType.Int64 ? DType.Float64 : DType;
        var result = Zeros(Shape, dtype);
        int i = 0;
        foreach (int offset in Offsets())
            result.Write(i++, operation(Read(offset)));
        return result;
    }
    public NDArray Negate()
    {
        if (DType is not (DType.Int32 or DType.Int64))
            return Unary(x => -x);
        var result = Zeros(Shape, DType);
        int i = 0;
        foreach (int offset in Offsets())
            result.WriteInteger(i++, checked(-ReadInteger(offset)));
        return result;
    }
    public NDArray Abs()
    {
        if (DType is not (DType.Int32 or DType.Int64))
            return Unary(Math.Abs);
        var result = Zeros(Shape, DType);
        int i = 0;
        foreach (int offset in Offsets())
            result.WriteInteger(i++, Math.Abs(ReadInteger(offset)));
        return result;
    }
    public NDArray Sqrt() => Unary(Math.Sqrt, true);
    public NDArray Exp() => Unary(Math.Exp, true);
    public NDArray Log() => Unary(Math.Log, true);
    public NDArray Clip(double minimum, double maximum)
    {
        if (double.IsNaN(minimum) || double.IsNaN(maximum) || minimum > maximum)
            throw new ArgumentException("Clip bounds must be ordered and not NaN.");
        // Clip is explicitly a floating operation for integer inputs.
        return Unary(x => Math.Min(Math.Max(x, minimum), maximum), true);
    }
    private static bool NumericEqual(NDArray a, int i, NDArray b, int j)
    {
        bool ai = a.DType is DType.Int32 or DType.Int64 or DType.Bool, bi = b.DType is DType.Int32 or DType.Int64 or DType.Bool;
        if (ai && bi)
            return a.ReadInteger(i) == b.ReadInteger(j);
        if (!ai && !bi)
            return a.Read(i) == b.Read(j);
        long integer = ai ? a.ReadInteger(i) : b.ReadInteger(j);
        double number = ai ? b.Read(j) : a.Read(i);
        return double.IsInteger(number) && number >= long.MinValue && number < 9223372036854775808.0 && (long)number == integer;
    }
    private static bool OrderedCompare(NDArray a, int i, NDArray b, int j, Func<double, double, bool> predicate)
    {
        bool ai = a.DType is DType.Int32 or DType.Int64 or DType.Bool, bi = b.DType is DType.Int32 or DType.Int64 or DType.Bool;
        if (ai && bi)
            return predicate(a.ReadInteger(i).CompareTo(b.ReadInteger(j)), 0);
        if (!ai && !bi)
            return predicate(a.Read(i), b.Read(j));
        long integer = ai ? a.ReadInteger(i) : b.ReadInteger(j);
        double number = ai ? b.Read(j) : a.Read(i);
        if (double.IsNaN(number))
            return false;
        int comparison;
        if (number >= 9223372036854775808.0)
            comparison = -1;
        else if (number < long.MinValue)
            comparison = 1;
        else
        {
            long truncated = (long)number;
            comparison = integer.CompareTo(truncated);
            if (comparison == 0)
                comparison = number == truncated ? 0 : number > truncated ? -1 : 1;
        }
        return predicate(ai ? comparison : -comparison, 0);
    }
    private NDArray Compare(NDArray other, Func<double, double, bool> predicate, bool equality = false)
    {
        ArgumentNullException.ThrowIfNull(other);
        var shape = Broadcasting(Shape, other.Shape);
        var result = Zeros(shape, DType.Bool);
        using var left = BroadcastTo(shape).Offsets().GetEnumerator();
        using var right = other.BroadcastTo(shape).Offsets().GetEnumerator();
        int i = 0;
        while (left.MoveNext() && right.MoveNext())
            result.Write(i++, (equality ? NumericEqual(this, left.Current, other, right.Current) : OrderedCompare(this, left.Current, other, right.Current, predicate)) ? 1 : 0);
        return result;
    }
    public NDArray EqualElements(NDArray other) => Compare(other, (a, b) => a == b, true);
    public NDArray Less(NDArray other) => Compare(other, (a, b) => a < b);
    public NDArray LessEqual(NDArray other) => Compare(other, (a, b) => a <= b);
    public NDArray Greater(NDArray other) => Compare(other, (a, b) => a > b);
    public NDArray GreaterEqual(NDArray other) => Compare(other, (a, b) => a >= b);
    public double Sum()
    {
        if (Size == 0)
            return 0;
        if (TryGetContiguousSpan(out var span))
            return TensorPrimitives.Sum<double>(span);
        double sum = 0;
        foreach (int offset in Offsets())
            sum += Read(offset);
        return sum;
    }
    private void RequireElements()
    {
        if (Size == 0)
            throw new InvalidOperationException("Reduction requires a nonempty array.");
    }
    public double Product()
    {
        double product = 1;
        foreach (int offset in Offsets())
            product *= Read(offset);
        return product;
    }
    public double Mean()
    {
        RequireElements();
        return Sum() / Size;
    }
    public double Min()
    {
        RequireElements();
        double minimum = double.PositiveInfinity;
        foreach (int offset in Offsets())
            minimum = Math.Min(minimum, Read(offset));
        return minimum;
    }
    public double Max()
    {
        RequireElements();
        double maximum = double.NegativeInfinity;
        foreach (int offset in Offsets())
            maximum = Math.Max(maximum, Read(offset));
        return maximum;
    }
    public double Variance(int degreesOfFreedom = 0)
    {
        if (degreesOfFreedom < 0 || Size <= degreesOfFreedom)
            throw new ArgumentOutOfRangeException(nameof(degreesOfFreedom), "Variance requires more elements than degrees of freedom.");
        double mean = Mean(), sum = 0;
        foreach (int offset in Offsets())
        {
            double difference = Read(offset) - mean;
            sum += difference * difference;
        }
        return sum / (Size - degreesOfFreedom);
    }
    public double Std(int degreesOfFreedom = 0) => Math.Sqrt(Variance(degreesOfFreedom));
    public bool All()
    {
        foreach (int offset in Offsets())
            if (Read(offset) == 0)
                return false;
        return true;
    }
    public bool Any()
    {
        foreach (int offset in Offsets())
            if (Read(offset) != 0)
                return true;
        return false;
    }
    public NDArray SumAxis(int axis, bool keepDimensions = false) => ReduceAxis(axis, keepDimensions, "sum");
    public NDArray ProductAxis(int axis, bool keepDimensions = false) => ReduceAxis(axis, keepDimensions, "product");
    public NDArray MinAxis(int axis, bool keepDimensions = false) => ReduceAxis(axis, keepDimensions, "min");
    public NDArray MaxAxis(int axis, bool keepDimensions = false) => ReduceAxis(axis, keepDimensions, "max");
    public NDArray MeanAxis(int axis, bool keepDimensions = false) => ReduceAxis(axis, keepDimensions, "mean");
    public NDArray VarianceAxis(int axis, bool keepDimensions = false, int degreesOfFreedom = 0) => ReduceAxis(axis, keepDimensions, "variance", degreesOfFreedom);
    public NDArray StdAxis(int axis, bool keepDimensions = false, int degreesOfFreedom = 0) => ReduceAxis(axis, keepDimensions, "std", degreesOfFreedom);
    public NDArray AllAxis(int axis, bool keepDimensions = false) => ReduceAxis(axis, keepDimensions, "all");
    public NDArray AnyAxis(int axis, bool keepDimensions = false) => ReduceAxis(axis, keepDimensions, "any");
    private NDArray ReduceAxis(int axis, bool keep, string operation, int ddof = 0)
    {
        if ((uint)axis >= (uint)Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));
        if (ddof < 0)
            throw new ArgumentOutOfRangeException(nameof(ddof));
        var dimensions = Shape.ToArray().ToList();
        if (keep)
            dimensions[axis] = 1;
        else
            dimensions.RemoveAt(axis);
        var dtype = operation is "all" or "any" ? DType.Bool : operation is "mean" or "variance" or "std" ? (DType == DType.Float32 ? DType.Float32 : DType.Float64) : operation is "sum" or "product" && DType is DType.Int32 or DType.Int64 or DType.Bool ? DType.Int64 : DType;
        var result = Zeros(new Shape(dimensions.ToArray()), dtype);
        var outputIndex = new int[result.Rank];
        var indices = new int[Rank];
        for (int n = 0; n < result.Size; n++)
        {
            for (int d = 0, j = 0; d < Rank; d++)
            {
                if (d != axis)
                    indices[d] = outputIndex[keep ? d : j++];
            }
            int count = Shape[axis];
            if (count == 0 && operation is not ("sum" or "product" or "all" or "any"))
                throw new InvalidOperationException("Reduction requires a nonempty axis.");
            if (operation is "variance" or "std" && count <= ddof)
                throw new ArgumentOutOfRangeException(nameof(ddof), "Variance requires more elements than degrees of freedom.");
            double value = operation is "product" or "all" ? 1 : operation == "min" ? double.PositiveInfinity : operation == "max" ? double.NegativeInfinity : 0;
            long integer = operation == "product" ? 1 : 0;
            for (int k = 0; k < count; k++)
            {
                indices[axis] = k;
                int position = Position(indices);
                double x = Read(position);
                if (dtype == DType.Int64 && operation is "sum" or "product")
                {
                    long y = ReadInteger(position);
                    integer = operation == "sum" ? checked(integer + y) : checked(integer * y);
                }
                else if (dtype is DType.Int32 or DType.Int64 && operation is "min" or "max")
                {
                    long y = ReadInteger(position);
                    integer = k == 0 ? y : operation == "min" ? Math.Min(integer, y) : Math.Max(integer, y);
                }
                else
                    value = operation switch
                    {
                        "product" => value * x,
                        "min" => Math.Min(value, x),
                        "max" => Math.Max(value, x),
                        "all" => value != 0 && x != 0 ? 1 : 0,
                        "any" => value != 0 || x != 0 ? 1 : 0,
                        _ => value + x
                    };
            }
            if (operation == "mean")
                value /= count;
            if (operation is "variance" or "std")
            {
                double mean = value / count;
                value = 0;
                for (int k = 0; k < count; k++)
                {
                    indices[axis] = k;
                    double x = Read(Position(indices)) - mean;
                    value += x * x;
                }
                value /= count - ddof;
                if (operation == "std")
                    value = Math.Sqrt(value);
            }
            if (dtype is DType.Int32 or DType.Int64)
                result.WriteInteger(n, integer);
            else
                result.Write(n, value);
            Increment(outputIndex, result.Shape);
        }
        return result;
    }
    public static NDArray Concatenate(IReadOnlyList<NDArray> arrays, int axis = 0)
    {
        ArgumentNullException.ThrowIfNull(arrays);
        if (arrays.Count == 0 || arrays.Any(a => a is null))
            throw new ArgumentException("Provide at least one non-null array.");
        var first = arrays[0];
        if ((uint)axis >= (uint)first.Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));
        var dimensions = first.Shape.ToArray();
        dimensions[axis] = 0;
        var dtype = first.DType;
        foreach (var a in arrays)
        {
            if (a.Rank != first.Rank || Enumerable.Range(0, a.Rank).Any(d => d != axis && a.Shape[d] != first.Shape[d]))
                throw new ArgumentException("Concatenate shapes must agree outside the selected axis.");
            dimensions[axis] = checked(dimensions[axis] + a.Shape[axis]);
            dtype = Promote(dtype, a.DType);
        }
        var result = Zeros(new Shape(dimensions), dtype);
        int start = 0;
        foreach (var a in arrays)
        {
            var indices = new int[a.Rank];
            var destination = new int[a.Rank];
            foreach (int offset in a.Offsets())
            {
                indices.CopyTo(destination, 0);
                destination[axis] += start;
                int position = result.Position(destination);
                if (dtype is DType.Int32 or DType.Int64)
                    result.WriteInteger(position, a.ReadInteger(offset));
                else
                    result.Write(position, a.Read(offset));
                Increment(indices, a.Shape);
            }
            start += a.Shape[axis];
        }
        return result;
    }
    public static NDArray Stack(IReadOnlyList<NDArray> arrays, int axis = 0)
    {
        ArgumentNullException.ThrowIfNull(arrays);
        if (arrays.Count == 0 || arrays.Any(a => a is null))
            throw new ArgumentException("Provide at least one non-null array.");
        if (arrays.Any(a => !a.Shape.Equals(arrays[0].Shape)))
            throw new ArgumentException("Stack requires equal shapes.");
        return Concatenate(arrays.Select(a => a.ExpandDims(axis)).ToArray(), axis);
    }
    public static NDArray Where(NDArray condition, NDArray whenTrue, NDArray whenFalse)
    {
        ArgumentNullException.ThrowIfNull(condition);
        ArgumentNullException.ThrowIfNull(whenTrue);
        ArgumentNullException.ThrowIfNull(whenFalse);
        if (condition.DType != DType.Bool)
            throw new ArgumentException("Where requires a boolean condition.");
        var shape = Broadcasting(condition.Shape, Broadcasting(whenTrue.Shape, whenFalse.Shape));
        var result = Zeros(shape, Promote(whenTrue.DType, whenFalse.DType));
        using var c = condition.BroadcastTo(shape).Offsets().GetEnumerator();
        using var a = whenTrue.BroadcastTo(shape).Offsets().GetEnumerator();
        using var b = whenFalse.BroadcastTo(shape).Offsets().GetEnumerator();
        int i = 0;
        while (c.MoveNext() && a.MoveNext() && b.MoveNext())
        {
            bool yes = condition.Read(c.Current) != 0;
            var source = yes ? whenTrue : whenFalse;
            int offset = yes ? a.Current : b.Current;
            if (result.DType is DType.Int32 or DType.Int64)
                result.WriteInteger(i++, source.ReadInteger(offset));
            else
                result.Write(i++, source.Read(offset));
        }
        return result;
    }
    private static void RequireFloating(NDArray array)
    {
        if (array.DType is not (DType.Float32 or DType.Float64))
            throw new ArgumentException("Linear algebra requires float32 or float64 arrays; convert explicitly.");
    }
    public double Dot(NDArray other)
    {
        ArgumentNullException.ThrowIfNull(other);
        RequireFloating(this);
        RequireFloating(other);
        if (Rank != 1 || other.Rank != 1 || Size != other.Size)
            throw new ArgumentException("Dot requires vectors of equal length.", nameof(other));
        if (Size == 0)
            return 0;
        if (TryGetContiguousSpan(out var a) && other.TryGetContiguousSpan(out var b))
            return TensorPrimitives.Dot<double>(a, b);
        double sum = 0;
        for (int k = 0; k < Size; k++)
            sum += Read(Offset + k * strides[0]) * other.Read(other.Offset + k * other.strides[0]);
        return sum;
    }
    public NDArray Matvec(NDArray vector)
    {
        ArgumentNullException.ThrowIfNull(vector);
        RequireFloating(this);
        RequireFloating(vector);
        if (Rank != 2 || vector.Rank != 1 || Shape[1] != vector.Size)
            throw new ArgumentException("Matvec requires a matrix and a vector matching its columns.", nameof(vector));
        var result = Zeros(new Shape(Shape[0]), Promote(DType, vector.DType));
        for (int i = 0; i < Shape[0]; i++)
        {
            double sum = 0;
            for (int k = 0; k < Shape[1]; k++)
                sum += Read(Offset + i * strides[0] + k * strides[1]) * vector.Read(vector.Offset + k * vector.strides[0]);
            result.Write(i, sum);
        }
        return result;
    }
    public NDArray Matmul(NDArray other)
    {
        ArgumentNullException.ThrowIfNull(other);
        RequireFloating(this);
        RequireFloating(other);
        if (Rank != 2 || other.Rank != 2 || Shape[1] != other.Shape[0])
            throw new ArgumentException("Matmul requires matrices with matching inner dimensions.", nameof(other));
        var result = Zeros(new Shape(Shape[0], other.Shape[1]), Promote(DType, other.DType));
        for (int i = 0; i < Shape[0]; i++)
            for (int j = 0; j < other.Shape[1]; j++)
            {
                double sum = 0;
                for (int k = 0; k < Shape[1]; k++)
                    sum += Read(Offset + i * strides[0] + k * strides[1]) * other.Read(other.Offset + k * other.strides[0] + j * other.strides[1]);
                result.Write(i * other.Shape[1] + j, sum);
            }
        return result;
    }
    public bool ContentEquals(NDArray other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (!Shape.Equals(other.Shape))
            return false;
        using var left = Offsets().GetEnumerator();
        using var right = other.Offsets().GetEnumerator();
        while (left.MoveNext() && right.MoveNext())
            if (!NumericEqual(this, left.Current, other, right.Current))
                return false;
        return true;
    }
    public bool AllClose(NDArray other, double absoluteTolerance, double relativeTolerance)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (!double.IsFinite(absoluteTolerance) || absoluteTolerance < 0 || !double.IsFinite(relativeTolerance) || relativeTolerance < 0)
            throw new ArgumentOutOfRangeException(nameof(absoluteTolerance), "Tolerances must be finite and nonnegative.");
        if (!Shape.Equals(other.Shape))
            return false;
        if (absoluteTolerance == 0 && relativeTolerance == 0)
            return ContentEquals(other);
        using var left = Offsets().GetEnumerator();
        using var right = other.Offsets().GetEnumerator();
        while (left.MoveNext() && right.MoveNext())
        {
            if (NumericEqual(this, left.Current, other, right.Current))
                continue;
            double a = Read(left.Current), b = other.Read(right.Current);
            if (!double.IsFinite(a) || !double.IsFinite(b))
                return false;
            double difference = DType is DType.Int32 or DType.Int64 && other.DType is DType.Int32 or DType.Int64 ? (double)Math.Abs((decimal)ReadInteger(left.Current) - other.ReadInteger(right.Current)) : Math.Abs(a - b);
            double scale = Math.Max(Math.Abs(a), Math.Abs(b)), tolerance = absoluteTolerance + relativeTolerance * scale;
            if (double.IsFinite(difference) && double.IsFinite(tolerance))
            {
                if (difference > tolerance)
                    return false;
            }
            else if (Math.Abs(a / scale - b / scale) > absoluteTolerance / scale + relativeTolerance)
                return false;
        }
        return true;
    }
}
