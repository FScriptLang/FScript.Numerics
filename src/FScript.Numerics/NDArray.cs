namespace FScript.Numerics;

/// <summary>An array with immutable layout and explicitly mutable, owner-anchored storage.</summary>
public sealed partial class NDArray
{
    private readonly Array storage;
    private readonly int[] strides;
    private readonly IReadOnlyList<int> strideView;
    private NDArray(Array storage, Shape shape, int[] strides, int offset, bool readOnly = false)
    {
        ArgumentNullException.ThrowIfNull(storage);
        ArgumentNullException.ThrowIfNull(shape);
        if (strides.Length != shape.Rank || strides.Any(s => s < 0))
            throw new ArgumentException("Invalid element strides.", nameof(strides));
        if (offset < 0 || offset > storage.Length)
            throw new ArgumentOutOfRangeException(nameof(offset));
        if (shape.Size > 0)
        {
            long last = offset;
            for (int d = 0; d < shape.Rank; d++)
                last = checked(last + checked((long)(shape[d] - 1) * strides[d]));
            if (last >= storage.Length)
                throw new ArgumentException("Layout extends beyond storage.", nameof(shape));
        }
        DType = storage switch
        {
            double[] => DType.Float64,
            float[] => DType.Float32,
            int[] => DType.Int32,
            long[] => DType.Int64,
            bool[] => DType.Bool,
            _ => throw new ArgumentException("Unsupported storage type.")
        };
        this.storage = storage;
        Shape = shape;
        this.strides = (int[])strides.Clone();
        strideView = Array.AsReadOnly(this.strides);
        Offset = offset;
        IsReadOnly = readOnly;
    }
    public Shape Shape
    {
        get;
    }
    public IReadOnlyList<int> Strides => strideView;
    public int Offset
    {
        get;
    }
    public int Rank => Shape.Rank;
    public int Size => Shape.Size;
    public DType DType
    {
        get;
    }
    public bool IsReadOnly
    {
        get;
    }
    public bool IsContiguous => Contiguous(false);
    public bool IsFortranContiguous => Contiguous(true);
    private bool Contiguous(bool fortran)
    {
        if (Size == 0)
            return true;
        int expected = 1;
        for (int i = 0; i < Rank; i++)
        {
            int d = fortran ? i : Rank - 1 - i;
            if (Shape[d] > 1 && strides[d] != expected)
                return false;
            expected = checked(expected * Shape[d]);
        }
        return true;
    }
    private static int[] LayoutStrides(Shape shape, bool fortran = false)
    {
        var result = new int[shape.Rank];
        int stride = 1;
        for (int i = 0; i < shape.Rank; i++)
        {
            int d = fortran ? i : shape.Rank - 1 - i;
            result[d] = stride;
            stride = checked(stride * shape[d]);
        }
        return result;
    }
    private static NDArray Own(Array data, Shape shape) => new(data, shape, LayoutStrides(shape), 0);
    internal bool SharesStorageWith(NDArray other) => ReferenceEquals(storage, other.storage);
    internal static NDArray CreateView(double[] data, Shape shape, int[] strides, int offset) => new(data, shape, strides, offset);
    internal Span<double> WritableSpan => IsContiguous && !IsReadOnly && storage is double[] data ? data.AsSpan(Offset, Size) : throw new InvalidOperationException("Writable contiguous float64 storage is required.");
    public bool TryGetContiguousSpan(out ReadOnlySpan<double> span)
    {
        if (IsContiguous && storage is double[] data)
        {
            span = data.AsSpan(Offset, Size);
            return true;
        }
        span = default;
        return false;
    }
    public bool TryGetContiguousFloat32Span(out ReadOnlySpan<float> span)
    {
        if (IsContiguous && storage is float[] data)
        {
            span = data.AsSpan(Offset, Size);
            return true;
        }
        span = default;
        return false;
    }
    private double Read(int position) => storage switch { double[] a => a[position], float[] a => a[position], int[] a => a[position], long[] a => a[position], bool[] a => a[position] ? 1 : 0, _ => throw new InvalidOperationException() };
    private long ReadInteger(int position) => storage switch { int[] a => a[position], long[] a => a[position], bool[] a => a[position] ? 1 : 0, _ => checked((long)Read(position)) };
    private void Write(int position, double value)
    {
        switch (storage)
        {
            case double[] a:
                a[position] = value;
                break;
            case float[] a:
                a[position] = (float)value;
                break;
            case int[] a:
                a[position] = checked((int)value);
                break;
            case long[] a:
                a[position] = checked((long)value);
                break;
            case bool[] a:
                a[position] = value != 0;
                break;
        }
    }
    private void WriteInteger(int position, long value)
    {
        switch (storage)
        {
            case int[] a:
                a[position] = checked((int)value);
                break;
            case long[] a:
                a[position] = value;
                break;
            case float[] a:
                a[position] = (float)value;
                break;
            case double[] a:
                a[position] = value;
                break;
            case bool[] a:
                a[position] = value != 0;
                break;
        }
    }
    private static Array Allocate(int size, DType dtype) => dtype switch
    {
        DType.Float64 => new double[size],
        DType.Float32 => new float[size],
        DType.Int32 => new int[size],
        DType.Int64 => new long[size],
        DType.Bool => new bool[size],
        _ => throw new ArgumentOutOfRangeException(nameof(dtype))
    };
    public static NDArray Scalar(double value, DType dtype = DType.Float64) => Full(new Shape(), value, dtype);
    public static NDArray FromArray(double[] values, DType dtype = DType.Float64)
    {
        ArgumentNullException.ThrowIfNull(values);
        return FromArray(values, new Shape(values.Length), dtype);
    }
    public static NDArray FromArray(ReadOnlySpan<double> values, Shape shape, DType dtype = DType.Float64)
    {
        ArgumentNullException.ThrowIfNull(shape);
        if (values.Length != shape.Size)
            throw new ArgumentException("Values must match the shape's element count.", nameof(values));
        if (dtype == DType.Float64)
            return Own(values.ToArray(), shape);
        var result = Zeros(shape, dtype);
        for (int i = 0; i < values.Length; i++)
            result.Write(i, values[i]);
        return result;
    }
    public static NDArray FromFloat32(float[] values, Shape? shape = null)
    {
        ArgumentNullException.ThrowIfNull(values);
        return FromTyped((float[])values.Clone(), shape);
    }
    public static NDArray FromInt32(int[] values, Shape? shape = null)
    {
        ArgumentNullException.ThrowIfNull(values);
        return FromTyped((int[])values.Clone(), shape);
    }
    public static NDArray FromInt64(long[] values, Shape? shape = null)
    {
        ArgumentNullException.ThrowIfNull(values);
        return FromTyped((long[])values.Clone(), shape);
    }
    public static NDArray FromBooleans(bool[] values, Shape? shape = null)
    {
        ArgumentNullException.ThrowIfNull(values);
        return FromTyped((bool[])values.Clone(), shape);
    }
    private static NDArray FromTyped(Array data, Shape? shape)
    {
        shape ??= new Shape(data.Length);
        if (data.Length != shape.Size)
            throw new ArgumentException("Values must match the shape's element count.", nameof(shape));
        return Own(data, shape);
    }
    public static NDArray FromRows(params double[][] rows) => FromRows(DType.Float64, rows);
    public static NDArray FromRows(DType dtype, params double[][] rows)
    {
        ArgumentNullException.ThrowIfNull(rows);
        if (rows.Any(row => row is null))
            throw new ArgumentException("Rows cannot be null.", nameof(rows));
        int columns = rows.Length == 0 ? 0 : rows[0].Length;
        if (rows.Any(row => row.Length != columns))
            throw new ArgumentException("Rows must be rectangular.", nameof(rows));
        var result = Zeros(new Shape(rows.Length, columns), dtype);
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < columns; j++)
                result.Write(i * columns + j, rows[i][j]);
        return result;
    }
    public static NDArray Zeros(Shape shape, DType dtype = DType.Float64)
    {
        ArgumentNullException.ThrowIfNull(shape);
        var layout = LayoutStrides(shape);
        return new(Allocate(shape.Size, dtype), shape, layout, 0);
    }
    public static NDArray Ones(Shape shape, DType dtype = DType.Float64) => Full(shape, 1, dtype);
    public static NDArray Full(Shape shape, double value, DType dtype = DType.Float64)
    {
        var result = Zeros(shape, dtype);
        for (int i = 0; i < result.Size; i++)
            result.Write(i, value);
        return result;
    }
    public static NDArray Identity(int size, DType dtype = DType.Float64)
    {
        var result = Zeros(new Shape(size, size), dtype);
        for (int i = 0; i < size; i++)
            result.Write(i * size + i, 1);
        return result;
    }
    public static NDArray Arange(double start, double stop, double step, DType dtype = DType.Float64)
    {
        if (!double.IsFinite(start) || !double.IsFinite(stop) || !double.IsFinite(step) || step == 0)
            throw new ArgumentException("Arange requires finite bounds and a finite nonzero step.");
        if (step > 0 ? start >= stop : start <= stop)
            return Zeros(new Shape(0), dtype);
        double difference = stop - start;
        double length = Math.Ceiling(double.IsFinite(difference) ? difference / step : stop / step - start / step);
        if (!double.IsFinite(length) || length > Array.MaxLength)
            throw new OverflowException("Arange exceeds the managed array element limit.");
        int count = Math.Max(1, checked((int)length));
        var data = new double[count];
        for (int i = 0; i < count; i++)
            data[i] = Math.FusedMultiplyAdd(i, step, start);
        while (count > 0 && (step > 0 ? data[count - 1] >= stop : data[count - 1] <= stop))
            count--;
        return FromArray(data.AsSpan(0, count), new Shape(count), dtype);
    }
    public static NDArray Linspace(double start, double stop, int count, DType dtype = DType.Float64)
    {
        if (!double.IsFinite(start) || !double.IsFinite(stop) || count < 0)
            throw new ArgumentException("Linspace requires finite endpoints and a nonnegative count.");
        var result = Zeros(new Shape(count), dtype);
        for (int i = 0; i < count; i++)
            result.Write(i, i == count - 1 && count > 1 ? stop : count <= 1 ? start : start * (1 - (double)i / (count - 1)) + stop * ((double)i / (count - 1)));
        return result;
    }
    private int Position(int[] indices)
    {
        ArgumentNullException.ThrowIfNull(indices);
        if (indices.Length != Rank)
            throw new ArgumentException("Provide exactly one index per dimension.", nameof(indices));
        int position = Offset;
        for (int d = 0; d < Rank; d++)
        {
            if ((uint)indices[d] >= (uint)Shape[d])
                throw new ArgumentOutOfRangeException(nameof(indices), "Index is outside its dimension.");
            position = checked(position + checked(indices[d] * strides[d]));
        }
        return position;
    }
    public double Get(params int[] indices) => Read(Position(indices));
    public long GetInt64(params int[] indices)
    {
        RequireInteger();
        return ReadInteger(Position(indices));
    }
    public bool GetBoolean(params int[] indices)
    {
        if (DType != DType.Bool)
            throw new InvalidOperationException("A bool array is required.");
        return Read(Position(indices)) != 0;
    }
    private void RequireWritable()
    {
        if (IsReadOnly)
            throw new InvalidOperationException("Broadcast views are read-only; copy first to obtain writable storage.");
    }
    private void RequireInteger()
    {
        if (DType is not (DType.Int32 or DType.Int64))
            throw new InvalidOperationException("An integer array is required.");
    }
    public void Set(double value, params int[] indices)
    {
        RequireWritable();
        Write(Position(indices), value);
    }
    public void SetInt64(long value, params int[] indices)
    {
        RequireWritable();
        RequireInteger();
        WriteInteger(Position(indices), value);
    }
    public void SetBoolean(bool value, params int[] indices)
    {
        RequireWritable();
        if (DType != DType.Bool)
            throw new InvalidOperationException("A bool array is required.");
        Write(Position(indices), value ? 1 : 0);
    }
    public NDArray Transpose() => Permute(Enumerable.Range(0, Rank).Reverse().ToArray());
    public NDArray Permute(params int[] axes)
    {
        ArgumentNullException.ThrowIfNull(axes);
        if (axes.Length != Rank || axes.Distinct().Count() != Rank || axes.Any(a => a < 0 || a >= Rank))
            throw new ArgumentException("Axes must be a permutation of every dimension.", nameof(axes));
        return new(storage, new Shape(axes.Select(a => Shape[a]).ToArray()), axes.Select(a => strides[a]).ToArray(), Offset, IsReadOnly);
    }
    public NDArray Reshape(Shape shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        if (shape.Size != Size)
            throw new ArgumentException("Reshape must preserve element count.", nameof(shape));
        if (!IsContiguous)
            throw new InvalidOperationException("Reshape requires a contiguous array; call Copy first.");
        return new(storage, shape, LayoutStrides(shape), Offset, IsReadOnly);
    }
    public NDArray Slice(int axis, int start, int stop, int step = 1)
    {
        if ((uint)axis >= (uint)Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));
        if (step <= 0)
            throw new ArgumentOutOfRangeException(nameof(step), "Slice step must be positive.");
        if (start < 0 || stop < start || stop > Shape[axis])
            throw new ArgumentOutOfRangeException(nameof(start), "Require 0 <= start <= stop <= dimension.");
        int length = stop - start, count = length / step + (length % step == 0 ? 0 : 1);
        var dimensions = Shape.ToArray();
        dimensions[axis] = count;
        var shape = new Shape(dimensions);
        var newStrides = (int[])strides.Clone();
        newStrides[axis] = checked(newStrides[axis] * step);
        int offset = shape.Size == 0 ? Offset : checked(Offset + checked(start * strides[axis]));
        return new(storage, shape, newStrides, offset, IsReadOnly);
    }
    public NDArray Copy()
    {
        var result = Zeros(Shape, DType);
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
    public NDArray AsContiguous() => IsContiguous ? this : Copy();
    public NDArray AsFortranContiguous()
    {
        if (IsFortranContiguous)
            return this;
        var result = new NDArray(Allocate(Size, DType), Shape, LayoutStrides(Shape, true), 0);
        var indices = new int[Rank];
        for (int n = 0; n < Size; n++)
        {
            int from = Position(indices), to = result.Position(indices);
            if (DType is DType.Int32 or DType.Int64)
                result.WriteInteger(to, ReadInteger(from));
            else
                result.Write(to, Read(from));
            Increment(indices, Shape);
        }
        return result;
    }
    public double[] ToArray()
    {
        if (TryGetContiguousSpan(out var span))
            return span.ToArray();
        var data = new double[Size];
        int i = 0;
        foreach (int offset in Offsets())
            data[i++] = Read(offset);
        return data;
    }
    public float[] ToFloat32Array()
    {
        if (TryGetContiguousFloat32Span(out var span))
            return span.ToArray();
        var data = new float[Size];
        int i = 0;
        foreach (int offset in Offsets())
            data[i++] = storage is long[] integers ? (float)integers[offset] : (float)Read(offset);
        return data;
    }
    public long[] ToInt64Array()
    {
        RequireInteger();
        var data = new long[Size];
        int i = 0;
        foreach (int offset in Offsets())
            data[i++] = ReadInteger(offset);
        return data;
    }
    public bool[] ToBooleanArray()
    {
        if (DType != DType.Bool)
            throw new InvalidOperationException("A bool array is required.");
        return ToArray().Select(x => x != 0).ToArray();
    }
    public double[][] ToRows()
    {
        if (Rank != 2)
            throw new InvalidOperationException("ToRows requires a matrix.");
        var rows = new double[Shape[0]][];
        for (int i = 0; i < rows.Length; i++)
        {
            rows[i] = new double[Shape[1]];
            for (int j = 0; j < rows[i].Length; j++)
                rows[i][j] = Read(Offset + i * strides[0] + j * strides[1]);
        }
        return rows;
    }
    private static void Increment(int[] indices, Shape shape)
    {
        for (int axis = indices.Length - 1; axis >= 0; axis--)
        {
            if (++indices[axis] < shape[axis])
                return;
            indices[axis] = 0;
        }
    }
    private IEnumerable<int> Offsets()
    {
        if (Size == 0)
            yield break;
        var indices = new int[Rank];
        for (int n = 0; n < Size; n++)
        {
            yield return Position(indices);
            Increment(indices, Shape);
        }
    }
}
