namespace FScript.Numerics;

/// <summary>A float64 array with immutable layout and potentially shared mutable storage.</summary>
public sealed partial class NDArray
{
    private readonly double[] storage;
    private readonly int[] strides;
    private readonly IReadOnlyList<int> strideView;
    private NDArray(double[] storage, Shape shape, int[] strides, int offset)
    {
        ArgumentNullException.ThrowIfNull(storage);
        ArgumentNullException.ThrowIfNull(shape);
        if (strides.Length != shape.Rank || strides.Any(s => s < 0)) throw new ArgumentException("Invalid element strides.", nameof(strides));
        if (offset < 0 || offset > storage.Length) throw new ArgumentOutOfRangeException(nameof(offset));
        if (shape.Size > 0)
        {
            long last = offset;
            for (int d = 0; d < shape.Rank; d++) last = checked(last + checked((long)(shape[d] - 1) * strides[d]));
            if (last >= storage.Length) throw new ArgumentException("Layout extends beyond storage.", nameof(shape));
        }
        this.storage = storage;
        Shape = shape;
        this.strides = (int[])strides.Clone();
        strideView = Array.AsReadOnly(this.strides);
        Offset = offset;
    }
    public Shape Shape { get; }
    public IReadOnlyList<int> Strides => strideView;
    public int Offset { get; }
    public int Rank => Shape.Rank;
    public int Size => Shape.Size;
    public DType DType => DType.Float64;
    public bool IsContiguous
    {
        get
        {
            if (Size == 0) return true;
            int expected = 1;
            for (int d = Rank - 1; d >= 0; d--)
            {
                if (Shape[d] > 1 && strides[d] != expected) return false;
                expected = checked(expected * Shape[d]);
            }
            return true;
        }
    }
    private static int[] RowMajorStrides(Shape shape)
    {
        var result = new int[shape.Rank];
        int stride = 1;
        for (int d = shape.Rank - 1; d >= 0; d--) { result[d] = stride; stride = checked(stride * shape[d]); }
        return result;
    }
    private static NDArray Own(double[] data, Shape shape) => new(data, shape, RowMajorStrides(shape), 0);
    internal bool SharesStorageWith(NDArray other) => ReferenceEquals(storage, other.storage);
    internal static NDArray CreateView(double[] data, Shape shape, int[] strides, int offset) => new(data, shape, strides, offset);
    internal Span<double> WritableSpan => IsContiguous ? storage.AsSpan(Offset, Size) : throw new InvalidOperationException("A contiguous layout is required.");
    public bool TryGetContiguousSpan(out ReadOnlySpan<double> span)
    {
        if (!IsContiguous) { span = default; return false; }
        span = storage.AsSpan(Offset, Size); return true;
    }
    public static NDArray Scalar(double value) => Own([value], new Shape());
    public static NDArray FromArray(double[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        return Own((double[])values.Clone(), new Shape(values.Length));
    }
    public static NDArray FromArray(ReadOnlySpan<double> values, Shape shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        if (values.Length != shape.Size) throw new ArgumentException("Values must match the shape's element count.", nameof(values));
        return Own(values.ToArray(), shape);
    }
    public static NDArray FromRows(params double[][] rows)
    {
        ArgumentNullException.ThrowIfNull(rows);
        if (rows.Any(row => row is null)) throw new ArgumentException("Rows cannot be null.", nameof(rows));
        int columns = rows.Length == 0 ? 0 : rows[0].Length;
        if (rows.Any(row => row.Length != columns)) throw new ArgumentException("Rows must be rectangular.", nameof(rows));
        var shape = new Shape(rows.Length, columns);
        var data = new double[shape.Size];
        for (int i = 0; i < rows.Length; i++) rows[i].CopyTo(data, checked(i * columns));
        return Own(data, shape);
    }
    public static NDArray Zeros(Shape shape) { ArgumentNullException.ThrowIfNull(shape); return Own(new double[shape.Size], shape); }
    public static NDArray Ones(Shape shape) => Full(shape, 1);
    public static NDArray Full(Shape shape, double value) { var result = Zeros(shape); result.WritableSpan.Fill(value); return result; }
    public static NDArray Identity(int size)
    {
        var result = Zeros(new Shape(size, size));
        for (int i = 0; i < size; i++) result.storage[i * size + i] = 1;
        return result;
    }
    public static NDArray Arange(double start, double stop, double step)
    {
        if (!double.IsFinite(start) || !double.IsFinite(stop) || !double.IsFinite(step) || step == 0)
            throw new ArgumentException("Arange requires finite bounds and a finite nonzero step.");
        if (step > 0 ? start >= stop : start <= stop) return FromArray([]);
        double difference = stop - start;
        double length = Math.Ceiling(double.IsFinite(difference) ? difference / step : stop / step - start / step);
        if (!double.IsFinite(length) || length > Array.MaxLength) throw new OverflowException("Arange exceeds the managed array element limit.");
        int count = Math.Max(1, checked((int)length));
        var data = new double[count];
        // FMA avoids intermediate overflow when start and i*step have opposite signs.
        for (int i = 0; i < count; i++) data[i] = Math.FusedMultiplyAdd(i, step, start);
        while (count > 0 && (step > 0 ? data[count - 1] >= stop : data[count - 1] <= stop)) count--;
        if (count != data.Length) Array.Resize(ref data, count);
        return Own(data, new Shape(count));
    }
    private int Position(int[] indices)
    {
        ArgumentNullException.ThrowIfNull(indices);
        if (indices.Length != Rank) throw new ArgumentException("Provide exactly one index per dimension.", nameof(indices));
        int position = Offset;
        for (int d = 0; d < Rank; d++)
        {
            if ((uint)indices[d] >= (uint)Shape[d]) throw new ArgumentOutOfRangeException(nameof(indices), "Index is outside its dimension.");
            position = checked(position + checked(indices[d] * strides[d]));
        }
        return position;
    }
    public double Get(params int[] indices) => storage[Position(indices)];
    public void Set(double value, params int[] indices) => storage[Position(indices)] = value;
    public NDArray Transpose() => Permute(Enumerable.Range(0, Rank).Reverse().ToArray());
    public NDArray Permute(params int[] axes)
    {
        ArgumentNullException.ThrowIfNull(axes);
        if (axes.Length != Rank || axes.Distinct().Count() != Rank || axes.Any(a => a < 0 || a >= Rank))
            throw new ArgumentException("Axes must be a permutation of every dimension.", nameof(axes));
        return new NDArray(storage, new Shape(axes.Select(a => Shape[a]).ToArray()), axes.Select(a => strides[a]).ToArray(), Offset);
    }
    public NDArray Reshape(Shape shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        if (shape.Size != Size) throw new ArgumentException("Reshape must preserve element count.", nameof(shape));
        if (!IsContiguous) throw new InvalidOperationException("Reshape requires a contiguous array; call Copy first.");
        return new NDArray(storage, shape, RowMajorStrides(shape), Offset);
    }
    public NDArray Slice(int axis, int start, int stop, int step = 1)
    {
        if ((uint)axis >= (uint)Rank) throw new ArgumentOutOfRangeException(nameof(axis));
        if (step <= 0) throw new ArgumentOutOfRangeException(nameof(step), "Slice step must be positive.");
        if (start < 0 || stop < start || stop > Shape[axis]) throw new ArgumentOutOfRangeException(nameof(start), "Require 0 <= start <= stop <= dimension.");
        int length = stop - start;
        int count = length / step + (length % step == 0 ? 0 : 1);
        var dimensions = Shape.ToArray(); dimensions[axis] = count;
        var shape = new Shape(dimensions);
        var newStrides = (int[])strides.Clone(); newStrides[axis] = checked(newStrides[axis] * step);
        int offset = shape.Size == 0 ? Offset : checked(Offset + checked(start * strides[axis]));
        return new NDArray(storage, shape, newStrides, offset);
    }
    public NDArray Copy() => Own(ToArray(), Shape);
    public NDArray AsContiguous() => IsContiguous ? this : Copy();
    public double[] ToArray()
    {
        var data = new double[Size];
        if (TryGetContiguousSpan(out var span)) span.CopyTo(data);
        else { int i = 0; foreach (int offset in Offsets()) data[i++] = storage[offset]; }
        return data;
    }
    public double[][] ToRows()
    {
        if (Rank != 2) throw new InvalidOperationException("ToRows requires a matrix.");
        var rows = new double[Shape[0]][];
        for (int i = 0; i < rows.Length; i++) { rows[i] = new double[Shape[1]]; for (int j = 0; j < rows[i].Length; j++) rows[i][j] = storage[Offset + i * strides[0] + j * strides[1]]; }
        return rows;
    }
    private IEnumerable<int> Offsets()
    {
        if (Size == 0) yield break;
        var indices = new int[Rank];
        long position = Offset;
        for (int n = 0; n < Size; n++)
        {
            yield return checked((int)position);
            if (n == Size - 1) yield break;
            for (int d = Rank - 1; d >= 0; d--)
            {
                indices[d]++; position += strides[d];
                if (indices[d] < Shape[d]) break;
                position -= (long)indices[d] * strides[d]; indices[d] = 0;
            }
        }
    }
}
