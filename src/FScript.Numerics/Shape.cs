using System.Collections;

namespace FScript.Numerics;

/// <summary>Immutable dimensions. The empty shape represents a scalar.</summary>
public sealed class Shape : IReadOnlyList<int>, IEquatable<Shape>
{
    private readonly int[] dimensions;
    public Shape(params int[] dimensions)
    {
        ArgumentNullException.ThrowIfNull(dimensions);
        this.dimensions = (int[])dimensions.Clone();
        if (dimensions.Any(d => d < 0))
            throw new ArgumentOutOfRangeException(nameof(dimensions), "Dimensions must be nonnegative.");
        if (dimensions.Contains(0))
        {
            Size = 0;
            return;
        }
        long size = 1;
        foreach (int dimension in dimensions)
        {
            size = checked(size * dimension);
            if (size > Array.MaxLength)
                throw new OverflowException("Shape exceeds the managed array element limit.");
        }
        Size = checked((int)size);
    }
    public int Size
    {
        get;
    }
    public int Rank => dimensions.Length;
    public int Count => Rank;
    public int this[int axis] => dimensions[axis];
    public int[] ToArray() => (int[])dimensions.Clone();
    public bool Equals(Shape? other) => other is not null && dimensions.AsSpan().SequenceEqual(other.dimensions);
    public override bool Equals(object? obj) => obj is Shape other && Equals(other);
    public override int GetHashCode()
    {
        var hash = new HashCode();
        foreach (int d in dimensions)
            hash.Add(d);
        return hash.ToHashCode();
    }
    public override string ToString() => $"[{string.Join(", ", dimensions)}]";
    public IEnumerator<int> GetEnumerator() => ((IEnumerable<int>)dimensions).GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
