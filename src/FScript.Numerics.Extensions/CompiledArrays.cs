using System.Reflection;
using FScript.Extensibility;

namespace FScript.Numerics.Extensions;

/// <summary>Typed boundaries retain the existing SDK host value, identity, and snapshots.</summary>
public static class CompiledArrayOperations
{
    private static T Invoke<T>(string name, Func<T> operation)
    {
        try { return operation(); }
        catch (Exception error) when (error is ArgumentException or OverflowException or InvalidOperationException or ExtensionException or ArithmeticException)
        { throw new ExtensionException($"NDArray.{name}: {error.Message}", error); }
    }
    private static NDArray Value(HostManagedValue value) => NumericsExtension.UnwrapHost(value);
    private static HostManagedValue Wrap(NDArray value) => ((ScriptValue.VHostManaged)NumericsExtension.Wrap(value)).Item;
    private static int[] Indices(IReadOnlyList<long> values)
    {
        var result = new int[values.Count];
        for (int i = 0; i < result.Length; i++) result[i] = checked((int)values[i]);
        return result;
    }
    private static bool BoundaryError(Exception error) => error is ArgumentException or OverflowException or InvalidOperationException or ExtensionException or ArithmeticException;
    private static ExtensionException Failure(string name, Exception error) => new($"NDArray.{name}: {error.Message}", error);

    public static HostManagedValue Arange(double start, double stop, double step) => Invoke("arange", () => Wrap(NDArray.Arange(start, stop, step)));
    public static HostManagedValue OfList(double[] values) => Invoke("ofList", () => Wrap(NDArray.FromArray(values)));
    public static HostManagedValue Zeros(IReadOnlyList<long> shape) => Invoke("zeros", () => Wrap(NDArray.Zeros(new Shape(Indices(shape)))));
    public static HostManagedValue Ones(IReadOnlyList<long> shape) => Invoke("ones", () => Wrap(NDArray.Ones(new Shape(Indices(shape)))));
    public static HostManagedValue Full(IReadOnlyList<long> shape, double value) => Invoke("full", () => Wrap(NDArray.Full(new Shape(Indices(shape)), value)));
    public static HostManagedValue Copy(HostManagedValue array) => Invoke("copy", () => Wrap(Value(array).Copy()));
    public static HostManagedValue Contiguous(HostManagedValue array) => Invoke("contiguous", () => Wrap(Value(array).AsContiguous()));
    public static HostManagedValue Transpose(HostManagedValue array) => Invoke("transpose", () => Wrap(Value(array).Transpose()));
    public static HostManagedValue AddScalar(HostManagedValue array, double value) => Invoke("addScalar", () => Wrap(Value(array).Add(value)));
    public static HostManagedValue MultiplyScalar(HostManagedValue array, double value) => Invoke("multiplyScalar", () => Wrap(Value(array).Multiply(value)));
    public static double Sum(HostManagedValue array) => Invoke("sum", () => Value(array).Sum());
    public static double Dot(HostManagedValue left, HostManagedValue right) => Invoke("dot", () => Value(left).Dot(Value(right)));
    public static double[] ToList(HostManagedValue array) => Invoke("toList", () => Value(array).ToArray());
    public static double Get(HostManagedValue array, IReadOnlyList<long> indices)
    {
        try { return Value(array).Get(Indices(indices)); }
        catch (Exception error) when (BoundaryError(error)) { throw Failure("get", error); }
    }
    public static long GetInt(HostManagedValue array, IReadOnlyList<long> indices)
    {
        try { return Value(array).GetInt64(Indices(indices)); }
        catch (Exception error) when (BoundaryError(error)) { throw Failure("getInt", error); }
    }
    public static bool GetBool(HostManagedValue array, IReadOnlyList<long> indices)
    {
        try { return Value(array).GetBoolean(Indices(indices)); }
        catch (Exception error) when (BoundaryError(error)) { throw Failure("getBool", error); }
    }
    public static void Set(HostManagedValue array, IReadOnlyList<long> indices, double value)
    {
        try { Value(array).Set(value, Indices(indices)); }
        catch (Exception error) when (BoundaryError(error)) { throw Failure("set", error); }
    }
    public static void SetInt(HostManagedValue array, IReadOnlyList<long> indices, long value)
    {
        try { Value(array).SetInt64(value, Indices(indices)); }
        catch (Exception error) when (BoundaryError(error)) { throw Failure("setInt", error); }
    }
    public static void SetBool(HostManagedValue array, IReadOnlyList<long> indices, bool value)
    {
        try { Value(array).SetBoolean(value, Indices(indices)); }
        catch (Exception error) when (BoundaryError(error)) { throw Failure("setBool", error); }
    }
}

/// <summary>Small public adapters opt into IL expansion. Checked operations remain calls.</summary>
public static class InlineArrayBindings
{
    public static double Get(HostManagedValue array, IReadOnlyList<long> indices) => CompiledArrayOperations.Get(array, indices);
    public static long GetInt(HostManagedValue array, IReadOnlyList<long> indices) => CompiledArrayOperations.GetInt(array, indices);
    public static bool GetBool(HostManagedValue array, IReadOnlyList<long> indices) => CompiledArrayOperations.GetBool(array, indices);
    public static void Set(HostManagedValue array, IReadOnlyList<long> indices, double value) => CompiledArrayOperations.Set(array, indices, value);
    public static void SetInt(HostManagedValue array, IReadOnlyList<long> indices, long value) => CompiledArrayOperations.SetInt(array, indices, value);
    public static void SetBool(HostManagedValue array, IReadOnlyList<long> indices, bool value) => CompiledArrayOperations.SetBool(array, indices, value);
}

public static partial class NumericsProvider
{
    // No FScriptCompiledTypes override: typed and legacy calls share HostManagedValue.
    public static IEnumerable<KeyValuePair<string, MethodInfo>> FScriptCompiledFunctions()
    {
        foreach (var name in new[] { "Arange", "OfList", "Zeros", "Ones", "Full", "Copy", "Contiguous", "Transpose", "AddScalar", "MultiplyScalar", "Sum", "Dot", "ToList" })
            yield return KeyValuePair.Create("NDArray." + char.ToLowerInvariant(name[0]) + name[1..], typeof(CompiledArrayOperations).GetMethod(name)!);
        foreach (var name in new[] { "Get", "GetInt", "GetBool", "Set", "SetInt", "SetBool" })
            yield return KeyValuePair.Create("NDArray." + char.ToLowerInvariant(name[0]) + name[1..], typeof(InlineArrayBindings).GetMethod(name)!);
    }
    public static IEnumerable<string> FScriptInlineFunctions() => new[]
    { "NDArray.get", "NDArray.getInt", "NDArray.getBool", "NDArray.set", "NDArray.setInt", "NDArray.setBool" };
}
