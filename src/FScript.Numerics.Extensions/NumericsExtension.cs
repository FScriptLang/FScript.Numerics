using System.Runtime.CompilerServices;
using FScript.Extensibility;
using Microsoft.FSharp.Collections;

namespace FScript.Numerics.Extensions;

/// <summary>SDK registration and lossless transport of core NDArray objects.</summary>
public static partial class NumericsExtension
{
    private sealed record Identity(string Value);
    private static readonly ConditionalWeakTable<NDArray, Identity> Identities = new();
    private static readonly ScriptType ArrayType = ScriptType.NewTNamed("NDArray");
    private static readonly ScriptType IntList = ScriptType.NewTList(ScriptType.TInt);
    private static readonly ScriptType FloatList = ScriptType.NewTList(ScriptType.TFloat);
    private static readonly ScriptType Rows = ScriptType.NewTList(FloatList);
    private static readonly HostTypeDescriptor Descriptor = new("NDArray", Sdk.Map(new[]
    {
        ("Id", ScriptType.TString), ("Rank", ScriptType.TInt), ("Size", ScriptType.TInt),
        ("Shape", IntList), ("DType", ScriptType.TString), ("Strides",IntList), ("IsContiguous",ScriptType.TBool), ("IsFortranContiguous",ScriptType.TBool), ("IsReadOnly",ScriptType.TBool)
    }), true);
    internal static readonly ExtensionRegistration TypeRegistration = new(Sdk.List(new[] { Descriptor }.Concat(ModelInterop.Descriptors)), Sdk.List(Array.Empty<HostUnionDescriptor>()), Sdk.List(Array.Empty<ExternalFunction>()));
    private static FSharpMap<string, ScriptValue> Snapshot(NDArray array) => Sdk.Map(new[]
    {
        ("Id", ScriptValue.NewVString(Identities.GetValue(array, _ => new Identity(Guid.NewGuid().ToString("N"))).Value)),
        ("Rank", ScriptValue.NewVInt(array.Rank)), ("Size", ScriptValue.NewVInt(array.Size)),
        ("Shape", ScriptValue.NewVList(Sdk.List(array.Shape.Select(d => ScriptValue.NewVInt(d))))),
        ("DType", ScriptValue.NewVString(array.DType.ToString().ToLowerInvariant())),
        ("Strides",ScriptValue.NewVList(Sdk.List(array.Strides.Select(s=>ScriptValue.NewVInt(s))))),
        ("IsContiguous",ScriptValue.NewVBool(array.IsContiguous)), ("IsFortranContiguous",ScriptValue.NewVBool(array.IsFortranContiguous)), ("IsReadOnly",ScriptValue.NewVBool(array.IsReadOnly))
    });
    public static ScriptValue Wrap(NDArray array)
    {
        ArgumentNullException.ThrowIfNull(array);
        return HostInterop.create(TypeRegistration, "NDArray", Snapshot(array), array);
    }
    public static NDArray Unwrap(ScriptValue value)
    {
        if (value is not ScriptValue.VHostManaged host || !host.Item.Descriptor.Equals(Descriptor) || host.Item.Payload is not NDArray array)
            throw new ExtensionException("Expected a registered NDArray with an NDArray payload.");
        if (!host.Item.Fields.Equals(Snapshot(array)))
            throw new ExtensionException("NDArray snapshot metadata does not match its payload.");
        return array;
    }
    public static ExtensionRegistration CreateRegistration()
    {
        var functions = new List<ExternalFunction>();
        void Add(string name, ScriptType result, ScriptType[] arguments, Func<ScriptValue[], ScriptValue> handler) => functions.Add(Sdk.Declare(name, result, arguments, handler));
        Add("scalar", ArrayType, [ScriptType.TFloat], a => Wrap(NDArray.Scalar(Sdk.Float(a[0]))));
        Add("ofList", ArrayType, [FloatList], a => Wrap(NDArray.FromArray(Sdk.Floats(a[0]))));
        Add("ofRows", ArrayType, [Rows], a => Wrap(NDArray.FromRows(Sdk.Items(a[0]).Select(Sdk.Floats).ToArray())));
        Add("zeros", ArrayType, [IntList], a => Wrap(NDArray.Zeros(new Shape(Sdk.Integers(a[0])))));
        Add("ones", ArrayType, [IntList], a => Wrap(NDArray.Ones(new Shape(Sdk.Integers(a[0])))));
        Add("full", ArrayType, [IntList, ScriptType.TFloat], a => Wrap(NDArray.Full(new Shape(Sdk.Integers(a[0])), Sdk.Float(a[1]))));
        Add("arange", ArrayType, [ScriptType.TFloat, ScriptType.TFloat, ScriptType.TFloat], a => Wrap(NDArray.Arange(Sdk.Float(a[0]), Sdk.Float(a[1]), Sdk.Float(a[2]))));
        Add("identity", ArrayType, [ScriptType.TInt], a => Wrap(NDArray.Identity(Sdk.Int(a[0]))));
        Add("get", ScriptType.TFloat, [ArrayType, IntList], a => ScriptValue.NewVFloat(Unwrap(a[0]).Get(Sdk.Integers(a[1]))));
        Add("set", ScriptType.TUnit, [ArrayType, IntList, ScriptType.TFloat], a => { Unwrap(a[0]).Set(Sdk.Float(a[2]), Sdk.Integers(a[1])); return ScriptValue.VUnit; });
        Add("toList", FloatList, [ArrayType], a => Sdk.FloatList(Unwrap(a[0]).ToArray()));
        Add("toRows", Rows, [ArrayType], a => ScriptValue.NewVList(Sdk.List(Unwrap(a[0]).ToRows().Select(Sdk.FloatList))));
        Add("transpose", ArrayType, [ArrayType], a => Wrap(Unwrap(a[0]).Transpose()));
        Add("permute", ArrayType, [ArrayType, IntList], a => Wrap(Unwrap(a[0]).Permute(Sdk.Integers(a[1]))));
        Add("reshape", ArrayType, [ArrayType, IntList], a => Wrap(Unwrap(a[0]).Reshape(new Shape(Sdk.Integers(a[1])))));
        Add("slice", ArrayType, [ArrayType, ScriptType.TInt, ScriptType.TInt, ScriptType.TInt, ScriptType.TInt], a => Wrap(Unwrap(a[0]).Slice(Sdk.Int(a[1]), Sdk.Int(a[2]), Sdk.Int(a[3]), Sdk.Int(a[4]))));
        Add("copy", ArrayType, [ArrayType], a => Wrap(Unwrap(a[0]).Copy()));
        Add("contiguous", ArrayType, [ArrayType], a => Wrap(Unwrap(a[0]).AsContiguous()));
        var operations = new (string Name, Func<NDArray, NDArray, NDArray> Array, Func<NDArray, double, NDArray> Scalar)[]
        {
            ("add", (a,b) => a.Add(b), (a,b) => a.Add(b)), ("subtract", (a,b) => a.Subtract(b), (a,b) => a.Subtract(b)),
            ("multiply", (a,b) => a.Multiply(b), (a,b) => a.Multiply(b)), ("divide", (a,b) => a.Divide(b), (a,b) => a.Divide(b))
        };
        foreach (var operation in operations)
        {
            Add(operation.Name, ArrayType, [ArrayType, ArrayType], a => Wrap(operation.Array(Unwrap(a[0]), Unwrap(a[1]))));
            Add(operation.Name + "Scalar", ArrayType, [ArrayType, ScriptType.TFloat], a => Wrap(operation.Scalar(Unwrap(a[0]), Sdk.Float(a[1]))));
        }
        Add("negate", ArrayType, [ArrayType], a => Wrap(Unwrap(a[0]).Negate()));
        foreach (var (name, reduction) in new (string, Func<NDArray, double>)[] { ("sum", a => a.Sum()), ("mean", a => a.Mean()), ("min", a => a.Min()), ("max", a => a.Max()) })
            Add(name, ScriptType.TFloat, [ArrayType], a => ScriptValue.NewVFloat(reduction(Unwrap(a[0]))));
        Add("dot", ScriptType.TFloat, [ArrayType, ArrayType], a => ScriptValue.NewVFloat(Unwrap(a[0]).Dot(Unwrap(a[1]))));
        Add("matvec", ArrayType, [ArrayType, ArrayType], a => Wrap(Unwrap(a[0]).Matvec(Unwrap(a[1]))));
        Add("matmul", ArrayType, [ArrayType, ArrayType], a => Wrap(Unwrap(a[0]).Matmul(Unwrap(a[1]))));
        Add("equals", ScriptType.TBool, [ArrayType, ArrayType], a => ScriptValue.NewVBool(Unwrap(a[0]).ContentEquals(Unwrap(a[1]))));
        Add("allClose", ScriptType.TBool, [ArrayType, ArrayType, ScriptType.TFloat, ScriptType.TFloat], a => ScriptValue.NewVBool(Unwrap(a[0]).AllClose(Unwrap(a[1]), Sdk.Float(a[2]), Sdk.Float(a[3]))));
        RegisterExpansion(functions);
        return new ExtensionRegistration(TypeRegistration.Types, TypeRegistration.Unions, Sdk.List(functions));
    }
}

public static class NumericsProvider
{
    [FScriptExternProvider]
    public static ExtensionRegistration Provide() => NumericsExtension.CreateRegistration();
}
