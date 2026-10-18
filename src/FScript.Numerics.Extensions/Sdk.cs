using FScript.Extensibility;
using Microsoft.FSharp.Collections;
using Microsoft.FSharp.Core;

namespace FScript.Numerics.Extensions;

internal static class Sdk
{
    internal static FSharpList<T> List<T>(IEnumerable<T> values) => ListModule.OfSeq(values);
    internal static FSharpMap<string, T> Map<T>(IEnumerable<(string Key, T Value)> values) => MapModule.OfSeq(values.Select(v => Tuple.Create(v.Key, v.Value)));
    internal static FSharpFunc<T, U> Function<T, U>(Func<T, U> function) => FSharpFunc<T, U>.FromConverter(value => function(value));
    internal static ExternalFunction Declare(string name, ScriptType result, ScriptType[] arguments, Func<ScriptValue[], ScriptValue> handler)
    {
        ScriptType signature = result;
        for (int i = arguments.Length - 1; i >= 0; i--)
            signature = ScriptType.NewTFun(arguments[i], signature);
        string qualified = name.Contains('.', StringComparison.Ordinal) ? name : $"NDArray.{name}";
        var scheme = FunctionScheme.NewForall(List(Array.Empty<int>()), signature);
        var implementation = Function<ExternalCallContext, FSharpFunc<FSharpList<ScriptValue>, ScriptValue>>(context =>
            Function<FSharpList<ScriptValue>, ScriptValue>(values =>
            {
                context.CheckRuntime.Invoke(null!);
                try
                {
                    var supplied = values.ToArray();
                    if (supplied.Length != arguments.Length)
                        throw new ArgumentException($"Expected {arguments.Length} arguments.");
                    var definitions = HostInterop.typeDefinitions(NumericsExtension.TypeRegistration, List(Array.Empty<string>()));
                    for (int i = 0; i < arguments.Length; i++)
                        HostInterop.validate(definitions, arguments[i], supplied[i]);
                    var value = handler(supplied);
                    HostInterop.validate(definitions, result, value);
                    context.CheckRuntime.Invoke(null!);
                    return value;
                }
                catch (Exception error) when (error is ArgumentException or OverflowException or InvalidOperationException or ExtensionException or ArithmeticException)
                {
                    throw new ExtensionException($"{qualified}: {error.Message}", error);
                }
            }));
        return new ExternalFunction(qualified, scheme, implementation, arguments.Length);
    }
    internal static double Float(ScriptValue value) => value is ScriptValue.VFloat number ? number.Item : throw new ArgumentException("Expected float.");
    internal static int Int(ScriptValue value) => value is ScriptValue.VInt number ? checked((int)number.Item) : throw new ArgumentException("Expected int.");
    internal static ScriptValue[] Items(ScriptValue value) => value is ScriptValue.VList list ? list.Item.ToArray() : throw new ArgumentException("Expected list.");
    internal static int[] Integers(ScriptValue value) => Items(value).Select(Int).ToArray();
    internal static double[] Floats(ScriptValue value) => Items(value).Select(Float).ToArray();
    internal static ScriptValue FloatList(IEnumerable<double> values) => ScriptValue.NewVList(List(values.Select(ScriptValue.NewVFloat)));
}
