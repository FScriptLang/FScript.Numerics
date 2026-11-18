using FScript.Extensibility;

namespace FScript.Numerics.Extensions;

public static partial class NumericsProvider
{
    // Successful arithmetic is not enough: mutable inputs, observable identities,
    // shape/dtype failures, and native failures prevent a Pure function promise.
    public static IEnumerable<string> FScriptPureFunctions() => Array.Empty<string>();

    // These objects own their fitted storage and expose copies, never writable aliases.
    // This says nothing about the purity of their factories or prediction functions.
    public static IEnumerable<string> FScriptPureTypes() => ModelInterop.Descriptors.Select(type => type.Name);
    public static IEnumerable<string> FScriptThreadSafeTypes() => FScriptPureTypes();

    // Factories and scalar distributions use local state. Operations receiving an
    // NDArray remain unsafe: a concurrent writer could share its backing storage.
    public static IEnumerable<string> FScriptThreadSafeFunctions() =>
        Provide().Functions.Where(function => SafeArguments(function.Scheme.Item2)).Select(function => function.Name);

    private static bool SafeArguments(ScriptType type) => type is not ScriptType.TFun function ||
        SafeInput(function.Item1) && SafeArguments(function.Item2);

    private static bool SafeInput(ScriptType type) => type switch
    {
        ScriptType.TNamed named => ModelInterop.Descriptors.Any(model => model.Name == named.Item),
        ScriptType.TList list => SafeInput(list.Item),
        ScriptType.TTuple tuple => tuple.Item.All(SafeInput),
        _ => type.IsTUnit || type.IsTInt || type.IsTFloat || type.IsTBool || type.IsTString
    };
}
