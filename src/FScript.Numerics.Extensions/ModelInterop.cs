using System.Runtime.CompilerServices;
using FScript.Extensibility;
using FScript.Numerics.ML;
using FScript.Numerics.Lapack;
namespace FScript.Numerics.Extensions;

/// <summary>Opaque immutable fitted objects. Snapshots contain bounded metadata, never coefficient storage.</summary>
public static class ModelInterop
{
    private sealed record Identity(string Value);
    private static readonly ConditionalWeakTable<object, Identity> Identities = new();
    private static readonly Type[] Types = [typeof(StandardScaler), typeof(MinMaxScaler), typeof(LinearRegression), typeof(LogisticRegression), typeof(KMeans), typeof(Pca), typeof(LuFactorization)];
    internal static readonly HostTypeDescriptor[] Descriptors = Types.Select(t => new HostTypeDescriptor(t.Name, Sdk.Map(Fields(t)), true)).ToArray();
    private static IEnumerable<(string, ScriptType)> Fields(Type type)
    {
        yield return ("Id", ScriptType.TString);
        if (type != typeof(LuFactorization))
            yield return ("FeatureCount", ScriptType.TInt);
        if (type == typeof(KMeans))
        {
            yield return ("ClusterCount", ScriptType.TInt);
            yield return ("Inertia", ScriptType.TFloat);
        }
        if (type == typeof(Pca))
            yield return ("ComponentCount", ScriptType.TInt);
        if (type == typeof(KMeans) || type == typeof(LogisticRegression))
        {
            yield return ("Iterations", ScriptType.TInt);
            yield return ("Converged", ScriptType.TBool);
        }
        if (type == typeof(LuFactorization))
            yield return ("Size", ScriptType.TInt);
    }
    private static IEnumerable<(string, ScriptValue)> Snapshot(object model)
    {
        yield return ("Id", ScriptValue.NewVString(Identities.GetValue(model, _ => new(Guid.NewGuid().ToString("N"))).Value));
        int features = model switch
        {
            StandardScaler m => m.FeatureCount,
            MinMaxScaler m => m.FeatureCount,
            LinearRegression m => m.FeatureCount,
            LogisticRegression m => m.FeatureCount,
            KMeans m => m.FeatureCount,
            Pca m => m.FeatureCount,
            _ => 0
        };
        if (model is not LuFactorization)
            yield return ("FeatureCount", ScriptValue.NewVInt(features));
        if (model is KMeans k)
        {
            yield return ("ClusterCount", ScriptValue.NewVInt(k.ClusterCount));
            yield return ("Inertia", ScriptValue.NewVFloat(k.Inertia));
            yield return ("Iterations", ScriptValue.NewVInt(k.Iterations));
            yield return ("Converged", ScriptValue.NewVBool(k.Converged));
        }
        if (model is LogisticRegression l)
        {
            yield return ("Iterations", ScriptValue.NewVInt(l.Iterations));
            yield return ("Converged", ScriptValue.NewVBool(l.Converged));
        }
        if (model is Pca p)
            yield return ("ComponentCount", ScriptValue.NewVInt(p.ComponentCount));
        if (model is LuFactorization lu)
            yield return ("Size", ScriptValue.NewVInt(lu.Order));
    }
    public static ScriptValue Wrap(object model)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (!Types.Contains(model.GetType()))
            throw new ExtensionException("Unsupported numerical model payload.");
        return HostInterop.create(NumericsExtension.TypeRegistration, model.GetType().Name, Sdk.Map(Snapshot(model)), model);
    }
    public static T Unwrap<T>(ScriptValue value) where T : class
    {
        var descriptor = Descriptors.SingleOrDefault(d => d.Name == typeof(T).Name);
        if (descriptor is null || value is not ScriptValue.VHostManaged host || !host.Item.Descriptor.Equals(descriptor) || host.Item.Payload is not T model || !host.Item.Fields.Equals(Sdk.Map(Snapshot(model))))
            throw new ExtensionException($"Expected a registered {typeof(T).Name} with valid payload and metadata.");
        return model;
    }
}
