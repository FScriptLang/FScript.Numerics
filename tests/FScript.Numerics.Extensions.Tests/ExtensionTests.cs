using FScript.Extensibility;
using FScript.Numerics.Extensions;
using FScript.Runtime;
using Microsoft.FSharp.Collections;
using Microsoft.FSharp.Core;
using NUnit.Framework;
using SV = FScript.Extensibility.ScriptValue;

namespace FScript.Numerics.Extensions.Tests;

public class ExtensionTests
{
    private static FSharpList<T> List<T>(IEnumerable<T> values) => ListModule.OfSeq(values);
    private static FSharpFunc<T, U> Fn<T, U>(Func<T, U> function) => FSharpFunc<T, U>.FromConverter(v => function(v));
    private static string Run(string source) => FScript.Language.Pretty.valueToString(ScriptHost.loadSourceWithRegistration(ExtensionBridge.toLanguageRegistrationWithPermissions(FScript.Language.HostPermissions.Native, NumericsExtension.CreateRegistration()), source).LastValue);
    private static ExternalCallContext Context() => new(
        Fn<SV, FSharpFunc<SV, SV>>(_ => Fn<SV, SV>(_ => throw new NotSupportedException())),
        Fn<SV, SV>(_ => throw new NotSupportedException()), Fn<SV, SV>(_ => throw new NotSupportedException()), Fn<Unit, Unit>(_ => null!));
    private static SV Invoke(string name, params SV[] arguments) => NumericsExtension.CreateRegistration().Functions.Single(f => f.Name == "NDArray." + name).Impl.Invoke(Context()).Invoke(List(arguments));

    [Test]
    public void Embedded_provider_functions_require_an_explicit_native_grant()
    {
        var denied = ExtensionBridge.toLanguageRegistration(NumericsExtension.CreateRegistration());
        Assert.Throws<FScript.Language.TypeException>(() =>
            ScriptHost.loadSourceWithRegistration(denied, "NDArray.sum (NDArray.ones [2])"));
        Assert.That(Run("NDArray.sum (NDArray.ones [2])"), Is.EqualTo("2"));
    }

    [Test]
    public void Registration_has_authoritative_monomorphic_schemes_and_visible_metadata()
    {
        var registration = NumericsProvider.Provide();
        Assert.That(registration.Functions.Count(), Is.EqualTo(137));
        Assert.That(registration.Functions.Select(f => f.Name).Distinct().Count(), Is.EqualTo(137));
        var descriptor = registration.Types.Single(t => t.Name == "NDArray");
        Assert.That(descriptor.Name, Is.EqualTo("NDArray"));
        Assert.That(descriptor.HostManaged, Is.True);
        Assert.That(descriptor.Fields.Keys, Is.EquivalentTo(new[] { "Id", "Rank", "Size", "Shape", "DType", "Strides", "IsContiguous", "IsFortranContiguous", "IsReadOnly" }));
        foreach (var function in registration.Functions)
        {
            var scheme = function.Scheme;
            Assert.That(scheme.Item1, Is.Empty);
            Assert.That(function.Arity, Is.GreaterThan(0));
        }
        Assert.That(Run("let a = NDArray.ones [2;3]\n(a.Rank, a.Size, a.Shape, a.DType)"), Is.EqualTo("(2, 6, [2;3], \"float64\")"));
    }

    [Test]
    public void Sdk2_descriptors_are_monomorphic_and_round_trip_through_the_published_host()
    {
        var registration = NumericsProvider.Provide();
        Assert.That(registration.Types.Count(), Is.EqualTo(8));
        foreach (var descriptor in registration.Types)
            Assert.That(descriptor.TypeParameters, Is.Empty, descriptor.Name);

        var definitions = HostInterop.typeDefinitions(registration, List(Array.Empty<string>()));
        foreach (var descriptor in registration.Types)
            Assert.That(definitions[descriptor.Name], Is.InstanceOf<ScriptType.THostManaged>());

        Assert.That(Run("""
            let values = NDArray.ofRows [[1.0];[3.0]]
            let model = StandardScaler.fit values
            let normalized = StandardScaler.transform model values
            (values.Shape, model.FeatureCount, NDArray.toRows normalized)
            """), Is.EqualTo("([2;1], 1, [[-1];[1]])"));
    }

    [Test]
    public void Sdk2_generic_descriptor_cannot_impersonate_a_numerical_value()
    {
        var wrapped = (SV.VHostManaged)NumericsExtension.Wrap(NDArray.FromArray([1, 2]));
        var host = wrapped.Item;
        var generic = new HostTypeDescriptor(host.Descriptor.Name, host.Descriptor.Fields,
            host.Descriptor.HostManaged, List(new[] { 0 }));
        var forged = SV.NewVHostManaged(new HostManagedValue(generic, host.Fields, host.Payload));
        Assert.Throws<ExtensionException>(() => NumericsExtension.Unwrap(forged));

        var model = FScript.Numerics.ML.StandardScaler.Fit(NDArray.FromRows([1], [3]));
        var fitted = ((SV.VHostManaged)ModelInterop.Wrap(model)).Item;
        generic = new HostTypeDescriptor(fitted.Descriptor.Name, fitted.Descriptor.Fields,
            fitted.Descriptor.HostManaged, List(new[] { 0 }));
        forged = SV.NewVHostManaged(new HostManagedValue(generic, fitted.Fields, fitted.Payload));
        Assert.Throws<ExtensionException>(() => ModelInterop.Unwrap<FScript.Numerics.ML.StandardScaler>(forged));
    }

    [Test]
    public void Wrappers_preserve_actual_payload_and_validate_metadata()
    {
        var array = NDArray.FromArray([1, 2]);
        var wrapped = NumericsExtension.Wrap(array);
        Assert.That(NumericsExtension.Unwrap(wrapped), Is.SameAs(array));
        Assert.That(wrapped, Is.EqualTo(NumericsExtension.Wrap(array)));
        Assert.That(wrapped, Is.Not.EqualTo(NumericsExtension.Wrap(array.Copy())));
        var host = ((SV.VHostManaged)wrapped).Item;
        Assert.Throws<ExtensionException>(() => NumericsExtension.Unwrap(SV.NewVHostManaged(new HostManagedValue(host.Descriptor, host.Fields, new object()))));
        var fields = MapModule.Add("Rank", SV.NewVInt(99), host.Fields);
        Assert.Throws<ExtensionException>(() => NumericsExtension.Unwrap(SV.NewVHostManaged(new HostManagedValue(host.Descriptor, fields, array))));
        var other = new HostTypeDescriptor("Other", host.Descriptor.Fields, true, List(Array.Empty<int>()));
        Assert.Throws<ExtensionException>(() => NumericsExtension.Unwrap(SV.NewVHostManaged(new HostManagedValue(other, host.Fields, array))));
        Assert.Throws<ExtensionException>(() => NumericsExtension.Unwrap(SV.VUnit));
    }

    [Test]
    public void Shared_views_mutate_while_copies_and_arithmetic_isolate_storage()
    {
        Assert.That(Run("""
            let a = NDArray.ofRows [[1.0;2.0];[3.0;4.0]]
            let view = NDArray.transpose a
            let copy = NDArray.copy a
            let result = NDArray.addScalar a 10.0
            NDArray.set view [1;0] 7.0
            (NDArray.toList a, NDArray.toList copy, NDArray.toList result, a = a, a = copy, NDArray.equals copy (NDArray.ofRows [[1.0;2.0];[3.0;4.0]]))
            """), Is.EqualTo("([1;7;3;4], [1;2;3;4], [11;12;13;14], true, false, true)"));
    }

    [Test]
    public void Identity_survives_script_functions_containers_and_host_round_trips()
    {
        Assert.That(Run("""
            let a = NDArray.ofList [1.0;2.0]
            let reader value =
                let rank = value.Rank
                value
            let identity value = value
            let values = [a; identity a]
            let pair = (a, Some a)
            let f x = a
            (a = reader a, a = f (), NDArray.contiguous a = a, NDArray.equals a (NDArray.copy a), a.Id = (identity a).Id)
            """), Is.EqualTo("(true, true, true, true, true)"));
        var native = ExtensionBridge.toLanguageRegistrationWithPermissions(FScript.Language.HostPermissions.Native, NumericsExtension.CreateRegistration());
        var loaded = ScriptHost.loadSourceWithRegistration(native, "export let a = NDArray.ofList [1.0;2.0]\nexport let values = [a;a]\na");
        var original = (FScript.Language.Value.VHostManaged)loaded.LastValue;
        var values = (FScript.Language.Value.VList)loaded.ExportedValues["values"];
        foreach (var value in values.Item)
            Assert.That(((FScript.Language.Value.VHostManaged)value).Item.Payload, Is.SameAs(original.Item.Payload));
    }

    [TestCase("NDArray.ofList [1;2]")]
    [TestCase("NDArray.get (NDArray.ones [2]) [0.0]")]
    [TestCase("let a = NDArray.ones [2]\na.Secret")]
    [TestCase("let a = NDArray.ones [2]\n{ a with Rank = 5 }")]
    [TestCase("let a = NDArray.ones [2]\n{| a with Rank = 5 |}")]
    [TestCase("NDArray.sum { Id = \"x\"; Rank = 1; Size = 2; Shape = [2]; DType = \"float64\" }")]
    public void Invalid_script_programs_are_rejected(string source) => Assert.Throws<FScript.Language.TypeException>(() => Run(source), "Rejection must occur before an invalid host value can be used");

    [Test]
    public void Boundary_errors_have_operation_names_and_inner_causes()
    {
        var exception = Assert.Throws<ExtensionException>(() => Invoke("zeros", SV.NewVList(List(new[] { SV.NewVInt(long.MaxValue) }))));
        Assert.That(exception!.Message, Does.StartWith("NDArray.zeros:"));
        Assert.That(exception.InnerException, Is.TypeOf<OverflowException>());
        Assert.Throws<ExtensionException>(() => Invoke("ofList", SV.NewVList(List(new[] { SV.NewVInt(1) }))));
        Assert.Throws<ExtensionException>(() => Invoke("sum", SV.NewVRecord(MapModule.OfSeq(Array.Empty<Tuple<string, SV>>()))));
        Assert.Throws<ExtensionException>(() => Invoke("get", NumericsExtension.Wrap(NDArray.Ones(new Shape(1))), SV.NewVList(List(new[] { SV.NewVInt(-1) }))));
        Assert.Throws<ExtensionException>(() => Invoke("mean", NumericsExtension.Wrap(NDArray.Zeros(new Shape(0)))));
    }

    [Test]
    public void All_function_schemes_execute_through_published_runtime()
    {
        Assert.That(Run("""
            let a = NDArray.reshape (NDArray.arange 0.0 6.0 1.0) [2;3]
            let b = NDArray.full [2;3] 2.0
            let x = NDArray.ofList [1.0;2.0;3.0]
            let z = NDArray.zeros [2;3]
            let s = NDArray.scalar 5.0
            let p = NDArray.permute a [1;0]
            let c = NDArray.contiguous p
            let t = NDArray.slice a 1 0 3 2
            let arithmetic = NDArray.divide (NDArray.multiply (NDArray.subtract (NDArray.add a b) b) b) b
            let scalarArithmetic = NDArray.divideScalar (NDArray.multiplyScalar (NDArray.subtractScalar (NDArray.addScalar a 3.0) 3.0) 2.0) 2.0
            (NDArray.equals arithmetic a, NDArray.allClose scalarArithmetic a 0.0 0.0, NDArray.toList (NDArray.negate x), NDArray.sum z, NDArray.mean x, NDArray.min x, NDArray.max x, NDArray.dot x x, NDArray.toList (NDArray.matvec a x), NDArray.toRows (NDArray.matmul a (NDArray.identity 3)), NDArray.get s [], NDArray.toList t, NDArray.toList c)
            """), Is.EqualTo("(true, true, [-1;-2;-3], 0, 2, 1, 3, 14, [8;26], [[0;1;2];[3;4;5]], 5, [0;2;3;5], [0;3;1;4;2;5])"));
    }

    [Test]
    public void Production_assembly_dependencies_respect_repository_boundaries()
    {
        Assert.That(typeof(NDArray).Assembly.GetReferencedAssemblies().Select(a => a.Name), Has.None.StartsWith("FScript").And.None.EqualTo("FSharp.Core"));
        Assert.That(typeof(NumericsExtension).Assembly.GetReferencedAssemblies().Select(a => a.Name), Has.None.EqualTo("FScript.Language").And.None.EqualTo("FScript.Runtime"));
        Assert.That(typeof(HostTypeDescriptor).Assembly.GetName().Version, Is.EqualTo(new Version(2, 0, 0, 0)));
    }
}
