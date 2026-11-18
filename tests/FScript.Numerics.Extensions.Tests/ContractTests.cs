using FScript.Numerics.Extensions;
using FScript.Numerics.ML;
using NUnit.Framework;

namespace FScript.Numerics.Extensions.Tests;

public class ContractTests
{
    [Test]
    public void Contracts_name_registered_members_without_promising_mutable_arrays_are_pure()
    {
        var registration = NumericsProvider.Provide();
        var models = registration.Types.Select(type => type.Name).Where(name => name != "NDArray").ToArray();
        Assert.That(NumericsProvider.FScriptPureFunctions(), Is.Empty);
        Assert.That(NumericsProvider.FScriptPureTypes(), Is.EquivalentTo(models));
        Assert.That(NumericsProvider.FScriptThreadSafeTypes(), Is.EquivalentTo(models));
        var safe = NumericsProvider.FScriptThreadSafeFunctions().ToArray();
        Assert.That(safe, Is.Unique);
        Assert.That(safe.Except(registration.Functions.Select(function => function.Name)), Is.Empty);
        Assert.That(safe, Does.Contain("NDArray.zeros").And.Contain("NDArray.ofIntList").And.Contain("Statistics.normal"));
        Assert.That(safe, Does.Not.Contain("NDArray.set").And.Not.Contain("NDArray.setInt").And.Not.Contain("NDArray.sum").And.Not.Contain("NDArray.copy").And.Not.Contain("BLAS.dot"));
    }

    [Test]
    public void Typed_bindings_and_inline_requests_are_registered_and_do_not_claim_purity()
    {
        var registered = NumericsProvider.Provide().Functions.Select(function => function.Name).ToArray();
        var bindings = NumericsProvider.FScriptCompiledFunctions().ToArray();
        Assert.That(bindings.Select(binding => binding.Key), Is.Unique);
        Assert.That(bindings.Select(binding => binding.Key).Except(registered), Is.Empty);
        Assert.That(NumericsProvider.FScriptInlineFunctions().Except(bindings.Select(binding => binding.Key)), Is.Empty);
        Assert.That(NumericsProvider.FScriptPureFunctions(), Is.Empty);
        Assert.That(bindings.All(binding => binding.Value.IsPublic && binding.Value.IsStatic), Is.True);
    }

    [Test]
    public void Typed_array_calls_keep_sdk_identity_aliasing_and_qualified_failures()
    {
        var array = CompiledArrayOperations.OfList([1.0, 2.0]);
        var view = CompiledArrayOperations.Contiguous(array);
        var copy = CompiledArrayOperations.Copy(array);
        InlineArrayBindings.Set(view, [0], 7.0);
        Assert.That(InlineArrayBindings.Get(array, [0]), Is.EqualTo(7.0));
        Assert.That(InlineArrayBindings.Get(copy, [0]), Is.EqualTo(1.0));
        Assert.That(view, Is.EqualTo(array));
        var error = Assert.Throws<FScript.Extensibility.ExtensionException>(() => InlineArrayBindings.Get(array, [-1]));
        Assert.That(error!.Message, Does.StartWith("NDArray.get:"));
        Assert.Throws<FScript.Extensibility.ExtensionException>(() => InlineArrayBindings.Get(array, [long.MaxValue]));
        Assert.Throws<FScript.Extensibility.ExtensionException>(() => CompiledArrayOperations.Zeros([long.MaxValue]));
        var integers = ((FScript.Extensibility.ScriptValue.VHostManaged)NumericsExtension.Wrap(NDArray.FromInt64([1L, 2L]))).Item;
        InlineArrayBindings.SetInt(integers, [0], long.MaxValue);
        Assert.That(InlineArrayBindings.GetInt(integers, [0]), Is.EqualTo(long.MaxValue));
        var booleans = ((FScript.Extensibility.ScriptValue.VHostManaged)NumericsExtension.Wrap(NDArray.FromBooleans([false]))).Item;
        InlineArrayBindings.SetBool(booleans, [0], true);
        Assert.That(InlineArrayBindings.GetBool(booleans, [0]), Is.True);
    }

    [Test]
    public void Snapshot_validation_rejects_each_forged_field_missing_extra_and_nested_dimensions()
    {
        var array = NDArray.FromRows([1.0, 2.0], [3.0, 4.0]);
        var host = ((FScript.Extensibility.ScriptValue.VHostManaged)NumericsExtension.Wrap(array)).Item;
        var descriptorCopy = new FScript.Extensibility.HostTypeDescriptor(host.Descriptor.Name, host.Descriptor.Fields, host.Descriptor.HostManaged);
        var equivalent = new FScript.Extensibility.HostManagedValue(descriptorCopy, host.Fields, array);
        Assert.That(NumericsExtension.Unwrap(FScript.Extensibility.ScriptValue.NewVHostManaged(equivalent)), Is.SameAs(array));
        void Reject(Microsoft.FSharp.Collections.FSharpMap<string, FScript.Extensibility.ScriptValue> fields)
        {
            var invalid = new FScript.Extensibility.HostManagedValue(host.Descriptor, fields, array);
            Assert.Throws<FScript.Extensibility.ExtensionException>(() => NumericsExtension.Unwrap(FScript.Extensibility.ScriptValue.NewVHostManaged(invalid)));
            Assert.Throws<FScript.Extensibility.ExtensionException>(() => InlineArrayBindings.Get(invalid, [0, 0]));
        }
        foreach (var field in host.Descriptor.Fields.Keys)
            Reject(Microsoft.FSharp.Collections.MapModule.Add(field, FScript.Extensibility.ScriptValue.VUnit, host.Fields));
        Reject(Microsoft.FSharp.Collections.MapModule.Remove("Rank", host.Fields));
        Reject(Microsoft.FSharp.Collections.MapModule.Add("Shape", FScript.Extensibility.ScriptValue.NewVList(null!), host.Fields));
        Reject(null!);
        Reject(Microsoft.FSharp.Collections.MapModule.Add("Extra", FScript.Extensibility.ScriptValue.VUnit, host.Fields));
        foreach (string name in new[] { "Shape", "Strides" })
        {
            var wrong = FScript.Extensibility.ScriptValue.NewVList(Microsoft.FSharp.Collections.ListModule.OfSeq(new[] { FScript.Extensibility.ScriptValue.NewVInt(2), FScript.Extensibility.ScriptValue.NewVInt(99) }));
            Reject(Microsoft.FSharp.Collections.MapModule.Add(name, wrong, host.Fields));
        }
        array.Set(7, 0, 0);
        Assert.That(NumericsExtension.Unwrap(FScript.Extensibility.ScriptValue.NewVHostManaged(host)), Is.SameAs(array));
        Assert.That(InlineArrayBindings.Get(host, [0, 0]), Is.EqualTo(7.0));
    }

    [Test]
    public void Repeated_valid_unwrapping_does_not_allocate_snapshot_maps()
    {
        var wrapped = NumericsExtension.Wrap(NDArray.FromRows([1.0, 2.0], [3.0, 4.0]));
        for (int i = 0; i < 1000; i++)
            NumericsExtension.Unwrap(wrapped);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++)
            NumericsExtension.Unwrap(wrapped);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.That(allocated, Is.Zero);
    }

    [Test]
    public void Typed_getters_allocate_only_the_checked_core_index_array()
    {
        var host = CompiledArrayOperations.OfList([1.0, 2.0]);
        long[] indices = [0];
        for (int i = 0; i < 1000; i++) CompiledArrayOperations.Get(host, indices);
        long before = GC.GetAllocatedBytesForCurrentThread();
        double total = 0;
        for (int i = 0; i < 1000; i++) total += CompiledArrayOperations.Get(host, indices);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.That(total, Is.EqualTo(1000.0));
        Assert.That(allocated, Is.LessThanOrEqualTo(40000), "No transport/validation maps, delegate closures or LINQ iterators on valid getter calls");
    }

    [Test]
    public void Model_arrays_are_copies_and_concurrent_model_observations_are_stable()
    {
        var input = NDArray.FromRows([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]);
        var scaler = StandardScaler.Fit(input);
        var wrapped = ModelInterop.Wrap(scaler);
        var expected = scaler.Mean.ToArray();
        scaler.Mean.Set(999, 0);
        input.Set(999, 0, 0);
        Assert.That(scaler.Mean.ToArray(), Is.EqualTo(expected));
        Parallel.For(0, 64, _ =>
        {
            Assert.That(ModelInterop.Wrap(scaler), Is.EqualTo(wrapped));
            Assert.That(scaler.Mean.ToArray(), Is.EqualTo(expected));
            Assert.That(ModelInterop.Unwrap<StandardScaler>(wrapped), Is.SameAs(scaler));
        });
    }
}
