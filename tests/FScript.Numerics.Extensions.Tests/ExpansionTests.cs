using FScript.Extensibility;
using FScript.Numerics.Extensions;
using FScript.Numerics.ML;
using FScript.Runtime;
using NUnit.Framework;
namespace FScript.Numerics.Extensions.Tests;

public class ExpansionTests
{
    private static string Run(string source) => FScript.Language.Pretty.valueToString(ScriptHost.loadSourceWithRegistration(ExtensionBridge.toLanguageRegistrationWithPermissions(FScript.Language.HostPermissions.Native, NumericsExtension.CreateRegistration()), source).LastValue);
    [Test]
    public void Script_broadcasting_integer_precision_and_readonly_views_are_authoritative()
    {
        Assert.That(Run("""
            let a = NDArray.ofIntList [9007199254740993; 9007199254740994]
            let b = NDArray.add a (NDArray.ofIntList [1])
            let mask = NDArray.greater b a
            (NDArray.toIntList b, NDArray.toBoolList mask, b.DType, (NDArray.broadcastTo a [3;2]).IsReadOnly)
            """), Is.EqualTo("([9007199254740994;9007199254740995], [true;true], \"int64\", true)"));
        Assert.Throws<FScript.Language.TypeException>(() => Run("NDArray.ofIntList [1.0]"));
        Assert.Throws<FScript.Language.EvalException>(() => Run("let a = NDArray.broadcastTo (NDArray.ofList [1.0]) [3]\nNDArray.set a [0] 2.0"));
    }
    [Test]
    public void Native_statistics_and_models_work_in_scripts_and_tasks()
    {
        Assert.That(Run("""
            let x = NDArray.ofRows [[0.0];[1.0];[2.0];[3.0]]
            let y = NDArray.ofList [1.0;3.0;5.0;7.0]
            let model = LinearRegression.fit x y true
            let predict () = LinearRegression.predict model x
            let result = predict ()
            let (q,r) = LAPACK.qr x
            let sample = Statistics.normal [3] 0.0 1.0 42
            (NDArray.allClose result y 0.000001 0.000001, model.FeatureCount, NDArray.allClose (BLAS.matmul q r) x 0.000001 0.000001, NDArray.equals sample (Statistics.normal [3] 0.0 1.0 42))
            """), Is.EqualTo("(true, 1, true, true)"));
    }
    [Test]
    public void Fitted_objects_validate_payload_and_hide_state()
    {
        var model = StandardScaler.Fit(NDArray.FromRows([1], [2]));
        var wrapped = ModelInterop.Wrap(model);
        Assert.That(ModelInterop.Unwrap<StandardScaler>(wrapped), Is.SameAs(model));
        var host = ((ScriptValue.VHostManaged)wrapped).Item;
        Assert.That(host.Fields.Keys, Is.EquivalentTo(new[] { "Id", "FeatureCount" }));
        Assert.Throws<ExtensionException>(() => ModelInterop.Unwrap<StandardScaler>(ScriptValue.NewVHostManaged(new HostManagedValue(host.Descriptor, host.Fields, new object()))));
        Assert.Throws<ExtensionException>(() => ModelInterop.Unwrap<KMeans>(wrapped));
        Assert.Throws<FScript.Language.TypeException>(() => Run("let m = StandardScaler.fit (NDArray.ofRows [[1.0]])\nm.Mean"));
        Assert.Throws<FScript.Language.TypeException>(() => Run("let m = StandardScaler.fit (NDArray.ofRows [[1.0]])\n{m with FeatureCount = 2}"));
        Assert.Throws<FScript.Language.TypeException>(() => Run("StandardScaler.transform {Id = \"fake\"; FeatureCount = 1} (NDArray.ones [1;1])"));
    }
}
