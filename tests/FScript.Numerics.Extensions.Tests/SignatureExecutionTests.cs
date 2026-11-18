using FScript.Numerics.Extensions;
using FScript.Runtime;
using NUnit.Framework;
namespace FScript.Numerics.Extensions.Tests;

public class SignatureExecutionTests
{
    private const string Prelude = "let a = NDArray.ofRows [[1.0;2.0];[3.0;4.0]]\nlet v = NDArray.ofList [1.0;2.0]\n";
    private static IEnumerable<TestCaseData> Cases()
    {
        foreach (var (expression, expected) in new (string, string)[]
        {
            ("(NDArray.ofFloat32List [1.0]).DType","\"float32\""),
            ("(NDArray.ofInt32List [1;2]).DType","\"int32\""),
            ("NDArray.toBoolList (NDArray.ofBoolList [true;false])","[true;false]"),
            ("let b = NDArray.ofBoolList [false]\nNDArray.setBool b [0] true\nNDArray.getBool b [0]","true"),
            ("let b = NDArray.ofIntList [9007199254740993]\nNDArray.setInt b [0] 9007199254740994\nNDArray.getInt b [0]","9007199254740994"),
            ("NDArray.toList (NDArray.linspace 1.0 3.0 3)","[1;2;3]"),
            ("NDArray.toList (NDArray.astype (NDArray.ofIntList [1;2]) \"float32\")","[1;2]"),
            ("NDArray.toList (NDArray.flatten (NDArray.transpose a))","[1;3;2;4]"),
            ("(NDArray.squeeze (NDArray.expandDims a 1)).Shape","[2;2]"),
            ("(NDArray.fortran a).IsFortranContiguous","true"),
            ("NDArray.toList (NDArray.abs (NDArray.negate v))","[1;2]"),
            ("NDArray.toList (NDArray.sqrt (NDArray.multiply v v))","[1;2]"),
            ("NDArray.toList (NDArray.log (NDArray.ones [2]))","[0;0]"),
            ("NDArray.toList (NDArray.exp (NDArray.zeros [2]))","[1;1]"),
            ("NDArray.toList (NDArray.power v (NDArray.full [2] 2.0))","[1;4]"),
            ("NDArray.toList (NDArray.powerScalar v 3.0)","[1;8]"),
            ("NDArray.toList (NDArray.minimum v (NDArray.scalar 1.5))","[1;1.5]"),
            ("NDArray.toList (NDArray.maximum v (NDArray.scalar 1.5))","[1.5;2]"),
            ("NDArray.toBoolList (NDArray.equal v (NDArray.scalar 1.0))","[true;false]"),
            ("NDArray.toBoolList (NDArray.less v (NDArray.scalar 2.0))","[true;false]"),
            ("NDArray.toBoolList (NDArray.lessEqual v (NDArray.scalar 2.0))","[true;true]"),
            ("NDArray.toBoolList (NDArray.greaterEqual v (NDArray.scalar 2.0))","[false;true]"),
            ("NDArray.toList (NDArray.clip v 1.5 3.0)","[1.5;2]"),
            ("NDArray.toList (NDArray.concatenate [v;v] 0)","[1;2;1;2]"),
            ("NDArray.toRows (NDArray.stack [v;v] 1)","[[1;1];[2;2]]"),
            ("NDArray.product a","24"), ("NDArray.variance a","1.25"),
            ("NDArray.std (NDArray.ofList [0.0;2.0])","1"),
            ("NDArray.all a","true"), ("NDArray.any (NDArray.zeros [0])","false"),
            ("NDArray.toList (NDArray.productAxis a 0 false)","[3;8]"),
            ("NDArray.toList (NDArray.minAxis a 1 false)","[1;3]"),
            ("NDArray.toList (NDArray.maxAxis a 1 false)","[2;4]"),
            ("NDArray.toList (NDArray.meanAxis a 1 false)","[1.5;3.5]"),
            ("NDArray.toList (NDArray.varianceAxis a 1 false 0)","[0.25;0.25]"),
            ("NDArray.toList (NDArray.stdAxis a 1 false 0)","[0.5;0.5]"),
            ("NDArray.toBoolList (NDArray.allAxis a 0 false)","[true;true]"),
            ("NDArray.toBoolList (NDArray.anyAxis a 1 false)","[true;true]"),
            ("BLAS.dot v v","5"),("BLAS.norm (NDArray.ofList [3.0;4.0])","5"),
            ("NDArray.toList (BLAS.matvec a v)","[5;11]"),
            ("NDArray.allClose (LAPACK.luSolve (LAPACK.lu a) v) (NDArray.ofList [0.0;0.5]) 0.00001 0.00001","true"),
            ("let (u,s,vt) = LAPACK.svd a\ns.Shape","[2]"),
            ("let (w,vectors) = LAPACK.symmetricEigen (NDArray.identity 2)\nNDArray.toList w","[1;1]"),
            ("NDArray.allClose (LAPACK.leastSquares a v) (NDArray.ofList [0.0;0.5]) 0.00001 0.00001","true"),
            ("Statistics.quantile a 0.25","1.75"),
            ("NDArray.toList (Statistics.quantileAxis a 0 0.5 false)","[2;3]"),
            ("NDArray.allClose (Statistics.correlation a) (NDArray.ones [2;2]) 0.000001 0.000001","true"),
            ("Statistics.normalCdf 0.0 0.0 1.0","0.5"),
            ("let p = Statistics.normalPdf 0.0 0.0 1.0\np > 0.398 && p < 0.400","true"),
            ("Statistics.uniformPdf 0.5 0.0 2.0","0.5"),
            ("Statistics.uniformCdf 0.5 0.0 2.0","0.25"),
            ("let p = Statistics.binomialPmf 1 2 0.5\np > 0.499 && p < 0.501","true"),
            ("let p = Statistics.binomialCdf 1 2 0.5\np > 0.749 && p < 0.751","true"),
            ("Statistics.poissonPmf 0 0.0","1"),("Statistics.poissonCdf 0 0.0","1"),
            ("(Statistics.uniform [2;3] 0.0 1.0 42).Shape","[2;3]"),
            ("NDArray.toIntList (Statistics.binomial [3] 10 1.0 42)","[10;10;10]"),
            ("NDArray.toIntList (Statistics.poisson [3] 0.0 42)","[0;0;0]"),
            ("(StandardScaler.transform (StandardScaler.fit a) a).Shape","[2;2]"),
            ("NDArray.toRows (MinMaxScaler.transform (MinMaxScaler.fit a) a)","[[0;0];[1;1]]"),
            ("let (model,x) = MinMaxScaler.fitTransform a\n(model.FeatureCount,x.Shape)","(2, [2;2])"),
            ("(LinearRegression.fitRidge a v true 1.0).FeatureCount","2"),
            ("let model = LogisticRegression.fit a (NDArray.ofIntList [0;1]) 0.01 0.1 1 0.00000001\n(model.Converged, model.Iterations, (LogisticRegression.predict model a).Shape, (LogisticRegression.predictProbability model a).Shape)","(false, 1, [2], [2])"),
            ("let (trainX,testX,trainY,testY) = ML.trainTestSplit a v 0.5 42\n(trainX.Shape,testX.Shape,trainY.Shape,testY.Shape)","([1;2], [1;2], [1], [1])"),
            ("Metrics.mae v v","0"),("Metrics.r2 v v","1"),
            ("Metrics.accuracy (NDArray.ofIntList [0;1]) (NDArray.ofIntList [1;1])","0.5"),
            ("NDArray.toIntList (Metrics.confusionMatrix (NDArray.ofIntList [0;1]) (NDArray.ofIntList [1;1]) 2)","[0;1;0;1]")
        })
            yield return new TestCaseData(expression, expected).SetName("Script: " + expression.Split('\n')[0]);
    }
    [TestCaseSource(nameof(Cases))]
    public void Registered_schemes_execute_with_expected_semantics(string expression, string expected)
    {
        var result = ScriptHost.loadSourceWithRegistration(ExtensionBridge.toLanguageRegistrationWithPermissions(FScript.Language.HostPermissions.Native, NumericsExtension.CreateRegistration()), Prelude + expression);
        Assert.That(FScript.Language.Pretty.valueToString(result.LastValue), Is.EqualTo(expected));
    }
}
