using FScript.Extensibility;
using FScript.Numerics.Blas;
using FScript.Numerics.Lapack;
using FScript.Numerics.Statistics;
using FScript.Numerics.ML;
namespace FScript.Numerics.Extensions;

public static partial class NumericsExtension
{
    private static ScriptType TupleType(params ScriptType[] items) => ScriptType.NewTTuple(Sdk.List(items));
    private static ScriptValue Tuple(params ScriptValue[] items) => ScriptValue.NewVTuple(Sdk.List(items));
    private static bool Boolean(ScriptValue value) => value is ScriptValue.VBool b ? b.Item : throw new ArgumentException("Expected bool.");
    private static string Text(ScriptValue value) => value is ScriptValue.VString s ? s.Item : throw new ArgumentException("Expected string.");
    private static long Integer(ScriptValue value) => value is ScriptValue.VInt i ? i.Item : throw new ArgumentException("Expected int.");
    private static DType ParseDType(ScriptValue value) => Text(value) switch { "float64" => DType.Float64, "float32" => DType.Float32, "int32" => DType.Int32, "int64" => DType.Int64, "bool" => DType.Bool, _ => throw new ArgumentException("Dtype must be float64, float32, int32, int64 or bool.") };
    private static void RegisterExpansion(List<ExternalFunction> functions)
    {
        var A = ArrayType;
        var I = ScriptType.TInt;
        var F = ScriptType.TFloat;
        var B = ScriptType.TBool;
        var S = ScriptType.TString;
        var IL = IntList;
        var AL = ScriptType.NewTList(A);
        void Add(string name, ScriptType result, ScriptType[] arguments, Func<ScriptValue[], ScriptValue> handler) => functions.Add(Sdk.Declare(name, result, arguments, handler));
        NDArray V(ScriptValue[] args, int i = 0) => Unwrap(args[i]);
        int Int(ScriptValue[] a, int i) => Sdk.Int(a[i]);
        double Float(ScriptValue[] a, int i) => Sdk.Float(a[i]);
        Add("ofIntList", A, [IL], a => Wrap(NDArray.FromInt64(Sdk.Items(a[0]).Select(Integer).ToArray())));
        Add("ofInt32List", A, [IL], a => Wrap(NDArray.FromInt32(Sdk.Integers(a[0]))));
        Add("ofBoolList", A, [ScriptType.NewTList(B)], a => Wrap(NDArray.FromBooleans(Sdk.Items(a[0]).Select(Boolean).ToArray())));
        Add("ofFloat32List", A, [FloatList], a => Wrap(NDArray.FromArray(Sdk.Floats(a[0]), DType.Float32)));
        Add("astype", A, [A, S], a => Wrap(V(a).Astype(ParseDType(a[1]))));
        Add("getInt", I, [A, IL], a => ScriptValue.NewVInt(V(a).GetInt64(Sdk.Integers(a[1]))));
        Add("setInt", ScriptType.TUnit, [A, IL, I], a => { V(a).SetInt64(Integer(a[2]), Sdk.Integers(a[1])); return ScriptValue.VUnit; });
        Add("getBool", B, [A, IL], a => ScriptValue.NewVBool(V(a).GetBoolean(Sdk.Integers(a[1]))));
        Add("setBool", ScriptType.TUnit, [A, IL, B], a => { V(a).SetBoolean(Boolean(a[2]), Sdk.Integers(a[1])); return ScriptValue.VUnit; });
        Add("toIntList", IL, [A], a => ScriptValue.NewVList(Sdk.List(V(a).ToInt64Array().Select(ScriptValue.NewVInt))));
        Add("toBoolList", ScriptType.NewTList(B), [A], a => ScriptValue.NewVList(Sdk.List(V(a).ToBooleanArray().Select(ScriptValue.NewVBool))));
        Add("linspace", A, [F, F, I], a => Wrap(NDArray.Linspace(Float(a, 0), Float(a, 1), Int(a, 2))));
        Add("broadcastTo", A, [A, IL], a => Wrap(V(a).BroadcastTo(new Shape(Sdk.Integers(a[1])))));
        Add("expandDims", A, [A, I], a => Wrap(V(a).ExpandDims(Int(a, 1))));
        foreach (var (name, op) in new (string, Func<NDArray, NDArray>)[] { ("flatten", x => x.Flatten()), ("squeeze", x => x.Squeeze()), ("fortran", x => x.AsFortranContiguous()), ("abs", x => x.Abs()), ("sqrt", x => x.Sqrt()), ("exp", x => x.Exp()), ("log", x => x.Log()) })
            Add(name, A, [A], a => Wrap(op(V(a))));
        foreach (var (name, op) in new (string, Func<NDArray, NDArray, NDArray>)[] { ("power", (x, y) => x.Power(y)), ("minimum", (x, y) => x.Minimum(y)), ("maximum", (x, y) => x.Maximum(y)), ("equal", (x, y) => x.EqualElements(y)), ("less", (x, y) => x.Less(y)), ("lessEqual", (x, y) => x.LessEqual(y)), ("greater", (x, y) => x.Greater(y)), ("greaterEqual", (x, y) => x.GreaterEqual(y)) })
            Add(name, A, [A, A], a => Wrap(op(V(a), V(a, 1))));
        Add("powerScalar", A, [A, F], a => Wrap(V(a).Power(Float(a, 1))));
        Add("clip", A, [A, F, F], a => Wrap(V(a).Clip(Float(a, 1), Float(a, 2))));
        Add("where", A, [A, A, A], a => Wrap(NDArray.Where(V(a), V(a, 1), V(a, 2))));
        Add("concatenate", A, [AL, I], a => Wrap(NDArray.Concatenate(Sdk.Items(a[0]).Select(Unwrap).ToArray(), Int(a, 1))));
        Add("stack", A, [AL, I], a => Wrap(NDArray.Stack(Sdk.Items(a[0]).Select(Unwrap).ToArray(), Int(a, 1))));
        foreach (var (name, op) in new (string, Func<NDArray, double>)[] { ("product", x => x.Product()), ("variance", x => x.Variance()), ("std", x => x.Std()) })
            Add(name, F, [A], a => ScriptValue.NewVFloat(op(V(a))));
        Add("all", B, [A], a => ScriptValue.NewVBool(V(a).All()));
        Add("any", B, [A], a => ScriptValue.NewVBool(V(a).Any()));
        foreach (var (name, op) in new (string, Func<NDArray, int, bool, NDArray>)[] { ("sumAxis", (x, i, k) => x.SumAxis(i, k)), ("productAxis", (x, i, k) => x.ProductAxis(i, k)), ("minAxis", (x, i, k) => x.MinAxis(i, k)), ("maxAxis", (x, i, k) => x.MaxAxis(i, k)), ("meanAxis", (x, i, k) => x.MeanAxis(i, k)), ("allAxis", (x, i, k) => x.AllAxis(i, k)), ("anyAxis", (x, i, k) => x.AnyAxis(i, k)) })
            Add(name, A, [A, I, B], a => Wrap(op(V(a), Int(a, 1), Boolean(a[2]))));
        Add("varianceAxis", A, [A, I, B, I], a => Wrap(V(a).VarianceAxis(Int(a, 1), Boolean(a[2]), Int(a, 3))));
        Add("stdAxis", A, [A, I, B, I], a => Wrap(V(a).StdAxis(Int(a, 1), Boolean(a[2]), Int(a, 3))));
        Add("BLAS.dot", F, [A, A], a => ScriptValue.NewVFloat(LinearAlgebra.Dot(V(a), V(a, 1))));
        Add("BLAS.norm", F, [A], a => ScriptValue.NewVFloat(LinearAlgebra.Norm(V(a))));
        Add("BLAS.matvec", A, [A, A], a => Wrap(LinearAlgebra.Matvec(V(a), V(a, 1))));
        Add("BLAS.matmul", A, [A, A], a => Wrap(LinearAlgebra.Matmul(V(a), V(a, 1))));
        var LU = ScriptType.NewTNamed(nameof(LuFactorization));
        Add("LAPACK.solve", A, [A, A], a => Wrap(Decompositions.Solve(V(a), V(a, 1))));
        Add("LAPACK.lu", LU, [A], a => ModelInterop.Wrap(Decompositions.Lu(V(a))));
        Add("LAPACK.luSolve", A, [LU, A], a => Wrap(ModelInterop.Unwrap<LuFactorization>(a[0]).Solve(V(a, 1))));
        Add("LAPACK.qr", TupleType(A, A), [A], a => { var r = Decompositions.Qr(V(a)); return Tuple(Wrap(r.Q), Wrap(r.R)); });
        Add("LAPACK.svd", TupleType(A, A, A), [A], a => { var r = Decompositions.Svd(V(a)); return Tuple(Wrap(r.U), Wrap(r.SingularValues), Wrap(r.VT)); });
        Add("LAPACK.symmetricEigen", TupleType(A, A), [A], a => { var r = Decompositions.SymmetricEigen(V(a)); return Tuple(Wrap(r.Values), Wrap(r.Vectors)); });
        Add("LAPACK.cholesky", A, [A], a => Wrap(Decompositions.Cholesky(V(a))));
        Add("LAPACK.leastSquares", A, [A, A], a => Wrap(Decompositions.LeastSquares(V(a), V(a, 1))));
        Add("Statistics.median", F, [A], a => ScriptValue.NewVFloat(Descriptive.Median(V(a))));
        Add("Statistics.quantile", F, [A, F], a => ScriptValue.NewVFloat(Descriptive.Quantile(V(a), Float(a, 1))));
        Add("Statistics.quantileAxis", A, [A, I, F, B], a => Wrap(Descriptive.QuantileAxis(V(a), Int(a, 1), Float(a, 2), Boolean(a[3]))));
        Add("Statistics.covariance", A, [A, I], a => Wrap(Descriptive.Covariance(V(a), Int(a, 1))));
        Add("Statistics.correlation", A, [A], a => Wrap(Descriptive.Correlation(V(a))));
        Add("Statistics.histogram", TupleType(A, A), [A, I, F, F], a => { var r = Descriptive.Histogram(V(a), Int(a, 1), Float(a, 2), Float(a, 3)); return Tuple(Wrap(r.Counts), Wrap(r.Edges)); });
        foreach (var (name, op) in new (string, Func<double, double, double, double>)[] { ("normalPdf", Distributions.NormalPdf), ("normalCdf", Distributions.NormalCdf), ("uniformPdf", Distributions.UniformPdf), ("uniformCdf", Distributions.UniformCdf) })
            Add("Statistics." + name, F, [F, F, F], a => ScriptValue.NewVFloat(op(Float(a, 0), Float(a, 1), Float(a, 2))));
        Add("Statistics.binomialPmf", F, [I, I, F], a => ScriptValue.NewVFloat(Distributions.BinomialPmf(Int(a, 0), Int(a, 1), Float(a, 2))));
        Add("Statistics.binomialCdf", F, [I, I, F], a => ScriptValue.NewVFloat(Distributions.BinomialCdf(Int(a, 0), Int(a, 1), Float(a, 2))));
        Add("Statistics.poissonPmf", F, [I, F], a => ScriptValue.NewVFloat(Distributions.PoissonPmf(Int(a, 0), Float(a, 1))));
        Add("Statistics.poissonCdf", F, [I, F], a => ScriptValue.NewVFloat(Distributions.PoissonCdf(Int(a, 0), Float(a, 1))));
        Add("Statistics.normal", A, [IL, F, F, I], a => Wrap(Distributions.Normal(new Shape(Sdk.Integers(a[0])), Float(a, 1), Float(a, 2), Integer(a[3]))));
        Add("Statistics.uniform", A, [IL, F, F, I], a => Wrap(Distributions.Uniform(new Shape(Sdk.Integers(a[0])), Float(a, 1), Float(a, 2), Integer(a[3]))));
        Add("Statistics.binomial", A, [IL, I, F, I], a => Wrap(Distributions.Binomial(new Shape(Sdk.Integers(a[0])), Int(a, 1), Float(a, 2), Integer(a[3]))));
        Add("Statistics.poisson", A, [IL, F, I], a => Wrap(Distributions.Poisson(new Shape(Sdk.Integers(a[0])), Float(a, 1), Integer(a[2]))));
        RegisterModels(functions);
    }
    private static void RegisterModels(List<ExternalFunction> functions)
    {
        var A = ArrayType;
        var I = ScriptType.TInt;
        var F = ScriptType.TFloat;
        var B = ScriptType.TBool;
        void Add(string name, ScriptType result, ScriptType[] arguments, Func<ScriptValue[], ScriptValue> handler) => functions.Add(Sdk.Declare(name, result, arguments, handler));
        NDArray V(ScriptValue[] a, int i = 0) => Unwrap(a[i]);
        void Scaler<T>(string name, Func<NDArray, T> fit, Func<T, NDArray, NDArray> transform) where T : class
        {
            var type = ScriptType.NewTNamed(name);
            Add(name + ".fit", type, [A], a => ModelInterop.Wrap(fit(V(a))));
            Add(name + ".transform", A, [type, A], a => Wrap(transform(ModelInterop.Unwrap<T>(a[0]), V(a, 1))));
            Add(name + ".fitTransform", TupleType(type, A), [A], a => { var model = fit(V(a)); return Tuple(ModelInterop.Wrap(model), Wrap(transform(model, V(a)))); });
        }
        Scaler("StandardScaler", StandardScaler.Fit, (StandardScaler m, NDArray x) => m.Transform(x));
        Scaler("MinMaxScaler", MinMaxScaler.Fit, (MinMaxScaler m, NDArray x) => m.Transform(x));
        var linear = ScriptType.NewTNamed("LinearRegression");
        var logistic = ScriptType.NewTNamed("LogisticRegression");
        var km = ScriptType.NewTNamed("KMeans");
        var pca = ScriptType.NewTNamed("Pca");
        Add("LinearRegression.fit", linear, [A, A, B], a => ModelInterop.Wrap(LinearRegression.Fit(V(a), V(a, 1), Boolean(a[2]))));
        Add("LinearRegression.fitRidge", linear, [A, A, B, F], a => ModelInterop.Wrap(LinearRegression.Fit(V(a), V(a, 1), Boolean(a[2]), Sdk.Float(a[3]))));
        Add("LinearRegression.predict", A, [linear, A], a => Wrap(ModelInterop.Unwrap<LinearRegression>(a[0]).Predict(V(a, 1))));
        Add("LogisticRegression.fit", logistic, [A, A, F, F, I, F], a => ModelInterop.Wrap(LogisticRegression.Fit(V(a), V(a, 1), Sdk.Float(a[2]), Sdk.Float(a[3]), Sdk.Int(a[4]), Sdk.Float(a[5]))));
        Add("LogisticRegression.predict", A, [logistic, A], a => Wrap(ModelInterop.Unwrap<LogisticRegression>(a[0]).Predict(V(a, 1))));
        Add("LogisticRegression.predictProbability", A, [logistic, A], a => Wrap(ModelInterop.Unwrap<LogisticRegression>(a[0]).PredictProbability(V(a, 1))));
        Add("KMeans.fit", km, [A, I, I, I, F], a => ModelInterop.Wrap(KMeans.Fit(V(a), Sdk.Int(a[1]), Integer(a[2]), Sdk.Int(a[3]), Sdk.Float(a[4]))));
        Add("KMeans.predict", A, [km, A], a => Wrap(ModelInterop.Unwrap<KMeans>(a[0]).Predict(V(a, 1))));
        Add("PCA.fit", pca, [A, I], a => ModelInterop.Wrap(Pca.Fit(V(a), Sdk.Int(a[1]))));
        Add("PCA.transform", A, [pca, A], a => Wrap(ModelInterop.Unwrap<Pca>(a[0]).Transform(V(a, 1))));
        Add("PCA.fitTransform", TupleType(pca, A), [A, I], a => { var m = Pca.Fit(V(a), Sdk.Int(a[1])); return Tuple(ModelInterop.Wrap(m), Wrap(m.Transform(V(a)))); });
        Add("ML.trainTestSplit", TupleType(A, A, A, A), [A, A, F, I], a => { var r = Preprocessing.TrainTestSplit(V(a), V(a, 1), Sdk.Float(a[2]), Integer(a[3])); return Tuple(Wrap(r.TrainFeatures), Wrap(r.TestFeatures), Wrap(r.TrainTargets), Wrap(r.TestTargets)); });
        foreach (var (name, op) in new (string, Func<NDArray, NDArray, double>)[] { ("mse", Metrics.Mse), ("mae", Metrics.Mae), ("r2", Metrics.R2), ("accuracy", Metrics.Accuracy) })
            Add("Metrics." + name, F, [A, A], a => ScriptValue.NewVFloat(op(V(a), V(a, 1))));
        Add("Metrics.confusionMatrix", A, [A, A, I], a => Wrap(Metrics.ConfusionMatrix(V(a), V(a, 1), Sdk.Int(a[2]))));
    }
}
