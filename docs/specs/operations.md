# Operation arguments and numerical policies

See [registration reference](registration.md) for every exact signature. Parameters below are in invocation order. All numerical array-taking functions use array first; fitted-model transformations take model then array. FScript function arguments are curried.

## NDArray

- `scalar value`, `ofList values`, `ofRows rows`, `zeros shape`, `ones shape`, `full shape value`, `identity size`, `arange start stop step`, `linspace start stop count` construct Float64 arrays. Lists contain script floats; integer literals are not silently accepted.
- `ofFloat32List values`, `ofIntList values`, `ofInt32List values`, `ofBoolList values` construct Float32, Int64, Int32, Bool arrays respectively. Integer lists are exact; Int32 checks narrowing.
- `astype array dtype` accepts strings `"float64"`, `"float32"`, `"int32"`, `"int64"`, `"bool"`. [Conversion and promotion rules](ndarray.md) apply.
- `get array indices` and `set array indices value` use script floats. `getInt`/`setInt` and `getBool`/`setBool` require integer/bool storage and corresponding script values. Every setter returns unit and mutates one element; broadcast writes fail.
- `toList array` flattens any rank as floats; `toRows` requires rank 2. `toIntList` and `toBoolList` preserve exact typed values. Metadata is available through visible fields, not expensive conversion calls.
- `transpose array`, `permute array axes`, `reshape array shape`, `slice array axis start stop step`, `expandDims array axis`, `squeeze array`, `broadcastTo array shape`, `flatten array`, `copy array`, `contiguous array`, `fortran array` follow the sharing table in the NDArray contract.
- `add`, `subtract`, `multiply`, `divide`, `power`, `minimum`, `maximum` take `array other`, with broadcasting. `addScalar`, `subtractScalar`, `multiplyScalar`, `divideScalar`, `powerScalar` take `array float`. `negate`, `abs`, `sqrt`, `exp`, `log` take one array. `clip array minimum maximum` is elementwise clamping.
- `equal`, `less`, `lessEqual`, `greater`, `greaterEqual` return elementwise Bool arrays. `equals array other` is a whole-array content bool. `allClose array other absoluteTolerance relativeTolerance` uses symmetric tolerance. Ordinary script equality uses Id.
- `sum`, `product`, `mean`, `min`, `max`, `variance`, `std` return a script float; variance/std use ddof=0. `all`, `any` return bool. `sumAxis`/`productAxis`/`minAxis`/`maxAxis`/`meanAxis`/`allAxis`/`anyAxis` take `array axis keepDimensions`; `varianceAxis`/`stdAxis` add a final `degreesOfFreedom` argument.
- `concatenate arrays axis`, `stack arrays axis`, `where mask whenTrue whenFalse` allocate new arrays.
- `dot array other`, `matvec matrix vector`, `matmul left right` use managed reference algorithms; native versions are below.

## BLAS and LAPACK

`BLAS.dot left right` and `BLAS.norm vector` return floats. `BLAS.matvec matrix vector` and `BLAS.matmul left right` allocate outputs. Float32+Float32 uses single-precision kernels; mixed floating inputs promote to Float64. Integer and Bool inputs must be explicitly converted. Empty compatible products return zeros without loading native code.

LAPACK requires finite Float32/Float64 matrices. Right-hand sides must have matching dtype and rows, rank 1 or 2. Results preserve input dtype; internal workspace is never returned as an alias of an input.

| Function | Arguments | Result and policy |
| --- | --- | --- |
| `LAPACK.solve` | matrix right | Solution; square nonsingular matrix |
| `LAPACK.lu` | matrix | Opaque `LuFactorization`; square nonsingular matrix |
| `LAPACK.luSolve` | factorization right | Solution; factorization reusable and unchanged |
| `LAPACK.qr` | matrix | `(Q,R)`, economy shapes `[m,k]`, `[k,n]`, k=min(m,n) |
| `LAPACK.leastSquares` | matrix right | Minimum-norm SVD solution, including rank-deficient / underdetermined systems |
| `LAPACK.cholesky` | matrix | Lower triangle L such that A=L Lᵀ; symmetric positive definite |
| `LAPACK.svd` | matrix | `(U,s,VT)`, economy shapes `[m,k]`, `[k]`, `[k,n]`; descending singular values |
| `LAPACK.symmetricEigen` | matrix | `(values,vectors)` ascending eigenvalues and eigenvectors in columns |

Both triangles must agree exactly for symmetric operations. Least-squares default relative cutoff is `max(m,n)*machineEpsilon(dtype)`; singular values at/below `largest*cutoff` are discarded. The .NET overload accepts an explicit relative cutoff. Empty systems/decompositions return consistently shaped empty/zero arrays. LAPACK `info` failures become named descriptive errors: singularity, non-positive-definiteness, nonconvergence, or invalid native arguments. See .NET `Decompositions` and immutable `LuFactorization`.

## Statistics

.NET entry points are `Descriptive` and `Distributions` in `FScript.Numerics.Statistics`.

| FScript operation | Arguments | Definition |
| --- | --- | --- |
| `Statistics.median` | array | Flattened median |
| `Statistics.quantile` | array probability | Linear interpolation at `(n-1)*p`, p in [0,1] |
| `Statistics.quantileAxis` | array axis probability keepDimensions | Same policy for each axis group |
| `Statistics.covariance` | observations degreesOfFreedom | Rows=samples, columns=variables; denominator n-ddof |
| `Statistics.correlation` | observations | Covariance scaled by standard deviations; constant columns yield NaN |
| `Statistics.histogram` | array bins minimum maximum | `(Int64 counts, Float64 edges)`; equal-width bins, final edge inclusive, outside bounds ignored |
| `Statistics.normalPdf` / `normalCdf` | x mean std | Finite mean and positive finite std |
| `Statistics.uniformPdf` / `uniformCdf` | x lower upper | Finite increasing bounds; PDF includes endpoints |
| `Statistics.binomialPmf` / `binomialCdf` | successes trials probability | Nonnegative Int32 trials, p in [0,1]; CDF P(X<=successes) |
| `Statistics.poissonPmf` / `poissonCdf` | count rate | Finite rate in [0,1e9], CDF P(X<=count) |
| `Statistics.normal` | shape mean std seed | Independent Float64 samples |
| `Statistics.uniform` | shape lower upper seed | Independent Float64 samples in [lower,upper), subject to float rounding |
| `Statistics.binomial` | shape trials probability seed | Int64 samples |
| `Statistics.poisson` | shape rate seed | Int64 samples |

Quantile rejects empty groups and propagates NaN; ordered infinities are allowed. Covariance requires n>ddof>=0; numerical NaNs propagate. Histogram rejects nonfinite observations/bounds, requires positive bins and minimum<maximum. Discrete PMFs/CDFs handle out-of-support counts and degenerate distributions explicitly. Normal CDF uses an erfc approximation (about 1.5e-7 maximum absolute error); discrete CDFs use incomplete beta/gamma with bounded iterative convergence. Validate tails numerically, not by subtracting nearly equal PMFs.

Sampling uses SplitMix64 with an explicit signed 64-bit seed, Box–Muller normals, and inverse-CDF discrete sampling. Each call owns its RNG, so simultaneous calls do not share state. Repeating arguments/seed repeats a sequence within this implementation; transcendental rounding need not be bit-identical on every architecture. Algorithm changes may change sequences in future versions. Public .NET `SeededRandom` instances are mutable and require caller synchronization when shared.

## Models and metrics

.NET classes live in `FScript.Numerics.ML`. Inputs use `[samples,features]`, finite Float32/64, at least one training row and feature. There is no missing-value imputation. Targets are finite rank-one arrays matching rows; classification requires documented labels. Fitted state is private. Transform/predict outputs use Float64 except class labels (Int64). Model arrays exposed by .NET properties are independent copies.

| Function | Arguments | Behavior |
| --- | --- | --- |
| `StandardScaler.fit` | features | Population column mean/std; constant scale becomes 1 |
| `MinMaxScaler.fit` | features | Column minimum/range; zero range becomes 1 |
| `StandardScaler.transform` / `MinMaxScaler.transform` | model features | Fresh transformed matrix; validates feature count |
| `StandardScaler.fitTransform` / `MinMaxScaler.fitTransform` | features | `(model,transformed)` |
| `ML.trainTestSplit` | features targets testFraction seed | `(trainX,testX,trainY,testY)`; seeded Fisher–Yates, test count ceil(n*f), both partitions nonempty; preserves dtypes |
| `LinearRegression.fit` | features targets fitIntercept | Minimum-norm SVD least squares |
| `LinearRegression.fitRidge` | features targets fitIntercept ridge | Augmented least squares; ridge>=0 penalizes coefficients only |
| `LinearRegression.predict` | model features | Fresh target vector |
| `LogisticRegression.fit` | features targets learningRate ridge maxIterations tolerance | Full-batch gradient descent; labels 0/1 and both classes required |
| `LogisticRegression.predictProbability` | model features | Stable sigmoid probabilities |
| `LogisticRegression.predict` | model features | Int64 labels, probability>=0.5 -> 1 |
| `KMeans.fit` | features clusters seed maxIterations tolerance | Seeded k-means++, Lloyd updates, squared Euclidean distances |
| `KMeans.predict` | model features | Nearest-center Int64 labels; ties choose first center |
| `PCA.fit` | features components | Centered SVD, at least two rows, 1..min(samples,features) components |
| `PCA.transform` | model features | Centered projection |
| `PCA.fitTransform` | features components | `(Pca model,projected)` |
| `Metrics.mse` / `mae` / `r2` | actual predicted | Equal nonempty finite vectors; R² requires >=2 values |
| `Metrics.accuracy` | actual predicted | Nonnegative Int32-valued class labels |
| `Metrics.confusionMatrix` | actual predicted classCount | Int64 `[classCount,classCount]`, rows actual, columns predicted |

Logistic loss is mean binary cross entropy + ridge/2*||w||²; intercept unpenalized. Convergence means gradient L2 norm <= tolerance. Positive learningRate/tolerance and maxIterations, nonnegative ridge; divergent arithmetic throws with scaling advice. Exhausting iterations returns a model with `Converged=false`, never silently claims convergence.

K-means stops when the largest center displacement <= tolerance. Empty clusters retain previous centers; duplicate observations may yield fewer distinct effective clusters. Exhausting iterations is visible through `Converged`. `Inertia` is recomputed for final centers. Seeds control initialization, not hidden global state.

PCA orders components by descending singular value, variance=s²/(n-1). Each component's largest-absolute loading is made nonnegative to stabilize signs; degenerate eigenspaces still need not have identical bases across native implementations. No whitening is performed. .NET offers `Mean`, `Components`, and `ExplainedVariance` copies. Script inspection shows only bounded metadata.

R² returns 1 for exact constant targets and 0 for imperfect constant targets. Confusion matrices use explicit positive classCount to retain absent classes, and reject out-of-range labels. Arithmetic with extreme finite values may still overflow; scaler/optimizer checks provide errors where a usable fitted model cannot be produced.

## .NET fitted objects and decomposition results

All array-valued fitted-model properties below return independent copies. Decomposition result arrays are fresh outputs that callers may explicitly mutate.

| .NET type | Public state and operations |
| --- | --- |
| `LuFactorization` | `Order`; `Factors` copy of packed LU; `Pivots` copy of **one-based LAPACK row-swap indices**; `Solve(right)` |
| `QrResult` | `Q`, `R` |
| `SvdResult` | `U`, `SingularValues`, `VT` |
| `EigenResult` | `Values`, `Vectors` (eigenvectors in columns) |
| `HistogramResult` | `Counts`, `Edges` |
| `SplitResult` | `TrainFeatures`, `TestFeatures`, `TrainTargets`, `TestTargets` |
| `StandardScaler` | `FeatureCount`, `Mean`, `Scale`; static `Fit`/`FitTransform`; instance `Transform` |
| `MinMaxScaler` | `FeatureCount`, `Minimum`, `Range`; static `Fit`/`FitTransform`; instance `Transform` |
| `LinearRegression` | `FeatureCount`, `Coefficients`, `Intercept`; static `Fit(features, targets, fitIntercept=true, ridge=0)`; instance `Predict` |
| `LogisticRegression` | `FeatureCount`, `Coefficients`, `Intercept`, `Iterations`, `Converged`; static `Fit(features, targets, learningRate=0.1, ridge=0, maxIterations=1000, tolerance=1e-8)`; instance `Predict`/`PredictProbability` |
| `KMeans` | `FeatureCount`, `ClusterCount`, `Centers`, `Iterations`, `Converged`, `Inertia`; static `Fit(features, clusters, seed, maxIterations=300, tolerance=1e-6)`; instance `Predict` |
| `Pca` | `FeatureCount`, `ComponentCount`, `Mean`, `Components`, `ExplainedVariance`; static `Fit`/`FitTransform(features, componentCount)`; instance `Transform` |
| `SeededRandom` | Constructor `(long seed)`; `NextUInt64`, `NextDouble` in [0,1), unbiased `NextInt(exclusiveMaximum)`; mutable instance, no shared global state |

`StandardScaler`, `MinMaxScaler` and `Pca` .NET `FitTransform` return named value tuples `(Model,Values)`. Static distribution sampling methods take `(Shape, parameters..., long seed)`; `Descriptive.QuantileAxis` takes `(array,axis,probability,keepDimensions=false)`. `LinearAlgebra.BackendConfiguration` explicitly initializes the backend and reports its configuration string; ordinary registration does not call it.
