using System.Runtime.InteropServices;
using FScript.Numerics.Blas;
namespace FScript.Numerics.Lapack;

public sealed record QrResult(NDArray Q, NDArray R);
public sealed record SvdResult(NDArray U, NDArray SingularValues, NDArray VT);
public sealed record EigenResult(NDArray Values, NDArray Vectors);

/// <summary>LU storage is private. Factors and pivots are returned as independent copies.</summary>
public sealed class LuFactorization
{
    private readonly NDArray factors;
    private readonly int[] pivots;
    internal LuFactorization(NDArray factors, int[] pivots)
    {
        this.factors = factors;
        this.pivots = (int[])pivots.Clone();
    }
    public int Order => factors.Shape[0];
    public NDArray Factors => factors.Copy();
    public int[] Pivots => (int[])pivots.Clone();
    public NDArray Solve(NDArray right) => Decompositions.SolveLu(factors, pivots, right);
}

/// <summary>Finite, floating LAPACKE operations, using private copies of all overwritten inputs.</summary>
public static class Decompositions
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int Gesv(int layout, int n, int nrhs, nint a, int lda, nint piv, nint b, int ldb);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int Getrf(int layout, int m, int n, nint a, int lda, nint piv);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int Getrs(int layout, byte trans, int n, int nrhs, nint a, int lda, nint piv, nint b, int ldb);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int Geqrf(int layout, int m, int n, nint a, int lda, nint tau);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int Orgqr(int layout, int m, int n, int k, nint a, int lda, nint tau);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int Potrf(int layout, byte triangle, int n, nint a, int lda);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int Gesdd(int layout, byte job, int m, int n, nint a, int lda, nint s, nint u, int ldu, nint vt, int ldvt);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int Syevd(int layout, byte vectors, byte triangle, int n, nint a, int lda, nint w);
    private sealed class Pin : IDisposable
    {
        private GCHandle handle;
        public Pin(Array array)
        {
            handle = GCHandle.Alloc(array, GCHandleType.Pinned);
        }
        public nint Pointer => handle.AddrOfPinnedObject();
        public void Dispose()
        {
            if (handle.IsAllocated)
                handle.Free();
        }
    }
    private static string Symbol(NDArray a, string suffix) => "LAPACKE_" + (a.DType == DType.Float32 ? "s" : "d") + suffix;
    private static Array Buffer(NDArray a) => a.DType == DType.Float32 ? a.ToFloat32Array() : a.ToArray();
    private static Array Empty(int n, DType dtype) => dtype == DType.Float32 ? new float[n] : new double[n];
    private static NDArray Wrap(Array a, Shape shape) => a is float[] f ? NDArray.FromFloat32(f, shape) : NDArray.FromArray((double[])a, shape);
    private static void Matrix(NDArray a, bool square = false)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Rank != 2 || (square && a.Shape[0] != a.Shape[1]))
            throw new ArgumentException(square ? "A square matrix is required." : "A rank-two matrix is required.");
        if (a.DType is not (DType.Float32 or DType.Float64))
            throw new ArgumentException("LAPACK requires float32 or float64 arrays.");
        if (a.ToArray().Any(x => !double.IsFinite(x)))
            throw new ArgumentException("LAPACK inputs must contain only finite values.");
    }
    private static void Info(int info, string operation, string positive)
    {
        if (info < 0)
            throw new InvalidOperationException($"{operation}: LAPACKE rejected argument {-info} (or could not allocate workspace), info={info}.");
        if (info > 0)
            throw new InvalidOperationException($"{operation}: {positive}, info={info}.");
    }
    private static int Right(NDArray a, NDArray b)
    {
        ArgumentNullException.ThrowIfNull(b);
        if (b.Rank is not (1 or 2) || b.Shape[0] != a.Shape[0] || b.DType != a.DType)
            throw new ArgumentException("Right-hand side must have matching rows and dtype and rank one or two.");
        if (b.ToArray().Any(x => !double.IsFinite(x)))
            throw new ArgumentException("Right-hand side must be finite.");
        return b.Rank == 1 ? 1 : b.Shape[1];
    }
    public static NDArray Solve(NDArray matrix, NDArray right)
    {
        Matrix(matrix, true);
        int n = matrix.Shape[0], nrhs = Right(matrix, right);
        if (n == 0 || nrhs == 0)
            return right.Copy();
        var a = Buffer(matrix);
        var b = Buffer(right);
        var piv = new int[n];
        using (var ap = new Pin(a))
        using (var bp = new Pin(b))
        using (var pp = new Pin(piv))
            Info(Native.Function<Gesv>(Symbol(matrix, "gesv"))(101, n, nrhs, ap.Pointer, n, pp.Pointer, bp.Pointer, nrhs), "solve", "matrix is singular");
        return Wrap(b, right.Shape);
    }
    public static LuFactorization Lu(NDArray matrix)
    {
        Matrix(matrix, true);
        int n = matrix.Shape[0];
        var a = Buffer(matrix);
        var piv = new int[n];
        if (n > 0)
        {
            using var ap = new Pin(a);
            using var pp = new Pin(piv);
            Info(Native.Function<Getrf>(Symbol(matrix, "getrf"))(101, n, n, ap.Pointer, n, pp.Pointer), "LU", "matrix is singular");
        }
        return new LuFactorization(Wrap(a, matrix.Shape), piv);
    }
    internal static NDArray SolveLu(NDArray factors, int[] pivots, NDArray right)
    {
        int nrhs = Right(factors, right), n = factors.Shape[0];
        if (n == 0 || nrhs == 0)
            return right.Copy();
        var a = Buffer(factors);
        var b = Buffer(right);
        using (var ap = new Pin(a))
        using (var bp = new Pin(b))
        using (var pp = new Pin(pivots))
            Info(Native.Function<Getrs>(Symbol(factors, "getrs"))(101, (byte)'N', n, nrhs, ap.Pointer, n, pp.Pointer, bp.Pointer, nrhs), "LU solve", "solution failed");
        return Wrap(b, right.Shape);
    }
    public static QrResult Qr(NDArray matrix)
    {
        Matrix(matrix);
        int m = matrix.Shape[0], n = matrix.Shape[1], k = Math.Min(m, n);
        if (k == 0)
            return new(NDArray.Zeros(new Shape(m, k), matrix.DType), NDArray.Zeros(new Shape(k, n), matrix.DType));
        var a = Buffer(matrix);
        var tau = Empty(k, matrix.DType);
        using (var ap = new Pin(a))
        using (var tp = new Pin(tau))
            Info(Native.Function<Geqrf>(Symbol(matrix, "geqrf"))(101, m, n, ap.Pointer, n, tp.Pointer), "QR", "factorization failed");
        var packed = Wrap(a, matrix.Shape);
        var r = NDArray.Zeros(new Shape(k, n), matrix.DType);
        var q = NDArray.Zeros(new Shape(m, k), matrix.DType);
        for (int i = 0; i < k; i++)
            for (int j = i; j < n; j++)
                r.Set(packed.Get(i, j), i, j);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
                q.Set(packed.Get(i, j), i, j);
        var qb = Buffer(q);
        using (var qp = new Pin(qb))
        using (var tp = new Pin(tau))
            Info(Native.Function<Orgqr>(Symbol(matrix, "orgqr"))(101, m, k, k, qp.Pointer, k, tp.Pointer), "QR Q", "generation failed");
        return new(Wrap(qb, q.Shape), r);
    }
    public static NDArray Cholesky(NDArray matrix)
    {
        Matrix(matrix, true);
        EnsureSymmetric(matrix);
        int n = matrix.Shape[0];
        var a = Buffer(matrix);
        if (n > 0)
        {
            using var ap = new Pin(a);
            Info(Native.Function<Potrf>(Symbol(matrix, "potrf"))(101, (byte)'L', n, ap.Pointer, n), "Cholesky", "matrix is not positive definite");
        }
        var result = Wrap(a, matrix.Shape);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                result.Set(0, i, j);
        return result;
    }
    public static SvdResult Svd(NDArray matrix)
    {
        Matrix(matrix);
        int m = matrix.Shape[0], n = matrix.Shape[1], k = Math.Min(m, n);
        var a = Buffer(matrix);
        var s = Empty(k, matrix.DType);
        var u = Empty(new Shape(m, k).Size, matrix.DType);
        var vt = Empty(new Shape(k, n).Size, matrix.DType);
        if (k > 0)
        {
            using var ap = new Pin(a);
            using var sp = new Pin(s);
            using var up = new Pin(u);
            using var vp = new Pin(vt);
            Info(Native.Function<Gesdd>(Symbol(matrix, "gesdd"))(101, (byte)'S', m, n, ap.Pointer, n, sp.Pointer, up.Pointer, k, vp.Pointer, n), "SVD", "iteration did not converge");
        }
        return new(Wrap(u, new Shape(m, k)), Wrap(s, new Shape(k)), Wrap(vt, new Shape(k, n)));
    }
    private static void EnsureSymmetric(NDArray matrix)
    {
        for (int i = 0; i < matrix.Shape[0]; i++)
            for (int j = 0; j < i; j++)
                if (matrix.Get(i, j) != matrix.Get(j, i))
                    throw new ArgumentException("A symmetric matrix is required (both triangles must agree).");
    }
    public static EigenResult SymmetricEigen(NDArray matrix)
    {
        Matrix(matrix, true);
        EnsureSymmetric(matrix);
        int n = matrix.Shape[0];
        var a = Buffer(matrix);
        var w = Empty(n, matrix.DType);
        if (n > 0)
        {
            using var ap = new Pin(a);
            using var wp = new Pin(w);
            Info(Native.Function<Syevd>(Symbol(matrix, "syevd"))(101, (byte)'V', (byte)'L', n, ap.Pointer, n, wp.Pointer), "symmetric eigen", "iteration did not converge");
        }
        return new(Wrap(w, new Shape(n)), Wrap(a, matrix.Shape));
    }
    /// <summary>Minimum-norm SVD solution, including rank-deficient and underdetermined systems.</summary>
    public static NDArray LeastSquares(NDArray matrix, NDArray right, double? relativeCutoff = null)
    {
        Matrix(matrix);
        int nrhs = Right(matrix, right);
        double threshold = relativeCutoff ?? (matrix.DType == DType.Float32 ? 1.1920928955078125e-7 : 2.220446049250313e-16) * Math.Max(matrix.Shape[0], matrix.Shape[1]);
        if (!double.IsFinite(threshold) || threshold < 0)
            throw new ArgumentOutOfRangeException(nameof(relativeCutoff));
        int m = matrix.Shape[0], n = matrix.Shape[1], k = Math.Min(m, n);
        var result = NDArray.Zeros(right.Rank == 1 ? new Shape(n) : new Shape(n, nrhs), matrix.DType);
        if (k == 0 || nrhs == 0)
            return result;
        var svd = Svd(matrix);
        double cutoff = svd.SingularValues.Get(0) * threshold;
        for (int j = 0; j < nrhs; j++)
            for (int l = 0; l < k; l++)
            {
                double sigma = svd.SingularValues.Get(l);
                if (sigma <= cutoff)
                    continue;
                double coefficient = 0;
                for (int i = 0; i < m; i++)
                    coefficient += svd.U.Get(i, l) * (right.Rank == 1 ? right.Get(i) : right.Get(i, j));
                coefficient /= sigma;
                for (int i = 0; i < n; i++)
                {
                    double value = coefficient * svd.VT.Get(l, i);
                    if (result.Rank == 1)
                        result.Set(result.Get(i) + value, i);
                    else
                        result.Set(result.Get(i, j) + value, i, j);
                }
            }
        return result;
    }
}
