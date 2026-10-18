using System.Runtime.InteropServices;
namespace FScript.Numerics.Blas;

/// <summary>Non-mutating float32/float64 BLAS operations. Strided inputs are copied into exclusive row-major buffers.</summary>
public static unsafe class LinearAlgebra
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate double DDot(int n, double* a, int ia, double* b, int ib);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate float SDot(int n, float* a, int ia, float* b, int ib);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate double DNorm(int n, double* a, int inc);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate float SNorm(int n, float* a, int inc);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate void DGemm(int order, int ta, int tb, int m, int n, int k, double alpha, double* a, int lda, double* b, int ldb, double beta, double* c, int ldc);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate void SGemm(int order, int ta, int tb, int m, int n, int k, float alpha, float* a, int lda, float* b, int ldb, float beta, float* c, int ldc);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate void DGemv(int order, int trans, int m, int n, double alpha, double* a, int lda, double* x, int incx, double beta, double* y, int incy);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate void SGemv(int order, int trans, int m, int n, float alpha, float* a, int lda, float* x, int incx, float beta, float* y, int incy);
    public static string BackendConfiguration => Native.Configuration;
    internal static DType Validate(params NDArray[] arrays)
    {
        foreach (var a in arrays)
        {
            ArgumentNullException.ThrowIfNull(a);
            if (a.DType is not (DType.Float32 or DType.Float64))
                throw new ArgumentException("Native linear algebra requires float32 or float64 arrays; convert explicitly.");
        }
        return arrays.All(a => a.DType == DType.Float32) ? DType.Float32 : DType.Float64;
    }
    // CBLAS never writes these inputs. The caller must synchronize explicit concurrent Set operations.
    private static ReadOnlySpan<double> Doubles(NDArray array) => array.TryGetContiguousSpan(out var span) ? span : array.ToArray();
    private static ReadOnlySpan<float> Singles(NDArray array) => array.TryGetContiguousFloat32Span(out var span) ? span : array.ToFloat32Array();
    private static NDArray MatrixLayout(NDArray array, out int transpose, out int leadingDimension)
    {
        if (!array.IsContiguous && array.IsFortranContiguous)
        {
            transpose = 112;
            leadingDimension = array.Shape[0];
            return array.Transpose();
        }
        transpose = 111;
        leadingDimension = array.Shape[1];
        return array;
    }
    public static double Dot(NDArray left, NDArray right)
    {
        var dtype = Validate(left, right);
        if (left.Rank != 1 || right.Rank != 1 || left.Size != right.Size)
            throw new ArgumentException("Dot requires equal-length vectors.");
        if (left.Size == 0)
            return 0;
        if (dtype == DType.Float32)
        {
            var a = Singles(left);
            var b = Singles(right);
            fixed (float* ap = a, bp = b)
                return Native.Function<SDot>("cblas_sdot")(a.Length, ap, 1, bp, 1);
        }
        else
        {
            var a = Doubles(left);
            var b = Doubles(right);
            fixed (double* ap = a, bp = b)
                return Native.Function<DDot>("cblas_ddot")(a.Length, ap, 1, bp, 1);
        }
    }
    public static double Norm(NDArray vector)
    {
        var dtype = Validate(vector);
        if (vector.Rank != 1)
            throw new ArgumentException("Norm requires a vector.");
        if (vector.Size == 0)
            return 0;
        if (dtype == DType.Float32)
        {
            var a = Singles(vector);
            fixed (float* p = a)
                return Native.Function<SNorm>("cblas_snrm2")(a.Length, p, 1);
        }
        else
        {
            var a = Doubles(vector);
            fixed (double* p = a)
                return Native.Function<DNorm>("cblas_dnrm2")(a.Length, p, 1);
        }
    }
    public static NDArray Matvec(NDArray matrix, NDArray vector)
    {
        var dtype = Validate(matrix, vector);
        if (matrix.Rank != 2 || vector.Rank != 1 || matrix.Shape[1] != vector.Size)
            throw new ArgumentException("Matvec requires a matrix and a vector matching its columns.");
        int m = matrix.Shape[0], n = matrix.Shape[1];
        if (m == 0 || n == 0)
            return NDArray.Zeros(new Shape(m), dtype);
        var physical = MatrixLayout(matrix, out int trans, out int lda);
        int rows = trans == 111 ? m : n, cols = trans == 111 ? n : m;
        if (dtype == DType.Float32)
        {
            var a = Singles(physical);
            var x = Singles(vector);
            var y = new float[m];
            fixed (float* ap = a, xp = x, yp = y)
                Native.Function<SGemv>("cblas_sgemv")(101, trans, rows, cols, 1, ap, lda, xp, 1, 0, yp, 1);
            return NDArray.FromFloat32(y);
        }
        else
        {
            var a = Doubles(physical);
            var x = Doubles(vector);
            var y = new double[m];
            fixed (double* ap = a, xp = x, yp = y)
                Native.Function<DGemv>("cblas_dgemv")(101, trans, rows, cols, 1, ap, lda, xp, 1, 0, yp, 1);
            return NDArray.FromArray(y);
        }
    }
    public static NDArray Matmul(NDArray left, NDArray right)
    {
        var dtype = Validate(left, right);
        if (left.Rank != 2 || right.Rank != 2 || left.Shape[1] != right.Shape[0])
            throw new ArgumentException("Matmul requires rank-two arrays with matching inner dimensions.");
        int m = left.Shape[0], k = left.Shape[1], n = right.Shape[1];
        var shape = new Shape(m, n);
        if (m == 0 || n == 0 || k == 0)
            return NDArray.Zeros(shape, dtype);
        var physicalLeft = MatrixLayout(left, out int ta, out int lda);
        var physicalRight = MatrixLayout(right, out int tb, out int ldb);
        if (dtype == DType.Float32)
        {
            var a = Singles(physicalLeft);
            var b = Singles(physicalRight);
            var c = new float[shape.Size];
            fixed (float* ap = a, bp = b, cp = c)
                Native.Function<SGemm>("cblas_sgemm")(101, ta, tb, m, n, k, 1, ap, lda, bp, ldb, 0, cp, n);
            return NDArray.FromFloat32(c, shape);
        }
        else
        {
            var a = Doubles(physicalLeft);
            var b = Doubles(physicalRight);
            var c = new double[shape.Size];
            fixed (double* ap = a, bp = b, cp = c)
                Native.Function<DGemm>("cblas_dgemm")(101, ta, tb, m, n, k, 1, ap, lda, bp, ldb, 0, cp, n);
            return NDArray.FromArray(c, shape);
        }
    }
}
