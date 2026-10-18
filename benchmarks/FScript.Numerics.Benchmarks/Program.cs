using System.Diagnostics;
using FScript.Numerics;
using FScript.Numerics.Blas;
Console.WriteLine($"Backend: {LinearAlgebra.BackendConfiguration}\nOPENBLAS_NUM_THREADS={Environment.GetEnvironmentVariable("OPENBLAS_NUM_THREADS") ?? "unset"}");
Console.WriteLine("operation,size,ms_per_call,managed_bytes_per_call");
void Measure(string name, int size, Action operation, int repetitions)
{
    for (int warmup = 0; warmup < 5; warmup++)
        operation();
    GC.Collect();
    long before = GC.GetAllocatedBytesForCurrentThread();
    var timer = Stopwatch.StartNew();
    for (int i = 0; i < repetitions; i++)
        operation();
    timer.Stop();
    Console.WriteLine(FormattableString.Invariant($"{name},{size},{timer.Elapsed.TotalMilliseconds / repetitions:F6},{(GC.GetAllocatedBytesForCurrentThread() - before) / repetitions}"));
}
foreach (int n in new[] { 16, 128, 512 })
{
    var m = NDArray.Full(new Shape(n, n), 0.5);
    var v = NDArray.Ones(new Shape(n));
    int reps = n == 512 ? 3 : 20;
    Measure("dot/native", n, () => LinearAlgebra.Dot(v, v), reps);
    Measure("gemv/native", n, () => LinearAlgebra.Matvec(m, v), reps);
    Measure("gemm/native", n, () => LinearAlgebra.Matmul(m, m), reps);
    Measure("gemm/transposed", n, () => LinearAlgebra.Matmul(m.Transpose(), m), reps);
    Measure("copy/contiguous", n, () => m.ToArray(), reps);
    Measure("copy/strided", n, () => m.Transpose().ToArray(), reps);
    Measure("sum/contiguous", n, () => m.Sum(), reps);
    Measure("sum/strided", n, () => m.Transpose().Sum(), reps);
}

foreach (int n in new[] { 256, 16384, 1048576 })
{
    var v = NDArray.Ones(new Shape(n));
    Measure("vector-dot/native", n, () => LinearAlgebra.Dot(v, v), 10);
    Measure("vector-dot/managed", n, () => v.Dot(v), 10);
}
