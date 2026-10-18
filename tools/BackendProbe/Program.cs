using FScript.Numerics;
using FScript.Numerics.Blas;
using FScript.Numerics.Extensions;
Console.WriteLine($"Registration without native loading: {NumericsExtension.CreateRegistration().Functions.Count()} functions");
try
{
    Console.WriteLine(LinearAlgebra.Dot(NDArray.FromArray([1, 2, 3]), NDArray.FromArray([4, 5, 6])));
    return args.Contains("--expect-failure") ? 1 : 0;
}
catch (InvalidOperationException error) when (args.Contains("--expect-failure"))
{
    if (!error.Message.Contains("OpenBLAS", StringComparison.Ordinal) || !error.Message.Contains(System.Runtime.InteropServices.RuntimeInformation.RuntimeIdentifier, StringComparison.Ordinal))
        throw;
    Console.WriteLine(error.Message);
    return 0;
}
