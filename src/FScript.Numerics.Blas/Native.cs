using System.Reflection;
using System.Runtime.InteropServices;

namespace FScript.Numerics.Blas;

/// <summary>Process-lifetime, LP64 C ABI loader. Loading is deferred until numerical execution.</summary>
internal static class Native
{
    private static readonly Lazy<nint> Library = new(Load, LazyThreadSafetyMode.ExecutionAndPublication);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate nint Config();
    private static nint Load()
    {
        bool supported = (OperatingSystem.IsMacOS() && RuntimeInformation.ProcessArchitecture == Architecture.Arm64) || ((OperatingSystem.IsLinux() || OperatingSystem.IsWindows()) && RuntimeInformation.ProcessArchitecture == Architecture.X64);
        if (!supported)
            throw new InvalidOperationException($"OpenBLAS native operations are unsupported on {RuntimeInformation.RuntimeIdentifier}. BlasSharp.OpenBlas 0.3.0 has no macOS x64 asset and its linux-arm64/win-x86 assets are mislabeled x64 binaries. Use osx-arm64, linux-x64 (glibc), or win-x64, or use managed NDArray operations.");
        string file = OperatingSystem.IsWindows() ? "libopenblas.dll" : OperatingSystem.IsMacOS() ? "libopenblas.dylib" : "libopenblas.so";
        var assembly = typeof(Native).Assembly;
        string directory = Path.GetDirectoryName(assembly.Location)!;
        nint handle = 0;
        try
        {
            var attempts = new List<string>();
            foreach (string path in new[] { Path.Combine(directory, file), Path.Combine(directory, "runtimes", RuntimeInformation.RuntimeIdentifier, "native", file) })
                if (File.Exists(path))
                {
                    try
                    {
                        handle = NativeLibrary.Load(path);
                        break;
                    }
                    catch (Exception error) when (error is DllNotFoundException or BadImageFormatException) { attempts.Add($"{path}: {error.Message}"); }
                }
            if (handle == 0)
            {
                try
                {
                    handle = NativeLibrary.Load(file, assembly, DllImportSearchPath.SafeDirectories);
                }
                catch (DllNotFoundException error) { throw new DllNotFoundException(string.Join(Environment.NewLine, attempts.Append(error.Message)), error); }
            }
            string config = Marshal.PtrToStringAnsi(Marshal.GetDelegateForFunctionPointer<Config>(NativeLibrary.GetExport(handle, "openblas_get_config"))()) ?? "";
            if (!config.StartsWith("OpenBLAS", StringComparison.Ordinal) || config.Contains("USE64BITINT", StringComparison.Ordinal))
                throw new InvalidOperationException("OpenBLAS must expose the LP64 (32-bit BLAS integer) ABI.");
            foreach (string symbol in new[] { "cblas_ddot", "cblas_sdot", "cblas_dnrm2", "cblas_snrm2", "cblas_dgemv", "cblas_sgemv", "cblas_dgemm", "cblas_sgemm", "LAPACKE_dgesv", "LAPACKE_sgesv", "LAPACKE_dgetrf", "LAPACKE_sgetrf", "LAPACKE_dgetrs", "LAPACKE_sgetrs", "LAPACKE_dgeqrf", "LAPACKE_sgeqrf", "LAPACKE_dorgqr", "LAPACKE_sorgqr", "LAPACKE_dpotrf", "LAPACKE_spotrf", "LAPACKE_dgesdd", "LAPACKE_sgesdd", "LAPACKE_dsyevd", "LAPACKE_ssyevd" })
                NativeLibrary.GetExport(handle, symbol);
            return handle;
        }
        catch (Exception error) when (error is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException or InvalidOperationException)
        {
            if (handle != 0)
                NativeLibrary.Free(handle);
            throw new InvalidOperationException($"Cannot initialize LP64 OpenBLAS for {RuntimeInformation.RuntimeIdentifier}. Restore the BlasSharp.OpenBlas 0.3.0 native assets and ensure the platform is supported. Linux also requires libgfortran.so.5 (Debian/Ubuntu: apt-get install libgfortran5). Windows needs x64 GNU libgfortran-5.dll and libgcc_s_seh-1.dll on PATH (MSYS2 MinGW64). {error.Message}", error);
        }
    }
    internal static T Function<T>(string name) where T : Delegate => Marshal.GetDelegateForFunctionPointer<T>(NativeLibrary.GetExport(Library.Value, name));
    public static string Configuration => Marshal.PtrToStringAnsi(Function<Config>("openblas_get_config")())!;
}
