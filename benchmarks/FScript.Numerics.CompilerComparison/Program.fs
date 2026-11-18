open System
open System.IO
open System.Reflection
open System.Runtime.CompilerServices
open System.Text.Json
open FScript.Language
open FScript.Compiler
open FScript.Runtime
open FScript.Numerics.Extensions
open FScript.DotNet.Comparison
open FScript.Numerics.CompilerComparison

let unwrap = function Ok value -> value | Error errors -> failwith (sprintf "%A" errors)
let median values = let sorted=List.sort values in sorted.[sorted.Length/2]

[<EntryPoint>]
let main args =
    try
        let positional=args |> Array.filter (fun arg -> arg<>"--verify-only")
        let iterations=if positional.Length>0 then Int32.Parse positional.[0] else 100
        let rounds=if positional.Length>1 then Int32.Parse positional.[1] else 7
        let output=Path.GetFullPath(if positional.Length>2 then positional.[2] else ".out/compiler-comparison")
        if iterations<=0 || rounds<=0 || positional.Length>3 then invalidArg "args" "Usage: [positive iterations] [positive rounds] [output] [--verify-only]"
        let source=File.ReadAllText(Path.Combine(AppContext.BaseDirectory,"workloads.fss"))
        let provider=typeof<NumericsProvider>.GetMethod("Provide") |> nonNull
        let registration=NumericsProvider.Provide() |> ExtensionBridge.toLanguageRegistration
        let purity=ClrBindings.providerPurity [provider]
        let watch=Diagnostics.Stopwatch.StartNew()
        let compiled=Compiler.compileWithRegistrationAndPurity registration purity {Program=Parser.parseProgram source;Externs=[]} |> unwrap
        let sdk=compiled.Lir |> ClrBindings.bindProvider (registration.Functions |> List.map _.Name) provider
        let inlineProgram=sdk |> ClrBindings.bindCompiledProviders [provider]
        let typed=NumericsProvider.FScriptCompiledFunctions() |> Seq.fold (fun (program:Lir.Program) binding ->
            if program.Declarations |> List.exists (fun declaration -> declaration.Name=binding.Key) then ClrBindings.bindMethod binding.Key binding.Value program else program) inlineProgram
        Directory.CreateDirectory output |> ignore
        File.WriteAllText(Path.Combine(output,"workloads.fss"),source)
        // Exercise a program requiring SDK host values but no SDK-dispatch functions.
        let onlyTyped=Compiler.compileWithRegistrationAndPurity registration purity {Program=Parser.parseProgram "NDArray.sum (NDArray.ofList [1.0;2.0])";Externs=[]} |> unwrap
        let onlyTyped=onlyTyped.Lir |> ClrBindings.bindProvider (registration.Functions |> List.map _.Name) provider |> ClrBindings.bindCompiledProviders [provider]
        let onlyOptions:DotNetOptions={AssemblyName="Numerics.TypedOnly";Namespace="Numerics.TypedOnly";ClassName="Program";OutputKind=Library;References=[provider.Module.Assembly.Location]}
        let onlyFiles=DotNetBackend.generate onlyOptions onlyTyped |> unwrap
        let onlyAssembly=Assembly.Load((onlyFiles |> List.find (fun file -> file.RelativePath="Numerics.TypedOnly.dll")).Content)
        let onlyType=onlyAssembly.GetType("Numerics.TypedOnly.Program") |> nonNull
        if (onlyType.GetMethod("Run") |> nonNull).Invoke(null,[||]) |> nonNull :?> double <> 3.0 then invalidOp "Typed-only host-value representation failed."
        let versions=["fscript-sdk",sdk;"fscript-typed",typed;"fscript-inline",inlineProgram] |> List.map (fun (name,program) ->
            let options:DotNetOptions={AssemblyName="Numerics.CompiledComparison";Namespace="Numerics.Compiled";ClassName="Workloads";OutputKind=Library;References=[provider.Module.Assembly.Location]}
            let files=DotNetBackend.generate options program |> unwrap
            let directory=Path.Combine(output,name)
            Directory.CreateDirectory directory |> ignore
            for file in files do File.WriteAllBytes(Path.Combine(directory,file.RelativePath),file.Content)
            let binary=files |> List.find (fun file -> file.RelativePath=options.AssemblyName+".dll") |> _.Content
            name,binary)
        watch.Stop()
        let assemblies=versions |> List.map (fun (name,binary) ->
            let assembly=Assembly.Load binary
            if assembly.GetReferencedAssemblies() |> Array.exists (fun reference -> reference.Name="FScript.Runtime" || reference.Name="FScript.Language") then invalidOp "Generated code hosts an interpreter."
            name,assembly)
        let exported (assembly:Assembly) name=
            let typ=assembly.GetType("Numerics.Compiled.Workloads") |> nonNull
            (typ.GetProperty("v_"+name) |> nonNull).GetValue(null) |> nonNull :?> Func<double,double>
        let seeds=[-97L;-1L;0L;1L;7L;42L;101L;999L] @ [for i in 1L..24L -> i*3L]
        let cases=["bulk32",Reference.ndarrayBulk 32,Reference.arrayBulk 32; "bulk4096",Reference.ndarrayBulk 4096,Reference.arrayBulk 4096
                   "bulk65536",Reference.ndarrayBulk 65536,Reference.arrayBulk 65536; "indexed1024",Reference.ndarrayIndexed,Reference.arrayIndexed]
        let workloads:Comparison.Workload list=cases |> List.map (fun (name,ndarray,array) ->
            let functions=(assemblies |> List.map (fun (variant,assembly) -> variant,exported assembly name)) @ ["fsharp-ndarray",Func<double,double>(ndarray); "fsharp-array",Func<double,double>(array)]
            for seed in seeds do
                let expected=array (float seed)
                for variant,invoke in functions do
                    let actual=invoke.Invoke(float seed)
                    if actual<>expected then invalidOp $"{name}/{variant}, seed {seed}: expected {expected}, got {actual}"
            let variants:Comparison.Variant list=functions |> List.map (fun (name,fn) ->
                {Name=name;Root=fn.Method;Function=Func<int64,int64>(fun seed -> fn.Invoke(float seed) |> int64)})
            {Name=name;Variants=variants})
        for name,assembly in assemblies do
            let call=exported assembly
            for seed in seeds do
                if (call "mutation").Invoke(float seed)<>float seed then invalidOp "Mutation was dropped."
                if (call "aliasing").Invoke(float seed)<>float seed+1.0 then invalidOp "View/copy semantics changed."
                if (call "mixed").Invoke(float seed)<>float seed+2.0 then invalidOp "Mixed typed/SDK calls or visible fields changed."
            let mutable failed=false
            try (call "invalid_index").Invoke(0.0) |> ignore
            with :? FScript.Extensibility.ExtensionException as error ->
                failed<-error.Message.StartsWith("NDArray.get:",StringComparison.Ordinal)
            if not failed then invalidOp "An unused failing access was pruned or lost its qualified diagnostic."
            for method in IlInspection.methods assembly do
                if not method.ContainsGenericParameters && not (method.DeclaringType |> nonNull).ContainsGenericParameters then RuntimeHelpers.PrepareMethod method.MethodHandle
            IlInspection.write (Path.Combine(output,name,"fscript.il.txt")) assembly
            let instructions=IlInspection.methods assembly |> Array.collect IlInspection.decode
            if name="fscript-inline" && instructions |> Array.exists (fun instruction -> instruction.Target |> Option.exists (fun target -> target.DeclaringType=typeof<InlineArrayBindings>)) then
                invalidOp "Opted-in array adapters survived IL expansion."
        IlInspection.write (Path.Combine(output,"fsharp.il.txt")) (Assembly.GetExecutingAssembly())
        let assembly=assemblies |> List.find (fst >> (=) "fscript-inline") |> snd
        let binary=versions |> List.find (fst >> (=) "fscript-inline") |> snd
        let prepared:Comparison.Prepared={Assembly=assembly;Binary=binary;CompilationMilliseconds=watch.Elapsed.TotalMilliseconds;Workloads=workloads;Source=source}
        let samples=if Array.contains "--verify-only" args then [] else Comparison.run iterations rounds prepared
        for (_,round),items in samples |> List.groupBy (fun sample -> sample.Workload,sample.Round) do
            if items |> List.map _.Checksum |> List.distinct |> List.length<>1 then invalidOp $"Measured batch checksums disagree in round {round}."
        let summaries=samples |> List.groupBy (fun sample -> sample.Workload,sample.Variant) |> List.map (fun ((name,variant),items) ->
            {|workload=name;variant=variant;milliseconds=items |> List.map _.Milliseconds |> median;allocatedBytes=items |> List.map _.AllocatedBytes |> median
              minimum=items |> List.map _.Milliseconds |> List.min;maximum=items |> List.map _.Milliseconds |> List.max|})
        let report={|runtime=Environment.Version.ToString();os=Runtime.InteropServices.RuntimeInformation.OSDescription;architecture=Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString()
                     tieredCompilation=Environment.GetEnvironmentVariable "DOTNET_TieredCompilation";readyToRun=Environment.GetEnvironmentVariable "DOTNET_ReadyToRun"
                     sourceSha256=Convert.ToHexString(Security.Cryptography.SHA256.HashData(Text.Encoding.UTF8.GetBytes source))
                     compilationMilliseconds=watch.Elapsed.TotalMilliseconds;validationSeeds=seeds.Length;iterations=iterations;rounds=rounds
                     pureFunctions=purity.PureFunctions |> Set.toArray;pureTypes=purity.PureTypes |> Set.toArray;measurements=samples;summaries=summaries|}
        File.WriteAllText(Path.Combine(output,"results.json"),JsonSerializer.Serialize(report,JsonSerializerOptions(WriteIndented=true)))
        printfn "Verified 4 workloads × %d seeds, mutation, aliasing, retained failures, and closed-method JIT validity." seeds.Length
        for row in summaries do printfn "%-12s %-15s %9.5f ms %12.0f B/call" row.workload row.variant row.milliseconds row.allocatedBytes
        printfn "Reports: %s" output
        0
    with error -> eprintfn "%O" error; 1
