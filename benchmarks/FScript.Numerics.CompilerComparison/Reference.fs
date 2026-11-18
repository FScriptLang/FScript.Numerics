namespace FScript.Numerics.CompilerComparison

open FScript.Numerics

module Reference =
    let ndarrayBulk (size:int) (seed:float) =
        let a = NDArray.Arange(0.0, float size, 1.0)
        let b = a.Add(seed)
        let c = b.Multiply(3.0)
        c.Dot(c)

    let arrayBulk (size:int) (seed:float) =
        let a = [|0.0..float(size-1)|]
        let b = a |> Array.map (fun value -> value + seed)
        let c = b |> Array.map (fun value -> value * 3.0)
        let mutable total = 0.0
        for i in 0..size-1 do total <- total + c.[i] * c.[i]
        total

    let ndarrayIndexed (seed:float) =
        let a = NDArray.Arange(0.0, 1024.0, 1.0)
        let rec loop i total =
            if i=1024 then total else loop (i+1) (total+a.Get([|i|]))
        loop 0 seed

    let arrayIndexed (seed:float) =
        let a = [|0.0..1023.0|]
        let rec loop i total =
            if i=1024 then total else loop (i+1) (total+a.[i])
        loop 0 seed
