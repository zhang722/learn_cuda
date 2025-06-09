#import "@preview/algorithmic:1.0.0"
#import algorithmic: algorithm

#let algo_upsweep = algorithm({
    import algorithmic: *
    Procedure(
      "upsweep",
      ("x", "n"),
      {
        Comment[$t$: Number of steps]
        Assign[$t$][$frac(n, 2)$] 
        LineBreak
        Comment[$s$: step]
        For($s<-1$ + strong(" to ") +  $frac(n, 2)$, {
          For($k<-0$ + strong(" to ") +  $n - 1$, {
              Assign[$i$][$k + s - 1$]
              Assign[$j$][$k + 2*s - 1$]
              Assign[$x[j]$][$x[i] + x[j]$]
              Assign[$k$][$k + 2 * s$]
          })

          Assign[$s$][$2* s$]
          Assign[$t$][$frac(t, 2)$]
        })

        Return[*null*]
      },
    )
})


#let algo_upsweep_parallel = algorithm({
    import algorithmic: *
    Procedure(
      "upsweep_parallel",
      ("x", "n"),
      {
        Comment[$t$: Number of steps]
        Assign[$t$][$frac(n, 2)$] 
        LineBreak
        Comment[$s$: step]
        For($s<-1$ + strong(" to ") +  $frac(n, 2)$, {
          Comment[Sync here]
          If($t i d < t$, {
            Assign($k$, $t i d * 2 * s$)
            Assign[$i$][$k + s - 1$]
            Assign[$j$][$k + 2*s - 1$]
            Assign[$x[j]$][$x[i] + x[j]$]

          })
          Assign[$s$][$2* s$]
          Assign[$t$][$frac(t, 2)$]
        })

        Return[*null*]
      },
    )
})



#let algo_downsweep = algorithm({
      import algorithmic: *
      Procedure(
        "downsweep",
        ("x", "n"),
        {
          Comment[$t$: Number of steps]
          Assign[$t$][$1$] 
          Assign($x[n - 1]$, $0$)

          LineBreak
          Comment[$s$: step]
          For($s <- frac(n, 2)$ + strong(" to ") +  $1$, {
            For($k<-0$ + strong(" to ") +  $n - 1$, {
                Assign[$i$][$k + s - 1$]
                Assign[$j$][$k + 2*s - 1$]
                Assign($t e m p$, $x[j]$)
                Assign($x[j]$, $x[i] + x[j]$)
                Assign($x[i]$, $t e m p$)
                Assign[$k$][$k + 2 * s$]
            })

            Assign[$s$][$frac(s, 2)$]
            Assign[$t$][$2 * t$]
          })

          Return[*null*]
        },
      )
})



#let algo_downsweep_parallel = algorithm({
      import algorithmic: *
      Procedure(
        "downsweep",
        ("x", "n"),
        {
          Comment[$t$: Number of steps]
          Assign[$t$][$1$] 
          If($t i d$ + strong(" is ") + $0$, {
            Assign($x[n - 1]$, $0$)
          })

          LineBreak
          Comment[$s$: step]

          For($s <- frac(n, 2)$ + strong(" to ") +  $1$, {
            Comment[Sync here]
            If($t i d < t$, {
              Assign($k$, $t i d * 2 * s$)
              Assign[$i$][$k + s - 1$]
              Assign[$j$][$k + 2*s - 1$]
              Assign($t e m p$, $x[j]$)
              Assign($x[j]$, $x[i] + x[j]$)
              Assign($x[i]$, $t e m p$)
            })

            Assign[$s$][$frac(s, 2)$]
            Assign[$t$][$2 * t$]
          })

          Return[*null*]
        },
      )
})


#let algo_exclusive_scan_block_parallel =  algorithm({
      import algorithmic: *
      Procedure(
        "excluseive_scan_block",
        ("x", "n"),
        {
          Comment[$t$: Number of steps]
          Assign[$t$][$frac(n, 2)$] 
          LineBreak
          Comment[$s$: step]
          For($s<-1$ + strong(" to ") +  $frac(n, 2)$, {
            Comment[Sync here]
            If($t i d < t$, {
              Assign($k$, $t i d * 2 * s$)
              Assign[$i$][$k + s - 1$]
              Assign[$j$][$k + 2*s - 1$]
              Assign[$x[j]$][$x[i] + x[j]$]

            })
            Assign[$s$][$2* s$]
            Assign[$t$][$frac(t, 2)$]
          })
          Assign[$t$][$1$] 
          If($t i d$ + strong(" is ") + $0$, {
            Assign($x[n - 1]$, $0$)
          })

          LineBreak
          Comment[$s$: step]

          For($s <- frac(n, 2)$ + strong(" to ") +  $1$, {
            Comment[Sync here]
            If($t i d < t$, {
              Assign($k$, $t i d * 2 * s$)
              Assign[$i$][$k + s - 1$]
              Assign[$j$][$k + 2*s - 1$]
              Assign($t e m p$, $x[j]$)
              Assign($x[j]$, $x[i] + x[j]$)
              Assign($x[i]$, $t e m p$)
            })

            Assign[$s$][$frac(s, 2)$]
            Assign[$t$][$2 * t$]
          })

          Return[*null*]
        },
      )
    })


#let algo_scan_warp = algorithm({
      import algorithmic: *
      Procedure(
        "scan_warp",
        ("x"),
        {
          Comment[$w$: Number of threads in a warp]
          Assign[$w$][$32$] 

          LineBreak
          Comment[$s$: step]
          For($s <- 1$ + strong(" to ") +  $frac(w, 2)$, {
            For($k<-s$ + strong(" to ") +  $w - 1$, {
                Assign($x[k]$, $x[k] + x[k - s]$)

                Assign[$k$][$k + 1$]
            })

            Assign[$s$][$2 * s$]
          })

          Return[*null*]
        },
      )
    })



#let algo_scan_warp_parallel = algorithm({
      import algorithmic: *
      Procedure(
        "scan_warp_parallel",
        ("x"),
        {
          Comment[$w$: Number of threads in a warp]
          Assign[$w$][$32$] 
          Assign($w i d$, $frac(t i d, w)$)

          LineBreak
          Comment[$s$: step]
          For($s <- 1$ + strong(" to ") +  $frac(w, 2)$, {
            Comment[Sync warp here]
            Assign($x[w i d]$, $x[w i d] + x[w i d - s]$)
            Assign[$s$][$2 * s$]
          })

          Return[*null*]
        },
      )
    })


#let algoBox = (body) => {
    box(
        inset: 6pt,
        stroke: 1pt,
        radius: 2pt,
        fill: white,
        [
            #body
        ]
    )
}

#let upsweep_box = algoBox(algo_upsweep)
#let upsweep_parallel_box = algoBox(algo_upsweep_parallel)
#let downsweep_box = algoBox(algo_downsweep)
#let downsweep_parallel_box = algoBox(algo_downsweep_parallel)
#let scan_warp_box = algoBox(algo_scan_warp)
#let scan_warp_parallel_box = algoBox(algo_scan_warp_parallel)
#let exclusive_scan_block_parallel_box = algoBox(algo_exclusive_scan_block_parallel)