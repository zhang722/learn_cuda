#import "template.typ": *
#import "@preview/algorithmic:1.0.0"
#import algorithmic: algorithm

#let title = "CUDA 并行scan实现"
#let authors = ("zs",)
#show: rest => project(title: title, authors: authors)[#rest]
@upsweep 是upsweep的图示，假设一共有 $n=8$ 个元素，则一共会进行 $log_2(n) = 3$ 次处理，每次处理以步长 $2s$ 遍历数组。

#figure(
  image("./upsweep.jpg", width: 100%),
  caption: [upsweep 图示],
  numbering: "1",
)<upsweep>

#let upsweep_box = box(
  inset: 6pt,
  stroke: 1pt,
  radius: 2pt,
  fill: white,
  [#algorithm({
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
  ]
)


#let upsweep_parallel_box = box(
  inset: 6pt,
  stroke: 1pt,
  radius: 2pt,
  fill: white,
  [#algorithm({
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
  ]
)

#columns(
  [
    #upsweep_box
    #upsweep_parallel_box
  ]
)

@downsweep 是downsweep的图示，处理的步长和upsweep一致，但是处理顺序相反，$s$ 应该从大到小。
#figure(
  image("./downsweep.jpg", width: 100%),
  caption: [upsweep 图示],
  numbering: "1",
)<downsweep>


#let downsweep_box = box(
  inset: 6pt,
  stroke: 1pt,
  radius: 2pt,
  fill: white,
  [
    #algorithm({
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
  ]
)

#let downsweep_parallel_box = box(
  inset: 6pt,
  stroke: 1pt,
  radius: 2pt,
  fill: white,
  [
    #algorithm({
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
  ]
)

#columns(
  [
    #downsweep_box
    #downsweep_parallel_box
  ]
)

#let exclusive_scan_block_parallel_box = box(
  inset: 6pt,
  stroke: 1pt,
  radius: 2pt,
  fill: white,
  [
    #algorithm({
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
  ]
)

#exclusive_scan_block_parallel_box