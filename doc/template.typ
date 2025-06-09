#let project(title: "", authors: (), body) = {
  // Set the document's basic properties.
  set document(author: authors, title: title)
  set page(numbering: "1", number-align: center)
  set text(
    12pt,
  font:("Times New Roman", "SimSun"),
  
  discretionary-ligatures:true,
  historical-ligatures:true,
  number-type: "lining",
  )
  show math.equation: set text(weight: 400,font:"New Computer Modern Math")
  show raw: set text(weight: 400,font:("Fira Code"))
  show raw.where(block: false): box.with(
    fill: luma(240),
    inset: (x: 3pt, y: 0pt),
    outset: (y: 3pt),
    radius: 2pt,
  )
  show raw.where(block: true): block.with(
    fill: luma(240),
    inset: 10pt,
    radius: 4pt,
  )
  show heading: it => [
  #set text(weight: 600,0.9em)
  #block(smallcaps(it.body))
  ]
  // Title row.
  align(center)[
    #block(text(weight: 600, 1.5em, title))
  ]

  // Author information.
  pad(
    top: 0.5em,
    bottom: 0.5em,
    x: 2em,
    grid(
      columns: (1fr,) * calc.min(3, authors.len()),
      gutter: 1em,
      ..authors.map(author => align(center, [#strong(author), #datetime.today().display()])),
    ),
  )

  // Main body.
  set par(justify: true, first-line-indent: (amount: 2em, all: true))

  body
}


// #let project(title:"", body) = {
//   set text(font: "SimSun")
//   body
// }