# Jinja
A minimalistic Swift implementation of the Jinja templating engine, specifically designed for parsing and rendering ML chat templates.

## SwiftPM

To use `Jinja` with SwiftPM, you can add this to your `Package.swift`:

```
dependencies: [
    .package(url: "https://github.com/maiqingqiang/Jinja", branch: "main")
]
```

And then, add the Transformers library as a dependency to your target:

```
targets: [
    .target(
        name: "YourTargetName",
        dependencies: [
            .product(name: "Jinja", package: "Jinja")
        ]
    )
]
```

## Usage

```swift
import Jinja

let template = """
{% for item in items %}
{{ item }}
{% endfor %}
"""

let context = [
    "items": [
        "item1", 
        "item2", 
        "item3"
    ]
]

let result = try Template(template).render(context)
```