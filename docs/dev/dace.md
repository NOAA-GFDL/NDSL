DaCe
============

DaCe is a parallel programming framework developed at SPCL. DaCe is a compiler framework that parses a subset of the Python/NumPy semantics. The intermediate representation that DaCe uses, the SDFG, can be optimizedby passes/transformations.

SDFGs are a transformable, interactive representation of code based on data movement. Since the input code and the SDFG are separate, it is possible to optimize a program without changing its source, so that it stays readable. On the other hand, the used optimizations are customizable and user-extensible, so they can be written once and reused in many applications. With data-centric parallel programming, we enable direct knowledge transfer of performance optimization, regardless of the application or the target processor.

For more detailed document about DaCe, please refer to the following link:
https://spcldace.readthedocs.io/en/latest/index.htm