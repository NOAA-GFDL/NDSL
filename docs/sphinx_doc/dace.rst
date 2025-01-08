Dace
============

DaCe is a parallel programming framework developed at Scalable Parallel Computing Laboratory (SPCL), DaCe is a high level intermediate representation (IR) that parses most of the Python/NumPy semantcs, and Fortran programming languages in the frontend to DaCe IR, and then optimizes the IR by passes/transformations, the DaCe IRs then used by the backend codegen to generate highly efficient C++ code for high-performance CPU, GPU, and FPGA hardware devices. 

DaCe IR uses the Stateful Dataflow multiGraphs (SDFG) data-centric intermediate representation: A transformable, interactive representation of code based on data movement. Since the input code and the SDFG are separate, it is possible to optimize a program without changing its source, so that it stays readable. On the other hand, the used optimizations are customizable and user-extensible, so they can be written once and reused in many applications. With data-centric parallel programming, we enable direct knowledge transfer of performance optimization, regardless of the application or the target processor.

For more detailed document about DaCe, please refer to the following link:
https://spcldace.readthedocs.io/en/latest/index.html