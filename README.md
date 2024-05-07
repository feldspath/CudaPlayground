
## Getting Started

Requirements:

* CUDA 12.4
* Cmake 3.26

To create build files, use cmake like this:

```
mkdir build
cd build
cmake ../
```

Note: Builds for VR projects are currently very windows/VisualStudio specific. 

## Credits & Licenses

CudaPlayground is licensed under MIT (see LICENSE.md), unless specified otherwise. Always check subfolders or top of source code files for licenses specific to subfolders or files. 

* ./libs contains third-party libraries, each subject to their own licences.
* ./modules/seascape is an adaption of the [Shadertoy Seascape demo by Alexander Alekseev](https://www.shadertoy.com/view/Ms2SD1)  aka TDM - 2014, under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License. 
* ./modules/sort/GPUSorting contains source code that was adapted from [GPUSorting by Thomas Smith](https://github.com/b0nes164/GPUSorting), which is published under the licenses specified at the top of the specific files. 

