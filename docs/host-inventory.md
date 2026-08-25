# Host inventory — baseline capture

Captured: 2026-08-20 19:19:47 +12:00

This is the baseline hardware/toolchain snapshot used by the deployment and
initial measurements. It is not a live hardware reading; rerun
`scripts/collect-host-inventory.ps1` when the host, drivers, or background GPU
workload changes. GPU assignments are explicit and must be rechecked before
starting a backend.

## Operating system and CPU

- Windows: Microsoft Windows 11 Pro, version 10.0.26200, build 26200, 64-bit
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor
- Cores / logical processors: 16 / 32
- System RAM: 127.6 GiB visible to Windows

## GPUs

| CUDA index | Name | UUID | PCI bus ID | Total MiB | Used MiB | Free MiB | Driver |
|---:|---|---|---|---:|---:|---:|---|
| 0 | NVIDIA GeForce RTX 5090 | GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0 | 00000000:01:00.0 | 32607 | 27780 | 4408 | 610.74 |
| 1 | NVIDIA GeForce RTX 4090 | GPU-eed52936-813f-8d68-1654-bfb56cb42bc3 | 00000000:03:00.0 | 24564 | 20763 | 3376 | 610.74 |

Deployment mapping:

- RTX 5090 backend: CUDA index 0, UUID GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0, PCI 00000000:01:00.0
- RTX 4090 backend: CUDA index 1, UUID GPU-eed52936-813f-8d68-1654-bfb56cb42bc3, PCI 00000000:03:00.0

The CUDA index is only used after the UUID/name check; the UUID and PCI identity are the authoritative mapping.

## CUDA and build tools

NVIDIA driver: 610.74

CUDA toolkit/compiler:

~~~text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Tue_Jun__9_14:30:19_Pacific_Daylight_Time_2026
Cuda compilation tools, release 13.3, V13.3.73
Build cuda_13.3.r13.3/compiler.38244171_0
~~~

CMake executable: C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe

~~~text
cmake version 3.31.6-msvc6

CMake suite maintained and supported by Kitware (kitware.com/cmake).
~~~

MSVC compiler executable: C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe

~~~text
Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35228 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.
~~~

## Baseline caveat

At capture time, both GPUs had substantial non-deployment Windows/LM Studio usage. The deployment setup does not terminate those processes.
VRAM acceptance measurements must be repeated with the intended workload and the actual background usage documented.
