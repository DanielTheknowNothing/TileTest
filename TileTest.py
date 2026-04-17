# -*- coding: utf-8 -*-
"""
AMD-optimized PyOpenCL tiled matrix multiplication benchmark
Refactored into a reusable class-based wrapper.
"""

import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import time

class CLMatMul:
    """
    A class to wrap the OpenCL tiled matrix multiplication kernel.
    
    This class handles:
    - OpenCL device and context setup.
    - Caching of compiled kernels.
    - Running the benchmark for different tile sizes.
    """
    
    # The kernel is a class-level constant template
    KERNEL_TEMPLATE = """
    #define TX {TX}
    #define TY {TY}

    __kernel void matmul_tiled(
        __global const float* A,
        __global const float* B,
        __global float* C,
        const int N)
    {{
        __local float Asub[TY][TX];
        __local float Bsub[TY][TX];

        int row = get_global_id(1);
        int col = get_global_id(0);
        int lx  = get_local_id(0);
        int ly  = get_local_id(1);

        float sum = 0.0f;
        
        for (int k0 = 0; k0 < N; k0 += TX) {{
            // Load tiles into local memory
            if (row < N && k0 + lx < N)
                Asub[ly][lx] = A[row * N + (k0 + lx)];
            else
                Asub[ly][lx] = 0.0f;

            if (k0 + ly < N && col < N)
                Bsub[ly][lx] = B[(k0 + ly) * N + col];
            else
                Bsub[ly][lx] = 0.0f;

            barrier(CLK_LOCAL_MEM_FENCE);

            // Perform multiplication from local memory
            for (int k = 0; k < TX; ++k)
                sum += Asub[ly][k] * Bsub[k][lx];

            barrier(CLK_LOCAL_MEM_FENCE);
        }}

        // Write result
        if (row < N && col < N)
            C[row * N + col] = sum;
    }}
    """

    def __init__(self):
        """
        Initializes the OpenCL context, device, and queue.
        """
        self.device = self._get_best_device()
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx, 
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        self._compiled_kernels = {} # Cache for compiled kernels

        print(f"Using platform: {self.device.platform.name}")
        print(f"Using device:   {self.device.name}")
        print(f"Max compute units: {self.device.max_compute_units}")
        print(f"Max work-group size: {self.device.max_work_group_size}")
        print("-" * 60)

    def _get_best_device(self):
        """
        Tries to find an AMD GPU, otherwise falls back to any GPU,
        and finally to any available device (e.g., CPU).
        """
        for p in cl.get_platforms():
            if "AMD" in p.name or "Advanced Micro" in p.name:
                try:
                    return p.get_devices(device_type=cl.device_type.GPU)[0]
                except cl.LogicError:
                    continue # No AMD GPU found in this platform
        
        # Fallback 1: Any GPU
        try:
            return cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)[0]
        except (cl.LogicError, IndexError):
            pass # No GPU found

        # Fallback 2: Any device
        try:
            return cl.get_platforms()[0].get_devices()[0]
        except IndexError:
            raise RuntimeError("No OpenCL devices found on this system.")

    def _get_compiled_kernel(self, tx, ty):
        """
        Builds the kernel for a specific tile size, or retrieves
        it from the cache if already built.
        """
        key = (tx, ty)
        if key not in self._compiled_kernels:
            # Build and cache the kernel
            try:
                formatted_kernel = self.KERNEL_TEMPLATE.format(TX=tx, TY=ty)
                program = cl.Program(self.ctx, formatted_kernel).build()
                self._compiled_kernels[key] = cl.Kernel(program, "matmul_tiled")
            except cl.LogicError as e:
                print(f"Tile {tx}x{ty} -> Build failed. {e}")
                self._compiled_kernels[key] = None # Cache failure
        
        return self._compiled_kernels[key]

    def _run_tile_benchmark(self, tx, ty, a_gpu, b_gpu, c_gpu, N):
        """
        Internal function to benchmark a single tile configuration.
        """
        wg_size = tx * ty
        if wg_size > self.device.max_work_group_size:
            print(f"Tile {tx:2d}x{ty:2d} → skipped (Work-group size {wg_size} > max {self.device.max_work_group_size})")
            return None

        kernel = self._get_compiled_kernel(tx, ty)
        if kernel is None:
            return None # Build failed earlier

        try:
            # Round global work size up to the nearest multiple of the local size
            gws = (int(np.ceil(N / tx)) * tx, int(np.ceil(N / ty)) * ty)
            lws = (tx, ty)

            # Warm-up run
            event = kernel(self.queue, gws, lws, a_gpu.data, b_gpu.data, c_gpu.data, np.int32(N))
            event.wait()

            # Timed run
            t0 = time.time()
            event = kernel(self.queue, gws, lws, a_gpu.data, b_gpu.data, c_gpu.data, np.int32(N))
            event.wait()
            t1 = time.time()

            host_time = t1 - t0
            device_time = (event.profile.end - event.profile.start) * 1e-9

            return host_time, device_time

        except cl.LogicError as e:
            print(f"Tile {tx}x{ty} → skipped ({e})")
            return None

    def benchmark(self, N, tile_shapes):
        """
        Runs the full benchmark for a given matrix size N and
        a list of tile shapes.
        
        :param N: Matrix size (N x N)
        :param tile_shapes: List of (tx, ty) tuples to test.
        :return: List of (tx, ty, host_time, device_time) results.
        """
        print(f"Setting up {N}x{N} matrices...")
        np.random.seed(0)
        # Host arrays
        a = np.random.rand(N, N).astype(np.float32)
        b = np.random.rand(N, N).astype(np.float32)

        # Device arrays (upload once)
        a_gpu = cl_array.to_device(self.queue, a)
        b_gpu = cl_array.to_device(self.queue, b)
        c_gpu = cl_array.empty_like(a_gpu)
        print("Data uploaded to GPU.")
        print("-" * 60)

        results = []
        for tx, ty in tile_shapes:
            res = self._run_tile_benchmark(tx, ty, a_gpu, b_gpu, c_gpu, N)
            if res:
                host, device_t = res
                print(f"Tile {tx:2d}x{ty:2d} → Host {host:.4f}s | Device {device_t:.4f}s")
                results.append((tx, ty, host, device_t))
        
        # Return host arrays for verification if needed
        return results, a, b, c_gpu

# ------------------------------------------------------------
# Main execution: Now much cleaner!
# ------------------------------------------------------------
if __name__ == "__main__":
    
    N_SIZE = 4096 # Reduced from 10000 for faster interactive testing
    TILES_TO_TEST = [(8, 32), (16, 16), (32, 8), (32, 16), (16, 32), (64, 4), (4, 64)]

    # 1. Initialize the wrapper
    matmul_benchmark = CLMatMul()

    # 2. Run the benchmark
    gpu_results, a_host, b_host, c_gpu = matmul_benchmark.benchmark(
        N=N_SIZE, 
        tile_shapes=TILES_TO_TEST
    )
    
    # 3. Compare with CPU NumPy
    print("-" * 60)
    print("🧠 Running CPU (NumPy) benchmark...")
    t0 = time.time()
    c_cpu = np.dot(a_host, b_host)
    t1 = time.time()
    print(f"🧠 CPU (NumPy) Time: {t1 - t0:.4f}s")

    # 4. (Optional) Verify results
    # c_from_gpu = c_gpu.get()
    # print(f"Verifying results... {np.allclose(c_cpu, c_from_gpu, atol=1e-5)}")

    # 5. Summary
    print("\n" + "=" * 60)
    print("Benchmark Summary (AMD GPU):")
    for tx, ty, host, device_t in gpu_results:
        print(f"Tile {tx:2d}x{ty:2d} | Host {host:.4f}s | Device {device_t:.4f}s")
    
    best_device = min(gpu_results, key=lambda x: x[3])
    print(f"\nBest Tile (Device): {best_device[0]}x{best_device[1]} @ {best_device[3]:.4f}s")
    print("\n✅ Done.")
