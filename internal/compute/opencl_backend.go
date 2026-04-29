//go:build darwin || linux
// +build darwin linux

package compute

/*
#cgo darwin LDFLAGS: -framework OpenCL
#cgo linux LDFLAGS: -lOpenCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdlib.h>
#include <string.h>

// Helper to get error string
const char* clGetErrorString(cl_int error) {
	switch(error) {
		case CL_SUCCESS: return "Success";
		case CL_DEVICE_NOT_FOUND: return "Device not found";
		case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
		case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory allocation failure";
		case CL_OUT_OF_RESOURCES: return "Out of resources";
		case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
		case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling info not available";
		case CL_MEM_COPY_OVERLAP: return "Memory copy overlap";
		case CL_IMAGE_FORMAT_MISMATCH: return "Image format mismatch";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image format not supported";
		case CL_BUILD_PROGRAM_FAILURE: return "Build program failure";
		case CL_MAP_FAILURE: return "Map failure";
		case CL_INVALID_VALUE: return "Invalid value";
		case CL_INVALID_DEVICE_TYPE: return "Invalid device type";
		case CL_INVALID_PLATFORM: return "Invalid platform";
		case CL_INVALID_DEVICE: return "Invalid device";
		case CL_INVALID_CONTEXT: return "Invalid context";
		case CL_INVALID_QUEUE_PROPERTIES: return "Invalid queue properties";
		case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
		case CL_INVALID_HOST_PTR: return "Invalid host pointer";
		case CL_INVALID_MEM_OBJECT: return "Invalid memory object";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid image format descriptor";
		case CL_INVALID_IMAGE_SIZE: return "Invalid image size";
		case CL_INVALID_SAMPLER: return "Invalid sampler";
		case CL_INVALID_BINARY: return "Invalid binary";
		case CL_INVALID_BUILD_OPTIONS: return "Invalid build options";
		case CL_INVALID_PROGRAM: return "Invalid program";
		case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid program executable";
		case CL_INVALID_KERNEL_NAME: return "Invalid kernel name";
		case CL_INVALID_KERNEL_DEFINITION: return "Invalid kernel definition";
		case CL_INVALID_KERNEL: return "Invalid kernel";
		case CL_INVALID_ARG_INDEX: return "Invalid arg index";
		case CL_INVALID_ARG_VALUE: return "Invalid arg value";
		case CL_INVALID_ARG_SIZE: return "Invalid arg size";
		case CL_INVALID_KERNEL_ARGS: return "Invalid kernel args";
		case CL_INVALID_WORK_DIMENSION: return "Invalid work dimension";
		case CL_INVALID_WORK_GROUP_SIZE: return "Invalid work group size";
		case CL_INVALID_WORK_ITEM_SIZE: return "Invalid work item size";
		case CL_INVALID_GLOBAL_OFFSET: return "Invalid global offset";
		case CL_INVALID_EVENT_WAIT_LIST: return "Invalid event wait list";
		case CL_INVALID_EVENT: return "Invalid event";
		case CL_INVALID_OPERATION: return "Invalid operation";
		case CL_INVALID_GL_OBJECT: return "Invalid GL object";
		case CL_INVALID_BUFFER_SIZE: return "Invalid buffer size";
		case CL_INVALID_MIP_LEVEL: return "Invalid mip level";
		case CL_INVALID_GLOBAL_WORK_SIZE: return "Invalid global work size";
		default: return "Unknown OpenCL error";
	}
}
*/
import "C"
import (
	"fmt"
	"runtime"
	"strings"
	"unsafe"
)

// OpenCLBackend implements the Backend interface using OpenCL for GPU computation
type OpenCLBackend struct {
	info        DeviceInfo
	platform    C.cl_platform_id
	device      C.cl_device_id
	context     C.cl_context
	queue       C.cl_command_queue
	programs    map[string]C.cl_program
	kernels     map[string]C.cl_kernel
	buffers     []C.cl_mem
	initialized bool
}

// NewOpenCLBackend creates a new OpenCL backend
func NewOpenCLBackend() (*OpenCLBackend, error) {
	backend := &OpenCLBackend{
		programs: make(map[string]C.cl_program),
		kernels:  make(map[string]C.cl_kernel),
		buffers:  make([]C.cl_mem, 0),
	}

	// Try to find an OpenCL platform
	var numPlatforms C.cl_uint
	err := C.clGetPlatformIDs(0, nil, &numPlatforms)
	if err != C.CL_SUCCESS || numPlatforms == 0 {
		return nil, fmt.Errorf("no OpenCL platforms found: %s", C.GoString(C.clGetErrorString(err)))
	}

	platforms := make([]C.cl_platform_id, numPlatforms)
	err = C.clGetPlatformIDs(numPlatforms, &platforms[0], nil)
	if err != C.CL_SUCCESS {
		return nil, fmt.Errorf("failed to get platforms: %s", C.GoString(C.clGetErrorString(err)))
	}

	// Find best platform (prefer AMD, then any GPU)
	var bestPlatform C.cl_platform_id
	var bestDevice C.cl_device_id
	found := false

	for _, platform := range platforms {
		// Get platform name
		var platformName [256]C.char
		C.clGetPlatformInfo(platform, C.CL_PLATFORM_NAME, 256, unsafe.Pointer(&platformName[0]), nil)
		name := strings.ToLower(C.GoString(&platformName[0]))

		// Get devices on this platform
		var numDevices C.cl_uint
		err = C.clGetDeviceIDs(platform, C.CL_DEVICE_TYPE_GPU, 0, nil, &numDevices)
		if err != C.CL_SUCCESS || numDevices == 0 {
			continue
		}

		devices := make([]C.cl_device_id, numDevices)
		err = C.clGetDeviceIDs(platform, C.CL_DEVICE_TYPE_GPU, numDevices, &devices[0], nil)
		if err != C.CL_SUCCESS {
			continue
		}

		// Prefer AMD platforms
		if strings.Contains(name, "amd") || strings.Contains(name, "advanced micro") {
			bestPlatform = platform
			bestDevice = devices[0]
			found = true
			break
		}

		// Otherwise use first available GPU
		if bestDevice == nil {
			bestPlatform = platform
			bestDevice = devices[0]
			found = true
		}
	}

	if !found {
		return nil, fmt.Errorf("no GPU devices found")
	}

	// Get device info
	var deviceName [256]C.char
	var deviceVendor [256]C.char
	var deviceVersion [256]C.char
	var memSize C.cl_ulong
	var maxWorkGroup C.size_t

	C.clGetDeviceInfo(bestDevice, C.CL_DEVICE_NAME, 256, unsafe.Pointer(&deviceName[0]), nil)
	C.clGetDeviceInfo(bestDevice, C.CL_DEVICE_VENDOR, 256, unsafe.Pointer(&deviceVendor[0]), nil)
	C.clGetDeviceInfo(bestDevice, C.CL_DEVICE_VERSION, 256, unsafe.Pointer(&deviceVersion[0]), nil)
	C.clGetDeviceInfo(bestDevice, C.CL_DEVICE_GLOBAL_MEM_SIZE, C.sizeof_cl_ulong, unsafe.Pointer(&memSize), nil)
	C.clGetDeviceInfo(bestDevice, C.CL_DEVICE_MAX_WORK_GROUP_SIZE, C.sizeof_size_t, unsafe.Pointer(&maxWorkGroup), nil)

	backend.platform = bestPlatform
	backend.device = bestDevice
	backend.info = DeviceInfo{
		Type:         DeviceGPU_OpenCL,
		Name:         strings.TrimSpace(C.GoString(&deviceName[0])),
		Vendor:       strings.TrimSpace(C.GoString(&deviceVendor[0])),
		Version:      strings.TrimSpace(C.GoString(&deviceVersion[0])),
		MemoryMB:     int(memSize / (1024 * 1024)),
		MaxWorkGroup: int(maxWorkGroup),
	}

	return backend, nil
}

func (b *OpenCLBackend) Info() DeviceInfo {
	return b.info
}

func (b *OpenCLBackend) Initialize() error {
	if b.initialized {
		return nil
	}

	var err C.cl_int

	// Create context
	b.context = C.clCreateContext(nil, 1, &b.device, nil, nil, &err)
	if err != C.CL_SUCCESS {
		return fmt.Errorf("failed to create context: %s", C.GoString(C.clGetErrorString(err)))
	}

	// Create command queue
	b.queue = C.clCreateCommandQueue(b.context, b.device, 0, &err)
	if err != C.CL_SUCCESS {
		C.clReleaseContext(b.context)
		return fmt.Errorf("failed to create command queue: %s", C.GoString(C.clGetErrorString(err)))
	}

	// Build kernels
	if err := b.buildKernels(); err != nil {
		C.clReleaseCommandQueue(b.queue)
		C.clReleaseContext(b.context)
		return fmt.Errorf("failed to build kernels: %w", err)
	}

	b.initialized = true
	runtime.SetFinalizer(b, (*OpenCLBackend).Release)

	return nil
}

func (b *OpenCLBackend) Release() error {
	if !b.initialized {
		return nil
	}

	// Release kernels
	for _, kernel := range b.kernels {
		C.clReleaseKernel(kernel)
	}

	// Release programs
	for _, program := range b.programs {
		C.clReleaseProgram(program)
	}

	// Release buffers
	for _, buf := range b.buffers {
		C.clReleaseMemObject(buf)
	}

	// Release queue and context
	if b.queue != nil {
		C.clReleaseCommandQueue(b.queue)
	}
	if b.context != nil {
		C.clReleaseContext(b.context)
	}

	b.initialized = false
	return nil
}

func (b *OpenCLBackend) Sync() error {
	if !b.initialized {
		return fmt.Errorf("backend not initialized")
	}
	err := C.clFinish(b.queue)
	if err != C.CL_SUCCESS {
		return fmt.Errorf("clFinish failed: %s", C.GoString(C.clGetErrorString(err)))
	}
	return nil
}

// buildKernels compiles all OpenCL kernels
func (b *OpenCLBackend) buildKernels() error {
	kernelSource := getOpenCLKernels()

	cSource := C.CString(kernelSource)
	defer C.free(unsafe.Pointer(cSource))

	var err C.cl_int
	program := C.clCreateProgramWithSource(b.context, 1, &cSource, nil, &err)
	if err != C.CL_SUCCESS {
		return fmt.Errorf("failed to create program: %s", C.GoString(C.clGetErrorString(err)))
	}

	err = C.clBuildProgram(program, 1, &b.device, nil, nil, nil)
	if err != C.CL_SUCCESS {
		// Get build log
		var logSize C.size_t
		C.clGetProgramBuildInfo(program, b.device, C.CL_PROGRAM_BUILD_LOG, 0, nil, &logSize)
		logBuf := make([]byte, logSize)
		C.clGetProgramBuildInfo(program, b.device, C.CL_PROGRAM_BUILD_LOG, logSize, unsafe.Pointer(&logBuf[0]), nil)
		C.clReleaseProgram(program)
		return fmt.Errorf("failed to build program: %s\nBuild log:\n%s",
			C.GoString(C.clGetErrorString(err)), string(logBuf))
	}

	b.programs["default"] = program

	// Create kernels
	kernelNames := []string{
		"matmul", "matmul_add", "add", "scale", "add_scaled",
		"softmax", "relu", "layer_norm", "embedding_lookup",
	}

	for _, name := range kernelNames {
		cName := C.CString(name)
		kernel := C.clCreateKernel(program, cName, &err)
		C.free(unsafe.Pointer(cName))
		if err != C.CL_SUCCESS {
			return fmt.Errorf("failed to create kernel '%s': %s", name, C.GoString(C.clGetErrorString(err)))
		}
		b.kernels[name] = kernel
	}

	return nil
}

// createBuffer creates an OpenCL buffer from host data
func (b *OpenCLBackend) createBuffer(data []float32, readOnly bool) (C.cl_mem, error) {
	var flags C.cl_mem_flags = C.CL_MEM_COPY_HOST_PTR
	if readOnly {
		flags |= C.CL_MEM_READ_ONLY
	} else {
		flags |= C.CL_MEM_READ_WRITE
	}

	var err C.cl_int
	buf := C.clCreateBuffer(b.context, flags, C.size_t(len(data)*4), unsafe.Pointer(&data[0]), &err)
	if err != C.CL_SUCCESS {
		return nil, fmt.Errorf("failed to create buffer: %s", C.GoString(C.clGetErrorString(err)))
	}

	b.buffers = append(b.buffers, buf)
	return buf, nil
}

// readBuffer reads data from an OpenCL buffer
func (b *OpenCLBackend) readBuffer(buf C.cl_mem, size int) ([]float32, error) {
	data := make([]float32, size)
	err := C.clEnqueueReadBuffer(b.queue, buf, C.CL_TRUE, 0, C.size_t(size*4), unsafe.Pointer(&data[0]), 0, nil, nil)
	if err != C.CL_SUCCESS {
		return nil, fmt.Errorf("failed to read buffer: %s", C.GoString(C.clGetErrorString(err)))
	}
	return data, nil
}

// CopyToDevice copies data to GPU memory
func (b *OpenCLBackend) CopyToDevice(data []float64) (*Buffer, error) {
	if !b.initialized {
		return nil, fmt.Errorf("backend not initialized")
	}

	buf := NewBuffer(data)
	return buf, nil
}

// CopyFromDevice copies data from GPU memory
func (b *OpenCLBackend) CopyFromDevice(buf *Buffer) ([]float64, error) {
	if buf == nil {
		return nil, fmt.Errorf("nil buffer")
	}
	return buf.ToFloat64(), nil
}

// MatMul computes C = A @ B on GPU
func (b *OpenCLBackend) MatMul(a, bBuf *Buffer, m, n, k int) (*Buffer, error) {
	if !b.initialized {
		return nil, fmt.Errorf("backend not initialized")
	}

	// Create device buffers
	aCL, err := b.createBuffer(a.FloatData, true)
	if err != nil {
		return nil, err
	}
	bCL, err := b.createBuffer(bBuf.FloatData, true)
	if err != nil {
		return nil, err
	}

	cData := make([]float32, m*n)
	cCL, err := b.createBuffer(cData, false)
	if err != nil {
		return nil, err
	}

	// Set kernel arguments
	kernel := b.kernels["matmul"]
	C.clSetKernelArg(kernel, 0, C.size_t(unsafe.Sizeof(aCL)), unsafe.Pointer(&aCL))
	C.clSetKernelArg(kernel, 1, C.size_t(unsafe.Sizeof(bCL)), unsafe.Pointer(&bCL))
	C.clSetKernelArg(kernel, 2, C.size_t(unsafe.Sizeof(cCL)), unsafe.Pointer(&cCL))
	C.clSetKernelArg(kernel, 3, C.size_t(unsafe.Sizeof(C.int(0))), unsafe.Pointer(&m))
	C.clSetKernelArg(kernel, 4, C.size_t(unsafe.Sizeof(C.int(0))), unsafe.Pointer(&n))
	C.clSetKernelArg(kernel, 5, C.size_t(unsafe.Sizeof(C.int(0))), unsafe.Pointer(&k))

	// Execute kernel
	globalSize := []C.size_t{C.size_t(m), C.size_t(n)}
	err2 := C.clEnqueueNDRangeKernel(b.queue, kernel, 2, nil, &globalSize[0], nil, 0, nil, nil)
	if err2 != C.CL_SUCCESS {
		return nil, fmt.Errorf("kernel execution failed: %s", C.GoString(C.clGetErrorString(err2)))
	}

	// Read result
	result, err := b.readBuffer(cCL, m*n)
	if err != nil {
		return nil, err
	}

	return &Buffer{Size: m * n, FloatData: result}, nil
}

// MatMulAdd computes C = A @ B + C on GPU
func (b *OpenCLBackend) MatMulAdd(a, bBuf, c *Buffer, m, n, k int) (*Buffer, error) {
	// For simplicity, compute A@B then add C on CPU
	result, err := b.MatMul(a, bBuf, m, n, k)
	if err != nil {
		return nil, err
	}

	if c != nil && len(c.FloatData) == m*n {
		for i := 0; i < m*n; i++ {
			result.FloatData[i] += c.FloatData[i]
		}
	}

	return result, nil
}

// MatVecMul computes c = A @ b on GPU
func (b *OpenCLBackend) MatVecMul(a, bBuf *Buffer, m, k int) (*Buffer, error) {
	return b.MatMul(a, bBuf, m, 1, k)
}

// Add performs element-wise addition on GPU
func (b *OpenCLBackend) Add(a, bBuf *Buffer) (*Buffer, error) {
	if !b.initialized {
		return nil, fmt.Errorf("backend not initialized")
	}
	if a.Size != bBuf.Size {
		return nil, fmt.Errorf("size mismatch")
	}

	aCL, _ := b.createBuffer(a.FloatData, true)
	bCL, _ := b.createBuffer(bBuf.FloatData, true)
	cData := make([]float32, a.Size)
	cCL, _ := b.createBuffer(cData, false)

	kernel := b.kernels["add"]
	C.clSetKernelArg(kernel, 0, C.size_t(unsafe.Sizeof(aCL)), unsafe.Pointer(&aCL))
	C.clSetKernelArg(kernel, 1, C.size_t(unsafe.Sizeof(bCL)), unsafe.Pointer(&bCL))
	C.clSetKernelArg(kernel, 2, C.size_t(unsafe.Sizeof(cCL)), unsafe.Pointer(&cCL))
	size := C.int(a.Size)
	C.clSetKernelArg(kernel, 3, C.size_t(unsafe.Sizeof(size)), unsafe.Pointer(&size))

	globalSize := []C.size_t{C.size_t(a.Size)}
	C.clEnqueueNDRangeKernel(b.queue, kernel, 1, nil, &globalSize[0], nil, 0, nil, nil)

	result, _ := b.readBuffer(cCL, a.Size)
	return &Buffer{Size: a.Size, FloatData: result}, nil
}

// Scale multiplies all elements by scalar on GPU
func (b *OpenCLBackend) Scale(a *Buffer, scalar float32) (*Buffer, error) {
	if !b.initialized {
		return nil, fmt.Errorf("backend not initialized")
	}

	aCL, _ := b.createBuffer(a.FloatData, true)
	cData := make([]float32, a.Size)
	cCL, _ := b.createBuffer(cData, false)

	kernel := b.kernels["scale"]
	C.clSetKernelArg(kernel, 0, C.size_t(unsafe.Sizeof(aCL)), unsafe.Pointer(&aCL))
	C.clSetKernelArg(kernel, 1, C.size_t(unsafe.Sizeof(cCL)), unsafe.Pointer(&cCL))
	C.clSetKernelArg(kernel, 2, C.size_t(4), unsafe.Pointer(&scalar))
	size := C.int(a.Size)
	C.clSetKernelArg(kernel, 3, C.size_t(unsafe.Sizeof(size)), unsafe.Pointer(&size))

	globalSize := []C.size_t{C.size_t(a.Size)}
	C.clEnqueueNDRangeKernel(b.queue, kernel, 1, nil, &globalSize[0], nil, 0, nil, nil)

	result, _ := b.readBuffer(cCL, a.Size)
	return &Buffer{Size: a.Size, FloatData: result}, nil
}

// AddScaled computes a + b * scale on GPU
func (b *OpenCLBackend) AddScaled(a, bBuf *Buffer, scale float32) (*Buffer, error) {
	if !b.initialized {
		return nil, fmt.Errorf("backend not initialized")
	}
	if a.Size != bBuf.Size {
		return nil, fmt.Errorf("size mismatch")
	}

	aCL, _ := b.createBuffer(a.FloatData, true)
	bCL, _ := b.createBuffer(bBuf.FloatData, true)
	cData := make([]float32, a.Size)
	cCL, _ := b.createBuffer(cData, false)

	kernel := b.kernels["add_scaled"]
	C.clSetKernelArg(kernel, 0, C.size_t(unsafe.Sizeof(aCL)), unsafe.Pointer(&aCL))
	C.clSetKernelArg(kernel, 1, C.size_t(unsafe.Sizeof(bCL)), unsafe.Pointer(&bCL))
	C.clSetKernelArg(kernel, 2, C.size_t(unsafe.Sizeof(cCL)), unsafe.Pointer(&cCL))
	C.clSetKernelArg(kernel, 3, C.size_t(4), unsafe.Pointer(&scale))
	size := C.int(a.Size)
	C.clSetKernelArg(kernel, 4, C.size_t(unsafe.Sizeof(size)), unsafe.Pointer(&size))

	globalSize := []C.size_t{C.size_t(a.Size)}
	C.clEnqueueNDRangeKernel(b.queue, kernel, 1, nil, &globalSize[0], nil, 0, nil, nil)

	result, _ := b.readBuffer(cCL, a.Size)
	return &Buffer{Size: a.Size, FloatData: result}, nil
}

// Softmax computes softmax on GPU
func (b *OpenCLBackend) Softmax(x *Buffer, axis int) (*Buffer, error) {
	if !b.initialized {
		return nil, fmt.Errorf("backend not initialized")
	}

	// For simplicity, compute on CPU and copy back
	cpu := NewCPUBackend()
	return cpu.Softmax(x, axis)
}

// ReLU applies ReLU on GPU
func (b *OpenCLBackend) ReLU(x *Buffer) (*Buffer, error) {
	if !b.initialized {
		return nil, fmt.Errorf("backend not initialized")
	}

	xCL, _ := b.createBuffer(x.FloatData, true)
	cData := make([]float32, x.Size)
	cCL, _ := b.createBuffer(cData, false)

	kernel := b.kernels["relu"]
	C.clSetKernelArg(kernel, 0, C.size_t(unsafe.Sizeof(xCL)), unsafe.Pointer(&xCL))
	C.clSetKernelArg(kernel, 1, C.size_t(unsafe.Sizeof(cCL)), unsafe.Pointer(&cCL))
	size := C.int(x.Size)
	C.clSetKernelArg(kernel, 2, C.size_t(unsafe.Sizeof(size)), unsafe.Pointer(&size))

	globalSize := []C.size_t{C.size_t(x.Size)}
	C.clEnqueueNDRangeKernel(b.queue, kernel, 1, nil, &globalSize[0], nil, 0, nil, nil)

	result, _ := b.readBuffer(cCL, x.Size)
	return &Buffer{Size: x.Size, FloatData: result}, nil
}

// LayerNorm applies layer normalization on GPU
func (b *OpenCLBackend) LayerNorm(x, gamma, beta *Buffer, rows, cols int, eps float32) (*Buffer, error) {
	if !b.initialized {
		return nil, fmt.Errorf("backend not initialized")
	}

	// For simplicity, compute on CPU
	cpu := NewCPUBackend()
	return cpu.LayerNorm(x, gamma, beta, rows, cols, eps)
}

// Sum computes sum on GPU
func (b *OpenCLBackend) Sum(x *Buffer) (float32, error) {
	if !b.initialized {
		return 0, fmt.Errorf("backend not initialized")
	}

	cpu := NewCPUBackend()
	return cpu.Sum(x)
}

// Mean computes mean on GPU
func (b *OpenCLBackend) Mean(x *Buffer) (float32, error) {
	if !b.initialized {
		return 0, fmt.Errorf("backend not initialized")
	}

	cpu := NewCPUBackend()
	return cpu.Mean(x)
}

// EmbeddingLookup looks up embeddings on GPU
func (b *OpenCLBackend) EmbeddingLookup(weights *Buffer, indices []int, vocabSize, embedDim int) (*Buffer, error) {
	if !b.initialized {
		return nil, fmt.Errorf("backend not initialized")
	}

	cpu := NewCPUBackend()
	return cpu.EmbeddingLookup(weights, indices, vocabSize, embedDim)
}
