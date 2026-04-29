package compute

// getOpenCLKernels returns the OpenCL kernel source code
func getOpenCLKernels() string {
	return `
// Matrix multiplication: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
__kernel void matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K
) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Matrix multiplication with addition: C = A @ B + C
__kernel void matmul_add(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K
) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        float sum = C[row * N + col];
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Element-wise addition: C = A + B
__kernel void add(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

// Scale: C = A * scalar
__kernel void scale(
    __global const float* A,
    __global float* C,
    const float scalar,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        C[idx] = A[idx] * scalar;
    }
}

// Add scaled: C = A + B * scale
__kernel void add_scaled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const float scale,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        C[idx] = A[idx] + B[idx] * scale;
    }
}

// Softmax (1D)
__kernel void softmax(
    __global const float* input,
    __global float* output,
    const int size
) {
    int idx = get_global_id(0);
    if (idx == 0) {
        // Find max for numerical stability
        float max_val = input[0];
        for (int i = 1; i < size; i++) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            output[i] = exp(input[i] - max_val);
            sum += output[i];
        }

        // Normalize
        if (sum > 0.0f) {
            for (int i = 0; i < size; i++) {
                output[i] /= sum;
            }
        }
    }
}

// ReLU activation
__kernel void relu(
    __global const float* input,
    __global float* output,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
    }
}

// Layer normalization
__kernel void layer_norm(
    __global const float* input,
    __global const float* gamma,
    __global const float* beta,
    __global float* output,
    const int rows,
    const int cols,
    const float epsilon
) {
    int row = get_global_id(0);
    if (row < rows) {
        // Compute mean
        float mean = 0.0f;
        for (int c = 0; c < cols; c++) {
            mean += input[row * cols + c];
        }
        mean /= cols;

        // Compute variance
        float variance = 0.0f;
        for (int c = 0; c < cols; c++) {
            float diff = input[row * cols + c] - mean;
            variance += diff * diff;
        }
        variance /= cols;

        // Normalize
        float std = sqrt(variance + epsilon);
        for (int c = 0; c < cols; c++) {
            float normalized = (input[row * cols + c] - mean) / std;
            output[row * cols + c] = normalized * gamma[c] + beta[c];
        }
    }
}

// Embedding lookup
__kernel void embedding_lookup(
    __global const float* weights,
    __global const int* indices,
    __global float* output,
    const int seq_len,
    const int embed_dim,
    const int vocab_size
) {
    int idx = get_global_id(0);
    if (idx < seq_len) {
        int token_idx = indices[idx];
        if (token_idx >= 0 && token_idx < vocab_size) {
            for (int d = 0; d < embed_dim; d++) {
                output[idx * embed_dim + d] = weights[token_idx * embed_dim + d];
            }
        }
    }
}

// Element-wise multiply
__kernel void mul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

// Transpose matrix
__kernel void transpose(
    __global const float* input,
    __global float* output,
    const int rows,
    const int cols
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}
`
}
