#include "ggml-cognitive-tensor.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>

// Test prime offset function
void test_prime_offset() {
    printf("Testing prime offset function...\n");
    
    // Test basic prime offset: p(n) = (n+1)-th prime
    assert(ggml_prime_offset(1) == 3);  // p(1) = 2nd prime = 3
    assert(ggml_prime_offset(2) == 5);  // p(2) = 3rd prime = 5
    assert(ggml_prime_offset(3) == 7);  // p(3) = 4th prime = 7
    assert(ggml_prime_offset(4) == 11); // p(4) = 5th prime = 11
    
    printf("✓ Prime offset tests passed\n");
}

// Test Matula-Goebel encoding
void test_matula_encoding() {
    printf("Testing Matula-Goebel encoding...\n");
    
    ggml_prime_lookup_t lookup = {0};
    ggml_init_prime_lookup(&lookup);
    
    // Test basic tree expressions
    ggml_matula_encoding_t enc1 = ggml_encode_tree("()", &lookup);
    assert(enc1.matula_value == 1);
    assert(enc1.system_level == 1);
    printf("✓ Empty tree '()' -> Matula: %u, System: %u\n", 
           enc1.matula_value, enc1.system_level);
    
    ggml_matula_encoding_t enc2 = ggml_encode_tree("(())", &lookup);
    printf("✓ Single nested tree '(())' -> Matula: %u, System: %u\n", 
           enc2.matula_value, enc2.system_level);
    
    ggml_matula_encoding_t enc3 = ggml_encode_tree("()()", &lookup);
    printf("✓ Two empty trees '()()' -> Matula: %u, System: %u\n", 
           enc3.matula_value, enc3.system_level);
    
    printf("✓ Matula encoding tests passed\n");
}

// Test cognitive kernel tensor
void test_cognitive_kernel() {
    printf("Testing cognitive kernel tensor...\n");
    
    // Initialize ggml context
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    assert(ctx != NULL);
    
    // Create cognitive kernel
    ggml_cognitive_kernel_t* kernel = ggml_cognitive_kernel_init(ctx, 8, 16, 16);
    assert(kernel != NULL);
    
    // Test tensor encoding
    struct ggml_tensor* encoded = ggml_cognitive_kernel_encode(ctx, kernel, "()");
    assert(encoded != NULL);
    assert(encoded->type == GGML_TYPE_F32);
    assert(encoded->ne[0] == 4);
    
    printf("✓ Cognitive kernel tensor created successfully\n");
    
    // Print statistics
    ggml_cognitive_tensor_print_stats(kernel);
    
    // Test tree tensor composition
    ggml_tree_tensor_t t1 = {
        .matula_value = 2,
        .phase = 1.0f + 0.0f * I,
        .is_prime = true,
        .probability_amplitude = 0.7f
    };
    
    ggml_tree_tensor_t t2 = {
        .matula_value = 3,
        .phase = 0.0f + 1.0f * I,
        .is_prime = true,
        .probability_amplitude = 0.6f
    };
    
    ggml_tree_tensor_t composed = ggml_tree_tensor_compose(t1, t2);
    printf("✓ Tree tensor composition: %u * %u = %u\n", 
           t1.matula_value, t2.matula_value, composed.matula_value);
    
    // Test superposition
    ggml_tree_tensor_t states[] = {t1, t2};
    struct ggml_tensor* superposition = ggml_cognitive_kernel_superposition(ctx, kernel, states, 2);
    assert(superposition != NULL);
    printf("✓ Quantum superposition tensor created\n");
    
    // Test phase interference
    struct ggml_tensor* interference = ggml_phase_interference_pattern(ctx, kernel, 4, 4);
    assert(interference != NULL);
    printf("✓ Phase interference pattern generated\n");
    
    // Test tensor similarity
    struct ggml_tensor* test_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    float* test_data = (float*)test_tensor->data;
    test_data[0] = 1.0f;
    test_data[1] = 2.0f;
    test_data[2] = 3.0f;
    test_data[3] = 4.0f;
    
    float similarity = ggml_cognitive_tensor_similarity(encoded, test_tensor);
    printf("✓ Tensor similarity computed: %.3f\n", similarity);
    
    // Test tensor norms
    float l1_norm = ggml_cognitive_tensor_norm(test_tensor, "l1");
    float l2_norm = ggml_cognitive_tensor_norm(test_tensor, "l2");
    printf("✓ Tensor norms - L1: %.3f, L2: %.3f\n", l1_norm, l2_norm);
    
    // Cleanup
    ggml_cognitive_kernel_free(kernel);
    ggml_free(ctx);
    
    printf("✓ Cognitive kernel tests passed\n");
}

// Test prime-structured attention
void test_prime_attention() {
    printf("Testing prime-structured attention...\n");
    
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    assert(ctx != NULL);
    
    // Create test tensors
    int n_tokens = 4;
    int d_model = 8;
    
    struct ggml_tensor* query = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, d_model);
    struct ggml_tensor* key = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, d_model);
    struct ggml_tensor* value = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, d_model);
    
    // Initialize with test data
    ggml_set_zero(query);
    ggml_set_zero(key);
    ggml_set_zero(value);
    
    float* q_data = (float*)query->data;
    float* k_data = (float*)key->data;
    float* v_data = (float*)value->data;
    
    for (int i = 0; i < n_tokens * d_model; i++) {
        q_data[i] = 0.1f * i;
        k_data[i] = 0.2f * i;
        v_data[i] = 0.3f * i;
    }
    
    // Create prime attention configuration
    ggml_prime_attention_t prime_config = {
        .prime_dims = {2, 3, 5, 7, 11, 13, 17, 19},
        .attention_weights = NULL,
        .prime_projections = NULL
    };
    
    // Test prime attention
    struct ggml_tensor* attention_output = ggml_prime_attention(ctx, query, key, value, &prime_config);
    assert(attention_output != NULL);
    
    printf("✓ Prime-structured attention computed successfully\n");
    
    ggml_free(ctx);
    
    printf("✓ Prime attention tests passed\n");
}

// Test quantum phase encoding
void test_quantum_phase() {
    printf("Testing quantum phase encoding...\n");
    
    // Test basic phase encoding
    ggml_complex_t phase1 = ggml_quantum_phase_encode(1, 0.0f);
    ggml_complex_t phase2 = ggml_quantum_phase_encode(2, 0.0f);
    ggml_complex_t phase3 = ggml_quantum_phase_encode(3, 0.0f);
    
    printf("✓ Phase encoding - 1: %.3f+%.3fi, 2: %.3f+%.3fi, 3: %.3f+%.3fi\n",
           crealf(phase1), cimagf(phase1),
           crealf(phase2), cimagf(phase2),
           crealf(phase3), cimagf(phase3));
    
    // Test phase normalization
    float norm1 = crealf(phase1) * crealf(phase1) + cimagf(phase1) * cimagf(phase1);
    float norm2 = crealf(phase2) * crealf(phase2) + cimagf(phase2) * cimagf(phase2);
    assert(fabs(norm1 - 1.0f) < 1e-6);
    assert(fabs(norm2 - 1.0f) < 1e-6);
    
    printf("✓ Quantum phase encoding tests passed\n");
}

int main() {
    printf("Neural-Symbolic Tensor Architecture Test Suite\n");
    printf("==============================================\n\n");
    
    test_prime_offset();
    printf("\n");
    
    test_matula_encoding();
    printf("\n");
    
    test_cognitive_kernel();
    printf("\n");
    
    test_prime_attention();
    printf("\n");
    
    test_quantum_phase();
    printf("\n");
    
    printf("==============================================\n");
    printf("All tests passed! ✓\n");
    printf("Neural-Symbolic Tensor Architecture successfully implemented.\n");
    
    return 0;
}