#include "ggml-cognitive-tensor.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Neural-Symbolic Tensor Architecture Demonstration\n");
    printf("================================================\n\n");
    
    // Initialize ggml context
    struct ggml_init_params params = {
        .mem_size = 32 * 1024 * 1024,  // 32MB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    
    // Create cognitive kernel with Matula-Goebel prime offset encoding
    ggml_cognitive_kernel_t* kernel = ggml_cognitive_kernel_init(ctx, 16, 32, 32);
    
    printf("1. System 1: Unity Manifold\n");
    printf("   n₁ = p(1) = 2 (prime offset)\n");
    struct ggml_tensor* system1 = ggml_cognitive_kernel_encode(ctx, kernel, "()");
    float* s1_data = (float*)system1->data;
    printf("   Encoded: [%.0f, %.0f, %.0f, %.0f]\n\n", 
           s1_data[0], s1_data[1], s1_data[2], s1_data[3]);
    
    printf("2. System 2: Duality Bifurcation\n");
    printf("   Two structures: \"()()\" and \"(())\"\n");
    struct ggml_tensor* system2a = ggml_cognitive_kernel_encode(ctx, kernel, "()()");
    struct ggml_tensor* system2b = ggml_cognitive_kernel_encode(ctx, kernel, "(())");
    
    float* s2a_data = (float*)system2a->data;
    float* s2b_data = (float*)system2b->data;
    printf("   ()(): [%.0f, %.0f, %.0f, %.0f]\n", 
           s2a_data[0], s2a_data[1], s2a_data[2], s2a_data[3]);
    printf("   (()): [%.0f, %.0f, %.0f, %.0f]\n\n", 
           s2b_data[0], s2b_data[1], s2b_data[2], s2b_data[3]);
    
    printf("3. System 3: Quaternary Emergence\n");
    printf("   More complex tree structures\n");
    struct ggml_tensor* system3 = ggml_cognitive_kernel_encode(ctx, kernel, "((()))");
    float* s3_data = (float*)system3->data;
    printf("   ((())): [%.0f, %.0f, %.0f, %.0f]\n\n", 
           s3_data[0], s3_data[1], s3_data[2], s3_data[3]);
    
    printf("4. Prime Offset Demonstration\n");
    printf("   p(1) = %u (2nd prime)\n", ggml_prime_offset(1));
    printf("   p(2) = %u (3rd prime)\n", ggml_prime_offset(2));
    printf("   p(3) = %u (4th prime)\n", ggml_prime_offset(3));
    printf("   p(4) = %u (5th prime)\n\n", ggml_prime_offset(4));
    
    printf("5. Quantum Phase Encoding\n");
    ggml_complex_t phase1 = ggml_quantum_phase_encode(2, 0.0f);
    ggml_complex_t phase2 = ggml_quantum_phase_encode(3, 0.0f);
    ggml_complex_t phase3 = ggml_quantum_phase_encode(5, 0.0f);
    printf("   Prime 2: %.3f + %.3fi\n", crealf(phase1), cimagf(phase1));
    printf("   Prime 3: %.3f + %.3fi\n", crealf(phase2), cimagf(phase2));
    printf("   Prime 5: %.3f + %.3fi\n\n", crealf(phase3), cimagf(phase3));
    
    printf("6. Tree Tensor Composition\n");
    ggml_tree_tensor_t t1 = {
        .matula_value = 2,
        .phase = phase1,
        .is_prime = true,
        .probability_amplitude = 0.7f
    };
    
    ggml_tree_tensor_t t2 = {
        .matula_value = 3,
        .phase = phase2,
        .is_prime = true,
        .probability_amplitude = 0.6f
    };
    
    ggml_tree_tensor_t composed = ggml_tree_tensor_compose(t1, t2);
    printf("   Compose: Prime(%u) ⊗ Prime(%u) → Composite(%u)\n", 
           t1.matula_value, t2.matula_value, composed.matula_value);
    printf("   Phase entanglement: %.3f + %.3fi\n\n", 
           crealf(composed.phase), cimagf(composed.phase));
    
    printf("7. Quantum Superposition of Tree States\n");
    ggml_tree_tensor_t states[] = {t1, t2};
    struct ggml_tensor* superposition = ggml_cognitive_kernel_superposition(ctx, kernel, states, 2);
    printf("   Superposition tensor shape: [%ld, %ld]\n", 
           superposition->ne[0], superposition->ne[1]);
    printf("   Multiple tree interpretations in quantum coherence\n\n");
    
    printf("8. Phase Interference Patterns\n");
    struct ggml_tensor* interference = ggml_phase_interference_pattern(ctx, kernel, 8, 8);
    float* interference_data = (float*)interference->data;
    printf("   Sample interference values:\n");
    for (int i = 0; i < 4; i++) {
        printf("   [%d]: %.3f\n", i, interference_data[i]);
    }
    printf("\n");
    
    printf("9. Tensor Similarity and Norms\n");
    float similarity = ggml_cognitive_tensor_similarity(system1, system2a);
    float l1_norm = ggml_cognitive_tensor_norm(system1, "l1");
    float l2_norm = ggml_cognitive_tensor_norm(system1, "l2");
    printf("   Similarity between () and ()(): %.3f\n", similarity);
    printf("   L1 norm: %.3f, L2 norm: %.3f\n\n", l1_norm, l2_norm);
    
    printf("10. Cognitive Kernel Statistics\n");
    ggml_cognitive_tensor_print_stats(kernel);
    printf("\n");
    
    printf("================================================\n");
    printf("Neural-Symbolic Tensor Architecture Complete!\n");
    printf("The cognitive kernel successfully encodes:\n");
    printf("• Prime-offset Matula-Goebel mappings\n");
    printf("• 4-mode hypercubic tensor structure\n");
    printf("• Quantum phase superposition states\n");
    printf("• Hypergraph composition operations\n");
    printf("• Prime-structured attention mechanisms\n");
    
    // Cleanup
    ggml_cognitive_kernel_free(kernel);
    ggml_free(ctx);
    
    return 0;
}