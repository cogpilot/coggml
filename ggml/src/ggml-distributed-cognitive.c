#include "ggml-distributed-cognitive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Generate unique IDs
static uint32_t generate_membrane_id(void) {
    static uint32_t counter = 1;
    return counter++;
}

static uint32_t generate_optimization_loop_id(void) {
    static uint32_t counter = 1;
    return counter++;
}

// Initialize distributed cognitive architecture
distributed_cognitive_architecture_t* distributed_cognitive_init(
    struct ggml_context* ctx,
    const char* endpoint) {
    
    distributed_cognitive_architecture_t* arch = malloc(sizeof(distributed_cognitive_architecture_t));
    if (!arch) return NULL;
    
    arch->ctx = ctx;
    arch->backend = NULL;  // Will be set by caller if needed
    
    // Initialize core systems
    arch->cogfluence = cogfluence_init(ctx);
    if (!arch->cogfluence) {
        free(arch);
        return NULL;
    }
    
    arch->atomspace = opencog_atomspace_init(ctx);
    if (!arch->atomspace) {
        cogfluence_free(arch->cogfluence);
        free(arch);
        return NULL;
    }
    
    arch->cognitive_kernel = ggml_cognitive_kernel_init(ctx, 16, 32, 32);
    if (!arch->cognitive_kernel) {
        cogfluence_free(arch->cogfluence);
        opencog_atomspace_free(arch->atomspace);
        free(arch);
        return NULL;
    }
    
    // Link systems together
    opencog_link_cogfluence(arch->atomspace, arch->cogfluence);
    
    // Initialize P-System membranes
    arch->membrane_capacity = DISTRIBUTED_COGNITIVE_MAX_MEMBRANES;
    arch->membranes = calloc(arch->membrane_capacity, sizeof(psystem_membrane_t));
    arch->membrane_count = 0;
    
    // Initialize dashboard
    arch->dashboard = calloc(1, sizeof(metacognitive_dashboard_t));
    arch->dashboard->history_capacity = 1000;
    arch->dashboard->performance_history = calloc(arch->dashboard->history_capacity, sizeof(float));
    arch->dashboard->history_length = 0;
    
    // Initialize optimization loops
    arch->optimization_loop_capacity = 16;
    arch->optimization_loops = calloc(arch->optimization_loop_capacity, sizeof(self_optimization_loop_t));
    arch->optimization_loop_count = 0;
    
    // Initialize system state
    arch->initialized = true;
    arch->self_optimization_active = false;
    arch->system_time = (uint64_t)time(NULL);
    
    // Initialize performance metrics
    arch->total_transductions = 0;
    arch->successful_transductions = 0;
    arch->system_efficiency = 0.0f;
    
    // Initialize network configuration
    strncpy(arch->endpoint, endpoint ? endpoint : "localhost:8080", sizeof(arch->endpoint) - 1);
    arch->endpoint[sizeof(arch->endpoint) - 1] = '\0';
    arch->agent_id = (uint32_t)time(NULL);
    
    printf("Distributed Cognitive Architecture initialized at %s (Agent ID: %u)\n",
           arch->endpoint, arch->agent_id);
    
    return arch;
}

// Free distributed cognitive architecture
void distributed_cognitive_free(distributed_cognitive_architecture_t* arch) {
    if (!arch) return;
    
    // Free core systems
    if (arch->cogfluence) cogfluence_free(arch->cogfluence);
    if (arch->atomspace) opencog_atomspace_free(arch->atomspace);
    if (arch->cognitive_kernel) ggml_cognitive_kernel_free(arch->cognitive_kernel);
    
    // Free membranes
    for (size_t i = 0; i < arch->membrane_count; i++) {
        if (arch->membranes[i].child_membranes) {
            free(arch->membranes[i].child_membranes);
        }
        if (arch->membranes[i].cogfluence_units) {
            free(arch->membranes[i].cogfluence_units);
        }
        if (arch->membranes[i].opencog_atoms) {
            free(arch->membranes[i].opencog_atoms);
        }
    }
    free(arch->membranes);
    
    // Free dashboard
    if (arch->dashboard) {
        if (arch->dashboard->performance_history) {
            free(arch->dashboard->performance_history);
        }
        if (arch->dashboard->activation_flows) {
            free(arch->dashboard->activation_flows);
        }
        if (arch->dashboard->membrane_depths) {
            free(arch->dashboard->membrane_depths);
        }
        free(arch->dashboard);
    }
    
    // Free optimization loops
    free(arch->optimization_loops);
    
    free(arch);
}

// Transduction: Cogfluence → OpenCog
bool transduction_cogfluence_to_opencog(
    distributed_cognitive_architecture_t* arch,
    uint64_t cogfluence_unit_id) {
    
    if (!arch || !arch->cogfluence || !arch->atomspace) return false;
    
    cogfluence_knowledge_unit_t* unit = cogfluence_get_knowledge_unit(arch->cogfluence, cogfluence_unit_id);
    if (!unit) return false;
    
    uint64_t atom_id = opencog_from_cogfluence_unit(arch->atomspace, unit);
    if (atom_id == 0) return false;
    
    arch->total_transductions++;
    arch->successful_transductions++;
    
    printf("Transduction Cogfluence→OpenCog: Unit '%s' → Atom %lu\n", unit->name, atom_id);
    
    return true;
}

// Transduction: OpenCog → GGML
bool transduction_opencog_to_ggml(
    distributed_cognitive_architecture_t* arch,
    uint64_t opencog_atom_id) {
    
    if (!arch || !arch->atomspace) return false;
    
    struct ggml_tensor* tensor = opencog_atom_to_tensor(arch->atomspace, opencog_atom_id);
    if (!tensor) return false;
    
    // Integrate with cognitive kernel
    // This is a simplified integration - in a full implementation,
    // this would involve complex tensor operations
    
    arch->total_transductions++;
    arch->successful_transductions++;
    
    printf("Transduction OpenCog→GGML: Atom %lu → Tensor [%ld]\n", 
           opencog_atom_id, tensor->ne[0]);
    
    return true;
}

// Transduction: GGML → Cogfluence
bool transduction_ggml_to_cogfluence(
    distributed_cognitive_architecture_t* arch,
    struct ggml_tensor* tensor,
    const char* unit_name) {
    
    if (!arch || !arch->cogfluence || !tensor || !unit_name) return false;
    
    cogfluence_knowledge_unit_t* unit = cogfluence_from_tensor(arch->cogfluence, tensor, unit_name);
    if (!unit) return false;
    
    arch->total_transductions++;
    arch->successful_transductions++;
    
    printf("Transduction GGML→Cogfluence: Tensor [%ld] → Unit '%s'\n", 
           tensor->ne[0], unit_name);
    
    return true;
}

// Full transduction pipeline
bool transduction_full_pipeline(
    distributed_cognitive_architecture_t* arch,
    const char* input_data,
    char* output_data,
    size_t output_size) {
    
    if (!arch || !input_data || !output_data) return false;
    
    printf("Running full transduction pipeline for input: '%s'\n", input_data);
    
    // Stage 1: Create Cogfluence knowledge unit
    float embedding[64];
    for (int i = 0; i < 64; i++) {
        embedding[i] = (float)((i + strlen(input_data)) % 256) / 255.0f;
    }
    
    struct ggml_tensor* input_tensor = ggml_new_tensor_1d(arch->ctx, GGML_TYPE_F32, 64);
    memcpy(input_tensor->data, embedding, sizeof(embedding));
    
    uint64_t cogfluence_unit_id = cogfluence_add_knowledge_unit(
        arch->cogfluence, input_data, COGFLUENCE_CONCEPT, input_tensor);
    
    if (cogfluence_unit_id == 0) return false;
    
    // Stage 2: Transduce to OpenCog
    if (!transduction_cogfluence_to_opencog(arch, cogfluence_unit_id)) {
        return false;
    }
    
    // Stage 3: Find corresponding atom and transduce to GGML
    cogfluence_knowledge_unit_t* unit = cogfluence_get_knowledge_unit(arch->cogfluence, cogfluence_unit_id);
    if (!unit) return false;
    
    uint64_t atom_id = unit->atomspace_id;  // Simple mapping for now
    
    if (!transduction_opencog_to_ggml(arch, atom_id)) {
        return false;
    }
    
    // Stage 4: Generate output
    snprintf(output_data, output_size, "Processed: %s (Cogfluence:%lu, OpenCog:%lu)", 
             input_data, cogfluence_unit_id, atom_id);
    
    printf("Full pipeline completed: %s\n", output_data);
    
    return true;
}

// Create P-System membrane
uint32_t psystem_create_membrane(
    distributed_cognitive_architecture_t* arch,
    const char* name,
    membrane_type_t type,
    uint32_t parent_id) {
    
    if (!arch || !name || arch->membrane_count >= arch->membrane_capacity) {
        return 0;
    }
    
    psystem_membrane_t* membrane = &arch->membranes[arch->membrane_count];
    uint32_t membrane_id = generate_membrane_id();
    
    // Initialize membrane
    membrane->membrane_id = membrane_id;
    strncpy(membrane->name, name, sizeof(membrane->name) - 1);
    membrane->name[sizeof(membrane->name) - 1] = '\0';
    membrane->type = type;
    membrane->parent_membrane_id = parent_id;
    
    // Initialize child membranes
    membrane->child_membranes = NULL;
    membrane->child_count = 0;
    membrane->child_capacity = 0;
    
    // Initialize contents
    membrane->cogfluence_units = NULL;
    membrane->cogfluence_unit_count = 0;
    membrane->opencog_atoms = NULL;
    membrane->opencog_atom_count = 0;
    
    // Initialize rules
    membrane->evolution_rules = ggml_new_tensor_2d(arch->ctx, GGML_TYPE_F32, 16, 16);
    membrane->communication_rules = ggml_new_tensor_2d(arch->ctx, GGML_TYPE_F32, 16, 16);
    ggml_set_zero(membrane->evolution_rules);
    ggml_set_zero(membrane->communication_rules);
    
    // Initialize state
    membrane->permeability = 0.5f;
    membrane->energy_level = 1.0f;
    membrane->active = true;
    
    // Initialize performance metrics
    membrane->evolution_cycles = 0;
    membrane->efficiency_score = 0.0f;
    
    arch->membrane_count++;
    
    printf("Created P-System membrane '%s' (ID %u, type %d)\n", name, membrane_id, type);
    
    return membrane_id;
}

// Update meta-cognitive dashboard
void dashboard_update(distributed_cognitive_architecture_t* arch) {
    if (!arch || !arch->dashboard) return;
    
    metacognitive_dashboard_t* dash = arch->dashboard;
    
    // Update global coherence
    dash->global_coherence = cogfluence_compute_coherence(arch->cogfluence);
    
    // Update cognitive load
    dash->cognitive_load = (float)arch->cogfluence->unit_count / COGFLUENCE_MAX_KNOWLEDGE_UNITS;
    
    // Update attention distribution (simplified)
    dash->attention_distribution[0] = 0.25f;  // Memory
    dash->attention_distribution[1] = 0.35f;  // Reasoning
    dash->attention_distribution[2] = 0.30f;  // Communication
    dash->attention_distribution[3] = 0.10f;  // Self-modification
    
    // Update performance metrics
    dash->total_operations = arch->total_transductions;
    dash->successful_operations = arch->successful_transductions;
    dash->success_rate = dash->total_operations > 0 ? 
        (float)dash->successful_operations / dash->total_operations : 0.0f;
    
    // Update network topology
    dash->active_agents = 1;  // This instance
    dash->active_workflows = arch->cogfluence->workflow_count;
    dash->active_membranes = arch->membrane_count;
    
    // Update performance history
    if (dash->history_length < dash->history_capacity) {
        dash->performance_history[dash->history_length++] = dash->success_rate;
    } else {
        // Shift history
        for (size_t i = 0; i < dash->history_capacity - 1; i++) {
            dash->performance_history[i] = dash->performance_history[i + 1];
        }
        dash->performance_history[dash->history_capacity - 1] = dash->success_rate;
    }
    
    printf("Dashboard updated: Coherence=%.2f, Load=%.2f, Success=%.2f\n",
           dash->global_coherence, dash->cognitive_load, dash->success_rate);
}

// Print meta-cognitive dashboard
void dashboard_print(distributed_cognitive_architecture_t* arch) {
    if (!arch || !arch->dashboard) return;
    
    metacognitive_dashboard_t* dash = arch->dashboard;
    
    printf("\n=== Meta-Cognitive Dashboard ===\n");
    printf("Global Coherence: %.2f\n", dash->global_coherence);
    printf("Cognitive Load: %.2f\n", dash->cognitive_load);
    printf("Success Rate: %.2f (%lu/%lu)\n", 
           dash->success_rate, dash->successful_operations, dash->total_operations);
    
    printf("\nAttention Distribution:\n");
    printf("  Memory: %.2f\n", dash->attention_distribution[0]);
    printf("  Reasoning: %.2f\n", dash->attention_distribution[1]);
    printf("  Communication: %.2f\n", dash->attention_distribution[2]);
    printf("  Self-modification: %.2f\n", dash->attention_distribution[3]);
    
    printf("\nNetwork Topology:\n");
    printf("  Active agents: %u\n", dash->active_agents);
    printf("  Active workflows: %u\n", dash->active_workflows);
    printf("  Active membranes: %u\n", dash->active_membranes);
    
    printf("\nTensor Statistics:\n");
    printf("  Memory usage: %.2f MB\n", dash->tensor_memory_usage);
    printf("  Computation load: %.2f\n", dash->tensor_computation_load);
    
    if (dash->history_length > 0) {
        printf("\nPerformance History (last 10):\n  ");
        size_t start = dash->history_length >= 10 ? dash->history_length - 10 : 0;
        for (size_t i = start; i < dash->history_length; i++) {
            printf("%.2f ", dash->performance_history[i]);
        }
        printf("\n");
    }
    
    printf("===============================\n");
}

// Create self-optimization loop
uint32_t optimization_create_loop(
    distributed_cognitive_architecture_t* arch,
    const char* target_system,
    const char* target_parameter,
    float initial_value,
    float target_value) {
    
    if (!arch || !target_system || !target_parameter || 
        arch->optimization_loop_count >= arch->optimization_loop_capacity) {
        return 0;
    }
    
    self_optimization_loop_t* loop = &arch->optimization_loops[arch->optimization_loop_count];
    uint32_t loop_id = generate_optimization_loop_id();
    
    // Initialize loop
    strncpy(loop->target_system, target_system, sizeof(loop->target_system) - 1);
    loop->target_system[sizeof(loop->target_system) - 1] = '\0';
    strncpy(loop->target_parameter, target_parameter, sizeof(loop->target_parameter) - 1);
    loop->target_parameter[sizeof(loop->target_parameter) - 1] = '\0';
    
    loop->current_value = initial_value;
    loop->target_value = target_value;
    loop->learning_rate = 0.01f;
    loop->momentum = 0.9f;
    
    loop->gradient = 0.0f;
    loop->previous_gradient = 0.0f;
    
    loop->baseline_performance = 0.0f;
    loop->current_performance = 0.0f;
    loop->optimization_cycles = 0;
    
    loop->min_value = initial_value * 0.1f;
    loop->max_value = initial_value * 10.0f;
    loop->converged = false;
    
    arch->optimization_loop_count++;
    
    printf("Created optimization loop for %s.%s (target: %.2f)\n",
           target_system, target_parameter, target_value);
    
    return loop_id;
}

// Update optimization loop
bool optimization_update_loop(
    distributed_cognitive_architecture_t* arch,
    uint32_t loop_id,
    float current_performance) {
    
    if (!arch || loop_id == 0 || loop_id > arch->optimization_loop_count) {
        return false;
    }
    
    self_optimization_loop_t* loop = &arch->optimization_loops[loop_id - 1];
    
    // Update performance
    if (loop->optimization_cycles == 0) {
        loop->baseline_performance = current_performance;
    }
    loop->current_performance = current_performance;
    
    // Compute gradient
    float performance_change = current_performance - loop->baseline_performance;
    loop->gradient = performance_change / (loop->current_value - loop->target_value + 1e-6f);
    
    // Apply momentum
    loop->gradient = loop->momentum * loop->previous_gradient + 
                    (1.0f - loop->momentum) * loop->gradient;
    
    // Update value
    float delta = loop->learning_rate * loop->gradient;
    loop->current_value += delta;
    
    // Apply constraints
    loop->current_value = fmaxf(loop->min_value, fminf(loop->max_value, loop->current_value));
    
    // Check convergence
    if (fabsf(loop->current_value - loop->target_value) < 0.01f) {
        loop->converged = true;
    }
    
    // Update for next iteration
    loop->previous_gradient = loop->gradient;
    loop->optimization_cycles++;
    
    printf("Optimization loop %u: value=%.3f, target=%.3f, performance=%.3f\n",
           loop_id, loop->current_value, loop->target_value, current_performance);
    
    return true;
}

// Run optimization cycle
bool optimization_run_cycle(distributed_cognitive_architecture_t* arch) {
    if (!arch || !arch->self_optimization_active) return false;
    
    bool any_updated = false;
    
    for (size_t i = 0; i < arch->optimization_loop_count; i++) {
        self_optimization_loop_t* loop = &arch->optimization_loops[i];
        
        if (loop->converged) continue;
        
        // Get current system performance
        float current_performance = dashboard_compute_coherence(arch);
        
        // Update the loop
        if (optimization_update_loop(arch, i + 1, current_performance)) {
            any_updated = true;
        }
    }
    
    return any_updated;
}

// Compute dashboard coherence
float dashboard_compute_coherence(distributed_cognitive_architecture_t* arch) {
    if (!arch) return 0.0f;
    
    float coherence = 0.0f;
    int components = 0;
    
    // Cogfluence coherence
    if (arch->cogfluence) {
        coherence += cogfluence_compute_coherence(arch->cogfluence);
        components++;
    }
    
    // OpenCog coherence (simplified)
    if (arch->atomspace) {
        float avg_truth = 0.0f;
        int atom_count = 0;
        
        for (size_t i = 0; i < arch->atomspace->atom_count; i++) {
            if (!arch->atomspace->atoms[i].is_deleted) {
                avg_truth += arch->atomspace->atoms[i].truth_value.strength;
                atom_count++;
            }
        }
        
        if (atom_count > 0) {
            coherence += avg_truth / atom_count;
            components++;
        }
    }
    
    // GGML coherence (simplified)
    if (arch->cognitive_kernel) {
        coherence += 0.7f;  // Simplified metric
        components++;
    }
    
    return components > 0 ? coherence / components : 0.0f;
}

// Print architecture overview
void distributed_cognitive_print_architecture(distributed_cognitive_architecture_t* arch) {
    if (!arch) return;
    
    printf("\n=== Distributed Cognitive Architecture ===\n");
    printf("Endpoint: %s (Agent ID: %u)\n", arch->endpoint, arch->agent_id);
    printf("Initialized: %s\n", arch->initialized ? "Yes" : "No");
    printf("Self-optimization: %s\n", arch->self_optimization_active ? "Active" : "Inactive");
    printf("System time: %lu\n", arch->system_time);
    
    printf("\nCore Systems:\n");
    if (arch->cogfluence) {
        printf("  Cogfluence: %zu knowledge units\n", arch->cogfluence->unit_count);
    }
    if (arch->atomspace) {
        printf("  OpenCog: %zu atoms\n", arch->atomspace->atom_count);
    }
    if (arch->cognitive_kernel) {
        printf("  GGML: Cognitive kernel initialized\n");
    }
    
    printf("\nP-System Membranes: %zu\n", arch->membrane_count);
    for (size_t i = 0; i < arch->membrane_count; i++) {
        printf("  %s (ID %u, type %d)\n", 
               arch->membranes[i].name, arch->membranes[i].membrane_id, arch->membranes[i].type);
    }
    
    printf("\nOptimization Loops: %zu\n", arch->optimization_loop_count);
    for (size_t i = 0; i < arch->optimization_loop_count; i++) {
        printf("  %s.%s: %.3f → %.3f %s\n",
               arch->optimization_loops[i].target_system,
               arch->optimization_loops[i].target_parameter,
               arch->optimization_loops[i].current_value,
               arch->optimization_loops[i].target_value,
               arch->optimization_loops[i].converged ? "(converged)" : "");
    }
    
    printf("\nPerformance Metrics:\n");
    printf("  Transductions: %lu/%lu (%.2f%%)\n",
           arch->successful_transductions, arch->total_transductions,
           arch->total_transductions > 0 ? 
           100.0f * arch->successful_transductions / arch->total_transductions : 0.0f);
    printf("  System efficiency: %.2f\n", arch->system_efficiency);
    
    printf("=========================================\n");
}

// Run basic test suite
bool distributed_cognitive_run_test_suite(distributed_cognitive_architecture_t* arch) {
    if (!arch) return false;
    
    printf("\n=== Running Distributed Cognitive Test Suite ===\n");
    
    bool all_passed = true;
    
    // Test 1: Basic transduction pipeline
    printf("Test 1: Basic transduction pipeline... ");
    char output[256];
    if (transduction_full_pipeline(arch, "test_concept", output, sizeof(output))) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
        all_passed = false;
    }
    
    // Test 2: P-System membrane creation
    printf("Test 2: P-System membrane creation... ");
    uint32_t membrane_id = psystem_create_membrane(arch, "test_membrane", MEMBRANE_ELEMENTARY, 0);
    if (membrane_id > 0) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
        all_passed = false;
    }
    
    // Test 3: Dashboard update
    printf("Test 3: Dashboard update... ");
    dashboard_update(arch);
    if (arch->dashboard->global_coherence >= 0.0f) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
        all_passed = false;
    }
    
    // Test 4: Optimization loop
    printf("Test 4: Optimization loop... ");
    uint32_t loop_id = optimization_create_loop(arch, "test_system", "test_param", 1.0f, 2.0f);
    if (loop_id > 0) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
        all_passed = false;
    }
    
    // Test 5: System coherence
    printf("Test 5: System coherence... ");
    float coherence = dashboard_compute_coherence(arch);
    if (coherence >= 0.0f && coherence <= 1.0f) {
        printf("PASS (%.2f)\n", coherence);
    } else {
        printf("FAIL (%.2f)\n", coherence);
        all_passed = false;
    }
    
    printf("===============================================\n");
    printf("Test Suite Result: %s\n", all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    
    return all_passed;
}

// Phase 2: Enhanced Distributed Communication Functions

// Enhanced cognitive message packet for Phase 2
typedef struct {
    uint64_t source_agent_id;
    uint64_t target_agent_id;
    uint32_t message_type;
    float attention_weight;
    float salience_score;
    uint32_t priority_level;
    char cognitive_context[512];
    
    // PLN reasoning payload
    opencog_truth_value_t truth_value;
    uint32_t reasoning_depth;
    
    // MOSES optimization payload
    float fitness_score;
    uint32_t generation_id;
    
    // Tensor payload
    struct ggml_tensor* tensor_data;
    size_t tensor_size;
    
    // Routing metadata
    uint32_t hop_count;
    uint64_t timestamp;
    char routing_path[256];
} enhanced_cognitive_message_t;

// Network topology node
typedef struct network_node {
    uint64_t agent_id;
    char endpoint[256];
    float reliability_score;
    float response_time;
    uint32_t connection_count;
    bool is_active;
    
    // Cognitive specialization
    float memory_capacity;
    float reasoning_capability;
    float attention_allocation;
    
    struct network_node* next;
} network_node_t;

// Enhanced cognitive network
typedef struct {
    network_node_t* nodes;
    size_t node_count;
    float network_coherence;
    float communication_efficiency;
    
    // Routing tables
    uint64_t** routing_matrix;
    float** distance_matrix;
    
    // Network resilience
    float fault_tolerance;
    uint32_t redundancy_level;
} enhanced_cognitive_network_t;

// Initialize enhanced distributed communication
enhanced_cognitive_network_t* enhanced_network_init(void) {
    enhanced_cognitive_network_t* network = malloc(sizeof(enhanced_cognitive_network_t));
    if (!network) return NULL;
    
    network->nodes = NULL;
    network->node_count = 0;
    network->network_coherence = 0.0f;
    network->communication_efficiency = 0.0f;
    
    // Initialize routing structures
    network->routing_matrix = NULL;
    network->distance_matrix = NULL;
    
    network->fault_tolerance = 0.8f;      // 80% fault tolerance
    network->redundancy_level = 2;        // 2-hop redundancy
    
    printf("Enhanced cognitive network initialized with fault tolerance\n");
    
    return network;
}

// Add agent to network topology
bool enhanced_network_add_agent(
    enhanced_cognitive_network_t* network,
    uint64_t agent_id,
    const char* endpoint,
    float memory_capacity,
    float reasoning_capability) {
    
    if (!network || !endpoint) return false;
    
    network_node_t* new_node = malloc(sizeof(network_node_t));
    if (!new_node) return false;
    
    new_node->agent_id = agent_id;
    strncpy(new_node->endpoint, endpoint, sizeof(new_node->endpoint) - 1);
    new_node->endpoint[sizeof(new_node->endpoint) - 1] = '\0';
    
    new_node->reliability_score = 1.0f;
    new_node->response_time = 0.1f;       // 100ms default
    new_node->connection_count = 0;
    new_node->is_active = true;
    
    new_node->memory_capacity = memory_capacity;
    new_node->reasoning_capability = reasoning_capability;
    new_node->attention_allocation = 0.5f;
    
    // Add to linked list
    new_node->next = network->nodes;
    network->nodes = new_node;
    network->node_count++;
    
    printf("Added agent %lu to network topology (%s)\n", agent_id, endpoint);
    
    return true;
}

// Discover agents in network using cognitive capabilities
uint64_t* enhanced_network_discover_agents(
    enhanced_cognitive_network_t* network,
    float min_memory_capacity,
    float min_reasoning_capability,
    size_t* result_count) {
    
    if (!network || !result_count) return NULL;
    
    *result_count = 0;
    
    // Count matching agents
    size_t count = 0;
    network_node_t* current = network->nodes;
    while (current) {
        if (current->is_active &&
            current->memory_capacity >= min_memory_capacity &&
            current->reasoning_capability >= min_reasoning_capability) {
            count++;
        }
        current = current->next;
    }
    
    if (count == 0) return NULL;
    
    uint64_t* results = malloc(count * sizeof(uint64_t));
    if (!results) return NULL;
    
    size_t idx = 0;
    current = network->nodes;
    while (current && idx < count) {
        if (current->is_active &&
            current->memory_capacity >= min_memory_capacity &&
            current->reasoning_capability >= min_reasoning_capability) {
            results[idx++] = current->agent_id;
        }
        current = current->next;
    }
    
    *result_count = count;
    
    printf("Discovered %zu agents with memory>=%.1f, reasoning>=%.1f\n", 
           count, min_memory_capacity, min_reasoning_capability);
    
    return results;
}

// Route cognitive message with attention-based priority
bool enhanced_network_route_message(
    enhanced_cognitive_network_t* network,
    enhanced_cognitive_message_t* message) {
    
    if (!network || !message) return false;
    
    // Find target node
    network_node_t* target = network->nodes;
    while (target && target->agent_id != message->target_agent_id) {
        target = target->next;
    }
    
    if (!target || !target->is_active) {
        printf("Target agent %lu not found or inactive\n", message->target_agent_id);
        return false;
    }
    
    // Attention-based routing priority
    float routing_priority = message->attention_weight * message->salience_score;
    
    // Update routing metadata
    message->hop_count++;
    message->timestamp = (uint64_t)time(NULL);
    
    // Simulate message delivery based on attention priority
    float delivery_probability = target->reliability_score * (0.5f + routing_priority * 0.5f);
    
    if ((float)rand() / RAND_MAX < delivery_probability) {
        printf("Message routed successfully from %lu to %lu (priority: %.2f)\n",
               message->source_agent_id, message->target_agent_id, routing_priority);
        
        // Update network statistics
        network->communication_efficiency = 
            network->communication_efficiency * 0.9f + delivery_probability * 0.1f;
        
        return true;
    } else {
        printf("Message routing failed from %lu to %lu\n", 
               message->source_agent_id, message->target_agent_id);
        return false;
    }
}

// Coordinate distributed reasoning across network
bool enhanced_network_coordinate_reasoning(
    enhanced_cognitive_network_t* network,
    const char* reasoning_task,
    uint64_t coordinator_agent_id) {
    
    if (!network || !reasoning_task) return false;
    
    printf("Coordinating distributed reasoning: '%s' (coordinator: %lu)\n", 
           reasoning_task, coordinator_agent_id);
    
    // Find agents with high reasoning capability
    size_t agent_count;
    uint64_t* capable_agents = enhanced_network_discover_agents(
        network, 0.3f, 0.7f, &agent_count);
    
    if (!capable_agents || agent_count == 0) {
        printf("No capable agents found for reasoning coordination\n");
        return false;
    }
    
    // Distribute reasoning tasks
    for (size_t i = 0; i < agent_count; i++) {
        enhanced_cognitive_message_t message = {0};
        message.source_agent_id = coordinator_agent_id;
        message.target_agent_id = capable_agents[i];
        message.message_type = 3;  // Reasoning task
        message.attention_weight = 0.8f;
        message.salience_score = 0.9f;
        message.priority_level = 1;  // High priority
        message.reasoning_depth = 2;
        
        snprintf(message.cognitive_context, sizeof(message.cognitive_context),
                "REASONING_TASK: %s (subtask %zu/%zu)", reasoning_task, i + 1, agent_count);
        
        bool sent = enhanced_network_route_message(network, &message);
        if (sent) {
            printf("  Subtask %zu assigned to agent %lu\n", i + 1, capable_agents[i]);
        }
    }
    
    free(capable_agents);
    
    printf("Distributed reasoning coordination completed\n");
    
    return true;
}

// Network resilience and fault tolerance
bool enhanced_network_handle_failure(
    enhanced_cognitive_network_t* network,
    uint64_t failed_agent_id) {
    
    if (!network) return false;
    
    // Find and mark agent as inactive
    network_node_t* current = network->nodes;
    while (current) {
        if (current->agent_id == failed_agent_id) {
            current->is_active = false;
            current->reliability_score *= 0.5f;  // Reduce reliability
            
            printf("Handling failure of agent %lu\n", failed_agent_id);
            
            // Attempt to redistribute load to other agents
            size_t active_count;
            uint64_t* active_agents = enhanced_network_discover_agents(
                network, 0.2f, 0.2f, &active_count);
            
            if (active_agents && active_count > 0) {
                printf("Redistributing load to %zu active agents\n", active_count);
                
                // Update attention allocation for remaining agents
                for (size_t i = 0; i < active_count; i++) {
                    network_node_t* node = network->nodes;
                    while (node && node->agent_id != active_agents[i]) {
                        node = node->next;
                    }
                    if (node) {
                        node->attention_allocation += 0.1f;  // Increase attention
                        node->attention_allocation = fminf(1.0f, node->attention_allocation);
                    }
                }
                
                free(active_agents);
                return true;
            } else {
                printf("WARNING: No active agents available for load redistribution\n");
                return false;
            }
        }
        current = current->next;
    }
    
    printf("Agent %lu not found in network\n", failed_agent_id);
    return false;
}

// Calculate network coherence metrics
float enhanced_network_calculate_coherence(enhanced_cognitive_network_t* network) {
    if (!network || network->node_count == 0) return 0.0f;
    
    float total_reliability = 0.0f;
    float total_reasoning = 0.0f;
    size_t active_count = 0;
    
    network_node_t* current = network->nodes;
    while (current) {
        if (current->is_active) {
            total_reliability += current->reliability_score;
            total_reasoning += current->reasoning_capability;
            active_count++;
        }
        current = current->next;
    }
    
    if (active_count == 0) return 0.0f;
    
    float avg_reliability = total_reliability / active_count;
    float avg_reasoning = total_reasoning / active_count;
    
    // Network coherence combines reliability, reasoning capability, and communication efficiency
    network->network_coherence = (avg_reliability + avg_reasoning + network->communication_efficiency) / 3.0f;
    
    return network->network_coherence;
}

// Print enhanced network statistics
void enhanced_network_print_stats(enhanced_cognitive_network_t* network) {
    if (!network) return;
    
    printf("\n=== Enhanced Cognitive Network Statistics ===\n");
    printf("Total nodes: %zu\n", network->node_count);
    
    size_t active_count = 0;
    float total_memory = 0.0f;
    float total_reasoning = 0.0f;
    
    network_node_t* current = network->nodes;
    while (current) {
        if (current->is_active) {
            active_count++;
            total_memory += current->memory_capacity;
            total_reasoning += current->reasoning_capability;
        }
        current = current->next;
    }
    
    printf("Active nodes: %zu\n", active_count);
    printf("Network coherence: %.3f\n", network->network_coherence);
    printf("Communication efficiency: %.3f\n", network->communication_efficiency);
    printf("Fault tolerance: %.1f%%\n", network->fault_tolerance * 100);
    
    if (active_count > 0) {
        printf("Average memory capacity: %.3f\n", total_memory / active_count);
        printf("Average reasoning capability: %.3f\n", total_reasoning / active_count);
    }
    
    printf("============================================\n");
}