#include "ggml-cogfluence.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Generate unique ID for knowledge units
static uint64_t generate_unit_id(void) {
    static uint64_t counter = 1;
    return counter++;
}

// Initialize Cogfluence system
cogfluence_system_t* cogfluence_init(struct ggml_context* ctx) {
    cogfluence_system_t* system = malloc(sizeof(cogfluence_system_t));
    if (!system) return NULL;
    
    system->ctx = ctx;
    
    // Initialize knowledge units
    system->unit_capacity = COGFLUENCE_MAX_KNOWLEDGE_UNITS;
    system->knowledge_units = calloc(system->unit_capacity, sizeof(cogfluence_knowledge_unit_t));
    system->unit_count = 0;
    
    // Initialize workflows
    system->workflow_capacity = COGFLUENCE_MAX_WORKFLOWS;
    system->workflows = calloc(system->workflow_capacity, sizeof(cogfluence_workflow_t));
    system->workflow_count = 0;
    
    // Initialize system state
    system->initialized = true;
    system->global_activation = 0.5f;
    system->system_time = (uint64_t)time(NULL);
    
    // Initialize performance tracking
    system->total_inferences = 0;
    system->successful_workflows = 0;
    system->system_coherence = 0.0f;
    
    printf("Cogfluence system initialized with capacity for %zu knowledge units and %zu workflows\n",
           system->unit_capacity, system->workflow_capacity);
    
    return system;
}

// Free Cogfluence system
void cogfluence_free(cogfluence_system_t* system) {
    if (!system) return;
    
    // Free knowledge units
    for (size_t i = 0; i < system->unit_count; i++) {
        if (system->knowledge_units[i].related_units) {
            free(system->knowledge_units[i].related_units);
        }
    }
    free(system->knowledge_units);
    
    // Free workflows
    for (size_t i = 0; i < system->workflow_count; i++) {
        if (system->workflows[i].step_units) {
            free(system->workflows[i].step_units);
        }
    }
    free(system->workflows);
    
    free(system);
}

// Add knowledge unit to system
uint64_t cogfluence_add_knowledge_unit(
    cogfluence_system_t* system,
    const char* name,
    cogfluence_unit_type_t type,
    struct ggml_tensor* embedding) {
    
    if (!system || !name || system->unit_count >= system->unit_capacity) {
        return 0;
    }
    
    cogfluence_knowledge_unit_t* unit = &system->knowledge_units[system->unit_count];
    uint64_t unit_id = generate_unit_id();
    
    // Initialize unit
    strncpy(unit->name, name, COGFLUENCE_MAX_CONCEPT_NAME - 1);
    unit->name[COGFLUENCE_MAX_CONCEPT_NAME - 1] = '\0';
    unit->type = type;
    unit->embedding = embedding;
    
    // Create tensor encoding if embedding provided
    if (embedding) {
        unit->tensor_encoding = ggml_dup(system->ctx, embedding);
    } else {
        // Create default tensor encoding
        unit->tensor_encoding = ggml_new_tensor_1d(system->ctx, GGML_TYPE_F32, 64);
        ggml_set_zero(unit->tensor_encoding);
    }
    
    // Initialize OpenCog mapping
    unit->atomspace_id = unit_id;  // Simple mapping for now
    unit->truth_value = 0.8f;      // Default truth value
    unit->confidence = 0.7f;       // Default confidence
    
    // Initialize metadata
    unit->creation_time = (uint64_t)time(NULL);
    unit->last_modified = unit->creation_time;
    unit->activation_level = system->global_activation;
    unit->attention_value = 0.5f;
    
    // Initialize relations
    unit->related_units = NULL;
    unit->relation_count = 0;
    unit->relation_capacity = 0;
    
    system->unit_count++;
    
    printf("Added knowledge unit '%s' (type %d, ID %lu)\n", name, type, unit_id);
    
    return unit_id;
}

// Get knowledge unit by ID
cogfluence_knowledge_unit_t* cogfluence_get_knowledge_unit(
    cogfluence_system_t* system,
    uint64_t unit_id) {
    
    if (!system || unit_id == 0) return NULL;
    
    // Simple linear search for now (could be optimized with hash table)
    for (size_t i = 0; i < system->unit_count; i++) {
        if (system->knowledge_units[i].atomspace_id == unit_id) {
            return &system->knowledge_units[i];
        }
    }
    
    return NULL;
}

// Add relation between two knowledge units
bool cogfluence_add_relation(
    cogfluence_system_t* system,
    uint64_t unit1_id,
    uint64_t unit2_id) {
    
    if (!system || unit1_id == 0 || unit2_id == 0 || unit1_id == unit2_id) {
        return false;
    }
    
    cogfluence_knowledge_unit_t* unit1 = cogfluence_get_knowledge_unit(system, unit1_id);
    cogfluence_knowledge_unit_t* unit2 = cogfluence_get_knowledge_unit(system, unit2_id);
    
    if (!unit1 || !unit2) return false;
    
    // Add relation from unit1 to unit2
    if (unit1->relation_count >= unit1->relation_capacity) {
        unit1->relation_capacity = unit1->relation_capacity == 0 ? 4 : unit1->relation_capacity * 2;
        unit1->related_units = realloc(unit1->related_units, 
                                     unit1->relation_capacity * sizeof(uint64_t));
    }
    
    unit1->related_units[unit1->relation_count++] = unit2_id;
    
    // Add bidirectional relation from unit2 to unit1
    if (unit2->relation_count >= unit2->relation_capacity) {
        unit2->relation_capacity = unit2->relation_capacity == 0 ? 4 : unit2->relation_capacity * 2;
        unit2->related_units = realloc(unit2->related_units, 
                                     unit2->relation_capacity * sizeof(uint64_t));
    }
    
    unit2->related_units[unit2->relation_count++] = unit1_id;
    
    printf("Added relation between units %lu and %lu\n", unit1_id, unit2_id);
    
    return true;
}

// Compute similarity between two knowledge units
float cogfluence_compute_similarity(
    cogfluence_knowledge_unit_t* unit1,
    cogfluence_knowledge_unit_t* unit2) {
    
    if (!unit1 || !unit2) return 0.0f;
    
    // Simple tensor similarity using dot product
    if (unit1->tensor_encoding && unit2->tensor_encoding) {
        // Simplified similarity calculation
        if (unit1->tensor_encoding->ne[0] == unit2->tensor_encoding->ne[0] &&
            unit1->tensor_encoding->type == GGML_TYPE_F32 &&
            unit2->tensor_encoding->type == GGML_TYPE_F32) {
            
            float* data1 = (float*)unit1->tensor_encoding->data;
            float* data2 = (float*)unit2->tensor_encoding->data;
            
            float dot_product = 0.0f;
            float norm1 = 0.0f, norm2 = 0.0f;
            
            for (int i = 0; i < unit1->tensor_encoding->ne[0]; i++) {
                dot_product += data1[i] * data2[i];
                norm1 += data1[i] * data1[i];
                norm2 += data2[i] * data2[i];
            }
            
            if (norm1 > 0 && norm2 > 0) {
                return dot_product / (sqrtf(norm1) * sqrtf(norm2));
            }
        }
    }
    
    // Fallback to type-based similarity
    if (unit1->type == unit2->type) {
        return 0.5f;
    }
    
    return 0.1f;
}

// Create workflow
uint64_t cogfluence_create_workflow(
    cogfluence_system_t* system,
    const char* name) {
    
    if (!system || !name || system->workflow_count >= system->workflow_capacity) {
        return 0;
    }
    
    cogfluence_workflow_t* workflow = &system->workflows[system->workflow_count];
    uint64_t workflow_id = generate_unit_id();
    
    // Initialize workflow
    strncpy(workflow->name, name, COGFLUENCE_MAX_CONCEPT_NAME - 1);
    workflow->name[COGFLUENCE_MAX_CONCEPT_NAME - 1] = '\0';
    workflow->workflow_id = workflow_id;
    
    // Initialize steps
    workflow->step_units = NULL;
    workflow->step_count = 0;
    workflow->step_capacity = 0;
    
    // Initialize execution state
    workflow->active = false;
    workflow->current_step = 0;
    workflow->completion_ratio = 0.0f;
    
    // Initialize performance metrics
    workflow->success_rate = 0.0f;
    workflow->efficiency_score = 0.0f;
    workflow->execution_count = 0;
    
    system->workflow_count++;
    
    printf("Created workflow '%s' (ID %lu)\n", name, workflow_id);
    
    return workflow_id;
}

// Add step to workflow
bool cogfluence_add_workflow_step(
    cogfluence_system_t* system,
    uint64_t workflow_id,
    uint64_t unit_id) {
    
    if (!system || workflow_id == 0 || unit_id == 0) return false;
    
    // Find workflow
    cogfluence_workflow_t* workflow = NULL;
    for (size_t i = 0; i < system->workflow_count; i++) {
        if (system->workflows[i].workflow_id == workflow_id) {
            workflow = &system->workflows[i];
            break;
        }
    }
    
    if (!workflow) return false;
    
    // Verify unit exists
    if (!cogfluence_get_knowledge_unit(system, unit_id)) return false;
    
    // Add step
    if (workflow->step_count >= workflow->step_capacity) {
        workflow->step_capacity = workflow->step_capacity == 0 ? 4 : workflow->step_capacity * 2;
        workflow->step_units = realloc(workflow->step_units, 
                                     workflow->step_capacity * sizeof(uint64_t));
    }
    
    workflow->step_units[workflow->step_count++] = unit_id;
    
    printf("Added step (unit %lu) to workflow %lu\n", unit_id, workflow_id);
    
    return true;
}

// Execute workflow
bool cogfluence_execute_workflow(
    cogfluence_system_t* system,
    uint64_t workflow_id) {
    
    if (!system || workflow_id == 0) return false;
    
    // Find workflow
    cogfluence_workflow_t* workflow = NULL;
    for (size_t i = 0; i < system->workflow_count; i++) {
        if (system->workflows[i].workflow_id == workflow_id) {
            workflow = &system->workflows[i];
            break;
        }
    }
    
    if (!workflow || workflow->step_count == 0) return false;
    
    printf("Executing workflow '%s' with %zu steps\n", workflow->name, workflow->step_count);
    
    workflow->active = true;
    workflow->current_step = 0;
    
    // Execute each step
    for (size_t step = 0; step < workflow->step_count; step++) {
        workflow->current_step = step;
        workflow->completion_ratio = (float)step / workflow->step_count;
        
        uint64_t unit_id = workflow->step_units[step];
        cogfluence_knowledge_unit_t* unit = cogfluence_get_knowledge_unit(system, unit_id);
        
        if (unit) {
            // Simulate step execution
            unit->activation_level = fminf(unit->activation_level + 0.1f, 1.0f);
            unit->attention_value = fminf(unit->attention_value + 0.05f, 1.0f);
            unit->last_modified = (uint64_t)time(NULL);
            
            printf("  Step %zu: Executed unit '%s' (activation: %.2f)\n", 
                   step, unit->name, unit->activation_level);
        }
        
        system->total_inferences++;
    }
    
    workflow->completion_ratio = 1.0f;
    workflow->active = false;
    workflow->execution_count++;
    
    // Update performance metrics
    workflow->success_rate = (float)(workflow->execution_count - 1) / workflow->execution_count * 
                           workflow->success_rate + 1.0f / workflow->execution_count;
    workflow->efficiency_score = fminf(workflow->efficiency_score + 0.1f, 1.0f);
    
    system->successful_workflows++;
    
    printf("Workflow '%s' completed successfully (executions: %lu)\n", 
           workflow->name, workflow->execution_count);
    
    return true;
}

// Convert knowledge unit to tensor
struct ggml_tensor* cogfluence_to_tensor(
    cogfluence_knowledge_unit_t* unit,
    struct ggml_context* ctx) {
    
    if (!unit || !ctx) return NULL;
    
    if (unit->tensor_encoding) {
        return ggml_dup(ctx, unit->tensor_encoding);
    }
    
    // Create default tensor
    struct ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_set_zero(tensor);
    
    return tensor;
}

// Create knowledge unit from tensor
cogfluence_knowledge_unit_t* cogfluence_from_tensor(
    cogfluence_system_t* system,
    struct ggml_tensor* tensor,
    const char* name) {
    
    if (!system || !tensor || !name) return NULL;
    
    uint64_t unit_id = cogfluence_add_knowledge_unit(system, name, COGFLUENCE_CONCEPT, tensor);
    
    return cogfluence_get_knowledge_unit(system, unit_id);
}

// Print system statistics
void cogfluence_print_statistics(cogfluence_system_t* system) {
    if (!system) return;
    
    printf("\n=== Cogfluence System Statistics ===\n");
    printf("Knowledge Units: %zu/%zu\n", system->unit_count, system->unit_capacity);
    printf("Workflows: %zu/%zu\n", system->workflow_count, system->workflow_capacity);
    printf("Total Inferences: %lu\n", system->total_inferences);
    printf("Successful Workflows: %lu\n", system->successful_workflows);
    printf("Global Activation: %.2f\n", system->global_activation);
    printf("System Coherence: %.2f\n", system->system_coherence);
    
    // Print knowledge unit types
    int type_counts[6] = {0};
    for (size_t i = 0; i < system->unit_count; i++) {
        if (system->knowledge_units[i].type >= 1 && system->knowledge_units[i].type <= 5) {
            type_counts[system->knowledge_units[i].type]++;
        }
    }
    
    printf("Unit Types:\n");
    printf("  Concepts: %d\n", type_counts[COGFLUENCE_CONCEPT]);
    printf("  Relations: %d\n", type_counts[COGFLUENCE_RELATION]);
    printf("  Workflows: %d\n", type_counts[COGFLUENCE_WORKFLOW]);
    printf("  Rules: %d\n", type_counts[COGFLUENCE_RULE]);
    printf("  Patterns: %d\n", type_counts[COGFLUENCE_PATTERN]);
    printf("=====================================\n");
}

// Compute system coherence
float cogfluence_compute_coherence(cogfluence_system_t* system) {
    if (!system || system->unit_count == 0) return 0.0f;
    
    float total_coherence = 0.0f;
    int coherence_count = 0;
    
    // Compute average pairwise similarity
    for (size_t i = 0; i < system->unit_count; i++) {
        for (size_t j = i + 1; j < system->unit_count; j++) {
            float similarity = cogfluence_compute_similarity(
                &system->knowledge_units[i], 
                &system->knowledge_units[j]);
            total_coherence += similarity;
            coherence_count++;
        }
    }
    
    if (coherence_count > 0) {
        system->system_coherence = total_coherence / coherence_count;
    }
    
    return system->system_coherence;
}

// Update activation levels
void cogfluence_update_activations(cogfluence_system_t* system) {
    if (!system) return;
    
    float decay_rate = 0.95f;
    float boost_factor = 1.05f;
    
    for (size_t i = 0; i < system->unit_count; i++) {
        cogfluence_knowledge_unit_t* unit = &system->knowledge_units[i];
        
        // Decay activation
        unit->activation_level *= decay_rate;
        
        // Boost based on relations
        if (unit->relation_count > 0) {
            unit->activation_level *= boost_factor;
        }
        
        // Clamp to valid range
        unit->activation_level = fmaxf(0.0f, fminf(1.0f, unit->activation_level));
        
        // Update attention value based on activation
        unit->attention_value = unit->activation_level * 0.8f + unit->attention_value * 0.2f;
    }
    
    system->system_time++;
}