#pragma once

//
// Cogfluence Knowledge Units Integration
//
// This header defines the Cogfluence knowledge representation system
// that integrates with OpenCog AtomSpace and GGML tensor operations
// to create a unified cognitive architecture.
//

#include "ggml.h"
#include "ggml-cognitive-tensor.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum limits for Cogfluence structures
#define COGFLUENCE_MAX_KNOWLEDGE_UNITS 1024
#define COGFLUENCE_MAX_WORKFLOWS 64
#define COGFLUENCE_MAX_CONCEPT_NAME 128
#define COGFLUENCE_MAX_RELATIONS 256

// Cogfluence knowledge unit types
typedef enum {
    COGFLUENCE_CONCEPT = 1,
    COGFLUENCE_RELATION = 2,
    COGFLUENCE_WORKFLOW = 3,
    COGFLUENCE_RULE = 4,
    COGFLUENCE_PATTERN = 5
} cogfluence_unit_type_t;

// Cogfluence knowledge unit structure
typedef struct {
    char name[COGFLUENCE_MAX_CONCEPT_NAME];
    cogfluence_unit_type_t type;
    
    // Semantic representation
    struct ggml_tensor* embedding;          // Vector embedding
    struct ggml_tensor* tensor_encoding;    // GGML tensor representation
    
    // OpenCog mapping
    uint64_t atomspace_id;                  // AtomSpace node/link ID
    float truth_value;                      // PLN truth value
    float confidence;                       // PLN confidence
    
    // Metadata
    uint64_t creation_time;
    uint64_t last_modified;
    float activation_level;                 // ECAN activation
    float attention_value;                  // ECAN attention value
    
    // Relations
    uint64_t* related_units;                // Array of related unit IDs
    size_t relation_count;
    size_t relation_capacity;
} cogfluence_knowledge_unit_t;

// Cogfluence workflow structure
typedef struct {
    char name[COGFLUENCE_MAX_CONCEPT_NAME];
    uint64_t workflow_id;
    
    // Workflow steps
    uint64_t* step_units;                   // Knowledge units in workflow
    size_t step_count;
    size_t step_capacity;
    
    // Execution state
    bool active;
    size_t current_step;
    float completion_ratio;
    
    // Performance metrics
    float success_rate;
    float efficiency_score;
    uint64_t execution_count;
} cogfluence_workflow_t;

// Cogfluence system structure
typedef struct {
    struct ggml_context* ctx;
    
    // Knowledge base
    cogfluence_knowledge_unit_t* knowledge_units;
    size_t unit_count;
    size_t unit_capacity;
    
    // Workflow system
    cogfluence_workflow_t* workflows;
    size_t workflow_count;
    size_t workflow_capacity;
    
    // System state
    bool initialized;
    float global_activation;
    uint64_t system_time;
    
    // Performance tracking
    uint64_t total_inferences;
    uint64_t successful_workflows;
    float system_coherence;
} cogfluence_system_t;

// Serialization format for inter-system communication
typedef struct {
    uint32_t magic;                         // Format identifier
    uint32_t version;                       // Version number
    uint32_t unit_count;                    // Number of knowledge units
    uint32_t workflow_count;                // Number of workflows
    uint32_t tensor_count;                  // Number of tensors
    uint32_t checksum;                      // Data integrity check
    
    // Variable-length data follows
    // - Knowledge units
    // - Workflows
    // - Tensor data
} cogfluence_serialization_header_t;

// Core Cogfluence functions
GGML_API cogfluence_system_t* cogfluence_init(struct ggml_context* ctx);
GGML_API void cogfluence_free(cogfluence_system_t* system);

// Knowledge unit management
GGML_API uint64_t cogfluence_add_knowledge_unit(
    cogfluence_system_t* system,
    const char* name,
    cogfluence_unit_type_t type,
    struct ggml_tensor* embedding);

GGML_API cogfluence_knowledge_unit_t* cogfluence_get_knowledge_unit(
    cogfluence_system_t* system,
    uint64_t unit_id);

GGML_API bool cogfluence_add_relation(
    cogfluence_system_t* system,
    uint64_t unit1_id,
    uint64_t unit2_id);

GGML_API float cogfluence_compute_similarity(
    cogfluence_knowledge_unit_t* unit1,
    cogfluence_knowledge_unit_t* unit2);

// Workflow management
GGML_API uint64_t cogfluence_create_workflow(
    cogfluence_system_t* system,
    const char* name);

GGML_API bool cogfluence_add_workflow_step(
    cogfluence_system_t* system,
    uint64_t workflow_id,
    uint64_t unit_id);

GGML_API bool cogfluence_execute_workflow(
    cogfluence_system_t* system,
    uint64_t workflow_id);

// Serialization functions
GGML_API size_t cogfluence_serialize_size(cogfluence_system_t* system);
GGML_API bool cogfluence_serialize(
    cogfluence_system_t* system,
    void* buffer,
    size_t buffer_size);

GGML_API cogfluence_system_t* cogfluence_deserialize(
    struct ggml_context* ctx,
    const void* buffer,
    size_t buffer_size);

// Integration functions
GGML_API struct ggml_tensor* cogfluence_to_tensor(
    cogfluence_knowledge_unit_t* unit,
    struct ggml_context* ctx);

GGML_API cogfluence_knowledge_unit_t* cogfluence_from_tensor(
    cogfluence_system_t* system,
    struct ggml_tensor* tensor,
    const char* name);

// Utility functions
GGML_API void cogfluence_print_statistics(cogfluence_system_t* system);
GGML_API float cogfluence_compute_coherence(cogfluence_system_t* system);
GGML_API void cogfluence_update_activations(cogfluence_system_t* system);

#ifdef __cplusplus
}
#endif