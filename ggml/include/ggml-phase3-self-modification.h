#pragma once

//
// Phase 3: Self-Modification Capabilities
//
// This header defines the self-modification and meta-evolution systems
// for the distributed cognitive architecture. It implements recursive
// self-improvement, automated architecture evolution, and emergent
// behavior analysis capabilities.
//

#include "ggml.h"
#include "ggml-opencog.h"
#include "ggml-moses.h"
#include "ggml-distributed-cognitive.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Phase 3 system limits
#define PHASE3_MAX_EVOLUTION_RULES 256
#define PHASE3_MAX_BEHAVIORAL_PATTERNS 128
#define PHASE3_MAX_CONSENSUS_NODES 64
#define PHASE3_MAX_COHERENCE_METRICS 32

// Self-modification operation types
typedef enum {
    SELF_MOD_RULE_CREATION = 1,      // Create new reasoning rules
    SELF_MOD_RULE_DELETION = 2,      // Remove ineffective rules
    SELF_MOD_RULE_MUTATION = 3,      // Modify existing rules
    SELF_MOD_ARCH_EXPANSION = 4,     // Expand architecture components
    SELF_MOD_ARCH_PRUNING = 5,       // Remove redundant components
    SELF_MOD_BEHAVIOR_ADAPTATION = 6  // Adapt behavioral patterns
} self_modification_type_t;

// Meta-evolution rule structure
typedef struct {
    uint32_t rule_id;
    char description[128];
    self_modification_type_t mod_type;
    
    // Performance metrics
    float effectiveness_score;       // How well this rule performs
    float novelty_score;            // How novel the rule is
    float stability_score;          // How stable the rule is
    uint32_t usage_count;           // How often it's been used
    
    // Rule logic (simplified as MOSES program reference)
    uint32_t moses_program_id;      // Reference to MOSES program
    float activation_threshold;     // When to trigger this rule
    
    // Metadata
    uint64_t creation_timestamp;
    uint64_t last_modification;
    bool is_active;
} meta_evolution_rule_t;

// Emergent behavior pattern
typedef struct {
    uint32_t pattern_id;
    char pattern_name[64];
    
    // Pattern characteristics
    float emergence_strength;       // How strongly the pattern emerges
    float coherence_level;         // How coherent the pattern is
    float stability_duration;      // How long the pattern persists
    
    // Agents exhibiting this pattern
    uint64_t* participating_agents;
    size_t agent_count;
    size_t agent_capacity;
    
    // Pattern evolution
    uint32_t generation;           // Evolution generation
    float fitness_score;           // Overall fitness
    
    // Metadata
    uint64_t first_observed;
    uint64_t last_observed;
    bool is_beneficial;
} emergent_behavior_pattern_t;

// Consensus protocol state
typedef struct {
    uint32_t consensus_id;
    char topic[128];
    
    // Participating nodes
    uint64_t* participant_agents;
    size_t participant_count;
    size_t participant_capacity;
    
    // Consensus state
    float agreement_level;         // Current level of agreement
    float confidence_level;        // Confidence in the consensus
    uint32_t voting_round;         // Current voting round
    
    // Decision making
    void* proposed_changes;        // Proposed system changes
    size_t change_count;
    bool consensus_reached;
    
    // Timing
    uint64_t start_timestamp;
    uint64_t timeout_duration;
} consensus_protocol_t;

// Global coherence metrics
typedef struct {
    char metric_name[64];
    float current_value;
    float target_value;
    float tolerance;
    bool is_within_bounds;
    
    // Historical tracking
    float* history_buffer;
    size_t history_size;
    size_t history_capacity;
    
    // Corrective actions
    uint32_t correction_rule_id;   // Rule to apply if out of bounds
    float correction_strength;
} coherence_metric_t;

// Main Phase 3 system structure
typedef struct {
    struct ggml_context* ctx;
    
    // Meta-evolution system
    meta_evolution_rule_t* evolution_rules;
    size_t rule_count;
    size_t rule_capacity;
    
    // Emergent behavior monitoring
    emergent_behavior_pattern_t* behavior_patterns;
    size_t pattern_count;
    size_t pattern_capacity;
    
    // Consensus protocols
    consensus_protocol_t* active_consensus;
    size_t consensus_count;
    size_t consensus_capacity;
    
    // Global coherence
    coherence_metric_t* coherence_metrics;
    size_t metric_count;
    size_t metric_capacity;
    
    // Integration with Phase 2 systems
    moses_system_t* moses_system;
    opencog_atomspace_t* atomspace;
    distributed_cognitive_architecture_t* distributed_arch;
    
    // Performance tracking
    uint32_t total_modifications;
    uint32_t successful_modifications;
    float system_improvement_rate;
} phase3_self_modification_system_t;

// Core Phase 3 functions

// System initialization
GGML_API phase3_self_modification_system_t* phase3_init(
    struct ggml_context* ctx,
    moses_system_t* moses_system,
    opencog_atomspace_t* atomspace,
    distributed_cognitive_architecture_t* distributed_arch
);

GGML_API void phase3_free(phase3_self_modification_system_t* system);

// Meta-evolution functions
GGML_API bool phase3_create_evolution_rule(
    phase3_self_modification_system_t* system,
    const char* description,
    self_modification_type_t mod_type,
    float activation_threshold
);

GGML_API bool phase3_execute_self_modification(
    phase3_self_modification_system_t* system,
    uint32_t rule_id
);

GGML_API void phase3_evolve_rules(
    phase3_self_modification_system_t* system
);

// Recursive improvement functions
GGML_API float phase3_measure_system_performance(
    phase3_self_modification_system_t* system
);

GGML_API bool phase3_recursive_self_improvement(
    phase3_self_modification_system_t* system
);

// Emergent behavior monitoring
GGML_API bool phase3_detect_emergent_behavior(
    phase3_self_modification_system_t* system,
    uint64_t* agent_ids,
    size_t agent_count
);

GGML_API void phase3_analyze_behavioral_patterns(
    phase3_self_modification_system_t* system
);

// Consensus protocol functions
GGML_API uint32_t phase3_initiate_consensus(
    phase3_self_modification_system_t* system,
    const char* topic,
    uint64_t* participants,
    size_t participant_count
);

GGML_API bool phase3_consensus_vote(
    phase3_self_modification_system_t* system,
    uint32_t consensus_id,
    uint64_t agent_id,
    bool agreement
);

GGML_API bool phase3_check_consensus_status(
    phase3_self_modification_system_t* system,
    uint32_t consensus_id
);

// Global coherence maintenance
GGML_API bool phase3_add_coherence_metric(
    phase3_self_modification_system_t* system,
    const char* metric_name,
    float target_value,
    float tolerance
);

GGML_API void phase3_update_coherence_metrics(
    phase3_self_modification_system_t* system
);

GGML_API bool phase3_maintain_global_coherence(
    phase3_self_modification_system_t* system
);

// Integration and coordination functions
GGML_API void phase3_coordinate_with_phase2(
    phase3_self_modification_system_t* system
);

GGML_API void phase3_update_system_state(
    phase3_self_modification_system_t* system
);

// Utility and diagnostic functions
GGML_API void phase3_print_system_status(
    const phase3_self_modification_system_t* system
);

GGML_API void phase3_print_evolution_rules(
    const phase3_self_modification_system_t* system
);

GGML_API void phase3_print_emergent_patterns(
    const phase3_self_modification_system_t* system
);

#ifdef __cplusplus
}
#endif