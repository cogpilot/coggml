#include "ggml-phase3-self-modification.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>

// Helper function to get current timestamp
static uint64_t get_current_timestamp(void) {
    return (uint64_t)time(NULL);
}

// Helper function to calculate fitness score
static float calculate_fitness_score(float effectiveness, float novelty, float stability) {
    return 0.5f * effectiveness + 0.3f * novelty + 0.2f * stability;
}

// System initialization
phase3_self_modification_system_t* phase3_init(
    struct ggml_context* ctx,
    moses_system_t* moses_system,
    opencog_atomspace_t* atomspace,
    distributed_cognitive_architecture_t* distributed_arch
) {
    printf("Initializing Phase 3: Self-Modification System\n");
    
    phase3_self_modification_system_t* system = calloc(1, sizeof(phase3_self_modification_system_t));
    if (!system) return NULL;
    
    system->ctx = ctx;
    system->moses_system = moses_system;
    system->atomspace = atomspace;
    system->distributed_arch = distributed_arch;
    
    // Initialize meta-evolution rules
    system->rule_capacity = PHASE3_MAX_EVOLUTION_RULES;
    system->evolution_rules = calloc(system->rule_capacity, sizeof(meta_evolution_rule_t));
    system->rule_count = 0;
    
    // Initialize behavior patterns
    system->pattern_capacity = PHASE3_MAX_BEHAVIORAL_PATTERNS;
    system->behavior_patterns = calloc(system->pattern_capacity, sizeof(emergent_behavior_pattern_t));
    system->pattern_count = 0;
    
    // Initialize consensus protocols
    system->consensus_capacity = PHASE3_MAX_CONSENSUS_NODES;
    system->active_consensus = calloc(system->consensus_capacity, sizeof(consensus_protocol_t));
    system->consensus_count = 0;
    
    // Initialize coherence metrics
    system->metric_capacity = PHASE3_MAX_COHERENCE_METRICS;
    system->coherence_metrics = calloc(system->metric_capacity, sizeof(coherence_metric_t));
    system->metric_count = 0;
    
    // Initialize performance tracking
    system->total_modifications = 0;
    system->successful_modifications = 0;
    system->system_improvement_rate = 0.0f;
    
    printf("✓ Phase 3 Self-Modification System initialized\n");
    printf("  - Evolution rules capacity: %zu\n", system->rule_capacity);
    printf("  - Behavior patterns capacity: %zu\n", system->pattern_capacity);
    printf("  - Consensus protocols capacity: %zu\n", system->consensus_capacity);
    printf("  - Coherence metrics capacity: %zu\n", system->metric_capacity);
    
    return system;
}

void phase3_free(phase3_self_modification_system_t* system) {
    if (!system) return;
    
    // Free behavior pattern agent arrays
    for (size_t i = 0; i < system->pattern_count; i++) {
        free(system->behavior_patterns[i].participating_agents);
    }
    
    // Free consensus protocol participant arrays
    for (size_t i = 0; i < system->consensus_count; i++) {
        free(system->active_consensus[i].participant_agents);
    }
    
    // Free coherence metric history buffers
    for (size_t i = 0; i < system->metric_count; i++) {
        free(system->coherence_metrics[i].history_buffer);
    }
    
    free(system->evolution_rules);
    free(system->behavior_patterns);
    free(system->active_consensus);
    free(system->coherence_metrics);
    free(system);
}

// Meta-evolution functions
bool phase3_create_evolution_rule(
    phase3_self_modification_system_t* system,
    const char* description,
    self_modification_type_t mod_type,
    float activation_threshold
) {
    if (!system || system->rule_count >= system->rule_capacity) {
        return false;
    }
    
    meta_evolution_rule_t* rule = &system->evolution_rules[system->rule_count];
    
    rule->rule_id = system->rule_count + 1;
    strncpy(rule->description, description, sizeof(rule->description) - 1);
    rule->mod_type = mod_type;
    rule->activation_threshold = activation_threshold;
    
    // Initialize performance metrics
    rule->effectiveness_score = 0.5f;  // Start neutral
    rule->novelty_score = 1.0f;        // New rules are novel
    rule->stability_score = 0.0f;      // Unknown stability initially
    rule->usage_count = 0;
    
    // Create corresponding MOSES program for this rule
    rule->moses_program_id = system->rule_count + 1;  // Simplified mapping
    
    rule->creation_timestamp = get_current_timestamp();
    rule->last_modification = rule->creation_timestamp;
    rule->is_active = true;
    
    system->rule_count++;
    
    printf("Created evolution rule %u: %s (type %d)\n", 
           rule->rule_id, rule->description, rule->mod_type);
    
    return true;
}

bool phase3_execute_self_modification(
    phase3_self_modification_system_t* system,
    uint32_t rule_id
) {
    if (!system || rule_id == 0 || rule_id > system->rule_count) {
        return false;
    }
    
    meta_evolution_rule_t* rule = &system->evolution_rules[rule_id - 1];
    
    if (!rule->is_active) {
        return false;
    }
    
    system->total_modifications++;
    
    printf("Executing self-modification rule %u: %s\n", rule->rule_id, rule->description);
    
    bool success = false;
    
    // Execute different types of self-modifications
    switch (rule->mod_type) {
        case SELF_MOD_RULE_CREATION: {
            // Create new reasoning rules in the system
            if (system->atomspace) {
                // Add new reasoning pattern to atomspace
                uint64_t new_node = opencog_add_node(system->atomspace, OPENCOG_CONCEPT_NODE, "SelfGeneratedRule");
                success = (new_node != 0);
            }
            break;
        }
        
        case SELF_MOD_RULE_MUTATION: {
            // Mutate existing rules using MOSES
            if (system->moses_system) {
                // Evolve the MOSES program associated with this rule
                // Note: Simplified for Phase 3 demo - full MOSES evolution integration in future
                success = true;  // Placeholder for MOSES evolution step
            }
            break;
        }
        
        case SELF_MOD_ARCH_EXPANSION: {
            // Expand architecture by adding new components
            if (system->distributed_arch) {
                // Add new cognitive membrane or workflow
                success = true;  // Simplified for demo
            }
            break;
        }
        
        case SELF_MOD_BEHAVIOR_ADAPTATION: {
            // Adapt behavioral patterns based on performance
            phase3_analyze_behavioral_patterns(system);
            success = true;
            break;
        }
        
        default:
            success = false;
            break;
    }
    
    // Update rule metrics
    rule->usage_count++;
    rule->last_modification = get_current_timestamp();
    
    if (success) {
        rule->effectiveness_score = fminf(1.0f, rule->effectiveness_score + 0.1f);
        system->successful_modifications++;
    } else {
        rule->effectiveness_score = fmaxf(0.0f, rule->effectiveness_score - 0.1f);
        if (rule->effectiveness_score < 0.2f) {
            rule->is_active = false;  // Deactivate ineffective rules
            printf("Deactivated ineffective rule %u\n", rule->rule_id);
        }
    }
    
    return success;
}

void phase3_evolve_rules(phase3_self_modification_system_t* system) {
    if (!system) return;
    
    printf("Evolving meta-evolution rules...\n");
    
    // Calculate system performance
    float performance = phase3_measure_system_performance(system);
    
    // Update all rule stability scores based on performance
    for (size_t i = 0; i < system->rule_count; i++) {
        meta_evolution_rule_t* rule = &system->evolution_rules[i];
        
        if (rule->is_active) {
            // Stability increases with consistent performance
            if (performance > 0.7f) {
                rule->stability_score = fminf(1.0f, rule->stability_score + 0.05f);
            } else {
                rule->stability_score = fmaxf(0.0f, rule->stability_score - 0.05f);
            }
            
            // Decay novelty over time
            rule->novelty_score = fmaxf(0.1f, rule->novelty_score * 0.95f);
        }
    }
    
    // Create new rules if performance is poor
    if (performance < 0.5f && system->rule_count < system->rule_capacity - 1) {
        char new_rule_desc[128];
        snprintf(new_rule_desc, sizeof(new_rule_desc), "PerformanceImprover_%zu", system->rule_count);
        
        phase3_create_evolution_rule(system, new_rule_desc, SELF_MOD_BEHAVIOR_ADAPTATION, 0.3f);
        printf("Created new rule due to poor performance: %s\n", new_rule_desc);
    }
}

// Recursive improvement functions
float phase3_measure_system_performance(phase3_self_modification_system_t* system) {
    if (!system) return 0.0f;
    
    float performance = 0.0f;
    int components = 0;
    
    // Factor in self-modification success rate
    if (system->total_modifications > 0) {
        performance += (float)system->successful_modifications / system->total_modifications;
        components++;
    }
    
    // Factor in rule effectiveness
    if (system->rule_count > 0) {
        float avg_effectiveness = 0.0f;
        int active_rules = 0;
        
        for (size_t i = 0; i < system->rule_count; i++) {
            if (system->evolution_rules[i].is_active) {
                avg_effectiveness += system->evolution_rules[i].effectiveness_score;
                active_rules++;
            }
        }
        
        if (active_rules > 0) {
            performance += avg_effectiveness / active_rules;
            components++;
        }
    }
    
    // Factor in coherence metrics
    if (system->metric_count > 0) {
        int coherent_metrics = 0;
        for (size_t i = 0; i < system->metric_count; i++) {
            if (system->coherence_metrics[i].is_within_bounds) {
                coherent_metrics++;
            }
        }
        performance += (float)coherent_metrics / system->metric_count;
        components++;
    }
    
    return components > 0 ? performance / components : 0.5f;
}

bool phase3_recursive_self_improvement(phase3_self_modification_system_t* system) {
    if (!system) return false;
    
    printf("Initiating recursive self-improvement cycle...\n");
    
    float initial_performance = phase3_measure_system_performance(system);
    printf("Initial system performance: %.3f\n", initial_performance);
    
    // Execute all active rules that meet their activation threshold
    bool improvements_made = false;
    
    for (size_t i = 0; i < system->rule_count; i++) {
        meta_evolution_rule_t* rule = &system->evolution_rules[i];
        
        if (rule->is_active && rule->effectiveness_score >= rule->activation_threshold) {
            if (phase3_execute_self_modification(system, rule->rule_id)) {
                improvements_made = true;
            }
        }
    }
    
    // Evolve the rules themselves
    phase3_evolve_rules(system);
    
    // Measure performance after improvements
    float final_performance = phase3_measure_system_performance(system);
    system->system_improvement_rate = final_performance - initial_performance;
    
    printf("Final system performance: %.3f (improvement: %+.3f)\n", 
           final_performance, system->system_improvement_rate);
    
    return improvements_made && (system->system_improvement_rate > 0.0f);
}

// Emergent behavior monitoring
bool phase3_detect_emergent_behavior(
    phase3_self_modification_system_t* system,
    uint64_t* agent_ids,
    size_t agent_count
) {
    if (!system || !agent_ids || agent_count == 0) return false;
    
    if (system->pattern_count >= system->pattern_capacity) return false;
    
    // Simple emergent behavior detection based on agent coordination
    emergent_behavior_pattern_t* pattern = &system->behavior_patterns[system->pattern_count];
    
    pattern->pattern_id = system->pattern_count + 1;
    snprintf(pattern->pattern_name, sizeof(pattern->pattern_name), 
             "EmergentPattern_%u", pattern->pattern_id);
    
    // Allocate and copy participating agents
    pattern->agent_capacity = agent_count + 4;  // Extra space for growth
    pattern->participating_agents = calloc(pattern->agent_capacity, sizeof(uint64_t));
    memcpy(pattern->participating_agents, agent_ids, agent_count * sizeof(uint64_t));
    pattern->agent_count = agent_count;
    
    // Calculate pattern characteristics
    pattern->emergence_strength = (float)agent_count / 10.0f;  // Simplified metric
    pattern->coherence_level = 0.7f + (rand() % 30) / 100.0f;  // Random coherence
    pattern->stability_duration = 1000.0f + (rand() % 5000);   // Random duration
    
    pattern->generation = 1;
    pattern->fitness_score = calculate_fitness_score(
        pattern->emergence_strength, 
        pattern->coherence_level, 
        pattern->stability_duration / 10000.0f
    );
    
    pattern->first_observed = get_current_timestamp();
    pattern->last_observed = pattern->first_observed;
    pattern->is_beneficial = pattern->fitness_score > 0.6f;
    
    system->pattern_count++;
    
    printf("Detected emergent behavior pattern %u: %s\n", 
           pattern->pattern_id, pattern->pattern_name);
    printf("  Agents: %zu, Fitness: %.3f, Beneficial: %s\n",
           pattern->agent_count, pattern->fitness_score,
           pattern->is_beneficial ? "Yes" : "No");
    
    return true;
}

void phase3_analyze_behavioral_patterns(phase3_self_modification_system_t* system) {
    if (!system) return;
    
    printf("Analyzing behavioral patterns...\n");
    
    for (size_t i = 0; i < system->pattern_count; i++) {
        emergent_behavior_pattern_t* pattern = &system->behavior_patterns[i];
        
        // Update pattern metrics based on time
        uint64_t current_time = get_current_timestamp();
        uint64_t age = current_time - pattern->first_observed;
        
        // Patterns that persist longer become more stable
        if (age > 300) {  // 5 minutes
            pattern->stability_duration *= 1.1f;
            pattern->coherence_level = fminf(1.0f, pattern->coherence_level + 0.05f);
        }
        
        // Recalculate fitness
        pattern->fitness_score = calculate_fitness_score(
            pattern->emergence_strength,
            pattern->coherence_level,
            fminf(1.0f, pattern->stability_duration / 10000.0f)
        );
        
        pattern->is_beneficial = pattern->fitness_score > 0.6f;
        pattern->last_observed = current_time;
        
        // Promote beneficial patterns
        if (pattern->is_beneficial && pattern->generation < 10) {
            pattern->generation++;
            printf("Promoted pattern %u to generation %u (fitness: %.3f)\n",
                   pattern->pattern_id, pattern->generation, pattern->fitness_score);
        }
    }
}

// Consensus protocol functions
uint32_t phase3_initiate_consensus(
    phase3_self_modification_system_t* system,
    const char* topic,
    uint64_t* participants,
    size_t participant_count
) {
    if (!system || !topic || !participants || participant_count == 0) return 0;
    
    if (system->consensus_count >= system->consensus_capacity) return 0;
    
    consensus_protocol_t* consensus = &system->active_consensus[system->consensus_count];
    
    consensus->consensus_id = system->consensus_count + 1;
    strncpy(consensus->topic, topic, sizeof(consensus->topic) - 1);
    
    // Allocate and copy participants
    consensus->participant_capacity = participant_count + 2;
    consensus->participant_agents = calloc(consensus->participant_capacity, sizeof(uint64_t));
    memcpy(consensus->participant_agents, participants, participant_count * sizeof(uint64_t));
    consensus->participant_count = participant_count;
    
    consensus->agreement_level = 0.0f;
    consensus->confidence_level = 0.0f;
    consensus->voting_round = 1;
    consensus->consensus_reached = false;
    
    consensus->start_timestamp = get_current_timestamp();
    consensus->timeout_duration = 300;  // 5 minutes timeout
    
    system->consensus_count++;
    
    printf("Initiated consensus protocol %u: '%s' with %zu participants\n",
           consensus->consensus_id, consensus->topic, consensus->participant_count);
    
    return consensus->consensus_id;
}

bool phase3_consensus_vote(
    phase3_self_modification_system_t* system,
    uint32_t consensus_id,
    uint64_t agent_id,
    bool agreement
) {
    if (!system || consensus_id == 0 || consensus_id > system->consensus_count) {
        return false;
    }
    
    consensus_protocol_t* consensus = &system->active_consensus[consensus_id - 1];
    
    if (consensus->consensus_reached) return false;
    
    // Verify agent is a participant
    bool is_participant = false;
    for (size_t i = 0; i < consensus->participant_count; i++) {
        if (consensus->participant_agents[i] == agent_id) {
            is_participant = true;
            break;
        }
    }
    
    if (!is_participant) return false;
    
    // Update agreement level (simplified)
    if (agreement) {
        consensus->agreement_level += 1.0f / consensus->participant_count;
    }
    consensus->confidence_level += 0.5f / consensus->participant_count;
    
    printf("Agent %lu voted %s on consensus %u\n", 
           agent_id, agreement ? "AGREE" : "DISAGREE", consensus_id);
    
    return true;
}

bool phase3_check_consensus_status(
    phase3_self_modification_system_t* system,
    uint32_t consensus_id
) {
    if (!system || consensus_id == 0 || consensus_id > system->consensus_count) {
        return false;
    }
    
    consensus_protocol_t* consensus = &system->active_consensus[consensus_id - 1];
    
    // Check for timeout
    uint64_t current_time = get_current_timestamp();
    if (current_time - consensus->start_timestamp > consensus->timeout_duration) {
        printf("Consensus %u timed out\n", consensus_id);
        return false;
    }
    
    // Check for consensus (70% agreement with 80% confidence)
    if (consensus->agreement_level >= 0.7f && consensus->confidence_level >= 0.8f) {
        consensus->consensus_reached = true;
        printf("Consensus %u reached! Agreement: %.1f%%, Confidence: %.1f%%\n",
               consensus_id, consensus->agreement_level * 100, consensus->confidence_level * 100);
        return true;
    }
    
    return false;
}

// Global coherence maintenance
bool phase3_add_coherence_metric(
    phase3_self_modification_system_t* system,
    const char* metric_name,
    float target_value,
    float tolerance
) {
    if (!system || !metric_name || system->metric_count >= system->metric_capacity) {
        return false;
    }
    
    coherence_metric_t* metric = &system->coherence_metrics[system->metric_count];
    
    strncpy(metric->metric_name, metric_name, sizeof(metric->metric_name) - 1);
    metric->target_value = target_value;
    metric->tolerance = tolerance;
    metric->current_value = target_value;  // Start at target
    metric->is_within_bounds = true;
    
    // Initialize history buffer
    metric->history_capacity = 100;
    metric->history_buffer = calloc(metric->history_capacity, sizeof(float));
    metric->history_size = 0;
    
    metric->correction_rule_id = 0;  // No correction rule initially
    metric->correction_strength = 0.1f;
    
    system->metric_count++;
    
    printf("Added coherence metric: %s (target: %.3f ± %.3f)\n",
           metric->metric_name, metric->target_value, metric->tolerance);
    
    return true;
}

void phase3_update_coherence_metrics(phase3_self_modification_system_t* system) {
    if (!system) return;
    
    for (size_t i = 0; i < system->metric_count; i++) {
        coherence_metric_t* metric = &system->coherence_metrics[i];
        
        // Simulate metric updates (in real implementation, these would be measured)
        float noise = ((rand() % 200) - 100) / 1000.0f;  // ±0.1 noise
        metric->current_value += noise;
        
        // Add to history
        if (metric->history_size < metric->history_capacity) {
            metric->history_buffer[metric->history_size] = metric->current_value;
            metric->history_size++;
        } else {
            // Shift history buffer
            memmove(metric->history_buffer, metric->history_buffer + 1, 
                    (metric->history_capacity - 1) * sizeof(float));
            metric->history_buffer[metric->history_capacity - 1] = metric->current_value;
        }
        
        // Check bounds
        float deviation = fabsf(metric->current_value - metric->target_value);
        metric->is_within_bounds = deviation <= metric->tolerance;
        
        if (!metric->is_within_bounds) {
            printf("WARNING: Coherence metric '%s' out of bounds: %.3f (target: %.3f ± %.3f)\n",
                   metric->metric_name, metric->current_value, 
                   metric->target_value, metric->tolerance);
        }
    }
}

bool phase3_maintain_global_coherence(phase3_self_modification_system_t* system) {
    if (!system) return false;
    
    phase3_update_coherence_metrics(system);
    
    bool coherence_maintained = true;
    int corrections_applied = 0;
    
    for (size_t i = 0; i < system->metric_count; i++) {
        coherence_metric_t* metric = &system->coherence_metrics[i];
        
        if (!metric->is_within_bounds) {
            coherence_maintained = false;
            
            // Apply corrective action
            float correction = (metric->target_value - metric->current_value) * metric->correction_strength;
            metric->current_value += correction;
            corrections_applied++;
            
            printf("Applied correction to %s: %+.3f\n", metric->metric_name, correction);
            
            // Create self-modification rule if needed
            if (metric->correction_rule_id == 0 && system->rule_count < system->rule_capacity - 1) {
                char rule_desc[128];
                snprintf(rule_desc, sizeof(rule_desc), "CoherenceCorrector_%s", metric->metric_name);
                
                if (phase3_create_evolution_rule(system, rule_desc, SELF_MOD_BEHAVIOR_ADAPTATION, 0.5f)) {
                    metric->correction_rule_id = system->rule_count;
                    printf("Created coherence correction rule %u for %s\n", 
                           metric->correction_rule_id, metric->metric_name);
                }
            }
        }
    }
    
    printf("Global coherence maintenance: %s (%d corrections applied)\n",
           coherence_maintained ? "STABLE" : "CORRECTED", corrections_applied);
    
    return coherence_maintained;
}

// Integration and coordination functions
void phase3_coordinate_with_phase2(phase3_self_modification_system_t* system) {
    if (!system) return;
    
    printf("Coordinating Phase 3 with Phase 2 systems...\n");
    
    // Sync with MOSES system
    if (system->moses_system) {
        // Use MOSES evolution to improve self-modification rules
        for (size_t i = 0; i < system->rule_count; i++) {
            meta_evolution_rule_t* rule = &system->evolution_rules[i];
            if (rule->is_active && rule->effectiveness_score < 0.7f) {
                // Trigger MOSES evolution for underperforming rules
                printf("Triggering MOSES evolution for rule %u\n", rule->rule_id);
                // In a full implementation, this would evolve the MOSES program
            }
        }
    }
    
    // Sync with OpenCog AtomSpace
    if (system->atomspace) {
        // Add behavioral patterns as concepts in AtomSpace
        for (size_t i = 0; i < system->pattern_count; i++) {
            emergent_behavior_pattern_t* pattern = &system->behavior_patterns[i];
            if (pattern->is_beneficial) {
                uint64_t pattern_node = opencog_add_node(system->atomspace, 
                                                        OPENCOG_CONCEPT_NODE, 
                                                        pattern->pattern_name);
                printf("Added beneficial pattern %u to AtomSpace (node %lu)\n", 
                       pattern->pattern_id, pattern_node);
            }
        }
    }
}

void phase3_update_system_state(phase3_self_modification_system_t* system) {
    if (!system) return;
    
    printf("Updating Phase 3 system state...\n");
    
    // Update behavioral patterns
    phase3_analyze_behavioral_patterns(system);
    
    // Maintain global coherence
    phase3_maintain_global_coherence(system);
    
    // Check active consensus protocols
    for (size_t i = 0; i < system->consensus_count; i++) {
        phase3_check_consensus_status(system, i + 1);
    }
    
    // Coordinate with Phase 2
    phase3_coordinate_with_phase2(system);
    
    printf("System state update completed\n");
}

// Utility and diagnostic functions
void phase3_print_system_status(const phase3_self_modification_system_t* system) {
    if (!system) return;
    
    printf("\n=== Phase 3 Self-Modification System Status ===\n");
    printf("Evolution Rules: %zu/%zu\n", system->rule_count, system->rule_capacity);
    printf("Behavior Patterns: %zu/%zu\n", system->pattern_count, system->pattern_capacity);
    printf("Active Consensus: %zu/%zu\n", system->consensus_count, system->consensus_capacity);
    printf("Coherence Metrics: %zu/%zu\n", system->metric_count, system->metric_capacity);
    printf("Total Modifications: %u (Success: %u, Rate: %.1f%%)\n",
           system->total_modifications, system->successful_modifications,
           system->total_modifications > 0 ? 
           (100.0f * system->successful_modifications / system->total_modifications) : 0.0f);
    printf("System Improvement Rate: %+.3f\n", system->system_improvement_rate);
    printf("===============================================\n\n");
}

void phase3_print_evolution_rules(const phase3_self_modification_system_t* system) {
    if (!system) return;
    
    printf("\n=== Evolution Rules ===\n");
    for (size_t i = 0; i < system->rule_count; i++) {
        const meta_evolution_rule_t* rule = &system->evolution_rules[i];
        printf("Rule %u: %s\n", rule->rule_id, rule->description);
        printf("  Type: %d, Active: %s, Usage: %u\n", 
               rule->mod_type, rule->is_active ? "Yes" : "No", rule->usage_count);
        printf("  Effectiveness: %.3f, Novelty: %.3f, Stability: %.3f\n",
               rule->effectiveness_score, rule->novelty_score, rule->stability_score);
        printf("  Activation Threshold: %.3f\n", rule->activation_threshold);
    }
    printf("=======================\n\n");
}

void phase3_print_emergent_patterns(const phase3_self_modification_system_t* system) {
    if (!system) return;
    
    printf("\n=== Emergent Behavior Patterns ===\n");
    for (size_t i = 0; i < system->pattern_count; i++) {
        const emergent_behavior_pattern_t* pattern = &system->behavior_patterns[i];
        printf("Pattern %u: %s\n", pattern->pattern_id, pattern->pattern_name);
        printf("  Agents: %zu, Generation: %u, Beneficial: %s\n",
               pattern->agent_count, pattern->generation, 
               pattern->is_beneficial ? "Yes" : "No");
        printf("  Emergence: %.3f, Coherence: %.3f, Fitness: %.3f\n",
               pattern->emergence_strength, pattern->coherence_level, pattern->fitness_score);
    }
    printf("===================================\n\n");
}