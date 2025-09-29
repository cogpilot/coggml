#pragma once

//
// MOSES (Meta-Optimizing Semantic Evolution) System
//
// This header defines the MOSES genetic algorithm framework for evolving
// cognitive programs and optimizing reasoning rules in the distributed
// cognitive architecture. MOSES integrates with PLN reasoning and OpenCog
// AtomSpace to automatically discover and improve cognitive patterns.
//

#include "ggml.h"
#include "ggml-opencog.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// MOSES system limits
#define MOSES_MAX_POPULATION 128
#define MOSES_MAX_PROGRAM_SIZE 256
#define MOSES_MAX_GENERATIONS 1000
#define MOSES_MAX_VARIABLES 32

// MOSES program operation types
typedef enum {
    MOSES_OP_CONSTANT = 1,      // Constant value
    MOSES_OP_VARIABLE = 2,      // Variable reference
    MOSES_OP_PLN_AND = 3,       // PLN logical AND
    MOSES_OP_PLN_OR = 4,        // PLN logical OR
    MOSES_OP_PLN_NOT = 5,       // PLN logical NOT
    MOSES_OP_PLN_IMPLIES = 6,   // PLN implication
    MOSES_OP_SIMILARITY = 7,    // Similarity computation
    MOSES_OP_INHERITANCE = 8,   // Inheritance reasoning
    MOSES_OP_PATTERN_MATCH = 9, // Pattern matching
    MOSES_OP_ATTENTION = 10     // Attention allocation
} moses_operation_type_t;

// MOSES program instruction
typedef struct {
    moses_operation_type_t op_type;
    union {
        float constant_value;               // For constants
        uint32_t variable_index;           // For variables
        struct {                           // For binary operations
            uint32_t arg1_index;
            uint32_t arg2_index;
        } binary_op;
        uint32_t unary_arg_index;          // For unary operations
    } operands;
    
    float output_value;                    // Execution result
    opencog_truth_value_t truth_value;     // PLN truth value result
} moses_instruction_t;

// MOSES cognitive program
typedef struct {
    moses_instruction_t* instructions;     // Program instructions
    size_t instruction_count;
    size_t instruction_capacity;
    
    // Fitness evaluation
    float fitness_score;                   // Overall fitness
    float reasoning_accuracy;              // PLN reasoning accuracy
    float efficiency_score;                // Computational efficiency
    
    // Program metadata
    uint64_t program_id;
    uint64_t generation;
    uint64_t parent1_id;                   // For crossover tracking
    uint64_t parent2_id;
    
    // Execution context
    float* variable_values;                // Variable assignments
    size_t variable_count;
    
    // Performance metrics
    uint64_t execution_count;
    float average_execution_time;
    float success_rate;
} moses_program_t;

// MOSES population
typedef struct {
    moses_program_t* programs;             // Population of programs
    size_t population_size;
    size_t population_capacity;
    
    // Evolution parameters
    float mutation_rate;                   // Probability of mutation
    float crossover_rate;                  // Probability of crossover
    float selection_pressure;              // Selection strength
    
    // Generation tracking
    uint64_t current_generation;
    uint64_t total_generations;
    
    // Best program tracking
    moses_program_t* best_program;         // Current best
    float best_fitness_history[100];       // Fitness evolution
    size_t history_index;
    
    // Statistics
    float average_fitness;
    float fitness_variance;
    uint64_t total_evaluations;
} moses_population_t;

// MOSES evolution system
typedef struct {
    struct ggml_context* ctx;
    opencog_atomspace_t* atomspace;
    
    // Evolution state
    moses_population_t* population;
    bool evolution_active;
    
    // Fitness evaluation environment
    struct ggml_tensor** test_cases;       // Training/test tensors
    size_t test_case_count;
    float* target_outputs;                 // Expected results
    
    // Evolution parameters
    float elitism_ratio;                   // Fraction to preserve
    float diversity_threshold;             // Minimum diversity
    uint32_t stagnation_limit;             // Generations without improvement
    
    // Performance tracking
    uint64_t total_mutations;
    uint64_t total_crossovers;
    uint64_t successful_mutations;
    uint64_t successful_crossovers;
    
    // Integration with cognitive systems
    bool integrate_with_pln;               // Use PLN for fitness
    bool integrate_with_attention;         // Use attention for selection
} moses_system_t;

// Core MOSES functions
GGML_API moses_system_t* moses_system_init(
    struct ggml_context* ctx,
    opencog_atomspace_t* atomspace);

GGML_API void moses_system_free(moses_system_t* moses);

// Population management
GGML_API moses_population_t* moses_population_create(
    moses_system_t* moses,
    size_t population_size);

GGML_API void moses_population_free(moses_population_t* population);

GGML_API moses_program_t* moses_program_create(moses_system_t* moses);
GGML_API void moses_program_free(moses_program_t* program);

// Program generation and modification
GGML_API bool moses_program_generate_random(
    moses_system_t* moses,
    moses_program_t* program,
    size_t max_instructions);

GGML_API moses_program_t* moses_program_mutate(
    moses_system_t* moses,
    const moses_program_t* parent);

GGML_API moses_program_t* moses_program_crossover(
    moses_system_t* moses,
    const moses_program_t* parent1,
    const moses_program_t* parent2);

// Program execution and evaluation
GGML_API bool moses_program_execute(
    moses_system_t* moses,
    moses_program_t* program,
    float* input_variables,
    size_t variable_count);

GGML_API float moses_program_evaluate_fitness(
    moses_system_t* moses,
    moses_program_t* program);

// Evolution operations
GGML_API bool moses_evolution_step(moses_system_t* moses);

GGML_API moses_program_t* moses_evolution_run(
    moses_system_t* moses,
    uint32_t max_generations);

GGML_API void moses_evolution_stop(moses_system_t* moses);

// Selection algorithms
GGML_API moses_program_t** moses_selection_tournament(
    moses_population_t* population,
    size_t tournament_size,
    size_t selection_count);

GGML_API moses_program_t** moses_selection_roulette(
    moses_population_t* population,
    size_t selection_count);

// Integration with cognitive systems
GGML_API bool moses_integrate_with_pln(
    moses_system_t* moses,
    bool enable_integration);

GGML_API bool moses_add_test_case(
    moses_system_t* moses,
    struct ggml_tensor* input,
    float expected_output);

GGML_API uint64_t moses_export_best_to_atomspace(
    moses_system_t* moses);

// Utility and statistics functions
GGML_API void moses_print_program(const moses_program_t* program);
GGML_API void moses_print_population_stats(const moses_population_t* population);
GGML_API void moses_print_evolution_summary(const moses_system_t* moses);

GGML_API float moses_compute_program_similarity(
    const moses_program_t* prog1,
    const moses_program_t* prog2);

GGML_API float moses_compute_population_diversity(
    const moses_population_t* population);

#ifdef __cplusplus
}
#endif