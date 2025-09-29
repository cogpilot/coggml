#include "ggml-moses.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Generate unique program ID
static uint64_t generate_program_id(void) {
    static uint64_t next_id = 1;
    return next_id++;
}

// Random float between 0 and 1
static float random_float(void) {
    return (float)rand() / (float)RAND_MAX;
}

// Random integer between min and max (inclusive)
static int random_int(int min, int max) {
    return min + rand() % (max - min + 1);
}

// Initialize MOSES system
moses_system_t* moses_system_init(
    struct ggml_context* ctx,
    opencog_atomspace_t* atomspace) {
    
    moses_system_t* moses = malloc(sizeof(moses_system_t));
    if (!moses) return NULL;
    
    moses->ctx = ctx;
    moses->atomspace = atomspace;
    moses->population = NULL;
    moses->evolution_active = false;
    
    // Initialize test cases
    moses->test_cases = NULL;
    moses->test_case_count = 0;
    moses->target_outputs = NULL;
    
    // Set evolution parameters
    moses->elitism_ratio = 0.1f;              // Keep top 10%
    moses->diversity_threshold = 0.05f;       // Minimum diversity
    moses->stagnation_limit = 50;             // 50 generations without improvement
    
    // Initialize performance tracking
    moses->total_mutations = 0;
    moses->total_crossovers = 0;
    moses->successful_mutations = 0;
    moses->successful_crossovers = 0;
    
    // Integration flags
    moses->integrate_with_pln = true;
    moses->integrate_with_attention = true;
    
    // Seed random number generator
    srand((unsigned int)time(NULL));
    
    printf("MOSES system initialized with PLN and attention integration\n");
    
    return moses;
}

// Free MOSES system
void moses_system_free(moses_system_t* moses) {
    if (!moses) return;
    
    if (moses->population) {
        moses_population_free(moses->population);
    }
    
    // Free test cases
    if (moses->test_cases) {
        free(moses->test_cases);
    }
    if (moses->target_outputs) {
        free(moses->target_outputs);
    }
    
    free(moses);
}

// Create population
moses_population_t* moses_population_create(
    moses_system_t* moses,
    size_t population_size) {
    
    if (!moses || population_size > MOSES_MAX_POPULATION) return NULL;
    
    moses_population_t* population = malloc(sizeof(moses_population_t));
    if (!population) return NULL;
    
    population->population_capacity = population_size;
    population->programs = calloc(population_size, sizeof(moses_program_t));
    if (!population->programs) {
        free(population);
        return NULL;
    }
    
    population->population_size = 0;
    
    // Evolution parameters
    population->mutation_rate = 0.1f;         // 10% mutation rate
    population->crossover_rate = 0.7f;        // 70% crossover rate
    population->selection_pressure = 1.5f;    // Moderate selection pressure
    
    // Generation tracking
    population->current_generation = 0;
    population->total_generations = 0;
    
    // Initialize fitness history
    population->best_program = NULL;
    population->history_index = 0;
    for (int i = 0; i < 100; i++) {
        population->best_fitness_history[i] = 0.0f;
    }
    
    // Statistics
    population->average_fitness = 0.0f;
    population->fitness_variance = 0.0f;
    population->total_evaluations = 0;
    
    moses->population = population;
    
    printf("MOSES population created with capacity %zu\n", population_size);
    
    return population;
}

// Free population
void moses_population_free(moses_population_t* population) {
    if (!population) return;
    
    // Free all programs
    for (size_t i = 0; i < population->population_size; i++) {
        moses_program_free(&population->programs[i]);
    }
    
    free(population->programs);
    free(population);
}

// Create a new program
moses_program_t* moses_program_create(moses_system_t* moses) {
    if (!moses) return NULL;
    
    moses_program_t* program = malloc(sizeof(moses_program_t));
    if (!program) return NULL;
    
    // Initialize instruction array
    program->instruction_capacity = MOSES_MAX_PROGRAM_SIZE;
    program->instructions = calloc(program->instruction_capacity, sizeof(moses_instruction_t));
    if (!program->instructions) {
        free(program);
        return NULL;
    }
    
    program->instruction_count = 0;
    
    // Initialize fitness
    program->fitness_score = 0.0f;
    program->reasoning_accuracy = 0.0f;
    program->efficiency_score = 0.0f;
    
    // Program metadata
    program->program_id = generate_program_id();
    program->generation = 0;
    program->parent1_id = 0;
    program->parent2_id = 0;
    
    // Initialize variables
    program->variable_count = MOSES_MAX_VARIABLES;
    program->variable_values = calloc(program->variable_count, sizeof(float));
    if (!program->variable_values) {
        free(program->instructions);
        free(program);
        return NULL;
    }
    
    // Performance metrics
    program->execution_count = 0;
    program->average_execution_time = 0.0f;
    program->success_rate = 0.0f;
    
    return program;
}

// Free program
void moses_program_free(moses_program_t* program) {
    if (!program) return;
    
    if (program->instructions) {
        free(program->instructions);
    }
    
    if (program->variable_values) {
        free(program->variable_values);
    }
    
    // Don't free the program itself if it's part of an array
}

// Generate random program
bool moses_program_generate_random(
    moses_system_t* moses,
    moses_program_t* program,
    size_t max_instructions) {
    
    if (!moses || !program || max_instructions > MOSES_MAX_PROGRAM_SIZE) {
        return false;
    }
    
    program->instruction_count = random_int(5, (int)max_instructions);
    
    for (size_t i = 0; i < program->instruction_count; i++) {
        moses_instruction_t* instr = &program->instructions[i];
        
        // Randomly select operation type
        instr->op_type = (moses_operation_type_t)random_int(MOSES_OP_CONSTANT, MOSES_OP_ATTENTION);
        
        switch (instr->op_type) {
            case MOSES_OP_CONSTANT:
                instr->operands.constant_value = random_float();
                break;
                
            case MOSES_OP_VARIABLE:
                instr->operands.variable_index = random_int(0, MOSES_MAX_VARIABLES - 1);
                break;
                
            case MOSES_OP_PLN_AND:
            case MOSES_OP_PLN_OR:
            case MOSES_OP_PLN_IMPLIES:
            case MOSES_OP_SIMILARITY:
            case MOSES_OP_INHERITANCE:
                // Binary operations reference previous instructions
                if (i > 0) {
                    instr->operands.binary_op.arg1_index = random_int(0, (int)i - 1);
                    instr->operands.binary_op.arg2_index = random_int(0, (int)i - 1);
                } else {
                    // Fallback to constants for first instruction
                    instr->op_type = MOSES_OP_CONSTANT;
                    instr->operands.constant_value = random_float();
                }
                break;
                
            case MOSES_OP_PLN_NOT:
            case MOSES_OP_PATTERN_MATCH:
            case MOSES_OP_ATTENTION:
                // Unary operations
                if (i > 0) {
                    instr->operands.unary_arg_index = random_int(0, (int)i - 1);
                } else {
                    instr->op_type = MOSES_OP_CONSTANT;
                    instr->operands.constant_value = random_float();
                }
                break;
        }
        
        // Initialize output values
        instr->output_value = 0.0f;
        instr->truth_value.strength = 0.0f;
        instr->truth_value.confidence = 0.0f;
        instr->truth_value.count = 0.0f;
    }
    
    printf("Generated random program %lu with %zu instructions\n", 
           program->program_id, program->instruction_count);
    
    return true;
}

// Execute a program
bool moses_program_execute(
    moses_system_t* moses,
    moses_program_t* program,
    float* input_variables,
    size_t variable_count) {
    
    if (!moses || !program || !input_variables) return false;
    
    // Copy input variables
    size_t copy_count = (variable_count < program->variable_count) ? 
                       variable_count : program->variable_count;
    memcpy(program->variable_values, input_variables, copy_count * sizeof(float));
    
    // Execute each instruction
    for (size_t i = 0; i < program->instruction_count; i++) {
        moses_instruction_t* instr = &program->instructions[i];
        
        switch (instr->op_type) {
            case MOSES_OP_CONSTANT:
                instr->output_value = instr->operands.constant_value;
                instr->truth_value.strength = instr->operands.constant_value;
                instr->truth_value.confidence = 0.9f;
                break;
                
            case MOSES_OP_VARIABLE:
                if (instr->operands.variable_index < program->variable_count) {
                    instr->output_value = program->variable_values[instr->operands.variable_index];
                    instr->truth_value.strength = instr->output_value;
                    instr->truth_value.confidence = 0.8f;
                } else {
                    instr->output_value = 0.0f;
                }
                break;
                
            case MOSES_OP_PLN_AND: {
                uint32_t arg1 = instr->operands.binary_op.arg1_index;
                uint32_t arg2 = instr->operands.binary_op.arg2_index;
                if (arg1 < i && arg2 < i) {
                    opencog_truth_value_t tv1 = program->instructions[arg1].truth_value;
                    opencog_truth_value_t tv2 = program->instructions[arg2].truth_value;
                    instr->truth_value = opencog_pln_and(tv1, tv2);
                    instr->output_value = instr->truth_value.strength;
                }
                break;
            }
            
            case MOSES_OP_PLN_OR: {
                uint32_t arg1 = instr->operands.binary_op.arg1_index;
                uint32_t arg2 = instr->operands.binary_op.arg2_index;
                if (arg1 < i && arg2 < i) {
                    opencog_truth_value_t tv1 = program->instructions[arg1].truth_value;
                    opencog_truth_value_t tv2 = program->instructions[arg2].truth_value;
                    instr->truth_value = opencog_pln_or(tv1, tv2);
                    instr->output_value = instr->truth_value.strength;
                }
                break;
            }
            
            case MOSES_OP_PLN_NOT: {
                uint32_t arg = instr->operands.unary_arg_index;
                if (arg < i) {
                    opencog_truth_value_t tv = program->instructions[arg].truth_value;
                    instr->truth_value = opencog_pln_not(tv);
                    instr->output_value = instr->truth_value.strength;
                }
                break;
            }
            
            case MOSES_OP_SIMILARITY: {
                // Simplified similarity computation
                uint32_t arg1 = instr->operands.binary_op.arg1_index;
                uint32_t arg2 = instr->operands.binary_op.arg2_index;
                if (arg1 < i && arg2 < i) {
                    float val1 = program->instructions[arg1].output_value;
                    float val2 = program->instructions[arg2].output_value;
                    float similarity = 1.0f - fabsf(val1 - val2);
                    instr->output_value = fmaxf(0.0f, similarity);
                    instr->truth_value.strength = instr->output_value;
                    instr->truth_value.confidence = 0.7f;
                }
                break;
            }
            
            default:
                instr->output_value = 0.0f;
                break;
        }
        
        // Clamp output values
        instr->output_value = fmaxf(0.0f, fminf(1.0f, instr->output_value));
        instr->truth_value.strength = fmaxf(0.0f, fminf(1.0f, instr->truth_value.strength));
        instr->truth_value.confidence = fmaxf(0.0f, fminf(1.0f, instr->truth_value.confidence));
    }
    
    program->execution_count++;
    
    return true;
}

// Evaluate program fitness
float moses_program_evaluate_fitness(
    moses_system_t* moses,
    moses_program_t* program) {
    
    if (!moses || !program || moses->test_case_count == 0) {
        return 0.0f;
    }
    
    float total_error = 0.0f;
    size_t valid_tests = 0;
    
    // Test on all test cases
    for (size_t test = 0; test < moses->test_case_count; test++) {
        struct ggml_tensor* input = moses->test_cases[test];
        float expected = moses->target_outputs[test];
        
        // Extract input variables from tensor
        float input_vars[MOSES_MAX_VARIABLES] = {0};
        if (input && input->data) {
            size_t n_elements = ggml_nelements(input);
            float* tensor_data = (float*)input->data;
            
            size_t copy_count = (n_elements < MOSES_MAX_VARIABLES) ? 
                               n_elements : MOSES_MAX_VARIABLES;
            
            for (size_t i = 0; i < copy_count; i++) {
                input_vars[i] = tensor_data[i];
            }
        }
        
        // Execute program
        if (moses_program_execute(moses, program, input_vars, MOSES_MAX_VARIABLES)) {
            // Get output from last instruction
            if (program->instruction_count > 0) {
                float output = program->instructions[program->instruction_count - 1].output_value;
                float error = fabsf(output - expected);
                total_error += error;
                valid_tests++;
            }
        }
    }
    
    // Calculate fitness (higher is better)
    float average_error = (valid_tests > 0) ? total_error / valid_tests : 1.0f;
    float fitness = 1.0f / (1.0f + average_error);
    
    // Integrate with PLN reasoning accuracy if enabled
    if (moses->integrate_with_pln && moses->atomspace) {
        float reasoning_accuracy = moses->atomspace->reasoning_accuracy;
        program->reasoning_accuracy = reasoning_accuracy;
        fitness = fitness * 0.7f + reasoning_accuracy * 0.3f;
    }
    
    // Efficiency bonus (shorter programs are better)
    float efficiency_bonus = 1.0f - (float)program->instruction_count / MOSES_MAX_PROGRAM_SIZE;
    program->efficiency_score = efficiency_bonus;
    fitness += efficiency_bonus * 0.1f;
    
    program->fitness_score = fitness;
    moses->population->total_evaluations++;
    
    return fitness;
}

// Print program
void moses_print_program(const moses_program_t* program) {
    if (!program) return;
    
    printf("Program %lu (Gen %lu) - Fitness: %.3f\n", 
           program->program_id, program->generation, program->fitness_score);
    printf("  Instructions: %zu, Executions: %lu\n", 
           program->instruction_count, program->execution_count);
    printf("  Reasoning accuracy: %.3f, Efficiency: %.3f\n",
           program->reasoning_accuracy, program->efficiency_score);
    
    for (size_t i = 0; i < program->instruction_count && i < 10; i++) {
        const moses_instruction_t* instr = &program->instructions[i];
        printf("    %zu: ", i);
        
        switch (instr->op_type) {
            case MOSES_OP_CONSTANT:
                printf("CONST %.3f", instr->operands.constant_value);
                break;
            case MOSES_OP_VARIABLE:
                printf("VAR[%u]", instr->operands.variable_index);
                break;
            case MOSES_OP_PLN_AND:
                printf("AND[%u,%u]", instr->operands.binary_op.arg1_index, 
                       instr->operands.binary_op.arg2_index);
                break;
            case MOSES_OP_PLN_OR:
                printf("OR[%u,%u]", instr->operands.binary_op.arg1_index,
                       instr->operands.binary_op.arg2_index);
                break;
            case MOSES_OP_PLN_NOT:
                printf("NOT[%u]", instr->operands.unary_arg_index);
                break;
            default:
                printf("OP%d", instr->op_type);
                break;
        }
        
        printf(" -> %.3f (TV: %.2f/%.2f)\n", 
               instr->output_value, instr->truth_value.strength, instr->truth_value.confidence);
    }
    
    if (program->instruction_count > 10) {
        printf("    ... (%zu more instructions)\n", program->instruction_count - 10);
    }
}

// Add test case
bool moses_add_test_case(
    moses_system_t* moses,
    struct ggml_tensor* input,
    float expected_output) {
    
    if (!moses || !input) return false;
    
    // Resize arrays if needed
    size_t new_size = moses->test_case_count + 1;
    
    struct ggml_tensor** new_cases = realloc(moses->test_cases, 
                                           new_size * sizeof(struct ggml_tensor*));
    if (!new_cases) return false;
    
    float* new_outputs = realloc(moses->target_outputs, new_size * sizeof(float));
    if (!new_outputs) {
        free(new_cases);
        return false;
    }
    
    moses->test_cases = new_cases;
    moses->target_outputs = new_outputs;
    
    moses->test_cases[moses->test_case_count] = input;
    moses->target_outputs[moses->test_case_count] = expected_output;
    moses->test_case_count++;
    
    printf("Added test case %zu with expected output %.3f\n", 
           moses->test_case_count, expected_output);
    
    return true;
}