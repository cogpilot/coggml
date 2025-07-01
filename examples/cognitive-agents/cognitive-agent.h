#pragma once

#include "ggml.h"
#include "ggml-rpc.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct cognitive_agent cognitive_agent;
typedef struct hypergraph_memory hypergraph_memory;
typedef struct task_orchestrator task_orchestrator;
typedef struct reasoning_engine reasoning_engine;
typedef struct attention_economy attention_economy;

// Cognitive tensor packet for enhanced RPC communication
typedef struct {
    rpc_tensor base_tensor;          // Standard ggml tensor
    float attention_weight;          // Economic attention value
    uint32_t cognitive_type;         // Type of cognitive operation
    uint64_t source_agent_id;        // Originating agent
    uint64_t target_agent_id;        // Target agent
    char meta_context[256];          // Context information
    float salience_score;            // Relevance measure
    uint32_t recursion_depth;        // Self-reference depth
    uint64_t timestamp;              // When created
} cognitive_tensor_packet;

// Cognitive operation types
enum cognitive_type {
    COGNITIVE_TYPE_MEMORY = 1,
    COGNITIVE_TYPE_TASK = 2,
    COGNITIVE_TYPE_REASONING = 3,
    COGNITIVE_TYPE_ATTENTION = 4,
    COGNITIVE_TYPE_COMMUNICATION = 5
};

// Hypergraph node types
enum node_type {
    NODE_TYPE_CONCEPT = 1,
    NODE_TYPE_LINK = 2,
    NODE_TYPE_RELATION = 3
};

// Hypergraph node structure
typedef struct {
    ggml_tensor* data;           // Node content as tensor
    ggml_tensor* embedding;      // Semantic embedding
    uint32_t node_type;          // Node type
    float truth_value;           // PLN truth value
    float confidence;            // PLN confidence
    uint64_t creation_time;      // Timestamp
    uint64_t last_access;        // Last accessed
} hypergraph_node;

// Hypergraph memory system
struct hypergraph_memory {
    ggml_context* ctx;
    
    // Storage
    hypergraph_node** nodes;
    size_t node_count;
    size_t capacity;
    
    // Statistics
    uint64_t total_accesses;
    uint64_t cache_hits;
};

// Attention economy state
struct attention_economy {
    float total_attention;       // Total available attention
    float allocated_attention;   // Currently allocated
    
    // Attention allocation per cognitive function
    float memory_allocation;
    float reasoning_allocation;
    float communication_allocation;
    float self_modification_allocation;
    
    // Economic parameters
    float min_threshold;         // Minimum attention to process
    float decay_rate;           // Attention decay over time
    float novelty_bonus;        // Bonus for novel information
    
    // Performance tracking
    float performance_history[100];
    size_t history_index;
    size_t history_size;
};

// Task orchestrator
struct task_orchestrator {
    // Current task state
    ggml_tensor* current_goal;
    ggml_tensor* context_state;
    
    // Task queue
    void** pending_tasks;
    size_t task_count;
    size_t task_capacity;
};

// Reasoning engine
struct reasoning_engine {
    ggml_context* ctx;
    
    // Reasoning state
    ggml_tensor* current_beliefs;
    ggml_tensor* inference_rules;
    
    // Performance metrics
    float reasoning_accuracy;
    uint64_t inferences_made;
};

// Main cognitive agent structure
struct cognitive_agent {
    ggml_context* ctx;
    ggml_backend_t backend;
    
    // Cognitive components
    hypergraph_memory* memory;
    task_orchestrator* tasks;
    reasoning_engine* reasoning;
    attention_economy* attention;
    
    // Network identity
    uint64_t agent_id;
    char endpoint[256];
    float attention_weight;
    
    // State tracking
    uint64_t messages_sent;
    uint64_t messages_received;
    uint64_t cycles_completed;
};

// Core agent functions
cognitive_agent* create_cognitive_agent(const char* endpoint);
void cleanup_cognitive_agent(cognitive_agent* agent);
uint64_t generate_agent_id(void);

// Memory system functions
hypergraph_memory* init_hypergraph_memory(ggml_context* ctx);
void cleanup_hypergraph_memory(hypergraph_memory* mem);
void add_knowledge(hypergraph_memory* mem, const char* concept, 
                   float* embedding, size_t emb_size);
hypergraph_node* find_concept(hypergraph_memory* mem, const char* concept);

// Attention economy functions
attention_economy* init_attention_economy(void);
void cleanup_attention_economy(attention_economy* attention);
float compute_salience(ggml_tensor* tensor, attention_economy* attention);
void allocate_attention(attention_economy* attention, float amount, uint32_t target);
void update_performance_history(attention_economy* attention, float performance);

// Task orchestration functions
task_orchestrator* init_task_orchestrator(void);
void cleanup_task_orchestrator(task_orchestrator* orch);

// Reasoning engine functions
reasoning_engine* init_reasoning_engine(ggml_context* ctx);
void cleanup_reasoning_engine(reasoning_engine* reasoning);

// Communication functions
void send_cognitive_tensor(cognitive_agent* sender, uint64_t target_agent_id,
                          ggml_tensor* tensor, float attention_weight);
void process_incoming_tensor(cognitive_agent* receiver, 
                           cognitive_tensor_packet* msg);

// Utility functions
uint64_t get_timestamp(void);
float compute_tensor_similarity(ggml_tensor* a, ggml_tensor* b);
uint32_t infer_cognitive_type(ggml_tensor* tensor);

#ifdef __cplusplus
}
#endif