#pragma once

//
// Financial Account Tensor Architecture
//
// This header implements financial account modeling through tensor embeddings,
// building on the cognitive tensor infrastructure with Matula-Goebel encodings.
// Financial accounts are represented as points in high-dimensional tensor space
// where similarity becomes distance and transaction patterns form geometric structures.
//

#include "ggml-cognitive-tensor.h"
#include "ggml.h"
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Financial tensor configuration
#define GGML_FINANCIAL_MAX_ACCOUNTS 1000
#define GGML_FINANCIAL_MAX_TRANSACTIONS 10000
#define GGML_FINANCIAL_EMBEDDING_DIM 64
#define GGML_FINANCIAL_TEMPORAL_DIM 12
#define GGML_FINANCIAL_BEHAVIORAL_DIM 32

// Account types with prime dimension encoding
typedef enum {
    GGML_ACCOUNT_CHECKING = 2,    // Prime dimension 2 (binary): active/inactive
    GGML_ACCOUNT_SAVINGS = 3,     // Prime dimension 3 (ternary): low/medium/high activity
    GGML_ACCOUNT_CREDIT = 5,      // Prime dimension 5: five utilization levels
    GGML_ACCOUNT_INVESTMENT = 7,  // Prime dimension 7: seven risk categories
    GGML_ACCOUNT_BUSINESS = 11,   // Prime dimension 11: business complexity levels
    GGML_ACCOUNT_SHELL = 13       // Prime dimension 13: shell company indicators
} ggml_financial_account_type_t;

// Transaction types
typedef enum {
    GGML_TRANSACTION_DEPOSIT,
    GGML_TRANSACTION_WITHDRAWAL,
    GGML_TRANSACTION_TRANSFER,
    GGML_TRANSACTION_WIRE,
    GGML_TRANSACTION_CHECK,
    GGML_TRANSACTION_ELECTRONIC
} ggml_financial_transaction_type_t;

// Financial account structure
typedef struct {
    uint32_t account_id;
    ggml_financial_account_type_t account_type;
    float balance;
    float average_balance;
    uint32_t transaction_count;
    time_t created_time;
    time_t last_activity;
    
    // Tensor embeddings
    struct ggml_tensor* primary_embedding;     // [EMBEDDING_DIM] - core account features
    struct ggml_tensor* temporal_embedding;    // [TEMPORAL_DIM] - time-based patterns
    struct ggml_tensor* behavioral_embedding;  // [BEHAVIORAL_DIM] - usage patterns
    
    // Risk and anomaly indicators
    float risk_score;
    float anomaly_score;
    bool flagged_for_review;
    
    // Matula encoding for account relationship tree
    uint32_t matula_encoding;
    ggml_complex_t relationship_phase;
} ggml_financial_account_t;

// Financial transaction structure
typedef struct {
    uint32_t transaction_id;
    uint32_t from_account_id;
    uint32_t to_account_id;
    ggml_financial_transaction_type_t type;
    float amount;
    time_t timestamp;
    
    // Transaction vector in tensor space
    struct ggml_tensor* transaction_vector;    // [EMBEDDING_DIM] - transaction characteristics
    
    // Pattern detection
    float structuring_score;    // Breaking large transactions into small ones
    float layering_score;      // Moving money through multiple accounts
    float integration_score;   // Mixing illicit with legitimate funds
} ggml_financial_transaction_t;

// Financial tensor system
typedef struct {
    // Core cognitive kernel
    ggml_cognitive_kernel_t* cognitive_kernel;
    
    // Financial-specific tensors
    struct ggml_tensor* account_embeddings;       // [MAX_ACCOUNTS × EMBEDDING_DIM]
    struct ggml_tensor* transaction_flows;        // [MAX_ACCOUNTS × MAX_ACCOUNTS × TEMPORAL_DIM]
    struct ggml_tensor* relationship_graph;       // [MAX_ACCOUNTS × MAX_ACCOUNTS × 3] - 3D relationship tensor
    struct ggml_tensor* anomaly_patterns;         // [PATTERN_COUNT × EMBEDDING_DIM]
    struct ggml_tensor* clustering_centroids;     // [CLUSTER_COUNT × EMBEDDING_DIM]
    
    // Account and transaction storage
    ggml_financial_account_t* accounts;
    ggml_financial_transaction_t* transactions;
    uint32_t account_count;
    uint32_t transaction_count;
    
    // Configuration
    uint32_t max_accounts;
    uint32_t max_transactions;
    uint32_t embedding_dim;
    float anomaly_threshold;
    float clustering_threshold;
} ggml_financial_tensor_system_t;

// Core system functions
GGML_API ggml_financial_tensor_system_t* ggml_financial_tensor_system_init(
    struct ggml_context* ctx,
    uint32_t max_accounts,
    uint32_t max_transactions);

GGML_API void ggml_financial_tensor_system_free(
    ggml_financial_tensor_system_t* system);

// Account operations
GGML_API uint32_t ggml_financial_add_account(
    ggml_financial_tensor_system_t* system,
    struct ggml_context* ctx,
    ggml_financial_account_type_t account_type,
    float initial_balance);

GGML_API void ggml_financial_update_account_embedding(
    ggml_financial_tensor_system_t* system,
    struct ggml_context* ctx,
    uint32_t account_id);

GGML_API float ggml_financial_account_similarity(
    ggml_financial_tensor_system_t* system,
    uint32_t account_id1,
    uint32_t account_id2);

// Transaction operations
GGML_API uint32_t ggml_financial_add_transaction(
    ggml_financial_tensor_system_t* system,
    struct ggml_context* ctx,
    uint32_t from_account_id,
    uint32_t to_account_id,
    ggml_financial_transaction_type_t type,
    float amount);

GGML_API void ggml_financial_update_transaction_flows(
    ggml_financial_tensor_system_t* system,
    struct ggml_context* ctx);

GGML_API struct ggml_tensor* ggml_financial_trace_money_flow(
    ggml_financial_tensor_system_t* system,
    struct ggml_context* ctx,
    uint32_t source_account_id,
    uint32_t max_hops);

// Anomaly detection
GGML_API void ggml_financial_detect_anomalies(
    ggml_financial_tensor_system_t* system,
    struct ggml_context* ctx);

GGML_API float ggml_financial_compute_anomaly_score(
    ggml_financial_tensor_system_t* system,
    uint32_t account_id);

GGML_API struct ggml_tensor* ggml_financial_find_similar_accounts(
    ggml_financial_tensor_system_t* system,
    struct ggml_context* ctx,
    uint32_t reference_account_id,
    uint32_t max_results);

// Pattern recognition
GGML_API float ggml_financial_detect_structuring(
    ggml_financial_tensor_system_t* system,
    uint32_t account_id);

GGML_API float ggml_financial_detect_layering(
    ggml_financial_tensor_system_t* system,
    uint32_t account_id);

GGML_API struct ggml_tensor* ggml_financial_detect_money_laundering_tree(
    ggml_financial_tensor_system_t* system,
    struct ggml_context* ctx,
    uint32_t root_account_id);

// Clustering and analysis
GGML_API void ggml_financial_cluster_accounts(
    ggml_financial_tensor_system_t* system,
    struct ggml_context* ctx,
    uint32_t num_clusters);

GGML_API uint32_t ggml_financial_get_account_cluster(
    ggml_financial_tensor_system_t* system,
    uint32_t account_id);

// Utility functions
GGML_API void ggml_financial_print_account_stats(
    ggml_financial_tensor_system_t* system,
    uint32_t account_id);

GGML_API void ggml_financial_print_system_stats(
    ggml_financial_tensor_system_t* system);

GGML_API void ggml_financial_export_visualization_data(
    ggml_financial_tensor_system_t* system,
    const char* filename);

#ifdef __cplusplus
}
#endif