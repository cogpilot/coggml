#include "ggml-financial-tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Initialize financial tensor system
ggml_financial_tensor_system_t* ggml_financial_tensor_system_init(
    struct ggml_context* ctx,
    uint32_t max_accounts,
    uint32_t max_transactions) {
    
    ggml_financial_tensor_system_t* system = calloc(1, sizeof(ggml_financial_tensor_system_t));
    if (!system) return NULL;
    
    // Initialize cognitive kernel
    system->cognitive_kernel = ggml_cognitive_kernel_init(ctx, 16, 32, 32);
    if (!system->cognitive_kernel) {
        free(system);
        return NULL;
    }
    
    // Set configuration
    system->max_accounts = max_accounts;
    system->max_transactions = max_transactions;
    system->embedding_dim = GGML_FINANCIAL_EMBEDDING_DIM;
    system->anomaly_threshold = 2.0f;
    system->clustering_threshold = 0.8f;
    
    // Allocate account and transaction arrays
    system->accounts = calloc(max_accounts, sizeof(ggml_financial_account_t));
    system->transactions = calloc(max_transactions, sizeof(ggml_financial_transaction_t));
    
    if (!system->accounts || !system->transactions) {
        ggml_financial_tensor_system_free(system);
        return NULL;
    }
    
    // Create financial tensor structures
    system->account_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
        GGML_FINANCIAL_EMBEDDING_DIM, max_accounts);
    
    system->transaction_flows = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
        max_accounts, max_accounts, GGML_FINANCIAL_TEMPORAL_DIM);
    
    system->relationship_graph = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
        max_accounts, max_accounts, 3);
    
    system->anomaly_patterns = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
        GGML_FINANCIAL_EMBEDDING_DIM, 16); // 16 anomaly patterns
    
    system->clustering_centroids = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
        GGML_FINANCIAL_EMBEDDING_DIM, 8); // 8 clusters
    
    // Initialize tensor data to zero
    ggml_set_zero(system->account_embeddings);
    ggml_set_zero(system->transaction_flows);
    ggml_set_zero(system->relationship_graph);
    ggml_set_zero(system->anomaly_patterns);
    ggml_set_zero(system->clustering_centroids);
    
    printf("Financial tensor system initialized: %u accounts, %u transactions\n", 
           max_accounts, max_transactions);
    
    return system;
}

// Free financial tensor system
void ggml_financial_tensor_system_free(ggml_financial_tensor_system_t* system) {
    if (!system) return;
    
    if (system->cognitive_kernel) {
        ggml_cognitive_kernel_free(system->cognitive_kernel);
    }
    
    free(system->accounts);
    free(system->transactions);
    free(system);
}

// Add financial account with tensor embedding
uint32_t ggml_financial_add_account(
    ggml_financial_tensor_system_t* system,
    struct ggml_context* ctx,
    ggml_financial_account_type_t account_type,
    float initial_balance) {
    
    if (!system || system->account_count >= system->max_accounts) {
        return UINT32_MAX;
    }
    
    uint32_t account_id = system->account_count++;
    ggml_financial_account_t* account = &system->accounts[account_id];
    
    // Initialize account data
    account->account_id = account_id;
    account->account_type = account_type;
    account->balance = initial_balance;
    account->average_balance = initial_balance;
    account->transaction_count = 0;
    account->created_time = time(NULL);
    account->last_activity = time(NULL);
    account->risk_score = 0.0f;
    account->anomaly_score = 0.0f;
    account->flagged_for_review = false;
    
    // Create tensor embeddings
    account->primary_embedding = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 
        GGML_FINANCIAL_EMBEDDING_DIM);
    account->temporal_embedding = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 
        GGML_FINANCIAL_TEMPORAL_DIM);
    account->behavioral_embedding = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 
        GGML_FINANCIAL_BEHAVIORAL_DIM);
    
    // Initialize embeddings based on account type using prime encoding
    float* primary_data = (float*)account->primary_embedding->data;
    float* temporal_data = (float*)account->temporal_embedding->data;
    float* behavioral_data = (float*)account->behavioral_embedding->data;
    
    // Prime-based encoding for account type
    uint32_t prime_base = (uint32_t)account_type;
    for (uint32_t i = 0; i < GGML_FINANCIAL_EMBEDDING_DIM; i++) {
        primary_data[i] = sinf(prime_base * i * 0.1f) * 0.1f;
    }
    
    // Temporal pattern initialization
    for (uint32_t i = 0; i < GGML_FINANCIAL_TEMPORAL_DIM; i++) {
        temporal_data[i] = cosf(prime_base * i * 0.05f) * 0.05f;
    }
    
    // Behavioral pattern initialization
    for (uint32_t i = 0; i < GGML_FINANCIAL_BEHAVIORAL_DIM; i++) {
        behavioral_data[i] = sinf(prime_base * i * 0.02f) * 0.02f;
    }
    
    // Encode account relationship using Matula encoding
    char tree_expression[64];
    snprintf(tree_expression, sizeof(tree_expression), "(%u)", account_type);
    ggml_matula_encoding_t encoding = ggml_encode_tree(tree_expression, 
        &system->cognitive_kernel->prime_cache);
    
    account->matula_encoding = encoding.matula_value;
    account->relationship_phase = encoding.phase;
    
    // Update system-wide account embeddings tensor
    ggml_financial_update_account_embedding(system, ctx, account_id);
    
    return account_id;
}

// Update account embedding in system tensor
void ggml_financial_update_account_embedding(
    ggml_financial_tensor_system_t* system,
    struct ggml_context* ctx,
    uint32_t account_id) {
    
    if (!system || account_id >= system->account_count) return;
    
    ggml_financial_account_t* account = &system->accounts[account_id];
    float* system_embeddings = (float*)system->account_embeddings->data;
    float* account_data = (float*)account->primary_embedding->data;
    
    // Copy account embedding to system tensor
    size_t offset = account_id * GGML_FINANCIAL_EMBEDDING_DIM;
    for (uint32_t i = 0; i < GGML_FINANCIAL_EMBEDDING_DIM; i++) {
        system_embeddings[offset + i] = account_data[i];
    }
    
    // Update based on recent transactions and balance changes
    float balance_factor = logf(1.0f + fabsf(account->balance)) * 0.01f;
    float activity_factor = logf(1.0f + account->transaction_count) * 0.01f;
    
    for (uint32_t i = 0; i < GGML_FINANCIAL_EMBEDDING_DIM; i++) {
        system_embeddings[offset + i] += balance_factor * sinf(i * 0.1f) + 
                                        activity_factor * cosf(i * 0.1f);
    }
}

// Compute similarity between accounts using cosine similarity
float ggml_financial_account_similarity(
    ggml_financial_tensor_system_t* system,
    uint32_t account_id1,
    uint32_t account_id2) {
    
    if (!system || account_id1 >= system->account_count || 
        account_id2 >= system->account_count) {
        return 0.0f;
    }
    
    float* embeddings = (float*)system->account_embeddings->data;
    float* emb1 = embeddings + account_id1 * GGML_FINANCIAL_EMBEDDING_DIM;
    float* emb2 = embeddings + account_id2 * GGML_FINANCIAL_EMBEDDING_DIM;
    
    // Compute cosine similarity
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (uint32_t i = 0; i < GGML_FINANCIAL_EMBEDDING_DIM; i++) {
        dot_product += emb1[i] * emb2[i];
        norm1 += emb1[i] * emb1[i];
        norm2 += emb2[i] * emb2[i];
    }
    
    if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;
    
    return dot_product / (sqrtf(norm1) * sqrtf(norm2));
}

// Add transaction with tensor encoding
uint32_t ggml_financial_add_transaction(
    ggml_financial_tensor_system_t* system,
    struct ggml_context* ctx,
    uint32_t from_account_id,
    uint32_t to_account_id,
    ggml_financial_transaction_type_t type,
    float amount) {
    
    if (!system || system->transaction_count >= system->max_transactions ||
        from_account_id >= system->account_count || 
        to_account_id >= system->account_count) {
        return UINT32_MAX;
    }
    
    uint32_t transaction_id = system->transaction_count++;
    ggml_financial_transaction_t* transaction = &system->transactions[transaction_id];
    
    // Initialize transaction data
    transaction->transaction_id = transaction_id;
    transaction->from_account_id = from_account_id;
    transaction->to_account_id = to_account_id;
    transaction->type = type;
    transaction->amount = amount;
    transaction->timestamp = time(NULL);
    transaction->structuring_score = 0.0f;
    transaction->layering_score = 0.0f;
    transaction->integration_score = 0.0f;
    
    // Create transaction vector tensor
    transaction->transaction_vector = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 
        GGML_FINANCIAL_EMBEDDING_DIM);
    
    float* trans_data = (float*)transaction->transaction_vector->data;
    
    // Encode transaction characteristics
    float amount_factor = logf(1.0f + amount) * 0.01f;
    float type_factor = (float)type * 0.1f;
    
    for (uint32_t i = 0; i < GGML_FINANCIAL_EMBEDDING_DIM; i++) {
        trans_data[i] = amount_factor * sinf(type_factor + i * 0.1f);
    }
    
    // Update account balances and statistics
    system->accounts[from_account_id].balance -= amount;
    system->accounts[to_account_id].balance += amount;
    system->accounts[from_account_id].transaction_count++;
    system->accounts[to_account_id].transaction_count++;
    system->accounts[from_account_id].last_activity = time(NULL);
    system->accounts[to_account_id].last_activity = time(NULL);
    
    // Update relationship graph
    float* graph_data = (float*)system->relationship_graph->data;
    size_t graph_offset = (from_account_id * system->max_accounts + to_account_id) * 3;
    
    graph_data[graph_offset + 0] += amount; // Total amount
    graph_data[graph_offset + 1] += 1.0f;   // Transaction count
    graph_data[graph_offset + 2] = (float)time(NULL); // Last transaction time
    
    // Update embeddings
    ggml_financial_update_account_embedding(system, ctx, from_account_id);
    ggml_financial_update_account_embedding(system, ctx, to_account_id);
    
    return transaction_id;
}

// Update transaction flow tensors
void ggml_financial_update_transaction_flows(
    ggml_financial_tensor_system_t* system,
    struct ggml_context* ctx) {
    
    if (!system) return;
    
    float* flow_data = (float*)system->transaction_flows->data;
    
    // Clear existing flow data
    size_t flow_size = system->max_accounts * system->max_accounts * GGML_FINANCIAL_TEMPORAL_DIM;
    memset(flow_data, 0, flow_size * sizeof(float));
    
    // Process all transactions to build flow patterns
    for (uint32_t i = 0; i < system->transaction_count; i++) {
        ggml_financial_transaction_t* trans = &system->transactions[i];
        
        // Get time-based index (hour of day)
        struct tm* time_info = localtime(&trans->timestamp);
        uint32_t hour_idx = time_info->tm_hour;
        
        // Update flow tensor
        size_t flow_offset = (trans->from_account_id * system->max_accounts + 
                             trans->to_account_id) * GGML_FINANCIAL_TEMPORAL_DIM + hour_idx;
        
        flow_data[flow_offset] += trans->amount;
    }
}

// Detect anomalies in account behavior
void ggml_financial_detect_anomalies(
    ggml_financial_tensor_system_t* system,
    struct ggml_context* ctx) {
    
    if (!system) return;
    
    // First, compute cluster centroids for normal behavior
    ggml_financial_cluster_accounts(system, ctx, 8);
    
    // Then compute anomaly scores for each account
    for (uint32_t i = 0; i < system->account_count; i++) {
        system->accounts[i].anomaly_score = ggml_financial_compute_anomaly_score(system, i);
        
        // Flag accounts with high anomaly scores
        if (system->accounts[i].anomaly_score > system->anomaly_threshold) {
            system->accounts[i].flagged_for_review = true;
        }
    }
}

// Compute anomaly score for specific account
float ggml_financial_compute_anomaly_score(
    ggml_financial_tensor_system_t* system,
    uint32_t account_id) {
    
    if (!system || account_id >= system->account_count) return 0.0f;
    
    float* embeddings = (float*)system->account_embeddings->data;
    float* centroids = (float*)system->clustering_centroids->data;
    float* account_emb = embeddings + account_id * GGML_FINANCIAL_EMBEDDING_DIM;
    
    // Find distance to nearest cluster centroid
    float min_distance = INFINITY;
    
    for (uint32_t c = 0; c < 8; c++) {
        float* centroid = centroids + c * GGML_FINANCIAL_EMBEDDING_DIM;
        float distance = 0.0f;
        
        for (uint32_t i = 0; i < GGML_FINANCIAL_EMBEDDING_DIM; i++) {
            float diff = account_emb[i] - centroid[i];
            distance += diff * diff;
        }
        
        distance = sqrtf(distance);
        if (distance < min_distance) {
            min_distance = distance;
        }
    }
    
    return min_distance;
}

// Detect structuring patterns (breaking large transactions into small ones)
float ggml_financial_detect_structuring(
    ggml_financial_tensor_system_t* system,
    uint32_t account_id) {
    
    if (!system || account_id >= system->account_count) return 0.0f;
    
    float structuring_score = 0.0f;
    uint32_t small_transaction_count = 0;
    float total_amount = 0.0f;
    
    // Analyze recent transactions from this account
    time_t current_time = time(NULL);
    time_t window_start = current_time - (24 * 60 * 60); // 24 hours
    
    for (uint32_t i = 0; i < system->transaction_count; i++) {
        ggml_financial_transaction_t* trans = &system->transactions[i];
        
        if (trans->from_account_id == account_id && 
            trans->timestamp >= window_start) {
            
            total_amount += trans->amount;
            
            // Count transactions just under reporting thresholds
            if (trans->amount > 9000.0f && trans->amount < 10000.0f) {
                small_transaction_count++;
            }
        }
    }
    
    // Calculate structuring score
    if (small_transaction_count > 3) {
        structuring_score = small_transaction_count * 0.25f;
    }
    
    return structuring_score;
}

// Detect layering patterns (moving money through multiple accounts)
float ggml_financial_detect_layering(
    ggml_financial_tensor_system_t* system,
    uint32_t account_id) {
    
    if (!system || account_id >= system->account_count) return 0.0f;
    
    float layering_score = 0.0f;
    uint32_t hop_count = 0;
    
    // Use relationship graph to detect rapid money movement
    float* graph_data = (float*)system->relationship_graph->data;
    
    for (uint32_t to_account = 0; to_account < system->account_count; to_account++) {
        size_t offset = (account_id * system->max_accounts + to_account) * 3;
        
        float transaction_count = graph_data[offset + 1];
        float last_transaction_time = graph_data[offset + 2];
        
        // Check for rapid consecutive transactions
        if (transaction_count > 2 && 
            (time(NULL) - last_transaction_time) < 3600) { // Within 1 hour
            hop_count++;
        }
    }
    
    layering_score = hop_count * 0.5f;
    return layering_score;
}

// Simple k-means clustering for account behavior
void ggml_financial_cluster_accounts(
    ggml_financial_tensor_system_t* system,
    struct ggml_context* ctx,
    uint32_t num_clusters) {
    
    if (!system || num_clusters == 0 || num_clusters > 8) return;
    
    float* embeddings = (float*)system->account_embeddings->data;
    float* centroids = (float*)system->clustering_centroids->data;
    
    // Initialize centroids randomly
    for (uint32_t c = 0; c < num_clusters; c++) {
        for (uint32_t i = 0; i < GGML_FINANCIAL_EMBEDDING_DIM; i++) {
            centroids[c * GGML_FINANCIAL_EMBEDDING_DIM + i] = 
                (float)rand() / RAND_MAX * 0.1f - 0.05f;
        }
    }
    
    // Simple k-means iterations
    for (uint32_t iter = 0; iter < 10; iter++) {
        // Assign accounts to clusters
        uint32_t* assignments = calloc(system->account_count, sizeof(uint32_t));
        
        for (uint32_t a = 0; a < system->account_count; a++) {
            float* account_emb = embeddings + a * GGML_FINANCIAL_EMBEDDING_DIM;
            float min_dist = INFINITY;
            uint32_t best_cluster = 0;
            
            for (uint32_t c = 0; c < num_clusters; c++) {
                float* centroid = centroids + c * GGML_FINANCIAL_EMBEDDING_DIM;
                float dist = 0.0f;
                
                for (uint32_t i = 0; i < GGML_FINANCIAL_EMBEDDING_DIM; i++) {
                    float diff = account_emb[i] - centroid[i];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            
            assignments[a] = best_cluster;
        }
        
        // Update centroids
        uint32_t* cluster_counts = calloc(num_clusters, sizeof(uint32_t));
        memset(centroids, 0, num_clusters * GGML_FINANCIAL_EMBEDDING_DIM * sizeof(float));
        
        for (uint32_t a = 0; a < system->account_count; a++) {
            uint32_t cluster = assignments[a];
            float* account_emb = embeddings + a * GGML_FINANCIAL_EMBEDDING_DIM;
            float* centroid = centroids + cluster * GGML_FINANCIAL_EMBEDDING_DIM;
            
            for (uint32_t i = 0; i < GGML_FINANCIAL_EMBEDDING_DIM; i++) {
                centroid[i] += account_emb[i];
            }
            cluster_counts[cluster]++;
        }
        
        // Average centroids
        for (uint32_t c = 0; c < num_clusters; c++) {
            if (cluster_counts[c] > 0) {
                float* centroid = centroids + c * GGML_FINANCIAL_EMBEDDING_DIM;
                for (uint32_t i = 0; i < GGML_FINANCIAL_EMBEDDING_DIM; i++) {
                    centroid[i] /= cluster_counts[c];
                }
            }
        }
        
        free(assignments);
        free(cluster_counts);
    }
}

// Print account statistics
void ggml_financial_print_account_stats(
    ggml_financial_tensor_system_t* system,
    uint32_t account_id) {
    
    if (!system || account_id >= system->account_count) return;
    
    ggml_financial_account_t* account = &system->accounts[account_id];
    
    printf("Account %u Statistics:\n", account_id);
    printf("  Type: %d\n", account->account_type);
    printf("  Balance: $%.2f\n", account->balance);
    printf("  Average Balance: $%.2f\n", account->average_balance);
    printf("  Transactions: %u\n", account->transaction_count);
    printf("  Risk Score: %.3f\n", account->risk_score);
    printf("  Anomaly Score: %.3f\n", account->anomaly_score);
    printf("  Flagged: %s\n", account->flagged_for_review ? "Yes" : "No");
    printf("  Matula Encoding: %u\n", account->matula_encoding);
    printf("  Structuring Score: %.3f\n", 
           ggml_financial_detect_structuring(system, account_id));
    printf("  Layering Score: %.3f\n", 
           ggml_financial_detect_layering(system, account_id));
}

// Print system-wide statistics
void ggml_financial_print_system_stats(
    ggml_financial_tensor_system_t* system) {
    
    if (!system) return;
    
    uint32_t flagged_count = 0;
    float total_balance = 0.0f;
    float avg_anomaly_score = 0.0f;
    
    for (uint32_t i = 0; i < system->account_count; i++) {
        if (system->accounts[i].flagged_for_review) flagged_count++;
        total_balance += system->accounts[i].balance;
        avg_anomaly_score += system->accounts[i].anomaly_score;
    }
    
    if (system->account_count > 0) {
        avg_anomaly_score /= system->account_count;
    }
    
    printf("Financial Tensor System Statistics:\n");
    printf("  Total Accounts: %u\n", system->account_count);
    printf("  Total Transactions: %u\n", system->transaction_count);
    printf("  Total Balance: $%.2f\n", total_balance);
    printf("  Average Anomaly Score: %.3f\n", avg_anomaly_score);
    printf("  Flagged Accounts: %u (%.1f%%)\n", 
           flagged_count, (float)flagged_count / system->account_count * 100.0f);
    printf("  Embedding Dimension: %u\n", system->embedding_dim);
    printf("  Anomaly Threshold: %.3f\n", system->anomaly_threshold);
}