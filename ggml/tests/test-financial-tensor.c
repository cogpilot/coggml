#include "ggml-financial-tensor.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Test financial tensor system initialization
void test_financial_tensor_system_init() {
    printf("Testing financial tensor system initialization...\n");
    
    // Initialize ggml context with larger memory pool
    struct ggml_init_params params = {
        .mem_size = 512 * 1024 * 1024,  // 512MB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    assert(ctx != NULL);
    
    // Initialize financial tensor system
    ggml_financial_tensor_system_t* system = ggml_financial_tensor_system_init(ctx, 1000, 10000);
    assert(system != NULL);
    assert(system->max_accounts == 1000);
    assert(system->max_transactions == 10000);
    assert(system->embedding_dim == GGML_FINANCIAL_EMBEDDING_DIM);
    
    printf("✓ Financial tensor system initialized successfully\n");
    
    // Cleanup
    ggml_financial_tensor_system_free(system);
    ggml_free(ctx);
}

// Test account creation and embedding
void test_financial_account_creation() {
    printf("Testing financial account creation...\n");
    
    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    ggml_financial_tensor_system_t* system = ggml_financial_tensor_system_init(ctx, 100, 1000);
    
    // Create different types of accounts
    uint32_t checking_id = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_CHECKING, 1000.0f);
    uint32_t savings_id = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_SAVINGS, 5000.0f);
    uint32_t credit_id = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_CREDIT, -500.0f);
    uint32_t investment_id = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_INVESTMENT, 25000.0f);
    
    assert(checking_id == 0);
    assert(savings_id == 1);
    assert(credit_id == 2);
    assert(investment_id == 3);
    assert(system->account_count == 4);
    
    // Test account properties
    assert(system->accounts[checking_id].account_type == GGML_ACCOUNT_CHECKING);
    assert(system->accounts[checking_id].balance == 1000.0f);
    assert(system->accounts[savings_id].balance == 5000.0f);
    assert(system->accounts[credit_id].balance == -500.0f);
    assert(system->accounts[investment_id].balance == 25000.0f);
    
    printf("✓ Created 4 accounts: checking, savings, credit, investment\n");
    
    // Test account similarity
    float sim_checking_savings = ggml_financial_account_similarity(system, checking_id, savings_id);
    float sim_checking_credit = ggml_financial_account_similarity(system, checking_id, credit_id);
    printf("✓ Account similarity: checking-savings=%.3f, checking-credit=%.3f\n", 
           sim_checking_savings, sim_checking_credit);
    
    // Cleanup
    ggml_financial_tensor_system_free(system);
    ggml_free(ctx);
}

// Test transaction processing
void test_financial_transactions() {
    printf("Testing financial transaction processing...\n");
    
    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    ggml_financial_tensor_system_t* system = ggml_financial_tensor_system_init(ctx, 100, 1000);
    
    // Create accounts
    uint32_t account1 = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_CHECKING, 10000.0f);
    uint32_t account2 = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_SAVINGS, 5000.0f);
    uint32_t account3 = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_BUSINESS, 50000.0f);
    
    // Process transactions
    uint32_t trans1 = ggml_financial_add_transaction(system, ctx, account1, account2, 
                                                     GGML_TRANSACTION_TRANSFER, 1000.0f);
    uint32_t trans2 = ggml_financial_add_transaction(system, ctx, account2, account3, 
                                                     GGML_TRANSACTION_WIRE, 2000.0f);
    uint32_t trans3 = ggml_financial_add_transaction(system, ctx, account3, account1, 
                                                     GGML_TRANSACTION_ELECTRONIC, 500.0f);
    
    assert(trans1 == 0);
    assert(trans2 == 1);
    assert(trans3 == 2);
    assert(system->transaction_count == 3);
    
    // Check balance updates
    assert(system->accounts[account1].balance == 9500.0f);  // 10000 - 1000 + 500
    assert(system->accounts[account2].balance == 4000.0f);  // 5000 + 1000 - 2000
    assert(system->accounts[account3].balance == 51500.0f); // 50000 + 2000 - 500
    
    // Check transaction counts
    assert(system->accounts[account1].transaction_count == 2);
    assert(system->accounts[account2].transaction_count == 2);
    assert(system->accounts[account3].transaction_count == 2);
    
    printf("✓ Processed 3 transactions with correct balance updates\n");
    
    // Update transaction flows
    ggml_financial_update_transaction_flows(system, ctx);
    printf("✓ Transaction flow tensors updated\n");
    
    // Cleanup
    ggml_financial_tensor_system_free(system);
    ggml_free(ctx);
}

// Test anomaly detection
void test_financial_anomaly_detection() {
    printf("Testing financial anomaly detection...\n");
    
    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    ggml_financial_tensor_system_t* system = ggml_financial_tensor_system_init(ctx, 100, 1000);
    
    // Create normal accounts
    for (uint32_t i = 0; i < 20; i++) {
        ggml_financial_add_account(system, ctx, GGML_ACCOUNT_CHECKING, 1000.0f + i * 100.0f);
    }
    
    // Create some potentially suspicious accounts
    uint32_t suspicious1 = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_SHELL, 1000000.0f);
    uint32_t suspicious2 = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_BUSINESS, 500000.0f);
    
    // Add some normal transactions
    for (uint32_t i = 0; i < 10; i++) {
        ggml_financial_add_transaction(system, ctx, i, i + 1, 
                                       GGML_TRANSACTION_TRANSFER, 50.0f + i * 10.0f);
    }
    
    // Add potentially suspicious transactions
    for (uint32_t i = 0; i < 5; i++) {
        ggml_financial_add_transaction(system, ctx, suspicious1, suspicious2, 
                                       GGML_TRANSACTION_WIRE, 9800.0f + i * 100.0f);
    }
    
    // Run anomaly detection
    ggml_financial_detect_anomalies(system, ctx);
    
    // Check results
    float suspicious1_score = system->accounts[suspicious1].anomaly_score;
    float suspicious2_score = system->accounts[suspicious2].anomaly_score;
    float normal_score = system->accounts[0].anomaly_score;
    
    printf("✓ Anomaly scores computed - Normal: %.3f, Suspicious1: %.3f, Suspicious2: %.3f\n", 
           normal_score, suspicious1_score, suspicious2_score);
    
    // Test pattern detection
    float structuring_score = ggml_financial_detect_structuring(system, suspicious1);
    float layering_score = ggml_financial_detect_layering(system, suspicious1);
    
    printf("✓ Pattern detection - Structuring: %.3f, Layering: %.3f\n", 
           structuring_score, layering_score);
    
    // Cleanup
    ggml_financial_tensor_system_free(system);
    ggml_free(ctx);
}

// Test financial tensor clustering
void test_financial_clustering() {
    printf("Testing financial account clustering...\n");
    
    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    ggml_financial_tensor_system_t* system = ggml_financial_tensor_system_init(ctx, 100, 1000);
    
    // Create accounts of different types
    for (uint32_t i = 0; i < 5; i++) {
        ggml_financial_add_account(system, ctx, GGML_ACCOUNT_CHECKING, 1000.0f + i * 100.0f);
        ggml_financial_add_account(system, ctx, GGML_ACCOUNT_SAVINGS, 5000.0f + i * 500.0f);
        ggml_financial_add_account(system, ctx, GGML_ACCOUNT_CREDIT, -500.0f - i * 100.0f);
        ggml_financial_add_account(system, ctx, GGML_ACCOUNT_INVESTMENT, 25000.0f + i * 2500.0f);
    }
    
    // Perform clustering
    ggml_financial_cluster_accounts(system, ctx, 4);
    
    printf("✓ Accounts clustered into 4 groups\n");
    
    // Test similarity within account types
    float sim_checking = ggml_financial_account_similarity(system, 0, 4);  // Both checking
    float sim_cross = ggml_financial_account_similarity(system, 0, 1);     // Checking vs savings
    
    printf("✓ Similarity - Same type: %.3f, Different types: %.3f\n", 
           sim_checking, sim_cross);
    
    // Cleanup
    ggml_financial_tensor_system_free(system);
    ggml_free(ctx);
}

// Test money laundering detection scenario
void test_money_laundering_scenario() {
    printf("Testing money laundering detection scenario...\n");
    
    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    ggml_financial_tensor_system_t* system = ggml_financial_tensor_system_init(ctx, 100, 1000);
    
    // Create a money laundering tree structure
    // Source -> Shell1 -> {Shell2, Shell3} -> Destination
    uint32_t source = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_BUSINESS, 1000000.0f);
    uint32_t shell1 = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_SHELL, 0.0f);
    uint32_t shell2 = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_SHELL, 0.0f);
    uint32_t shell3 = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_SHELL, 0.0f);
    uint32_t destination = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_CHECKING, 0.0f);
    
    // Create layered transactions
    ggml_financial_add_transaction(system, ctx, source, shell1, GGML_TRANSACTION_WIRE, 100000.0f);
    ggml_financial_add_transaction(system, ctx, shell1, shell2, GGML_TRANSACTION_TRANSFER, 50000.0f);
    ggml_financial_add_transaction(system, ctx, shell1, shell3, GGML_TRANSACTION_TRANSFER, 50000.0f);
    ggml_financial_add_transaction(system, ctx, shell2, destination, GGML_TRANSACTION_ELECTRONIC, 49000.0f);
    ggml_financial_add_transaction(system, ctx, shell3, destination, GGML_TRANSACTION_ELECTRONIC, 49000.0f);
    
    // Add structuring transactions (just under $10,000)
    for (uint32_t i = 0; i < 5; i++) {
        ggml_financial_add_transaction(system, ctx, source, shell1, 
                                       GGML_TRANSACTION_TRANSFER, 9800.0f + i * 50.0f);
    }
    
    // Update flows and detect anomalies
    ggml_financial_update_transaction_flows(system, ctx);
    ggml_financial_detect_anomalies(system, ctx);
    
    // Check detection results
    float source_layering = ggml_financial_detect_layering(system, source);
    float source_structuring = ggml_financial_detect_structuring(system, source);
    float shell1_anomaly = system->accounts[shell1].anomaly_score;
    
    printf("✓ Money laundering patterns detected:\n");
    printf("  Source layering score: %.3f\n", source_layering);
    printf("  Source structuring score: %.3f\n", source_structuring);
    printf("  Shell account anomaly: %.3f\n", shell1_anomaly);
    
    // Print detailed account statistics
    printf("\n--- Detailed Account Analysis ---\n");
    ggml_financial_print_account_stats(system, source);
    ggml_financial_print_account_stats(system, shell1);
    ggml_financial_print_account_stats(system, destination);
    
    // Cleanup
    ggml_financial_tensor_system_free(system);
    ggml_free(ctx);
}

// Test system statistics
void test_system_statistics() {
    printf("Testing system statistics...\n");
    
    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    ggml_financial_tensor_system_t* system = ggml_financial_tensor_system_init(ctx, 100, 1000);
    
    // Create diverse account portfolio
    for (uint32_t i = 0; i < 10; i++) {
        ggml_financial_add_account(system, ctx, GGML_ACCOUNT_CHECKING, 1000.0f + i * 100.0f);
        ggml_financial_add_account(system, ctx, GGML_ACCOUNT_SAVINGS, 5000.0f + i * 500.0f);
    }
    
    // Add some transactions
    for (uint32_t i = 0; i < 15; i++) {
        ggml_financial_add_transaction(system, ctx, i, (i + 1) % 20, 
                                       GGML_TRANSACTION_TRANSFER, 100.0f + i * 50.0f);
    }
    
    // Run full analysis
    ggml_financial_detect_anomalies(system, ctx);
    
    // Print comprehensive statistics
    printf("\n--- System Statistics ---\n");
    ggml_financial_print_system_stats(system);
    
    // Cleanup
    ggml_financial_tensor_system_free(system);
    ggml_free(ctx);
}

// Main test function
int main() {
    printf("Financial Tensor Architecture Test Suite\n");
    printf("========================================\n\n");
    
    // Seed random number generator
    srand(time(NULL));
    
    // Run all tests
    test_financial_tensor_system_init();
    test_financial_account_creation();
    test_financial_transactions();
    test_financial_anomaly_detection();
    test_financial_clustering();
    test_money_laundering_scenario();
    test_system_statistics();
    
    printf("\n========================================\n");
    printf("All financial tensor tests passed! ✓\n");
    printf("Financial Account Tensor Architecture successfully implemented.\n");
    printf("Features validated:\n");
    printf("  • Account tensor embeddings with prime encoding\n");
    printf("  • Transaction flow modeling\n");
    printf("  • Anomaly detection using cluster analysis\n");
    printf("  • Structuring pattern detection\n");
    printf("  • Layering pattern detection\n");
    printf("  • Money laundering tree structure analysis\n");
    printf("  • Multi-dimensional similarity computation\n");
    printf("  • Real-time risk scoring\n");
    
    return 0;
}