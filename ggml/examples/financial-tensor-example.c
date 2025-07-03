#include "ggml-financial-tensor.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

// Example usage demonstrating financial tensor capabilities
int main() {
    printf("Financial Tensor Architecture Example\n");
    printf("====================================\n");
    
    // Initialize context
    struct ggml_init_params params = {
        .mem_size = 128 * 1024 * 1024,  // 128MB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    assert(ctx != NULL);
    
    // Initialize financial tensor system
    ggml_financial_tensor_system_t* system = ggml_financial_tensor_system_init(ctx, 20, 50);
    assert(system != NULL);
    
    // Create accounts representing different entities
    uint32_t corp_account = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_BUSINESS, 1000000.0f);
    uint32_t shell_company = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_SHELL, 0.0f);
    uint32_t personal_account = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_CHECKING, 5000.0f);
    uint32_t savings_account = ggml_financial_add_account(system, ctx, GGML_ACCOUNT_SAVINGS, 25000.0f);
    
    printf("Created accounts:\n");
    printf("  Corporate: %u ($%.0f)\n", corp_account, system->accounts[corp_account].balance);
    printf("  Shell Company: %u ($%.0f)\n", shell_company, system->accounts[shell_company].balance);
    printf("  Personal: %u ($%.0f)\n", personal_account, system->accounts[personal_account].balance);
    printf("  Savings: %u ($%.0f)\n", savings_account, system->accounts[savings_account].balance);
    
    // Process some transactions
    printf("\nProcessing transactions:\n");
    
    // Large corporate to shell transfer
    ggml_financial_add_transaction(system, ctx, corp_account, shell_company, 
                                   GGML_TRANSACTION_WIRE, 100000.0f);
    printf("  $100,000 wire: Corporate -> Shell Company\n");
    
    // Shell company to personal (structured amounts)
    ggml_financial_add_transaction(system, ctx, shell_company, personal_account, 
                                   GGML_TRANSACTION_TRANSFER, 9800.0f);
    printf("  $9,800 transfer: Shell Company -> Personal\n");
    
    ggml_financial_add_transaction(system, ctx, shell_company, personal_account, 
                                   GGML_TRANSACTION_TRANSFER, 9900.0f);
    printf("  $9,900 transfer: Shell Company -> Personal\n");
    
    // Normal personal to savings
    ggml_financial_add_transaction(system, ctx, personal_account, savings_account, 
                                   GGML_TRANSACTION_TRANSFER, 5000.0f);
    printf("  $5,000 transfer: Personal -> Savings\n");
    
    // Show final balances
    printf("\nFinal balances:\n");
    printf("  Corporate: $%.0f\n", system->accounts[corp_account].balance);
    printf("  Shell Company: $%.0f\n", system->accounts[shell_company].balance);
    printf("  Personal: $%.0f\n", system->accounts[personal_account].balance);
    printf("  Savings: $%.0f\n", system->accounts[savings_account].balance);
    
    // Demonstrate account similarity
    float sim_personal_savings = ggml_financial_account_similarity(system, personal_account, savings_account);
    float sim_corporate_shell = ggml_financial_account_similarity(system, corp_account, shell_company);
    
    printf("\nAccount similarities:\n");
    printf("  Personal-Savings: %.3f\n", sim_personal_savings);
    printf("  Corporate-Shell: %.3f\n", sim_corporate_shell);
    
    // Pattern detection
    float structuring_corp = ggml_financial_detect_structuring(system, corp_account);
    float layering_shell = ggml_financial_detect_layering(system, shell_company);
    
    printf("\nPattern detection:\n");
    printf("  Corporate structuring score: %.3f\n", structuring_corp);
    printf("  Shell layering score: %.3f\n", layering_shell);
    
    // Show system statistics
    printf("\nSystem Statistics:\n");
    printf("Total accounts: %u\n", system->account_count);
    printf("Total transactions: %u\n", system->transaction_count);
    
    // Cleanup
    ggml_financial_tensor_system_free(system);
    ggml_free(ctx);
    
    printf("\n✓ Financial tensor demonstration completed successfully!\n");
    printf("\nThis example shows how financial accounts are modeled as tensor embeddings\n");
    printf("where account similarity, transaction patterns, and anomaly detection\n");
    printf("are computed using multi-dimensional tensor operations.\n");
    
    return 0;
}