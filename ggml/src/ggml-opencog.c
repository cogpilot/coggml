#include "ggml-opencog.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Generate unique atom ID
static uint64_t generate_atom_id(opencog_atomspace_t* atomspace) {
    return atomspace->next_atom_id++;
}

// Initialize OpenCog AtomSpace
opencog_atomspace_t* opencog_atomspace_init(struct ggml_context* ctx) {
    opencog_atomspace_t* atomspace = malloc(sizeof(opencog_atomspace_t));
    if (!atomspace) return NULL;
    
    atomspace->ctx = ctx;
    
    // Initialize atom storage
    atomspace->atom_capacity = OPENCOG_MAX_ATOMS;
    atomspace->atoms = calloc(atomspace->atom_capacity, sizeof(opencog_atom_t));
    atomspace->atom_count = 0;
    atomspace->next_atom_id = 1;
    
    // Initialize ECAN parameters
    atomspace->attention_decay_rate = 0.95f;
    atomspace->attention_threshold = 0.1f;
    atomspace->importance_diffusion_rate = 0.1f;
    
    // Initialize PLN parameters
    atomspace->default_strength = 0.8f;
    atomspace->default_confidence = 0.9f;
    
    // Initialize performance metrics
    atomspace->total_inferences = 0;
    atomspace->successful_inferences = 0;
    atomspace->reasoning_accuracy = 0.0f;
    
    atomspace->initialized = true;
    atomspace->cogfluence_system = NULL;
    
    printf("OpenCog AtomSpace initialized with capacity for %zu atoms\n", 
           atomspace->atom_capacity);
    
    return atomspace;
}

// Free OpenCog AtomSpace
void opencog_atomspace_free(opencog_atomspace_t* atomspace) {
    if (!atomspace) return;
    
    // Free atom storage
    for (size_t i = 0; i < atomspace->atom_count; i++) {
        if (atomspace->atoms[i].outgoing) {
            free(atomspace->atoms[i].outgoing);
        }
        if (atomspace->atoms[i].incoming) {
            free(atomspace->atoms[i].incoming);
        }
    }
    
    free(atomspace->atoms);
    free(atomspace);
}

// Add node to AtomSpace
uint64_t opencog_add_node(
    opencog_atomspace_t* atomspace,
    opencog_atom_type_t type,
    const char* name) {
    
    if (!atomspace || !name || atomspace->atom_count >= atomspace->atom_capacity) {
        return 0;
    }
    
    opencog_atom_t* atom = &atomspace->atoms[atomspace->atom_count];
    uint64_t atom_id = generate_atom_id(atomspace);
    
    // Initialize atom
    atom->atom_id = atom_id;
    strncpy(atom->name, name, OPENCOG_MAX_ATOM_NAME - 1);
    atom->name[OPENCOG_MAX_ATOM_NAME - 1] = '\0';
    atom->type = type;
    
    // Initialize truth value
    atom->truth_value.strength = atomspace->default_strength;
    atom->truth_value.confidence = atomspace->default_confidence;
    atom->truth_value.count = 1.0f;
    
    // Initialize attention value
    atom->attention_value.sti = 0.0f;
    atom->attention_value.lti = 0.0f;
    atom->attention_value.vlti = 0.0f;
    
    // Create tensor encoding
    atom->tensor_encoding = ggml_new_tensor_1d(atomspace->ctx, GGML_TYPE_F32, 128);
    ggml_set_zero(atom->tensor_encoding);
    
    // Initialize name-based encoding
    if (atom->tensor_encoding->type == GGML_TYPE_F32) {
        float* data = (float*)atom->tensor_encoding->data;
        for (int i = 0; i < 128 && i < strlen(name); i++) {
            data[i] = (float)name[i] / 255.0f;
        }
    }
    
    // Initialize links
    atom->outgoing = NULL;
    atom->outgoing_count = 0;
    atom->outgoing_capacity = 0;
    atom->incoming = NULL;
    atom->incoming_count = 0;
    atom->incoming_capacity = 0;
    
    // Initialize metadata
    atom->creation_time = (uint64_t)time(NULL);
    atom->last_access = atom->creation_time;
    atom->is_deleted = false;
    atom->cogfluence_unit_id = 0;
    
    atomspace->atom_count++;
    
    printf("Added OpenCog node '%s' (type %d, ID %lu)\n", name, type, atom_id);
    
    return atom_id;
}

// Add link to AtomSpace
uint64_t opencog_add_link(
    opencog_atomspace_t* atomspace,
    opencog_atom_type_t type,
    uint64_t* outgoing,
    size_t outgoing_count) {
    
    if (!atomspace || !outgoing || outgoing_count == 0 || 
        atomspace->atom_count >= atomspace->atom_capacity) {
        return 0;
    }
    
    // Verify all outgoing atoms exist
    for (size_t i = 0; i < outgoing_count; i++) {
        if (!opencog_get_atom(atomspace, outgoing[i])) {
            return 0;
        }
    }
    
    opencog_atom_t* atom = &atomspace->atoms[atomspace->atom_count];
    uint64_t atom_id = generate_atom_id(atomspace);
    
    // Initialize atom
    atom->atom_id = atom_id;
    snprintf(atom->name, OPENCOG_MAX_ATOM_NAME, "Link_%lu", atom_id);
    atom->type = type;
    
    // Initialize truth value
    atom->truth_value.strength = atomspace->default_strength;
    atom->truth_value.confidence = atomspace->default_confidence;
    atom->truth_value.count = 1.0f;
    
    // Initialize attention value
    atom->attention_value.sti = 0.0f;
    atom->attention_value.lti = 0.0f;
    atom->attention_value.vlti = 0.0f;
    
    // Create tensor encoding (aggregate from outgoing)
    atom->tensor_encoding = ggml_new_tensor_1d(atomspace->ctx, GGML_TYPE_F32, 128);
    ggml_set_zero(atom->tensor_encoding);
    
    // Initialize outgoing links
    atom->outgoing_capacity = outgoing_count;
    atom->outgoing = malloc(atom->outgoing_capacity * sizeof(uint64_t));
    memcpy(atom->outgoing, outgoing, outgoing_count * sizeof(uint64_t));
    atom->outgoing_count = outgoing_count;
    
    // Initialize incoming links
    atom->incoming = NULL;
    atom->incoming_count = 0;
    atom->incoming_capacity = 0;
    
    // Add incoming links to outgoing atoms
    for (size_t i = 0; i < outgoing_count; i++) {
        opencog_atom_t* outgoing_atom = opencog_get_atom(atomspace, outgoing[i]);
        if (outgoing_atom) {
            if (outgoing_atom->incoming_count >= outgoing_atom->incoming_capacity) {
                outgoing_atom->incoming_capacity = outgoing_atom->incoming_capacity == 0 ? 
                    4 : outgoing_atom->incoming_capacity * 2;
                outgoing_atom->incoming = realloc(outgoing_atom->incoming,
                    outgoing_atom->incoming_capacity * sizeof(uint64_t));
            }
            outgoing_atom->incoming[outgoing_atom->incoming_count++] = atom_id;
        }
    }
    
    // Initialize metadata
    atom->creation_time = (uint64_t)time(NULL);
    atom->last_access = atom->creation_time;
    atom->is_deleted = false;
    atom->cogfluence_unit_id = 0;
    
    atomspace->atom_count++;
    
    printf("Added OpenCog link (type %d, ID %lu) with %zu outgoing atoms\n", 
           type, atom_id, outgoing_count);
    
    return atom_id;
}

// Get atom by ID
opencog_atom_t* opencog_get_atom(
    opencog_atomspace_t* atomspace,
    uint64_t atom_id) {
    
    if (!atomspace || atom_id == 0) return NULL;
    
    // Linear search for now (could be optimized with hash table)
    for (size_t i = 0; i < atomspace->atom_count; i++) {
        if (atomspace->atoms[i].atom_id == atom_id && !atomspace->atoms[i].is_deleted) {
            atomspace->atoms[i].last_access = (uint64_t)time(NULL);
            return &atomspace->atoms[i];
        }
    }
    
    return NULL;
}

// PLN AND operation
opencog_truth_value_t opencog_pln_and(
    opencog_truth_value_t tv1,
    opencog_truth_value_t tv2) {
    
    opencog_truth_value_t result;
    
    // PLN AND formula: Min(s1, s2) with confidence combination
    result.strength = fminf(tv1.strength, tv2.strength);
    result.confidence = (tv1.confidence * tv2.confidence) / 
                       (tv1.confidence + tv2.confidence - tv1.confidence * tv2.confidence);
    result.count = fminf(tv1.count, tv2.count);
    
    return result;
}

// PLN OR operation
opencog_truth_value_t opencog_pln_or(
    opencog_truth_value_t tv1,
    opencog_truth_value_t tv2) {
    
    opencog_truth_value_t result;
    
    // PLN OR formula: Max(s1, s2) with confidence combination
    result.strength = fmaxf(tv1.strength, tv2.strength);
    result.confidence = (tv1.confidence * tv2.confidence) / 
                       (tv1.confidence + tv2.confidence - tv1.confidence * tv2.confidence);
    result.count = fmaxf(tv1.count, tv2.count);
    
    return result;
}

// PLN NOT operation
opencog_truth_value_t opencog_pln_not(opencog_truth_value_t tv) {
    opencog_truth_value_t result;
    
    result.strength = 1.0f - tv.strength;
    result.confidence = tv.confidence;
    result.count = tv.count;
    
    return result;
}

// Set truth value
void opencog_set_truth_value(
    opencog_atomspace_t* atomspace,
    uint64_t atom_id,
    float strength,
    float confidence) {
    
    opencog_atom_t* atom = opencog_get_atom(atomspace, atom_id);
    if (!atom) return;
    
    atom->truth_value.strength = fmaxf(0.0f, fminf(1.0f, strength));
    atom->truth_value.confidence = fmaxf(0.0f, fminf(1.0f, confidence));
    atom->truth_value.count = 1.0f;
}

// Get truth value
opencog_truth_value_t opencog_get_truth_value(
    opencog_atomspace_t* atomspace,
    uint64_t atom_id) {
    
    opencog_truth_value_t default_tv = {0.0f, 0.0f, 0.0f};
    
    opencog_atom_t* atom = opencog_get_atom(atomspace, atom_id);
    if (!atom) return default_tv;
    
    return atom->truth_value;
}

// Set attention value
void opencog_set_attention_value(
    opencog_atomspace_t* atomspace,
    uint64_t atom_id,
    float sti,
    float lti,
    float vlti) {
    
    opencog_atom_t* atom = opencog_get_atom(atomspace, atom_id);
    if (!atom) return;
    
    atom->attention_value.sti = fmaxf(-1.0f, fminf(1.0f, sti));
    atom->attention_value.lti = fmaxf(0.0f, fminf(1.0f, lti));
    atom->attention_value.vlti = fmaxf(0.0f, fminf(1.0f, vlti));
}

// Get attention value
opencog_attention_value_t opencog_get_attention_value(
    opencog_atomspace_t* atomspace,
    uint64_t atom_id) {
    
    opencog_attention_value_t default_av = {0.0f, 0.0f, 0.0f};
    
    opencog_atom_t* atom = opencog_get_atom(atomspace, atom_id);
    if (!atom) return default_av;
    
    return atom->attention_value;
}

// Update attention values (ECAN)
void opencog_update_attention_values(opencog_atomspace_t* atomspace) {
    if (!atomspace) return;
    
    // Attention decay
    for (size_t i = 0; i < atomspace->atom_count; i++) {
        if (atomspace->atoms[i].is_deleted) continue;
        
        opencog_atom_t* atom = &atomspace->atoms[i];
        
        // Apply decay
        atom->attention_value.sti *= atomspace->attention_decay_rate;
        atom->attention_value.lti *= atomspace->attention_decay_rate;
        
        // Convert STI to LTI over time
        if (atom->attention_value.sti > atomspace->attention_threshold) {
            float transfer = atom->attention_value.sti * 0.1f;
            atom->attention_value.lti += transfer;
            atom->attention_value.sti -= transfer;
        }
        
        // Clamp values
        atom->attention_value.sti = fmaxf(-1.0f, fminf(1.0f, atom->attention_value.sti));
        atom->attention_value.lti = fmaxf(0.0f, fminf(1.0f, atom->attention_value.lti));
        atom->attention_value.vlti = fmaxf(0.0f, fminf(1.0f, atom->attention_value.vlti));
    }
}

// Spread attention
void opencog_spread_attention(
    opencog_atomspace_t* atomspace,
    uint64_t source_atom_id,
    float amount) {
    
    opencog_atom_t* source_atom = opencog_get_atom(atomspace, source_atom_id);
    if (!source_atom) return;
    
    // Spread to outgoing atoms
    if (source_atom->outgoing_count > 0) {
        float spread_amount = amount / source_atom->outgoing_count;
        
        for (size_t i = 0; i < source_atom->outgoing_count; i++) {
            opencog_atom_t* target_atom = opencog_get_atom(atomspace, source_atom->outgoing[i]);
            if (target_atom) {
                target_atom->attention_value.sti += spread_amount;
                target_atom->attention_value.sti = fmaxf(-1.0f, fminf(1.0f, target_atom->attention_value.sti));
            }
        }
    }
    
    // Spread to incoming atoms
    if (source_atom->incoming_count > 0) {
        float spread_amount = amount / source_atom->incoming_count;
        
        for (size_t i = 0; i < source_atom->incoming_count; i++) {
            opencog_atom_t* target_atom = opencog_get_atom(atomspace, source_atom->incoming[i]);
            if (target_atom) {
                target_atom->attention_value.sti += spread_amount;
                target_atom->attention_value.sti = fmaxf(-1.0f, fminf(1.0f, target_atom->attention_value.sti));
            }
        }
    }
}

// Link with Cogfluence system
bool opencog_link_cogfluence(
    opencog_atomspace_t* atomspace,
    cogfluence_system_t* cogfluence_system) {
    
    if (!atomspace || !cogfluence_system) return false;
    
    atomspace->cogfluence_system = cogfluence_system;
    
    printf("Linked OpenCog AtomSpace with Cogfluence system\n");
    
    return true;
}

// Create atom from Cogfluence unit
uint64_t opencog_from_cogfluence_unit(
    opencog_atomspace_t* atomspace,
    cogfluence_knowledge_unit_t* unit) {
    
    if (!atomspace || !unit) return 0;
    
    // Map Cogfluence unit type to OpenCog atom type
    opencog_atom_type_t atom_type = OPENCOG_CONCEPT_NODE;
    switch (unit->type) {
        case COGFLUENCE_CONCEPT: atom_type = OPENCOG_CONCEPT_NODE; break;
        case COGFLUENCE_RELATION: atom_type = OPENCOG_INHERITANCE_LINK; break;
        case COGFLUENCE_RULE: atom_type = OPENCOG_IMPLICATION_LINK; break;
        default: atom_type = OPENCOG_CONCEPT_NODE; break;
    }
    
    uint64_t atom_id = opencog_add_node(atomspace, atom_type, unit->name);
    
    if (atom_id > 0) {
        opencog_atom_t* atom = opencog_get_atom(atomspace, atom_id);
        if (atom) {
            // Copy truth value
            atom->truth_value.strength = unit->truth_value;
            atom->truth_value.confidence = unit->confidence;
            
            // Copy attention value
            atom->attention_value.sti = unit->attention_value;
            atom->attention_value.lti = unit->activation_level;
            
            // Link back to Cogfluence unit
            atom->cogfluence_unit_id = unit->atomspace_id;
            
            // Copy tensor encoding
            if (unit->tensor_encoding) {
                atom->tensor_encoding = ggml_dup(atomspace->ctx, unit->tensor_encoding);
            }
            
            printf("Created OpenCog atom from Cogfluence unit '%s' (ID %lu)\n", 
                   unit->name, atom_id);
        }
    }
    
    return atom_id;
}

// Convert atom to tensor
struct ggml_tensor* opencog_atom_to_tensor(
    opencog_atomspace_t* atomspace,
    uint64_t atom_id) {
    
    opencog_atom_t* atom = opencog_get_atom(atomspace, atom_id);
    if (!atom) return NULL;
    
    if (atom->tensor_encoding) {
        return ggml_dup(atomspace->ctx, atom->tensor_encoding);
    }
    
    // Create default tensor
    struct ggml_tensor* tensor = ggml_new_tensor_1d(atomspace->ctx, GGML_TYPE_F32, 128);
    ggml_set_zero(tensor);
    
    return tensor;
}

// Print atom
void opencog_print_atom(
    opencog_atomspace_t* atomspace,
    uint64_t atom_id) {
    
    opencog_atom_t* atom = opencog_get_atom(atomspace, atom_id);
    if (!atom) return;
    
    printf("Atom %lu: %s (type %d)\n", atom->atom_id, atom->name, atom->type);
    printf("  Truth: strength=%.2f, confidence=%.2f\n", 
           atom->truth_value.strength, atom->truth_value.confidence);
    printf("  Attention: sti=%.2f, lti=%.2f, vlti=%.2f\n",
           atom->attention_value.sti, atom->attention_value.lti, atom->attention_value.vlti);
    printf("  Outgoing: %zu, Incoming: %zu\n", 
           atom->outgoing_count, atom->incoming_count);
    
    if (atom->cogfluence_unit_id > 0) {
        printf("  Cogfluence unit: %lu\n", atom->cogfluence_unit_id);
    }
}

// Print AtomSpace statistics
void opencog_print_atomspace_statistics(opencog_atomspace_t* atomspace) {
    if (!atomspace) return;
    
    printf("\n=== OpenCog AtomSpace Statistics ===\n");
    printf("Atoms: %zu/%zu\n", atomspace->atom_count, atomspace->atom_capacity);
    printf("Total inferences: %lu\n", atomspace->total_inferences);
    printf("Successful inferences: %lu\n", atomspace->successful_inferences);
    printf("Reasoning accuracy: %.2f\n", atomspace->reasoning_accuracy);
    
    // Count atoms by type
    int type_counts[9] = {0};
    for (size_t i = 0; i < atomspace->atom_count; i++) {
        if (!atomspace->atoms[i].is_deleted && 
            atomspace->atoms[i].type >= 1 && atomspace->atoms[i].type <= 8) {
            type_counts[atomspace->atoms[i].type]++;
        }
    }
    
    printf("Atom types:\n");
    printf("  Concept nodes: %d\n", type_counts[OPENCOG_CONCEPT_NODE]);
    printf("  Predicate nodes: %d\n", type_counts[OPENCOG_PREDICATE_NODE]);
    printf("  Variable nodes: %d\n", type_counts[OPENCOG_VARIABLE_NODE]);
    printf("  Inheritance links: %d\n", type_counts[OPENCOG_INHERITANCE_LINK]);
    printf("  Evaluation links: %d\n", type_counts[OPENCOG_EVALUATION_LINK]);
    printf("  Implication links: %d\n", type_counts[OPENCOG_IMPLICATION_LINK]);
    printf("  Similarity links: %d\n", type_counts[OPENCOG_SIMILARITY_LINK]);
    printf("  Member links: %d\n", type_counts[OPENCOG_MEMBER_LINK]);
    
    // Average attention values
    float avg_sti = 0.0f, avg_lti = 0.0f;
    int active_atoms = 0;
    
    for (size_t i = 0; i < atomspace->atom_count; i++) {
        if (!atomspace->atoms[i].is_deleted) {
            avg_sti += atomspace->atoms[i].attention_value.sti;
            avg_lti += atomspace->atoms[i].attention_value.lti;
            active_atoms++;
        }
    }
    
    if (active_atoms > 0) {
        avg_sti /= active_atoms;
        avg_lti /= active_atoms;
        printf("Average attention: STI=%.2f, LTI=%.2f\n", avg_sti, avg_lti);
    }
    
    printf("=====================================\n");
}