# Faster-GCG Implementation Guide

This document explains how to use the Faster-GCG optimizations implemented in your codebase.

## What Was Implemented

### Technique 1: Distance Regularization
- **File**: `llm_attacks/minimal_gcg/opt_utils.py`
- **Function**: `token_gradients()` (modified)
- **New parameter**: `reg_weight` (default: 0.0)
- **What it does**: Adds a regularization term `w · ||X_j - X_k||` to penalize tokens that are far apart in embedding space, improving gradient approximation accuracy.

### Technique 2: Greedy Sampling
- **File**: `llm_attacks/minimal_gcg/opt_utils.py`
- **Function**: `sample_control_greedy()` (new function)
- **What it does**: Deterministically selects candidates from best to worst according to gradient, eliminating randomness for faster convergence.

## How to Use in Your Notebook

### Option 1: Use Technique 1 Only (Regularized Gradients)

Replace this line in your notebook:
```python
# OLD (Original GCG)
coordinate_grad = token_gradients(model,
                input_ids,
                suffix_manager._control_slice,
                suffix_manager._target_slice,
                suffix_manager._loss_slice)
```

With this:
```python
# NEW (Faster-GCG Technique 1)
REG_WEIGHT = 4.0  # Use 4.0 for Llama models, 5.0 for Vicuna (from paper)
coordinate_grad = token_gradients(model,
                input_ids,
                suffix_manager._control_slice,
                suffix_manager._target_slice,
                suffix_manager._loss_slice,
                reg_weight=REG_WEIGHT)  # <-- Added parameter
```

### Option 2: Use Technique 2 Only (Greedy Sampling)

Replace this line in your notebook:
```python
# OLD (Original GCG - Random Sampling)
from llm_attacks.minimal_gcg.opt_utils import sample_control

new_adv_suffix_toks = sample_control(adv_suffix_tokens,
                   coordinate_grad,
                   batch_size,
                   topk=topk,
                   temp=1,
                   not_allowed_tokens=not_allowed_tokens)
```

With this:
```python
# NEW (Faster-GCG Technique 2 - Greedy Sampling)
from llm_attacks.minimal_gcg.opt_utils import sample_control_greedy

new_adv_suffix_toks = sample_control_greedy(adv_suffix_tokens,
                   coordinate_grad,
                   batch_size,
                   topk=topk,
                   temp=1,
                   not_allowed_tokens=not_allowed_tokens)
```

### Option 3: Use Both Techniques (Recommended - Full Faster-GCG)

Combine both modifications:

```python
# Technique 1: Regularized gradients
REG_WEIGHT = 4.0  # Recommended for Llama-3.2
coordinate_grad = token_gradients(model,
                input_ids,
                suffix_manager._control_slice,
                suffix_manager._target_slice,
                suffix_manager._loss_slice,
                reg_weight=REG_WEIGHT)

# Technique 2: Greedy sampling
from llm_attacks.minimal_gcg.opt_utils import sample_control_greedy

new_adv_suffix_toks = sample_control_greedy(adv_suffix_tokens,
                   coordinate_grad,
                   batch_size,
                   topk=topk,
                   temp=1,
                   not_allowed_tokens=not_allowed_tokens)
```

## Complete Modified Loop

Here's your complete optimization loop with both techniques:

```python
import importlib
import llm_attacks.minimal_gcg.opt_utils

# Reload the module
importlib.reload(llm_attacks.minimal_gcg.opt_utils)

# Import functions
from llm_attacks.minimal_gcg.opt_utils import token_gradients, sample_control_greedy, get_logits, target_loss, get_filtered_cands

plotlosses = PlotLosses()
not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
adv_suffix = adv_string_init
correctas = []

# Faster-GCG parameter
REG_WEIGHT = 4.0  # Recommended for Llama-3.2

for i in range(num_steps):

    # Step 1. Encode user prompt (behavior + adv suffix) as tokens
    input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix)
    input_ids = input_ids.to(device)

    # Step 2. Compute Coordinate Gradient with Regularization (Technique 1)
    coordinate_grad = token_gradients(model,
                    input_ids,
                    suffix_manager._control_slice,
                    suffix_manager._target_slice,
                    suffix_manager._loss_slice,
                    reg_weight=REG_WEIGHT)  # <-- NEW: Technique 1

    # Step 3. Sample a batch of new tokens
    with torch.no_grad():

        # Step 3.1 Slice the input to locate the adversarial suffix
        adv_suffix_tokens = input_ids[suffix_manager._control_slice].to(device)

        # Step 3.2 Greedy sample replacements (Technique 2)
        new_adv_suffix_toks = sample_control_greedy(adv_suffix_tokens,  # <-- NEW: Technique 2
                       coordinate_grad,
                       batch_size,
                       topk=topk,
                       temp=1,
                       not_allowed_tokens=not_allowed_tokens)

        # Step 3.3 Filter candidates
        print(len(new_adv_suffix_toks), new_adv_suffix_toks.shape)

        new_adv_suffix = get_filtered_cands(tokenizer,
                                            new_adv_suffix_toks,
                                            filter_cand=True,
                                            curr_control=adv_suffix)

        # Step 3.4 Compute loss and select best
        logits, ids = get_logits(model=model,
                                 tokenizer=tokenizer,
                                 input_ids=input_ids,
                                 control_slice=suffix_manager._control_slice,
                                 test_controls=new_adv_suffix,
                                 return_ids=True,
                                 batch_size=512)

        losses = target_loss(logits, ids, suffix_manager._target_slice)

        best_new_adv_suffix_id = losses.argmin()
        best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]
        current_loss = losses[best_new_adv_suffix_id]

        # Update the running adv_suffix
        adv_suffix = best_new_adv_suffix
        is_success, res_model = check_for_attack_success(model,
                                 tokenizer,
                                 suffix_manager.get_input_ids(adv_string=adv_suffix).to(device),
                                 suffix_manager._assistant_role_slice,
                                 test_prefixes)

    # Plot and print results
    plotlosses.update({'Loss': current_loss.detach().cpu().numpy()})
    plotlosses.send()

    print(f"Iteracion:{i}\nPassed:{is_success}\nCurrent Suffix:{best_new_adv_suffix}\nRes model:{res_model}", end='\n\n')

    if is_success:
        correctas.append(best_new_adv_suffix)

    # Clean up
    del coordinate_grad, adv_suffix_tokens, logits, new_adv_suffix
    gc.collect()
    torch.cuda.empty_cache()
```

## Expected Performance Improvements

According to the Faster-GCG paper:
- **Faster convergence**: Lower loss in fewer iterations
- **Higher success rate**: Better jailbreak performance
- **Reduced computational cost**: Can achieve better results with fewer iterations (e.g., 100 instead of 500)

## Recommended Settings

For **Llama-3.2-3B-Instruct**:
- `REG_WEIGHT = 4.0` (similar to Llama-2 in the paper)
- `batch_size = 256` (reduced from 512)
- `num_steps = 100` (reduced from 500 for 1/10 computational cost)

You can also keep `batch_size = 512` and `num_steps = 500` for even better results.

## Notes

- Both techniques work independently but perform best when combined
- The regularization weight `w` is not very sensitive (3-7 works well according to the paper)
- Greedy sampling eliminates randomness, so results are more deterministic
- You should see lower loss values faster compared to original GCG
