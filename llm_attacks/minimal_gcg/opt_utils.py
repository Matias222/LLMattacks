import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_attacks import get_embedding_matrix, get_embeddings

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # New format for better compression
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed. -> Adversarial suffix
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    
    #print(embed_weights.shape)
    #print(input_ids[input_slice].shape[0],embed_weights.shape[0])

    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0) #proyeccion a las dimensiones del embedding
    
    #print(input_embeds.shape)

    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach() #embedding originales
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], #Embeddings before suffix
            input_embeds, #Nuevos embeddings del sufijo
            embeds[:,input_slice.stop:,:] #Embeddings despues
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
    
    loss.backward()
    
    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True) #[adv length,vocab size], direccion de la gradiente para cada uno de los tokens del vocab
    
    #print(grad.shape)

    return grad

def sample_control_all_tokens_cosine(model,control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.inf

    embed_weights = get_embedding_matrix(model)

    seq_len = control_toks.shape[0]
    vocab_size = grad.shape[1]

    # ── 2. Top-k candidates for every position ─────────────────────
    top_idx = (-grad).topk(topk, dim=1).indices         # [seq_len, topk]

    # ── 3. Sample one candidate per (batch,row) uniformly in [0,k) ─
    rand_idx        = torch.randint(0, topk, (batch_size, seq_len), device=grad.device)
    top_idx_exp     = top_idx.unsqueeze(0).expand(batch_size, -1, -1)   # [B,S,k]
    repl_tokens     = torch.gather(top_idx_exp, 2, rand_idx.unsqueeze(-1)).squeeze(-1)  # [B,S]


    # ── 4. Get embeddings (orig & replacement) ─────────────────────

    emb_orig = get_embeddings(model, control_toks).detach() #embedding originales
    emb_repl = get_embeddings(model, repl_tokens).detach() #embedding originales


    emb_orig=emb_orig.unsqueeze(0).expand(batch_size, -1, -1)

    print(emb_orig.shape)
    print(emb_repl.shape)
    #print()

    diff_emb = F.normalize(emb_orig - emb_repl, dim=2)                        # [B,S,D]
 
    print(diff_emb.shape)
    print(embed_weights.shape)

    logits = diff_emb @ embed_weights.T      # [B,S,V]
    
    if not_allowed_tokens is not None:
        logits[:, :, not_allowed_tokens.to(grad.device)] = -np.inf
    
    print("ACA ESTAMOS",logits.shape)

    new_tokens  = logits.argmax(dim=-1).long()                 # [B,S]

    print(control_toks[0])
    print(new_tokens[0])

    return new_tokens

def sample_control_all_tokens(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.inf  # prevent them from being selected

    top_indices = (-grad).topk(topk, dim=1).indices     # [seq_len, topk]
    seq_len = control_toks.shape[0]
    control_toks = control_toks.to(grad.device)

    #for i in range(top_indices.shape[0]):
    #    print(top_indices[i])
    #    continue

    #for i in range(seq_len):
    #    for j in range(seq_len):
    #        if not torch.equal(top_indices[i], top_indices[j]):
    #            print("ACAAA",i,j)

    #torch.set_printoptions(threshold=float('inf'))
    #print(top_indices)

    #print(top_indices)

    # Repeat the original tokens across the batch
    original_control_toks = control_toks.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_len]

    #print(original_control_toks)

    # Generate random indices from 0 to topk-1 for each token in each batch
    rand_idx = torch.randint(0, topk, (batch_size, seq_len), device=grad.device)  # [batch_size, seq_len]

    #print(rand_idx)
    #print(rand_idx.shape)

    # Expand top_indices to [batch_size, seq_len, topk]
    top_indices_exp = top_indices.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, topk]

    #print(top_indices)
    #print("*"*50)
    #print(top_indices_exp)

    # Use gather to select one random token from topk for each position in each sequence
    rand_replacements = torch.gather(top_indices_exp, 2, rand_idx.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]

    

    #print(rand_replacements)

    return rand_replacements


def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.inf

    #print(control_toks.shape)
    #print(len(control_toks),batch_size)

    top_indices = (-grad).topk(topk, dim=1).indices

    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)

    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)

    new_token_val = torch.gather(  #Para cada batch eligue un random
        top_indices[new_token_pos],  # [batch_size, topk] (el adv_length esta colapsado)
        1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )


    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    #Lo que realmente esta haciendo, es diviendo las cadenas en batch/seq_length, queda algo de 5 reemplazos por posicion
    #Entonces para cada indice de seq_length, vamos a tener 5 posibles reemplazos en un espacio muestral de 256!

    #print(new_control_toks)

    return new_control_toks


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        #print(control_cand[i][:26].shape)
        decoded_str = tokenizer.decode(control_cand[i][:100], skip_special_tokens=True)

        #print(decoded_str)
        #print("*"*50)

        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str.strip())
            else:
                count += 1
        else:
            cands.append(decoded_str.strip())

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
    return cands


def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    
    print("Batch size",batch_size)

    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device) #reemplazando en la seccion adversarial del prompt
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), " 
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids ; gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits
    

def forward(*, model, input_ids, attention_mask, batch_size=512):

    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        
        batch_input_ids = input_ids[i:i+batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask
    
    return torch.cat(logits, dim=0)

def target_loss(logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction='none')
    target_len = target_slice.stop - target_slice.start
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])

    #probemos con el decay

    #print(loss.shape)

    return loss.mean(dim=-1) #Esta sacando el promedio pero el loss mas importante es el primero y asi y asi

    weights = torch.exp(-0.1 * torch.arange(target_len)).to(loss.device)
    weights = weights / weights.sum()

    weighted_loss = (loss * weights.unsqueeze(0)).sum(dim=1)  # shape: (batch_size), esta metrica no funciona, no llega a encontrar la cadena inicial, tal vez porque el desfase sea muy achorado? en ese caso veriamos cambios en el primer token, e igual tenemos I

    return weighted_loss

def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            **kwargs
        ).to(device).eval()
    
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if 'Llama-3.2' in tokenizer_path:
        
        print("Llama-3.2 Tokenizer")

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer