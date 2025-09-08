import torch
import torch.nn.functional as F

class TokenMixture:
    """
    Mantiene logits de mezcla por posición. alpha = softmax(logits / T).
    """
    def __init__(self, S_emb: torch.Tensor, device: str):
        # S_emb: [L, k, d]
        L, k, _ = S_emb.shape
        self.alpha_logits = torch.zeros(1, L, k, device=device, requires_grad=True)

    def probs(self, T: float = 1.0):
        return F.softmax(self.alpha_logits / T, dim=-1)

def mixture_to_embeddings(alpha_probs: torch.Tensor, S_emb: torch.Tensor) -> torch.Tensor:
    """
    alpha_probs: [B, L, k], S_emb: [L, k, d] -> embeddings: [B, L, d]
    """
    # Ensure both tensors have the same dtype
    alpha_probs = alpha_probs.to(S_emb.dtype)
    return torch.einsum('bls,lsd->bld', alpha_probs, S_emb)

def straight_through_onehot(alpha_probs: torch.Tensor):
    """
    STE opcional para redondeo en forward con gradientes suaves en backward.
    """
    idx = alpha_probs.argmax(dim=-1)                                # [B, L]
    onehot = F.one_hot(idx, num_classes=alpha_probs.shape[-1]).float()
    ste = (onehot - alpha_probs).detach() + alpha_probs
    return ste, idx                                                 # ste: [B, L, k]
    
@torch.no_grad()
def alpha_to_tokens(alpha_probs: torch.Tensor, S_ids: torch.Tensor) -> torch.Tensor:
    """
    Proyecta a ids discretos por posición: [L]
    """
    idx = alpha_probs.argmax(dim=-1)                                # [B, L]
    toks = torch.gather(S_ids.unsqueeze(0), 2, idx.unsqueeze(-1)).squeeze(-1)  # [B, L]
    return toks[0]