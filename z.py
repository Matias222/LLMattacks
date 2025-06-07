import torch


def decay(alpha):

    weights = torch.exp(-alpha * torch.arange(20))
    
    #print(weights)
    
    weights = weights / weights.sum()
    
    print(weights)

#decay(0.75)
#decay(0.5)
#decay(0.35)
#decay(0.2)
decay(0.1)
