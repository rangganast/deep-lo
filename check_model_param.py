import torch, gc
from model import PointTransformerLO38

DEVICE = "cpu"

model = PointTransformerLO38().to(DEVICE)

print('total param',  sum([param.nelement() for param in model.parameters()]))

gc.collect()
torch.cuda.empty_cache()