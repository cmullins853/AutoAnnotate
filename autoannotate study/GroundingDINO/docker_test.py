from groundingdino.util.inference import load_model
import torch

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.pyy", "weights/groundingdino_swint_ogc.pth")
model = model.to('cuda:0')
print(torch.cuda.is_available())
print('DONE!')