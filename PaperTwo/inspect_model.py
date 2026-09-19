import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:",device)
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor=CLIPProcessor.from_pretrained(model_name)
model = model.to(device)
model.eval()
print(model)

print("Number of vision transformer layers:",len(model.vision_model.encoder.layers))

print("Vision hidden size:", model.config.vision_config.hidden_size)
print("Patch size:", model.config.vision_config.patch_size)
print("Image size:", model.config.vision_config.image_size)


# Loading One Test Image
image = Image.open("data/test.jpeg").convert("RGB")
#CLIPProcessor
inputs = processor(images=image,return_tensors="pt")
pixel_values=inputs["pixel_values"].to(device)
print("Input tensor shape:",pixel_values.shape)

# only vision side 
with torch.no_grad():
    outputs=model.vision_model(pixel_values=pixel_values,output_hidden_states=True)

print("Number of hidden state tensors:",len(outputs.hidden_states))

for i, h in enumerate(outputs.hidden_states):
    print(i,h.shape)

layer_output = outputs.hidden_states[1]
cls_rep=layer_output[:,0,:]
print("CLS representation shape:",cls_rep.shape)
