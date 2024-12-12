from src.open_clip import create_model_from_pretrained, get_tokenizer
import torch

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
text = tokenizer("A diagram of the human heart").cuda()
model.cuda().eval()
x = model.text.transformer.embeddings.word_embeddings(text)
# print(x)
# y = model.text.pooler(model.text.transformer(text),attention_mask = None)
# print(model.text.pooler(model.text.transformer(text),attention_mask = None))
# print(model.text.pooler(model.text.transformer(inputs_embeds=x),attention_mask = None))
print(model.encode_text(x,True))
print(model.encode_text(text,False))