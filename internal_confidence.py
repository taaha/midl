from methods.paligemma_utils import load_paligemma_state, retrieve_logit_lens_paligemma

model_state = load_paligemma_state()

retrieve_logit_lens = retrieve_logit_lens_paligemma

img_path = "images/COCO_val2014_000000562150.jpg"

caption, softmax_probs = retrieve_logit_lens(model_state, img_path)

print(caption)
print(softmax_probs)

