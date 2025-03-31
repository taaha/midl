# midl


vlm_logit_lens.py:
Vibe coded using AI


vlm_logit_lens_1.py:
getting logit lens successfully
```
    # len(outputs.sequences[0][input_len:]) -> 20 (number of generated tokens)
    # output.keys() -> odict_keys(['sequences', 'logits', 'hidden_states', 'past_key_values'])
    # len(outputs.hidden_states) -> 20 (number of generated tokens)
    # len(outputs.hidden_states[0]) -> 27 (number of layers)
    # outputs.hidden_states[0][0].shape -> torch.Size([1, 1029, 2304])    (1029 is number of input tokens)
    # outputs.hidden_states[1][0].shape -> torch.Size([1, 1, 2304])
    # outputs.hidden_states[19][0].shape -> torch.Size([1, 1, 2304])
    # outputs.hidden_states[19][26][0][0].shape -> torch.Size([2304])       (19 is token number in sentence, 26 is final layer)

    # model.language_model.lm_head. -> (lm_head): Linear(in_features=2304, out_features=257216, bias=False)
    # model.language_model.lm_head(outputs.hidden_states[19][26][0][0]).shape -> torch.Size([257216])
    # model.language_model.lm_head(outputs.hidden_states[19][26][0][0]) ->
    #                               tensor([ -1.5703,   6.2188, -11.3750,  ...,  -1.5547,  -1.5703,  -1.5703],
    #                               device='cuda:0', dtype=torch.bfloat16, grad_fn=<SqueezeBackward4>)


    """
    working code:

    # Get logits for the last token from the last layer
    last_token_logits = model.language_model.lm_head(outputs.hidden_states[19][26][0][0])
    
    # Convert to probabilities using softmax
    probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
    
    # Get top 5 probabilities and their indices
    top_probs, top_indices = torch.topk(probs, k=5)
    
    # Convert to CPU and regular Python types for easier printing
    top_probs = top_probs.cpu().tolist()
    top_indices = top_indices.cpu().tolist()

    processor.tokenizer.decode(top_indices)
    """
```

conf_checker.py:
```
# tokenizer.encode(class_) -> [4991]
# tokenizer.encode(class_)[0] -> 4991
```

internal_confidence.py:
find internal confidence. has to do with passing
image tokens through lm_head and getting the softmaxx probs
an decode token (i think)

internal_confidence_gemma10b_a.py:
getting logits in npy_files

internal_confidence_gemma10b_b.py:
plotting logits in heatmap
segmentation good but is their bug in confidence heatmap?
what should happen in final layers happening in first layers?
checked with numpy_checker.py but checks out fine? what the hell?

gemma_2b/train_gemma2b.py:
training gemma 2b model on vqav2 dataset (qlora)

gemma_2b/comp_test1.py:
comparing trained and untrained gemma 2b model