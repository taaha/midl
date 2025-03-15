from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)

processor = PaliGemmaProcessor.from_pretrained("google/paligemma2-3b-mix-448")
tokenizer = processor.tokenizer

class_ = "cat"

# tokenizer.encode(class_) -> [4991]
# tokenizer.encode(class_)[0] -> 4991

breakpoint()
