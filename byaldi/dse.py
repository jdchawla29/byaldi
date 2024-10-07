from typing import List, Union

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class DSEModel:
    def __init__(self, model_name: str = "MrLight/dse-qwen2-2b-mrl-v1", device: str = "cuda", use_flash_attention: bool = True):
        self.device = device
        self.pretrained_model_name_or_path = model_name
        
        self.min_pixels = 1 * 28 * 28
        self.max_pixels = 2560 * 28 * 28

        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2" if use_flash_attention else "eager",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).to(device).eval()

        self.processor.tokenizer.padding_side = "left"
        self.model.padding_side = "left"
        print(f"Initialized DSEModel with model_name: {model_name}, device: {device}")

    def get_embedding(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        reps = last_hidden_state[:, -1]
        reps = torch.nn.functional.normalize(reps[:, :512], p=2, dim=-1)
        return reps

    def encode_query(self, queries: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(queries, str):
            queries = [queries]

        query_messages = [
            [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.new("RGB", (28, 28)),
                        "resized_height": 1,
                        "resized_width": 1,
                    },
                    {"type": "text", "text": f"Query: {query}"},
                ],
            }]
            for query in queries
        ]

        query_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            + "<|endoftext|>"
            for msg in query_messages
        ]
        query_image_inputs, query_video_inputs = process_vision_info(query_messages)
        query_inputs = self.processor(
            text=query_texts,
            images=query_image_inputs,
            videos=query_video_inputs,
            padding="longest",
            return_tensors="pt",
        ).to(self.device)
        cache_position = torch.arange(0, len(query_texts))
        query_inputs = self.model.prepare_inputs_for_generation(
            **query_inputs, cache_position=cache_position, use_cache=False
        )

        with torch.no_grad():
            output = self.model(**query_inputs, return_dict=True, output_hidden_states=True)
            query_embeddings = self.get_embedding(output.hidden_states[-1])

        return query_embeddings

    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]

        doc_messages = [
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "resized_height": 680, "resized_width": 680},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            }]
            for image in images
        ]

        doc_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            + "<|endoftext|>"
            for msg in doc_messages
        ]
        doc_image_inputs, doc_video_inputs = process_vision_info(doc_messages)
        doc_inputs = self.processor(
            text=doc_texts,
            images=doc_image_inputs,
            videos=doc_video_inputs,
            padding="longest",
            return_tensors="pt",
        ).to(self.device)
        cache_position = torch.arange(0, len(doc_texts))
        doc_inputs = self.model.prepare_inputs_for_generation(
            **doc_inputs, cache_position=cache_position, use_cache=False
        )

        with torch.no_grad():
            output = self.model(**doc_inputs, return_dict=True, output_hidden_states=True)
            doc_embeddings = self.get_embedding(output.hidden_states[-1])

        return doc_embeddings

    def score(self, query: Union[str, List[str]], document_embeddings: List[torch.Tensor]) -> torch.Tensor:
        # Encode the query
        query_embeddings = self.encode_query(query)
        
        # Stack document embeddings
        document_embeddings_tensor = torch.stack(document_embeddings)
        
        # Calculate scores
        scores = torch.matmul(query_embeddings, document_embeddings_tensor.t())

        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        
        return scores.cpu()
