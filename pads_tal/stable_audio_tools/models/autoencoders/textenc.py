import torch
import math
import logging, warnings
import gc

import typing as tp
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from .encdec import ResidualUnit
class T5Encoder(nn.Module):

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl", "google/t5-v1_1-xl", "google/t5-v1_1-xxl"]
    
    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/t5-v1_1-xl": 2048,
        "google/t5-v1_1-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
            self,
            t5_model_name: str = "t5-base",
            latent_dim : int=128,
            inner_latent_dim : int=2048,
            max_length: str = 128,
            target_length: str = 32,
            enable_grad: bool = False,
            project_out: bool = False,
            use_decoding : bool=True,
            use_outer_encoder : bool=False,
    ):

        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__()
        
        from transformers import T5EncoderModel, AutoTokenizer

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)

        if not use_outer_encoder:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                    self.encoder = T5EncoderModel.from_pretrained(t5_model_name).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)
                finally:
                    logging.disable(previous_level)
        else:
            self.tokenizer=None
            self.encoder=None
            
        dim = self.T5_MODEL_DIMS[t5_model_name]
        self.inner_latent_dim = inner_latent_dim
        self.linear_proj = nn.Linear(dim, inner_latent_dim)
        self.latent_dim = latent_dim
        self.linear_proj2 = nn.Conv1d(in_channels=max_length, out_channels=target_length, kernel_size=1) if max_length!=target_length else nn.Identity()
        
        self.use_decoding = use_decoding
        if not use_outer_encoder and use_decoding:
            from transformers import T5ForConditionalGeneration
            # Load tokenizer and full T5 model
            logging.disable(logging.ERROR)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self.decoder = T5ForConditionalGeneration.from_pretrained(t5_model_name).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)
                finally:
                    logging.disable(previous_level)


    def forward(self, texts: tp.Union[tp.List[str], tp.Tuple[torch.Tensor]], 
            device: tp.Union[torch.device, str], 
            return_ids=False,
            return_before_proj=False,
            attention_mask=None,
        ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor], tp.Tuple[torch.Tensor, torch.Tensor]]:
        
        self.linear_proj.to(device)
        self.linear_proj2.to(device)
        if self.encoder!=None and self.tokenizer!=None:
            self.encoder.to(device)

            encoded = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

            with torch.amp.autocast(device_type="cuda") and torch.set_grad_enabled(self.enable_grad):
                embeddings_real= self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )["last_hidden_state"]    
        else:
            return_before_proj=False
            embeddings_real = texts
            
        if not isinstance(self.linear_proj, nn.Identity):
            linear_proj_dtype = next(self.linear_proj.parameters()).dtype
            embeddings = embeddings_real.to(linear_proj_dtype)

        embeddings = self.linear_proj(embeddings)

        if self.inner_latent_dim != self.latent_dim : 
            embeddings = torch.nn.functional.adaptive_avg_pool1d(
                embeddings, self.latent_dim,
            )
        if attention_mask!=None:
            embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        embeddings = self.linear_proj2(embeddings)

        if return_ids:
            if return_before_proj :
                return embeddings, input_ids, embeddings_real, attention_mask
            else:
                return embeddings, input_ids
        else:
            if return_before_proj :
                return embeddings, embeddings_real, attention_mask
            else:
                return embeddings

    def generate(
            self,
            encoder_embeddings: torch.Tensor,
            encoder_attention_mask: torch.Tensor,
            device: tp.Union[torch.device, str],
            return_encoder_ids=False,
    ) -> tp.Union[tp.List[str], tp.Tuple[tp.List[str], torch.Tensor]]:
        """
        Args:
            encoder_embeddings: [B, seq_len, dim] from T5Encoder
            encoder_attention_mask: [B, seq_len] boolean mask
            device: torch device
        Returns:
            List of decoded strings
        """
        if not self.use_decoding:
            print("[MultiModalVAE Text Encoder] generator option not ON")
            return encoder_embeddings

        self.decoder.to(device)
        self.decoder.eval()

        encoder_embeddings = encoder_embeddings.to(device=device, dtype=torch.float16)
        encoder_attention_mask = encoder_attention_mask.to(device=device)
        
        length = (encoder_attention_mask == 0).nonzero(as_tuple=True)[1][0].item()

        with torch.amp.autocast(device_type="cuda") and torch.set_grad_enabled(self.enable_grad):
            generated_ids = self.decoder.generate(
                encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=encoder_embeddings),
                attention_mask=encoder_attention_mask,
                max_length=length+1,
                num_beams=4,
            )
        
        generated_strings =  self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        if return_encoder_ids :
            encoded = self.tokenizer(
                generated_strings,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            output_ids = encoded["input_ids"].to(device)


            return generated_strings, output_ids
        else:
            return generated_strings



class TextDecoder(nn.Module):
    def __init__(
        self,
        tokenizer,
        latent_dim : int = 64,
        target_length=32,
        max_length=128,
        dropout=0.1,
        enable_grad: bool = True,
        tf_num_layers=2,
        tf_nhead:int=8,
        tf_dim_feedforward:int = 512, # 128*4
        conv_kernel_size=3,
    ):
        super().__init__()
        self.enable_grad = enable_grad
        self.tokenizer= tokenizer

        self.linear_proj = nn.Linear(latent_dim, 768)
        self.linear_proj2 = nn.Conv1d(in_channels=target_length, out_channels=max_length, kernel_size=1)

        self.pos_embedding = nn.Embedding(max_length, 768)

        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=tf_nhead,
                                                   dim_feedforward=tf_dim_feedforward, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=tf_num_layers, 
                                            norm=nn.LayerNorm(768))
        self.proj_to_vocab = nn.Linear(768, tokenizer.vocab_size)

        self.pos_embedding = self.pos_embedding.train(enable_grad).requires_grad_(enable_grad)

        self.decoder = self.decoder.train(enable_grad).requires_grad_(enable_grad)
        self.proj_to_vocab = self.proj_to_vocab.train(enable_grad).requires_grad_(enable_grad)
        if not self.enable_grad:
            self.pos_embedding.eval()
            self.decoder.eval()
            self.proj_to_vocab.eval()

    def forward(self, latents):
        """
        x: [B, the_result_of_encoder, sequence_length]
        return: [B, seq_len(sequence_length), vocab_size]
        """

        self.pos_embedding.to(latents.device)
        self.decoder.to(latents.device)
        self.proj_to_vocab.to(latents.device)

        latents = self.linear_proj(latents)
        
        latents = self.linear_proj2(latents)

        positions = torch.arange(0, latents.size(1), device=latents.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        latents = latents + pos_emb

        latents = self.decoder(latents)
        latents = self.proj_to_vocab(latents)
        return latents

    def generate(self,logits):
        """
        logits: [B, seq_len, vocab_size]
        """
        logits = logits.to(dtype=torch.float16)

        pred_ids = logits.argmax(dim=-1)  # [B, seq_len] - Select token with highest probability

        eos_trimmed_ids = self.truncate_at_eos(pred_ids, eos_token_id=1)
        decoded_texts = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in eos_trimmed_ids]

        return decoded_texts

    def truncate_at_eos(self, ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
        result = []
        for seq in ids:
            # Find EOS token position
            eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                cut_idx = eos_positions[0].item() + 1
                result.append(seq[:cut_idx])
            else:
                result.append(seq)
        return result


