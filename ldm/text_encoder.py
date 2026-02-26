import torch
from torch import nn
from torch.nn import LayerNorm
from einops import rearrange

class TextContextEncoder(nn.Module):
    def __init__(self, context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=512,
                pretrained='./clip/ViT-B-16.pt', **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = None

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.llm = False
        self.noise = False
        # Note: We don't need a custom rotary_emb here because each Llama layer
        # has its own internal rotary embedding that's properly configured

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
            state_dict = {}
            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]
             
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    
    ### emb_txt = self.text_encoder(tok_txt.to(x_in.device), self.contexts) 
    def forward(self, text, context=None, train=False, alpha=None):

        ### text [batch, 128] - token IDs
        ### x_text [batch, 128, 2560] - token embeddings
        x_text = self.token_embedding(text)  # batch, n_text, C

        # Ensure x_text is float32 for stability
        x_text = x_text.float()

        B, N1, C = x_text.shape

        # Calculate base eos_indx (last non-padding token position)
        non_zero_mask = (text != 0)
        base_eos_indx = non_zero_mask.sum(dim=-1) - 1  # Last non-padding position
        base_eos_indx = base_eos_indx.clamp(min=0)  # Ensure non-negative

        # prompting with learnable contexts
        if context is not None:
            ### context [P, N2, C] e.g., [2, 8, 2560]
            # Ensure context is float32
            context = context.float()
            P, N2, C_ctx = context.shape

            # eos_indx shifts by N2 because we insert context tokens after first token
            eos_indx = base_eos_indx + N2
            eos_indx = eos_indx.reshape(1, B).expand(P, B).reshape(-1)  # [P*B]

            x_text = x_text.reshape(1, B, N1, C).expand(P, B, N1, C)     # [P, B, N1, C]
            context = context.reshape(P, 1, N2, C_ctx).expand(P, B, N2, C_ctx)  # [P, B, N2, C]

            # Insert context after first token: [CLS, context_tokens, rest_of_text]
            x = torch.cat([x_text[:,:,0:1], context, x_text[:, :, 1:]], dim=2)  # [P, B, N1+N2, C]
            x = rearrange(x, 'p b n c -> (p b) n c')  # [P*B, N1+N2, C]

        else:
            P = 1
            eos_indx = base_eos_indx
            x = x_text

        # llm
        if self.llm:

            seqlen = x.shape[1]
            batch_size = x.shape[0]

            # Clamp eos_indx to valid range to prevent out-of-bounds indexing
            eos_indx = eos_indx.clamp(max=seqlen - 1)

            position_ids = torch.arange(
                0, seqlen, dtype=torch.long, device=x.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # [batch, seqlen]

            # Create causal attention mask
            attention_mask = None
            if seqlen > 1:
                attention_mask = torch.ones((batch_size, seqlen), dtype=torch.long, device=x.device)

            # Use the transformer's forward method directly
            # Disable autocast and use float32 for numerical stability with quantized models
            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.transformer(
                    inputs_embeds=x,  # Already float32
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=False,
                    return_dict=True,
                )

            x = outputs.last_hidden_state  # [batch, seq, hidden]
            x = x[torch.arange(x.shape[0], device=x.device), eos_indx]  # [batch, hidden]

            # Ensure output is float32 for UNet cross-attention compatibility
            x = x.float()

        # clip
        else:

            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection

        x = rearrange(x, '(p b) c -> b p c', b=B, p=P)
        return x
