# Research Question
Do we need an image encoder in Vision Language Models?

If you want to cite this work:
```
@misc{MSciancalepore2025,
  author       = {Mauro Sciancalepore},
  title        = {Do we need an image encoder in Vision Language Models?},
  month        = {june},
  year         = {2025},
  note         = {\url{https://github.com/masc-it/neural-activations/1-unified-gpt.md}}
}
```

# Use cases

- Simplify a VLM architecture even more, by removing the need of a separate image encoder on-top.

# Idea

Instead of having a full blown image encoder on-top, e.g. SigLIP or a generic ViT, I propose to have a simple Image Patchifying module: 

```
patchifier = nn.Conv2d(3, hidden_dim, stride=patch_size, kernel_size=patch_size)

def forward(self, x: torch.Tensor):
    x = self.patchifier(x).flatten(2) # [B, hidden_dim, seq_len]
    x = x.transpose(1,2)
    return x

```

Then, the deocder will do its job to learn meaningful patterns.

Regarding attention, tokens associated to image will communicate bidirectionally.

# Pros and cons

## Pros

- simplify neural architecture

## Cons
- monolith architecture
- image representation is learnt indirectly, in the sense that they are updated w.r.t. the next token prediction task. Only the text tokens are going to participate in the cross entropy loss. The network will optimize the image weights, since they condition the text gen. But yeah, it should be a weaker signal, so optim can take longer.

# Results

It works, I mean it's not something groundbreaking, there are folks which have already validated similar ideas, but I couldn't find papers that just feed a Conv2D outputs to the decoder.

I have trained such decoder-only Vision Language Model on internal data, to perform both structured information extraction and image tagging (generate a short tag, similar to classification).
The model has converged to ~0.4 cross entropy loss.

# Dev logs

This was my first decoder-only experiment run from scratch, so I implemented all the components in pytorch:

- Image Patchifier
- Transformer blocks (RMSNorm, SwiGLU act, parallel attention from PALM paper)
- Causal Self Attention
    - causal attention mask + padding management
    - F.scaled_dot_product_attention
- KV cache scaffolding for optimized inference

# Next steps

- train the model and evaluate using some benchmarks
