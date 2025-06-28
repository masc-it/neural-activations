# Research Question
What does it take to port a pytorch vison language model to ONNX?

If you want to cite this work:
```
@misc{MSciancalepore2025,
  author       = {Mauro Sciancalepore},
  title        = {What does it take to port a pytorch vison language model to ONNX?},
  month        = {april},
  year         = {2025},
  note         = {\url{https://github.com/masc-it/neural-activations/blob/main/1b-vision-language-model-onnx.md}}
}
```

# Use cases

- ONNX can be used to run model inference in a platform and language independent way

# Trust the Process

From past experience, I know that ONNX require to compile the model in a graph, that captures all the shapes and operators in the network.

A language model, in terms of operators is quite simple. If we want to do a quick list, we'd have the following (not an exhaustive one):

- Attention (I have used F.scaled_dot_product_attention)
- RMSNorm (torch.mean, torch.rsqrt, nn.Parameter )
- SwiGLU (nn.Linear + F.silu, .chunk(..))
- nn.Embedding
- SinCos positional embedding module (nn.Parameter, data dependent slicing operator)
- Language Modeling head (just a nn.Linear)

Fortunately for me, all of these are extensively supported in ONNX, nice!

Now that we're figured out operators support, we need to talk about input and output shapes.

ONNX needs to know in advance the shape of your data and the outputs of your network. 

In my case, this is the network forward:

```
def forward(self, 
  images: torch.Tensor,    # [B, C, H, W]
  input_ids: torch.Tensor, # [B, S]
)
```

When running inference, the model is called with such tensors, the next token is sampled and added back to the sequence. The autoregressive process is repeated up until the eos token is sampled or max length is reached. So far, so good.

With this setup, the onnx export has been butter smooth.

There's a catch though: autoregressive generation with full input sequence, is slow, utterly slow. I need to make onnx work with KV cache, which I already had in place in the pytorch model.

So, with KV cache things are a little bit different..

```
def forward(self, 
  images: Union[torch.Tensor, None],    # [B, C, H, W]
  input_ids: torch.Tensor, # [B, S]   
  kv_cache: Union[list[tuple[torch.Tensor]], None]
)
```

With KV cache on, generation can be conceptually split in two steps:

- prefill stage: the prompt is processed. in my case, the image + prompt. the keys and values are stored in the kv cache.
- generation stage: only the next token is fed, kv cache will be used to retrieve the past keys and values and compute attention properly. kv cache is then updated.

As I told before, ONNX require a static computational graph.

With KV cache, you're gonna have a lot of ifs in your pytorch code: NO GOOD.

In a vision language model tho, things are more complicated, due to the extra input: images

We need to process the image tensor only in the prefill stage, and in order to do so, I tried to pass it as an Optional field, which is None in the generation stage. Well, this didn't work out, onnx was not able to capture this case and things broke.

Even the KV cache, I couldn't pass it as an Optional list. Nope.

First thing I had to fix, was the images tensor. I came up with this idea:


```
def forward(self, 
  images: torch.Tensor,    # [B, N, C, H, W]
  input_ids: torch.Tensor, # [B, S]   
  kv_cache: Union[list[tuple[torch.Tensor]], None]
)
```

Practically, during the prefill, the image tensor has shape [B, 1, C, H, W], which is then reshaped to [B*1, C, H, W] and everything's fine.

During generation, the image tensor has shape [B, 0, C, H, W], which is just a dummy tensor. Turns out that Conv2d works well with this so at the end of the day, the image encoder shape is [B, 0, D]. perfect.

With this, I can now concat the image_embeds with the text_embeds in a **branchless** way:

```

tokens = torch.concat(
  [
    image_embeds, # [B, 0, D]
    text_embeds   # [B, N, D]
  ],
  dim=1
)
```

Now let's fix the kv cache input.

I quickly understood that the fastest way to make things work out with onnx is to go **branchless**. None is forbidden. ifs are forbidden. Just Tensors and shapes shenanigans.

Updated forward:
```
def forward(self, 
  images: torch.Tensor,    # [B, N, C, H, W]
  input_ids: torch.Tensor, # [B, S]   
  kv_cache: tuple[torch.Tensor, ...], # Tuple containing past Key/Value tensors for each layer.
                                      # Format: (k_past_layer0, v_past_layer0, k_past_layer1, v_past_layer1, ...)
)
```

then I could build the kv cache tensor like this:

```
past_key_values = torch.stack(
            past_kv,
            dim=1,
        )
```

and get the current sequence length with `pos_start_offset = past_key_values.size(3)` which is used to properly compute positional embeddings for the next token.

Finally, it worked. What a journey.

# Notes

- An alternative is compile the model in two separate graphs: the prefill graph and the generation graph.
  - That is just skill issue. Too easy for me. I wanted one graph :)
- you need to set dynamic axes for the sequence length dimension of your input_ids and output
- you need to set dynamic axes for the sequence length dimension of your images

