# Research Question
Can we semantically encode an image as plain text?

If you want to cite this work:
```
@misc{MSciancalepore2025,
  author       = {Mauro Sciancalepore},
  title        = {Can we semantically encode an image as plain text?},
  month        = {may},
  year         = {2025},
  note         = {\url{https://github.com/masc-it/neural-activations/2-encode-image-as-text.md}}
}
```

# Use cases

- In a multimodal setting, with images and text, it would be much easier to work with a unified input: just a discrete token sequence. No separate image encoder on top.

# Idea

Build a preprocessing algorithm that, given an image: 

1. it quantizes it from RGB space to a custom, reduced, N-color space, with a fixed palette.
2. groups pixels by colour, having each row of the encoding containing all the pixels with that colour.

So, an encoded image would look like this:
```
gray-0_0,0_10,10_1,..
red-0_1,0_14,12_1,..
darkred-2_1,3_14,11_1,..
..
..
..
```

Having such encoded representation, one could feed it to a language model, as normal text.

# Pros and cons

## Pros

- image becomes text
- simplify neural architecture

## Cons
- Sequence length can easily explode. Even a 32x32px image, encoded with 128 colors, can quickly reach a ~2000 sequence length.
- Does this even make sense? lol

# Tokenization

In order to keep the encoding length under control, I have added all the colors as special tokens, along with the pixel coordinates `x_y`:

```
...
"c_brown",
"c_lime",
"c_navy",
"c_burgundy",
"c_indigo",
"c_sgreen",
"c_beige",
"c_vdgray",
*[f"{i}" for i in range(128)],
*[f"{i}_" for i in range(128)],
*[f"{i}," for i in range(128)],
```

# Results

Right now I am training a GPT model on an internal company dataset. 
Given the image representation + prompt, the network has to predict some textual data, contained in the image.

It takes ~3 hours per epoch on my RTX A4000.

Cross entropy loss logs:
- start: ~6
- 20%: 3.3
- 50%: 2.30

# Dev logs

**Update:**
Interestingly, at ~30% of the first epoch, the loss started to drop more sharply and gradient norm jumped from a steady 0.5 to a more fluctating range, between 10 to 40. I have grad clipping set to 3.0, so let's see how it goes.
It should be related to the scheduler as well, I am using a cosine scheduler with warmup, at this stage we are at 2e-5. warmup is set to last 30% of training steps, I have 5 epochs so that's gonna end sometime during the 2nd epoch.

---