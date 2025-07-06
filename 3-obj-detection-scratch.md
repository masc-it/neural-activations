# Research Question
How to build a simple object detection model from scratch?

If you want to cite this work:
```
@misc{MSciancalepore2025,
  author       = {Mauro Sciancalepore},
  title        = {How to build a simple object detection model from scratch?},
  month        = {march},
  year         = {2025},
  note         = {\url{https://github.com/masc-it/neural-activations/blob/main/3-obj-detection-scratch.md}}
}
```

# Use cases

- learning
- ship a lightweight, understandable object detection model on CPU

# Recipe

## Pretraining

Most of the object detection models out in the wild are pretrained on COCO or other datasets with hundreds of object classes.

For this experiment, I to try implement an object-detection friendly Self-Supervised Learning (SSL) pretraining technique: Learn To Detect (LTD).

The self-supervised setup is pretty simple: you have a reference image (R) and a random sub-image (S) of it. The goal is to make the network predict the coordinates of S, in R.

> note: I did some research and I found just one paper using something similar to this, to boost performance (will try to find it, I think it was a DETR-related paper).

> fun fact: I got this idea while waiting for my gf. if she wasn't late, I wouldn't had it perhaps. LoL

Back to us. The intuition is pretty similar to what happens when you are working on a puzzle: you have the bigger picture (R), with some holes in it, and you're trying to find the right spot for the puzzle pieces in your hand (S).

In my ssl setup, the reference image is left intact, without masking out the random patch R. I did an ablation on this and it made the task waaay easier for the network. I wanted it to sweat and learn something real good. You know, neural nets can get very lazy. (like us)

So, that's what I did: I pretrained a resnet-18 on this ssl task up until convergence.

## Object detection architecture

I wanted to start simple: a CNN, resnet-18.

The idea is very primitive: make the network output a grid of cells, where each cell is responsible for a single bounding box.

And CNNs are perfect for this, since they will output a feature map of shape [B, C, H, W].

Schema of shapes:

image: [B, C_IN, H, W]
cnn: from [B, C_IN, H, W] -> [B, C_OUT, F_H, F_W]
transpose: [B, C_OUT, F_H, F_W] -> [B, F_H, F_W, C_OUT]
flatten spatiality: [B, F_H*F_W, C_OUT]

At this point, we need to output for each cell the coordinates (cx,cy,w,h) and class idx, so we'll use a nn.Linear to go from C_OUT -> 4 + NUM_CLASSES

linear: [B, F_H*F_W, C_OUT] -> [B, F_H*F_W, 4+NUM_CLASSES]

then with .split you can extract the coordinates and classes logits vector.

## Loss function

How do we assign a bounding box to a grid cell?

There exist specific algorithms to do this, like the hungarian algorithms (used in DETR). But I want something dumb and effective for this experiment, no gradient instabilities and other crap to deal with.

Idea: Assign a bbox B to a grid cell C (of shape cell_size), if B center falls into the cell coordinates. Yeah, this dumb. of course there are countless limitations, but it's fine for now.

In this way, we are able to compare ground truth with predictions, since now we have a deterministic way to assign a bbox to a grid cell.

For the loss I am using a classic SmoothL1 for the coords + crossentropy for the classification/objectness score.

## Data

I have used `detection-datasets/fashionpedia`

## results

80%+ mAP@90

[](https://github.com/masc-it/neural-activations/blob/main/3-obj-detection-scratch/odd_scratch_results.png)

[](https://github.com/masc-it/neural-activations/blob/main/3-obj-detection-scratch/odd_scratch_results_objectness.png)