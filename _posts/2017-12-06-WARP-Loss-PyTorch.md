---
layout: page
title:  "WARP Loss, automatic differentiation and PyTorch"
date:   2017-12-06 10:17:15 -0500
---

Note: This post was originally published on the [Canopy Labs website](https://canopylabs.com/resources/intro-warp-loss-automatic-differentiation-pytorch/),
and describes work I've been lucky to do as a data scientist there.

Loss functions are one of the most important parts of a machine learning algorithm; by telling the algorithm what it got
right or wrong, they essentially define what it is learning. A loss function is a scalar value, where — in general — a higher
value means the model is more wrong.

When training recommenders, we often don’t care about the absolute score of the items being recommended as much as their
rank relative to one another. However, few loss functions actually optimize for this.

In this post, we investigate a loss function which does optimize for rank — WARP loss. We also implement it in PyTorch,
a machine learning framework. Along the way, we take the hood off PyTorch, and look at how it implements neural network
layers. PyTorch implements a tool called automatic differentiation to keep track of gradients — we also take a look at
how this works.

### Weighted Approximate-Rank Pairwise Loss

WARP loss was first introduced in [2011](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37180.pdf),
not for recommender systems but for image annotation. It was used to assign to
an image the correct label from a very large sample of possible labels. Originally, the motivation for developing this
loss — which in particular, has a novel sampling technique — was one of memory efficiency. However, the sampling technique
also has additional benefits which make it well suited to training a recommender system.

So how does WARP loss work?

At a high level, WARP loss will randomly sample output labels of a model, until it finds a pair which it knows are wrongly
labelled, and will then only apply an update to these two incorrectly labelled examples.

Consider the following example: I have developed a recommender system to recommend one of the following 5 candy bars.
I’ve input a customer’s customer journey through my recommender, and it has generated an output vector, which assigns to
each candy bar a probability that this customer will purchase it. To train my recommender, I also have a target vector,
which describes the customer’s actual behavior using 1s if the customer purchased a specific candy bar, and 0 if they did not:

![WARP_1](../../../assets/img/2017-12-06/WARP_1.png "WARP_1")

Highlighted in red is the candy bar the customer actually bought (note that for simplicity, I’m only considering a single
purchase, but this loss extends to the case where the customer has made multiple purchases). This is known as the correct
label; let’s label it \\( x^{+}_{3}\\) for clarity (where the + highlights that this was the purchased item, and the subscript
indicates where the element is in the vector).

I’m now going to randomly sample the other labels, until I find one which the model assigned a higher probability of purchase
to the customer (or I run out of labels to sample). I’ll know that this randomly sampled label is wrongly labelled, because
I know that the Milky Way bar should have the highest probability — since this is the one the customer actually bought!

For instance, if the first random sample I look at is the Mars bar:

![WARP_2](../../../assets/img/2017-12-06/WARP_2.png "WARP_2")

Now, I have two variables: my correct label, \\( x_{3}^{+}\\) , and my sampled label, which I’ll call a sampled negative
label, \\( x^{-}_{5}\\) (negative because since the customer didn’t buy it).

In this case, my model was correct; 0.59 > 0.17 (or \\( x_{3}^{+} > x^{-}_{5}\\)) so my model correctly ranked the Milky
Way higher than the Mars bar. When this happens, I sample another label — and I’ll keep doing this until I find a case
where the model was wrong.

Say the second random sample I take is of the Kit Kat (which becomes the sampled negative label, \\( x^{-}_{2}\\)):

![WARP_3](../../../assets/img/2017-12-06/WARP_3.png "WARP_3")

In this case, 0.59 < 0.63 (or \\( x_{3}^{+} < x_{2}^{-}\\)). My model was wrong here, since it thought the customer would be more likely to
buy the Kit Kat. To tell my model to correct this, \\( x_{3}^{+}\\) and \\( x^{-}_{2}\\) are the two examples I’ll use for the WARP loss, where the
loss is the difference between the two values.

\\[ \textrm{Loss} = x_{2}^{-} - x_{3}^{+} \\]

We’re not done yet! In addition to this pair, I want to have an idea of how well my model did in general; was the Milky
Way bar ranked near the top of all the candy bars? Or did the model do poorly, and stick it near the bottom?

To avoid having to look at all the examples (remember; efficiency!), I can keep track of this while I do the random sampling.
If it takes me lots of random samples to find an example where my model was wrong, then I can assume it did pretty well.
On the other hand, if the first random sample I look at had a higher score than my correct label, then I can assume it
did pretty poorly.

I therefore multiply my loss by the following function:

\\[ ln(\frac{X-1}{N}) \\]

where X is the total number of labels (5, in this case) and N is the number of samples needed to find an example where
the model was wrong (2, in this case — the Mars bar, and the Kit Kat).

This makes sense; as I have to take more samples (and N gets larger), it indicates my model is more correct, so I want
my loss to be small. I also take the natural logarithm of this function, just to prevent the loss from exploding if N
gets small (and since X is generally large).

So now, my loss function looks like this:

\\[ \textrm{Loss} = ln(\frac{X-1}{N})(x_{2}^{-} - x_{3}^{+}) \\]

It’s interesting to note that the loss only depends on these two examples which I have sampled (and so only weights for
those two examples will be updated). Nothing is going to be done about the fact that the Twix bar was also ranked higher
than the Milky Way, or the fact that Snickers got a 0.35 chance of being bought even though the customer didn’t buy it
(so in the best model, it should have a 0). The model will only learn that the Milky Way bar should be ranked above the Kit Kat.

For a recommender, this is much more desirable than a model which learns that it should output 1s for all positive examples
and 0 for all negative examples, because often for recommenders, a 0 does not mean a negative interaction. Just because
the customer didn’t buy a Twix, it doesn’t mean they didn’t want to buy it — many other factors could have contributed to
their not purchasing it, most notably (considering the case where there are not 5 but 500 products to recommend) that they
just didn’t see it.

Therefore, having a model which learns to rank items it knows are positive above others can yield a better outcome than a
model which learns to rank a few items as 1s, and everything else as a 0. Losses which adopt this approach are known as ranking losses.

## Implementing WARP Loss in PyTorch

Although public implementations of WARP loss do exist (notably in Mendeley’s mrec, and lysts’s lightFM), at Canopy we were
interested in implementing WARP loss not for matrix factorisation (as it is used in mrec and lightFM), but to train
neural networks.

We’ve been using the PyTorch framework to experiment with different network architectures, but found it lacked an
equivalent to WARP loss (in particular, we were looking to avoid the problem of learning negative examples, instead of
positives).

In implementing our own WARP loss function, we got to open the hood on exactly how PyTorch implements loss functions, and
also take a closer look at automatic differentiation (autodiff), PyTorch’s approach to calculating gradients to
update neural networks.

### How are losses implemented in PyTorch?

First, a quick overview of how PyTorch works: PyTorch is a python library which allows tensors (PyTorch’s equivalent of
numpy nd-arrays) to be manipulated. However, unlike numpy, you can also wrap tensors in `Variables()`; these allow you to
keep track of the gradients of the tensors as you manipulate them, using a technique called automatic differentiation.

PyTorch treats losses as an additional layer of the neural network, so that when I am writing a loss ‘layer’, its actually
an `nn.Module` class (the same class as any layer in PyTorch’s neural networks). `nn.Module` classes consists of a constructor
( `__init__` ), and a single (required) method: `forward()`, which describes the operations to execute on the Variables.

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self, conv1(x))
        return F.relu(self.conv2(x))
```

This is sufficient to write a loss function; using automatic differentiation, PyTorch can calculate the gradients of the
Variables as a result of the `forward()` method for free, allowing backpropagation to occur. However, you can go a step
further and define a function to calculate those gradients yourself. This has two advantages: you are sure the gradients
are being calculated correctly, and it can be more efficient.

To implement a function to calculate gradients of the Variables, the approach is a little different; instead of writing
an `nn.Module` class, you actually implement an `autograd.Function` class, which is then ‘applied’ in the `nn.Module` class
(this is actually what happens in the example above; `F.relu` is written as an `autograd.Function`, and then applied
using `nn.Module`).

It helped me a lot to work through a very simple example — the [Linear function](http://pytorch.org/docs/master/notes/extending.html)
used as an example in the PyTorch docs — before implementing a WARP function, so I’ll do the same here.

```python
# Inherit from Function
class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias
```

This function has three inputs, and returns a single output; it’s a very straightforward transformation:

\\[ \textrm{output} = \textrm{weight} \times \textrm{input} + \textrm{bias} \\]

or, more succinctly

\\[ Y = W \cdot X + b\\]

This is implemented in the Function’s forward method ( `.mm()` is PyTorch syntax for a dot product, and the bias term is
optional, but `forward()` otherwise easy to read).

Now, for the backward method (warning — some math ahead).

The `backward()` method has for input `grad_output` – it will receive as many of these inputs as there were outputs from
the `forward()` method, so in this case, 1.

This `grad_output` represents the gradient of the *layer output* with respect to the *entire computational graph*
(in this – and most -cases, a neural network with a loss function). So for our toy one-layer network, where F is the
function representing the computational graph and Y is the layer output,

\\[ \textrm{grad output} = \frac{\partial F}{\partial Y}\\]

I can calculate this using the chain rule:

\\[ \frac{\partial F}{\partial X} = \frac{\partial F}{\partial Y} \cdot \frac{\partial Y}{\partial X}\\]

Since I know that \\(Y=WX+b \\), I can easily calculate

\\[ \frac{\partial Y}{\partial X} = W\\]

Substituting this into my expression for `grad_input`,

\\[ \textrm{grad input} = \frac{\partial F}{\partial X} = \frac{\partial F}{\partial Y} \cdot W\\]

As expected, in the `backward()` function of `LinearFunction`, `grad_input` is the dot product (`.mm()`) of `grad_output` and the
weight matrix!

*Note*: in this example, where \\( F(X)=Y, \frac{\partial F}{\partial Y} \\) may seem to be trivially 1. However, remember that in practice, F defines not the single
layer, but the entire computational graph (i.e. the neural network including the loss function). These partial derivatives allow me to find how this
layer’s weights affect the loss function, even if there are a bunch of layers after this one! For a loss function, \\(\frac{\partial F}{\partial Y} \\) is indeed 1.
However, if there are layers after this Linear one, then the \\(\frac{\partial F}{\partial X} \\) output from the previous layer will
become the \\(\frac{\partial F}{\partial Y} \\) input of this layer during backpropagation.

This is a really cool illustration of how autodiff works, and in particular how it combines both numerical and symbolic
differentiation:

Since I only care about the gradient matrix (or more precisely, the Jacobian matrix), the `backward()` method gets as input
some matrix, and returns some matrix; it is blind to whatever operations happened before it received the matrix, and to
whatever operations happen afterwards. This is akin to numerical differentiation, and is far more memory efficient than
trying to save a whole mathematical expression and then analytically differentiate it, which is what happens in symbolic
differentiation (especially when you consider how massive these expressions can get for neural networks).

However, the actual operations that happen within the `backward()` method are analytically motivated, and are therefore exact
(none of numerical differentiation’s rounding or truncation errors). This is similar to symbolic differentiation.

### WARP Loss in PyTorch

To implement the WARP function, I need to implement a `forward()` and a `backward()` method.

#### `forward()`

The algorithm for the `forward()` method works exactly as described at the beginning of this page, with a few additions for
efficiency. The locations of the selected positive and negative samples are stored in matrices of 1s and 0s of the same shape
as the input matrix, so that an element-by-element multiplication of these matrices with the input matrix zeroes out all non
selected elements.

To clarify what this means, lets return to the candy bar example from above. As a reminder, here were the ultimately selected samples:

![WARP_3](../../../assets/img/2017-12-06/WARP_3.png "WARP_3")

My negative example is the Kit Kat, and my positive example is the Milky Way, so my positive indices and negative indices
‘matrices’ (vectors, since there is only one sample) would look like this:

![WARP_6](../../../assets/img/2017-12-06/WARP_6.png "WARP_6")

so that, multiplying this with the original input,

![WARP_7](../../../assets/img/2017-12-06/WARP_7.png "WARP_7")

I also add all of the loss multiples, \\( ln((X-1)/N) \\), into a vector L. This allows me to calculate my loss in a single
calculation, instead of adding to it with every loop:

```python
loss = L * (1-torch.sum(positive_indices*input, dim = 1) + torch.sum( negative_indices*input, dim = 1))
```

Note that the margin here is 1. The positive_indices and negative_indices matrices are also useful for the `backward()` method.

#### `backward()`

The `backward()` method needs to return `grad_input`, the gradient with respect to the input matrix, \\(\frac{\partial F}{ \partial X} \\).

Since random sampling is a non-differentiable operation, the rank multiplier \\(ln(\frac{X-1}{N}) \\) is considered constant for
the purpose of differentiation. The way I approach this is to imagine that WARP is simply a sampling technique which
defines the loss function, and therefore a unique loss function is created for every training example the recommender
sees — each with unique (‘randomly chosen’) samples, and weights (the rank multiplier).

In this case, applying the same notation as for the linear function,

\\[ Y = F(X) = L( \textrm{margin} - X^{-} - X^{+} )\\]

where

\\[ X^{+} = \textrm{positive indices} \cdot X \\]
\\[ X^{-} = \textrm{negative indices} \cdot X \\]

Then, given \\(\textrm{grad_output} = \frac{\partial F}{\partial Y} \\), I want to find the grad input,
\\(\frac{\partial F}{\partial X} \\). As before, this can be achieved by chain rule,

\\[ \frac{\partial F}{\partial X} = \frac{\partial F}{\partial Y} \cdot \frac{\partial Y}{\partial X}\\]

where

\\[ \frac{\partial Y}{\partial X} = L \cdot (\textrm{margin} + \textrm{positive indices} - \textrm{negative indices})\\]

Substituting this into the expression for `grad_input` yields the `backward()` method:

\\[ \frac{\partial F}{\partial Y} = \frac{\partial F}{\partial Y} \cdot L \cdot (\textrm{margin} + \textrm{positive indices} - \textrm{negative indices})\\]

or, in code

```python
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import random

class WARP(Function):
    '''
    autograd function of WARP loss
    '''
    @staticmethod
    def forward(ctx, input, target, max_num_trials = None):

        batch_size = target.size()[0]
        if max_num_trials is None:
            max_num_trials = target.size()[1] - 1

        positive_indices = torch.zeros(input.size())
        negative_indices = torch.zeros(input.size())
        L = torch.zeros(input.size()[0])

        all_labels_idx = np.arange(target.size()[1])

        Y = float(target.size()[1])
        J = torch.nonzero(target)

        for i in range(batch_size):

            msk = np.ones(target.size()[1], dtype = bool)

            # Find the positive label for this example
            j = J[i, 1]
            positive_indices[i, j] = 1
            msk[j] = False

            # initialize the sample_score_margin
            sample_score_margin = -1
            num_trials = 0

            neg_labels_idx = all_labels_idx[msk]

            while ((sample_score_margin < 0) and (num_trials < max_num_trials)):

                #randomly sample a negative label
                neg_idx = random.sample(neg_labels_idx, 1)[0]
                msk[neg_idx] = False
                neg_labels_idx = all_labels_idx[msk]

                num_trials += 1
                # calculate the score margin
                sample_score_margin = 1 + input[i, neg_idx] - input[i, j]

            if sample_score_margin < 0:
                # checks if no violating examples have been found
                continue
            else:
                loss_weight = np.log(math.floor((Y-1)/(num_trials)))
                L[i] = loss_weight
                negative_indices[i, neg_idx] = 1

        loss = L * (1-torch.sum(positive_indices*input, dim = 1) + torch.sum(negative_indices*input, dim = 1))

        ctx.save_for_backward(input, target)
        ctx.L = L
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices

        return torch.sum(loss , dim = 0, keepdim = True)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_variables
        L = Variable(torch.unsqueeze(ctx.L, 1), requires_grad = False)

        positive_indices = Variable(ctx.positive_indices, requires_grad = False)
        negative_indices = Variable(ctx.negative_indices, requires_grad = False)
        grad_input = grad_output*L*(negative_indices - positive_indices)

        return grad_input, None, None


class WARPLoss(nn.Module):
    def __init__(self, max_num_trials = None):
        super(WARPLoss, self).__init__()
        self.max_num_trials = max_num_trials

    def forward(self, input, target):
        return WARP.apply(input, target, self.max_num_trials)
```

## Conclusion

To conclude, WARP loss is a powerful loss function, which can be applied to deep neural networks to provide
recommendations based on customer journeys. In particular, its approach of optimizing for ranking instead of absolute
outputs makes it particularly well suited to recommenders, and in addition to its current applications in matrix factorisation
recommenders, it can also be applied to neural network — based recommenders.




