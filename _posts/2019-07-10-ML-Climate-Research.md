---
layout: page
title:  "A Machine Learning Pipeline for Climate Research"
date:   2019-07-10 10:17:15 -0500
---

![VHI](../../../assets/img/2019-07-10/vhi_cropped.png "VHI")

Over the last few weeks, I have been working with [Tommy Lees](https://tommylees112.github.io/) to develop a
machine learning pipeline to predict drought, as part of the ECMWF’s
[Summer of Weather Code program](https://www.ecmwf.int/en/learning/workshops/ecmwf-summer-weather-code-2019).

As we developed the pipeline, there were a few requirements we wanted to satisfy. The pipeline has to:

- **Be easily extensible** (to both new features and new data sources), to allow us to experiment with many different
ways of predicting drought, since there is no accepted approach.
- **Be robust, and thoroughly tested**, so that we can be confident the pipeline is doing what we expect.
- **Be easy to validate and analyze**, so that we can assess the performance of our models and understand what
patterns they are learning.

The result consists of 5 steps, which I’ll discuss in more detail:

- **Exporters**, which download raw data
- **Preprocessors**, which transform the raw data into a uniform data format
- **Engineers**, which prepare the inputs to the machine learning models
- (Machine Learning) **Models**
- **Analysis** of the model predictions, and of what the models have learnt

![pipeline](../../../assets/img/2019-07-10/pipeline.png "pipeline")

## Extensibility

The pipeline is split into 2 “halves”, each of which tackle a specific type of extensibility:

####  Data extensibility
At the front end of the pipeline, we focused on data extensibility; making it easy to incorporate new data sources.

This is achieved by the exporters and the preprocessors: the [exporters](https://github.com/esowc/ml_drought/tree/master/src/exporters)
handle interactions with data stores, and have to be customized for each unique store (e.g. talking to a specific API, or to an FTP).

The [preprocessors](https://github.com/esowc/ml_drought/tree/master/src/preprocess) handle quirks in the data itself
(e.g. dimensions named “time1” instead of “time”). In addition, they are responsible for putting all the data onto a
uniform spatial and temporal grid; this makes different datasets easy to combine.

Ensuring all the data has the same format means that future steps are decoupled from the data sources. This allows new
data sources to be added with the addition of only two classes: an exporter and a preprocessor. Nothing downstream needs
to be changed for that new data to be included in experiments.

#### Experimental extensibility
The back end of the pipeline focuses on experimental extensibility (for instance, experimenting with different
machine learning algorithms). The biggest challenge here was balancing ease of use with flexibility.

Using experiments with different machine learning algorithms as an example, we imposed some constraints on different
algorithms:

- Most significantly, the engineer is responsible for splitting the training and test datasets. This is to prevent a
model from accidentally introducing leakage.
- Each model has access to the same training data format (i.e. a year of data to predict the next timestep). Although
this may not be optimal for all models (e.g. RNNs might prefer to predict on a rolling basis) it makes it much easier
to alternate between models for experiments.

The training and testing data is stored in [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) format to
maximize the amount of information available to the models (and therefore the type of models which can be implemented)
by ensuring the spatial and temporal grids associated with the data can still be accessed.

## Robustness

![travis_build](../../../assets/img/2019-07-10/travis_build.png "travis_build")

Particularly when developing a machine learning pipeline, which can often fail silently, we have found it super
helpful to use tests to make sure every step does what’s expected.

We use [pytest](https://docs.pytest.org/en/latest/) to extensively [unit test](https://github.com/esowc/ml_drought/tree/master/tests)
everything. Beyond just making sure the code works, pytest is really useful in allowing us to communicate what a piece of
code is supposed to achieve.

We also use type hints (checked using [mypy](http://mypy-lang.org/)). In addition to catching bugs, this is another
useful way to communicate what a piece of code expects (and what it returns), which makes it easier for us to
work on each other’s code.

A little bit of overhead at the beginning of the development process (adding
[continuous integration](https://github.com/esowc/ml_drought/blob/master/.travis.yml) and the framework for all the
testing) has made it super easy to ensure robustness is baked into our pipeline as we add experiments.

## Validation & Analysis

We validate all our models by comparing them to extremely simple models, which we understand well. This is especially
useful because - unlike some other applications of machine learning - there aren’t well established baselines for
machine learning applied to climate sciences.

The baseline we currently use is a [persistence model](https://github.com/esowc/ml_drought/blob/master/src/models/parsimonious.py#L9),
which predicts the vegetation health in month N to be vegetation health in month N - 1 (e.g. we predict vegetation
health in June to be identical to vegetation health in May). This gives us a good idea of how the models should be performing.

In addition, we have focused on leveraging interpretable machine learning techniques to analyze what the models are
learning. This allows us to ensure the patterns being learnt by the models make sense (e.g. data from months close to
the month being predicted should be more important to the model's prediction), and also to better understand
the relationship between the input variables and the target variable.

## Conclusion

In conclusion, the pipeline we have developed allows us to quickly iterate through different experiments, whilst
ensuring the code we are writing is robust.

Analyzing what the machine learning algorithms are learning and validating the models against a very simple baseline
allows us to sanity check the entire pipeline, and may allow us to discover new things about the relationships between
climate patterns and drought.

The pipeline can be explored at [https://github.com/esowc/ml_drought](https://github.com/esowc/ml_drought)

Please let us know if you have any questions!
