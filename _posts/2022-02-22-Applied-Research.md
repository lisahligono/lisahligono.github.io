---
layout: page
title:  "Getting started with applied Machine Learning Research"
date:   2022-02-22 10:17:15 -0500
---

I recently had a conversation discussing how I structure my (applied) machine learning research projects.
I've written my thoughts up here, as a checklist for getting started.

### 1. Understand the metrics
The number one thing to do is to understand what success looks like. This can be highly multifaceted, but here are some questions I would ask stakeholders when starting a project:

- **What metric do we care about?** This is important because the output of a machine learning model may be processed in some way before being used, so a metric which makes sense in the “machine learning world” may not transfer to impact. For example, an image segmentation may ultimately be used to (e.g.) count leaves in an image. In this case, traditional segmentation metrics may not be as useful as just looking at whether you have predicted the number of leaves correctly.
- **What error is acceptable / usable?** This goes both ways; at what point does the model become useful? And when is improved performance on the metric marginal (at which point it may not be worth continuing to try and improve performance)
- **Is {metric} the only consideration?** Do you have computational constraints on which models you can use? What will it take for practitioners to trust your model?

This should be continually revisited throughout the project. This avoids the scenario where you build an awesome model after many months / years of hard work, but it ends up being useless to the stakeholder because you have missed some key requirement that they have.

### 2. Build an end to end pipeline
This is the next priority. The reason for this is that for machine learning, everything affects everything else. This means I find it difficult to work on different stages in isolation, since (as I add complexity) it’s hard to know if changes are ultimately beneficial or detrimental. Having an end-to-end pipeline early on mitigates this to some extent.

I typically think of my pipelines in 4 stages (and work on them sequentially):

#### 2.1. Data Export
This stage consists of getting the data from the world onto your machine. It’s highly dependent on what data you are working on (and can often take a significant amount of time to run).

As an example, for [crop-type mapping](https://github.com/nasaharvest/cropharvest) this stage consists of a) getting labels, b) getting satellite data to link to those labels.

The output of this step (for me) might be a folder containing:

1. a bunch of [tiff files](https://en.wikipedia.org/wiki/TIFF) exported from [google earth engine](https://earthengine.google.com/) stored on an instance
2. a labels [geojson](https://en.wikipedia.org/wiki/GeoJSON) that tells me what agriculture is captured in the tiff files (e.g. does the tiff file contain an image of maize? Or of soybean?)

I’ll typically have some identifier which will be in the labels geojson which maps to the tiff filename (i.e. the tiff filename will be “{id}.tif”)

#### 2.2. Data processing / engineering
The next step consists of getting the raw data into something which can be ingested into a machine learning model.

The output of this stage is another folder, containing a bunch of (X,y) numpy arrays. They can be stored in different ways - e.g. in [hdf5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) - but ultimately I should be able to load something and (nearly) immediately pass it to an ML model.

Another thing you should do here is split the data into training and validation data. By creating the splits here, you significantly reduce the likelihood of data leakage. You also ensure different models can be compared, since they will all have the same train / val splits.

#### 2.3. Writing a Dataset
The final step is a dataset. Both PyTorch and Tensorflow have a concept of datasets, so creating this class should allow you to plug into a rich set of tools (e.g. DataLoaders, which have nice multiprocessing capabilities or data augmentation functionality).

A dataset is a class which implements two functions:
- `__len__`, which returns the length of the dataset (how many (X,y) pairs do you have?)
- `__getitem__`, which returns the ith (X,y) pair when it is called

If step 2.2. has gone well, this should be relatively simple. Let’s say I have a folder with 2n numpy files, paired into X and Y arrays: “X_{n}.npy” and “Y_{n}.npy”. Then, my MVP dataset would look like this:

```python
class MyDataset:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.x_files = list(self.data_folder.glob(“X_*.npy”))

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self):
        x_filepath = self.x_files[idx]
        y_filepath = x_filepath.parent / f”Y_{x_filepath.name.split(‘_’)[1]}”

        return np.load(x_filepath), np.load(y_filepath)
```
Now, I can just wrap this with a dataloader and use a normal PyTorch workflow:

```python
from torch.utils.data import DataLoader

mydataset  = MyDataset(path_to_data)
myloader = DataLoader(mydataset)

for x, y in myloader:
    ...
```

#### 2.4. The model!
The key output here is not the predictions, it is the performance of the model on the validation set (although predictions may also be saved for analysis). So for instance, the output may be a json file that has certain metrics of performance:
```json
{“accuracy”: 0.6, “f1_score”: 0.7, ...}
```
The way I typically do this is write a BaseModel class, which implements a “validate” function. This function expects a trained `self.model` to make some predictions, and then saves them:
```python
class myBaseModel:
    model: nn.Module = None

    def validate(self, validation_data_path, model_save_folder):
        # given a path to the validation data, save the results json to model_save_folder

        val_dataset = MyDataset(validation_data_path)
        myloader = DataLoader(val_dataset)

        true_y, pred_y = [], []
        for x, y in myloader:
            true_y.append(y)
            pred_y.append(self.model(x))

        with open(model_save_folder / “results.json”, “w”) as f:
                json.dump({“accuracy”: accuracy_score(true_y, pred_y)})
```
You can then extend the Base Model with different models which have a train function (which trains the self.model attribute).

This also helps with model comparison! Since every other class only needs to care about training some `self.model` attribute - you can be sure they will all be evaluated in exactly the same way.

So for instance let’s say I have some fancy model to implement:
```python
class myFancyModel(myBaseModel):

    def train(self, train_data_path):

        self.model = myFancyPytorchModule()

        # training loop goes here

my_fancy_model = myFancyModel()
my_fancy_model.train()
my_fancy_model.validate()
```

The other nice thing about this is that it makes it very easy to implement simple baselines (which is always a good place to start).

### 3. Iteration
This is where the ~~magic~~ research happens. Some (very non-comprehensive) notes:

- Make sure every change ultimately benefits the final metrics
- Use git, and keep track of which git commit resulted in a model run. An example of this is [here](https://github.com/nasaharvest/cropharvest/blob/main/benchmarks/dl/maml.py#L74) - this allows me to couple experiments with states of the code, which is helpful when making changes to (e.g.) the data pipeline instead of the model itself or hyperparameters
- I typically see much larger gains from looking at the data (e.g. normalisation, augmentation, finding additional sources of data etc.) then from looking at the model.
