# Repository overview
This repository is designed to acquire images from the remote sensing dataset Sentinel-2, train and predict using vision deep learning models using various balanced training methods, and visualize results using the LOTS visualization technique.

## Image Extraction Script
The script `image.ipynb` is designed to extract images from the Sentinel-2 and Dynamic World V1 datasets using Google Earth Engine. The main functionality involves retrieving the least cloudy satellite images for specific geographical locations and time periods, processing these images, and exporting them for further analysis. 

To extract images from the Sentinel dataset, please refer to the `extract_GPS_images` directory, specifically the script `image.ipynb`. This will generate TFRecord files of the satellite images.

### Wealth index data
The script utilize `labels.csv` to get save labels in the TFRecords. Unfortunately, this dataset is private. However, if you have numerical values, you can still get images by saving `labels.csv` in the folder `extract_GPS_images`. The structure of the data is:

| Column             | Description                          |
|--------------------|--------------------------------------|
| year               | The year to sample data.             |
| cell id            | Unique id as identifier.             |
| centroid latitude  | Latitude of the location to sample data |
| centroid longitude | Longitude of the location to sample data |
| label              | Numerical label containing one number.|

## Split and Batch Processing Script
The script `split_and_batch.ipynb` is designed to manage and process TFRecord files. It performs the following main functions: unzipping TFRecord files, reading them, splitting into training/val/test based on location, concatenating into HDF5 batches, and handling corrupted files by moving them to a designated folder. 

To process and batch your TFRecord files, please refer to the  `split_and_batch.ipynb` script.

## Train and validating models:

The script `main.ipynb` contains all the experiments conducted in the thesis. It serves as an example of how to load data into a dataloader, use different loss functions, initialize various models, and more.

This script is where all code ties together.

## Visualization with LOTS:
The script `visualization.ipynb` contains examples of how to generate a LOTS adversarial example and visualize it as a heatmap.

## Other helper:
Folder `loss`: Contains all loss functions used for balanced training.

Folder `metrics`: Contains the balanced_MAE class, used to validate the performance of trained models.

Folder `utils`: Contains various utility functions and classes to assist with training and validation:
- `checkpoint.py`: Saves and loads model checkpoints.
- dataloader.py: Provides dataloaders, datasets, and necessary transformations for model inputs.
- inference.py: Performs inference using a trained model and a dataloader.
- train.py: Trains models with specified loaders and parameters, saves the best checkpoint after each epoch, and prints training and validation loss along with other metrics.
- utils.py: Contains miscellaneous utility functions used in multiple scripts.


Folder `visualization`: Contains lots.py for generating LOTS adversarial examples and heatmaps.