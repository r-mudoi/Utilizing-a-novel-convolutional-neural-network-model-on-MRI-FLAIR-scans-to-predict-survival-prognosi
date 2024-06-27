# Utilizing a novel convolutional neural network model on MRI FLAIR scans to predict survival prognosis of high grade glioma patients

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
pip install opendatasets
pip install pandas
!pip install medpy
!pip install nibabel
!pip install nilearn
!pip install pybids
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...
Download "Multimodal Brain Tumor Segmentation Challenge 2019: Data" dataset.

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
model.fit(X_train_scaled, y_train_reshaped, epochs=10, batch_size=60, validation_data=(X_test_scaled, y_test_reshaped))
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.


## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
predicted_values = model.predict(X_test_scaled)
y_test = y_test.tolist()
mse = mean_squared_error(predicted_values, y_test)
print(mse)
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).
Evaluate the model using the validation dataset. Identify the highest performing epoch based on the lowest val_mae. You may also graph the MAE over epochs.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 MAE  | Top 5 MAE |
| ------------------ |----------- | --------- |
| CNN                |    227     |    248    |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
