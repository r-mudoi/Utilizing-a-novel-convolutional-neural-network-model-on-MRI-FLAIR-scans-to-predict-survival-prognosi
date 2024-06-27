# Utilizing a novel convolutional neural network model on MRI FLAIR scans to predict survival prognosis of high grade glioma patients

This repository is the official implementation of "Utilizing a novel convolutional neural network model on MRI FLAIR scans to predict survival prognosis of high grade glioma patients." 


## Requirements

To install requirements:

```setup
pip install opendatasets
pip install pandas
!pip install medpy
!pip install nibabel
!pip install nilearn
!pip install pybids
```

>📋 
Download appropriate medical and neuro- imaging libraries.
Download "Multimodal Brain Tumor Segmentation Challenge 2019: Data" dataset.

## Training

To train the model(s) in the paper, run this command:

```train
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
model.fit(X_train_scaled, y_train_reshaped, epochs=10, batch_size=60, validation_data=(X_test_scaled, y_test_reshaped))
```

>  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.
Train/test split of 0.75/0.25
>Adam optimizer, loss function = MAE, metric = MAE


## Evaluation

To evaluate my model, run:

```eval
predicted_values = model.predict(X_test_scaled)
y_test = y_test.tolist()
mse = mean_squared_error(predicted_values, y_test)
print(mse)
```

> 
Evaluate the model using the validation dataset. Identify the highest performing epoch based on the lowest val_mae. You may also graph the MAE over epochs.

## Results

Our model achieves the following performance on the validation test set :


|          | Top 1 MAE  | Top 5 MAE |
| ------------------ |----------- | --------- |
| CNN                |    212     |    237.2    |
  
