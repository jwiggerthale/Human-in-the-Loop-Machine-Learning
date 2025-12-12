# Human-in-the-Loop-Machine-Learning

This repo implements the code for our paper "Operationalizing the R4VR-framework: Safe Human-in-the-Loop 4
Machine Learning for Image Recognition". In the course of the paper we demonstrate hoe to create highly reliable ML-based inspection systems based on the *R4VR*-framework using explainable AI (XAI) and uncertainty quantification (UQ). The basic workflow of the framework is shown below: 

![Fig. 1: Basic workflow of *R4VR*-framework](https://github.com/jwiggerthale/Human-in-the-Loop-Machine-Learning/blob/main/ims/R4VR.PNG)
*Fig. 1: Basic workflow of R4VR-framework; developer undergoes the phases "Reliability", "Validation" and "Verification" to create highly reliable models; end user undergoes the steps of "Verification", "Validation" and "Reliability" to apply modesl properly*

Our exact guidelines on how to adopt the framework in visual inspection is shown here: 

![Fig. 2: Guidelines on how to adopt the *R4VR*-framework in visual inspection](https://github.com/jwiggerthale/Human-in-the-Loop-Machine-Learning/blob/main/ims/R4VRInspection.jpg)
*Fig. 2: Guidelines on how to adopt the *R4VR*-framework in visual inspection*

The process of data collection is shown in Fig. 3. 

![Fig. 3: Visualization of reliable and efficient data collection process](https://github.com/jwiggerthale/Human-in-the-Loop-Machine-Learning/blob/main/ims/DataCollection.jpg)
*Fig. 3: Visualization of reliable and efficient data collection process; baseline dataset is curated manuylly; baseline model is trained; new images are labeled by baseline model automatically and human operator only corrects wrong predictions; baseline model is retrained and improved iteratively*

For demonstration of the workflow, we utilized the [severstal dataset](https://datasetninja.com/severstal). Also, we conducted exploratory experiments using the NEU metal surface defects dataset from [kaggle](https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data/data).

In the first step, we deonstrate how the data generation process scales and examine how different parameter settings affect manual labeling effort as well as computing time. To replicate the experiment, you can run the script "train_model_data_generation.py". This will create a json-file with statistics from the modeling process for each parameter setting as well as one .csv-file with runtime data for the different settings. You can use these files to create a plot like this (Fig. 4). 

![Fig. 4: Manual labeling effort and computing time depending on the step size and the initial size of the training dataset](https://github.com/jwiggerthale/Human-in-the-Loop-Machine-Learning/blob/main/HiL%20ML/NEU%20metal%20surface%20defects%20data/ims/ManualLabelingEffortBaseModel.png)
*Fig. 4: Manual labeling effort and computing time depending on the step size and the initial size of the training dataset*

Afterwards, we train two different models (ResNet18 and VGG16 in case of the NEU dataset and ResNet18 as well as EfficientNet-B0 in case of severstal dataset) on the entire dataset. You can replicate this training by running the script "train_model.py" with argument "use_aug" = "False" in the appropriate folders. 

Based on the model, we implement saliency maps (XAI) as well as Monte Carlo Dropout (UQ). We use these techniques to examine and improve the model. To examine the distributions of uncertaities, you can run the script "get_mc_scores.py". To find out how the uncertainties behave when progressively increasing the number of pixels available, you can run the script "saliency_map_uncertaities.py". 

To implement the training with data augmentation as described in Sec. 4.2 of our paper, you can run the scrpit "train_model.py" with argument "use_aug" = "True". We kindly ask you to create the dataset splits yourself as we can't upload the files due to their size. For more detils on the approach, please refer to our paper or reach out to julius.wiggerthale@hs-furtwangen.de

