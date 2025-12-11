Running the script [train_model.py](https://github.com/jwiggerthale/Human-in-the-Loop-Machine-Learning/blob/main/HiL%20ML/Severstal/train_model.py)
will create four folders: 

- {out_dir}/resnet_fine/seed_{seed}
- {out_dir}/effnet_fine/seed_{seed}
- {out_dir}/resnet_pre/seed_{seed}
- {out_dir}/effnet_pre/seed_{seed}

In each of these folders, a file 'model_acc.pth', a model 'model_loss.pth' as well as a file 'train_stats.txt' will be saved. The 'train_stats.txt' file contains information on loss and accuracy for each epoch of training. 
We uploaded one example. However, these values refer to the training and validation data. More relevant is performance on test data.
In course of our research, we trained models 10 times with 10 different seed. Performance on test data for each seed and training cycle is reported below. 


# Accuracies Without Augmentation 

  |Seed|resnet pre|effnet pre|resnet fine|effnet fine|
  |---|---|---|---|---|
  |1|0.84049379|0.86939836|0.84890486|0.87553926|
  |17|0.83828362|0.85241545|0.84827138|0.879214|
  |22|0.83896215|0.86049516|0.8615439|0.90008717|
  |33|0.84159936|0.86172069|0.84354698|0.87674305| 
  |42|0.84511539|0.85640487|0.84230945|0.87322723|
  |59|0.84302783|0.86164495|0.8460466|0.8810602|
  |66|0.84914144|0.8717689|0.85497525|0.87653728|
  |73|0.83430735|0.8618716|0.85121618|0.86731785|
  |88|0.84956194|0.86595571|0.84152673|0.89424046|
  |90|0.82832924|0.87069692|0.86711351|0.87399414|
  


# Accuracies With Augmentation 

  |Seed|resnet pre|effnet pre|resnet fine|effnet fine|
  |---|---|---|---|---|
  |1|0.9250851|0.94061886|0.978002|0.96445063|
  |17|0.928539|0.93622878|0.97726345|0.97133829|
  |22|0.92532094|0.93407558|0.98786269|0.9680612|
  |33|0.93448472|0.94584528|0.98335925|0.9654817| 
  |42|0.93279712|0.93092644|0.97924123|0.96793534|
  |59|0.9281662|0.94657436|0.97844493|0.97568381|
  |66|0.93039119|0.94528096|0.97966038|0.97604294|
  |73|0.92237932|0.93932903|0.97752891|0.98170732|
  |88|0.93241666|0.94639486|0.98168739|0.97155302|
  |90|0.93418797|0.93871312|0.97766844|0.96716833|
  
