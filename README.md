
#Preproccessing of data - 

Organized the images into the respective folders using preprocessing function named folder_creation()

#Image classification of data

We use transfer learning method for this classification. For this I used retrain.py function released by Tensorflow for 
retraining a given model.

I used the default parameters for training the model-

Model used: Inception v3.0
Training_steps: 4,000
Learning_rate: 0.01
Testing_percentage: 10
Valiation_percentage: 10
Batch size for training: 100

#Inference data

run inference.py filename.jpg





