# Code from the Federated Learning section of the Uscalt BT Young Scientist project

## IMPORTANT AMMENDMENTS

We have made ammendments to files due to there being a mistake. The files contained a training function that would load the current model after loading a stale model. This is incorrect and was not how we trained our models. However, we did use this when training our 20 client model and we have rectified and retrained. We apologise for this. In addition to retriaining, we trained a buffered Async FL model (fedbuff.py) to compare it to. Below are updated figures for the task

![](0.01_40.png)
Training with Learning Rate = 0.01 (to reach 40% accuracy)
![](0.1.png)
Training with Learning Rate = 0.1 (to reach 40% accuracy)
![](0.01_50.png)
Training with Learning Rate = 0.01 (to reach 50% accuracy)

-------
All of the files contain code to load and create ***different*** dataset splits for each client. 

However, the dataset splits remain the same throughout each experiment (the datasets were shuffled and split once, and they were saved and loaded as needed).
