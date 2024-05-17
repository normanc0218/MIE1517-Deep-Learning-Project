## **Overview**
Our project is focused on building a pipeline for a household object locator. The pipeline will use our voice as input and process the object that we want to detect. Then we will use object localization and detection to search for the object in question using image feed from a camera. We will then use an LLM to process the outputs of the localization algorithm to help guide the user where to look. Currently, we have results for two sections. **Part 1** will be focused on the Object Localization and Detection module results and **Part 2** will be focused on our Speech to Text module. Finally, **Part 3** will explain what we have left to do and why we believe we will succeed in this project.
## **Part 1: Object Localization and Detection**
##Overview
We are using Faster RCNN model architecture for object localization and detection. The Faster RCNN is implemented in 2 stages namely: Region Proposal Network (RPN) and Object Detection/Classification.

**Stage 1: Region Proposal Network (RPN)**

Objective: To generate region proposals where there might be an object based on objectness scores.
Architecture: It comprises a feature extraction backbone (using a pre-trained model such as MobileNet) and a proposal module (network of 2 convolutional layers)
Outputs: RPN generates proposed regions with probabilities of having an object in them

**Stage 2: Object Detection/Classification Module**

Objective: To classify the proposed regions into specific object categories and further refine the bounding box positions.
Architecture: currently we have 1 convolutional layer and 2 linear layers (not shown in this progress report)
Outputs: class labels and coordinates of the bounding boxes for each detected object.
Dataset

We are training our model on a subset of COCO dataset; a large-scale object detection, segmentation, and captioning dataset.
Classes of interest are: "mouse", "keyboard", "laptop","cellphone"

## **Part 2: Automatic Speech Recognition (Speech to Text Pipeline)**
The second part of our pipeline focuses on training a speech to text model so that we tell our object detection model which object to look for through our voice.
## **Part 3: What we have left to do and why we will succeed?**
Given our currently trained models, it is evident that we have some strong  results that we will further need to finetune.

1.   **R-CNN:** Regarding the R-CNN, we currently have an RPN model that works decently with the COCO dataset. The model is able successfully predict object bounding boxes and overfit on a small dataset. The next step for this model is to incorporate the object classifier (last part of Faster RCNN), refine the model backbone (feature extractor) and track training/validation loss and accuracies on a larger dataset. We will further finetune the prediction confidence and test the module with image data from a real camera. This will give us a unique testing dataset with which we can verify the true performance of the model.
2.   **Speech to Text:** We have a decently performative model for automatic speech recognition that works with voice data that is not within the dataset. We, however, need to improve the WER and CER of the test dataset as the error still appears to be large. We can do this by the steps detailed at the end of Section 2.
3. **LLM:** We currently have some progress on the LLM pipeline, which was not shown in this document. Our next steps for us will be to integrate the outputs of the R-CNN and Speech to Text module to work with the LLM interface.

Considering we have made significant progress in each element of our pipeline, we believe we will be able to successfully complete the project within the given timeframe. Our next week will be focused on fine tuning our models, and the week after will be integration of our LLM and our presentation.

