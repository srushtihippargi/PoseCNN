# **PoseCNN: 6D Object Pose Estimation**

This repository provides a **PoseCNN** implementation for **6D object pose estimation**. It includes a **step-by-step guide** in a Jupyter notebook to train, test, and visualize model predictions.

## **Dataset**
We use the **props dataset** from the **University of Michigan's PROGRESS Lab**. This dataset contains a variety of objects captured from multiple angles, with corresponding pose annotations. Below is an example of how the dataset appears after preprocessing:

![Screenshot from 2025-02-02 17-32-02](https://github.com/user-attachments/assets/27a65058-ca57-4a34-b04e-0ec1b074f836)

## **Segmentation Branch**
The segmentation branch fuses extracted features from the backbone network to segment objects within the scene. After training, the segmentation inference should resemble the following results:


![Screenshot from 2025-02-02 17-32-24](https://github.com/user-attachments/assets/d3dcf32f-631c-4cad-8c08-666c134c7f15)

## **Rotation & Translation Estimation**
PoseCNN predicts **rotation and translation** using separate branches, combined with a **Hough voting** mechanism for refinement. Once the model is trained, inference results should look like this:

![Screenshot from 2025-02-02 17-49-57](https://github.com/user-attachments/assets/cc6a8b4e-b775-479f-b544-e1545f8fb963)


