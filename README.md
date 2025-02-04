# **PoseCNN Implementation for 6D Pose Estimation**

This repository provides an implementation of **PoseCNN** for **6D pose estimation**, following the step-by-step approach outlined in the paper. The provided **Jupyter notebook** guides you through training your own model.

## **Dataset**
We use the **props dataset** from the **University of Michigan's PROGRESS Lab**, led by **Professor Chad Jenkins**. Below is a snapshot of the dataset after passing through the dataloader:

![Screenshot from 2025-02-02 17-32-02](https://github.com/user-attachments/assets/27a65058-ca57-4a34-b04e-0ec1b074f836)

## **Segmentation Branch Implementation**
The segmentation branch integrates features extracted from the backbone network. After training, the inference results should resemble the following:

![Screenshot from 2025-02-02 17-32-24](https://github.com/user-attachments/assets/d3dcf32f-631c-4cad-8c08-666c134c7f15)

## **Rotation and Translation Branches with Hough Voting**
The **rotation and translation branches**, along with the **Hough voting layer**, are implemented in the corresponding Python files. Once trained, you can load your model in the notebook for inference. Below is an example output from the test split of the dataset:

![Screenshot from 2025-02-02 17-49-57](https://github.com/user-attachments/assets/cc6a8b4e-b775-479f-b544-e1545f8fb963)


