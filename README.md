# CNN-Image-Classifier---Deep-Learning


**Project Report:** Fruit Freshness Classification

**Course**: Introduction to Deep Learning

**Date**: April 20, 2026

**Dataset**: Fruit Freshness Dataset (Kaggle)

***Introduction & Problem Statement***

The objective of this project is to develop a Convolutional Neural Network (CNN) to automate the quality control process for produce. By classifying images of fruit as either "Fresh" or "Rotten," this model aims to provide a scalable solution for food inventory management, helping to reduce waste and ensure food safety within retail or supply chain environments. A key focus of this study is the application of deep learning to automatically identify visual patterns such as bruising, discoloration, and texture changes. These features would then be used to distinguish fresh fruit from decaying fruit. By implementing a CNN, I aim to determine how effectively a multi-layered neural network can handle image classification compared to traditional manual inspection, providing a reliable baseline for automated sorting technologies.

***Data Preprocessing and Feature Engineering***

To prepare the dataset for the CNN model, several preprocessing steps were performed stated down below:

• **Data Collection and Labeling**: An automated crawl of the dataset directory was implemented to assign labels based on folder names (0 for Fresh and 1 for Rotten). The dataset contains 529 total images, with 342 Fresh and 187 Rotten instances.

• **Normalization**: To assist in model convergence and ensure numerical stability, all pixel values were rescaled from the standard 0–255 range to a normalized range of 0.0–1.0.

• **Image Resizing**: All images were standardized to a resolution of 224x224 pixels. This ensures that the input layer of the CNN receives consistent tensor shapes regardless of the original image dimensions.

• **Data Pipeline**: A TensorFlow processing pipeline was created to shuffle the training data and group it into batches of 32. Prefetching was enabled to ensure that the CPU prepares data while the model is training, optimizing computational efficiency.

# Methodology and Model Implementation

The dataset was split into a training set (70%) and a testing set (30%) using a random_state of 42 to ensure reproducibility. A Sequential CNN architecture was implemented to extract features through a hierarchy of layers:

A. **Convolutional Layers**: Three blocks of Conv2D layers were used (starting with 32 filters and moving up to 128). These layers use the ReLU activation function to identify spatial features like edges and textures.

B. **Pooling Layers**: MaxPooling2D layers were placed after convolutional blocks to reduce spatial dimensions, helping the model focus on the most important features and reducing the risk of overfitting.

**Baseline for the Model:**

Important to note that during development, multiple split ratios were tested(80/20), but a 70/30 split was chosen for its balanced performance. The model was trained over 15 epochs using the Adam optimizer with a learning rate of $1 \times 10^{-4}$.

Initially, the model achieved very high accuracy quickly, but I investigated potential overfitting. To address this, a Dropout layer (0.5) was added before the final output. This forced the network to learn more robust features rather than relying on specific pixel patterns. By doing so, the model became much more reliable for real-world application, ensuring that the high accuracy was a result of genuine pattern recognition rather than memorization of the training set.

Taking note that the labels were automatically generated from the directory structure to ensure the target variable for predictions was accurate.

# Performance Evaluation

The model was evaluated using Accuracy, a Confusion Matrix, and a Classification Report.

***4.1 CNN Results***

• **Accuracy**: 94.97%

• **Precision (Rotten)**: 0.90

• **Recall (Rotten)**: 0.96

• **Confusion Matrix Observations**: The model correctly identified 97 fresh items and 54 rotten items, with only 2 false negatives (rotten fruit misclassified as fresh).

***Model Comparison and Interpretation***
Comparing the CNN results to the baseline, the model performed exceptionally well, achieving a validation accuracy of nearly 95%. Unlike simpler linear models, the CNN was able to capture the non-linear, complex visual cues associated with fruit decay. The high recall for the "Rotten" class is particularly significant.

In a supply chain context, the Recall for "Rotten" items is the most critical metric. Missing a "Rotten" item (a False Negative) is significantly more detrimental for a retailer, as it leads to spoiled inventory reaching consumers. My results show that the CNN is highly effective in its classification approach, successfully flagging the vast majority of decaying produce. Although the real-world use of the model is not life or death (ie wildfires) it is still important to understand the models uses and risks.

**Conclusion**

This project highlights a critical lesson in deep learning: visual feature extraction is a powerful tool for quality assurance. While high overall accuracy was achieved, it was important to look beyond that metric to ensure the model was actually identifying decay. The implementation of Dropout and pooling layers proved essential in creating a model that generalizes well to new data.

In contrast to manual sorting, the CNN demonstrated superior utility by capturing subtle interactions in pixel data that indicate spoilage. In a commercial context, the cost of a False Negative (allowing rotten fruit to pass) far outweighs the cost of a False Positive. This project has shown the importance of robust architecture design and the necessity of questioning high accuracy to ensure it stems from meaningful feature learning rather than data leakage or overfitting.
