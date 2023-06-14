# Facial-Recognition

**Title: Facial Expression Recognition using Convolutional Neural Networks**

**1. Introduction**
The project aims to develop a facial expression recognition system using Convolutional Neural Networks (CNN). The dataset consists of 48x48 grayscale images of faces categorized into seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The goal is to build a CNN model that can accurately classify the emotions based on the facial expressions.

**2. Data Preparation**
The dataset is loaded from the provided CSV file, which contains labels and pixel data for each image. The data is divided into training, validation, and test sets. The pixel values are reshaped and scaled to fit the input requirements of the CNN model. Helper functions are defined to prepare and visualize the data.

**3. Exploratory Data Analysis**
Visualizations are created to understand the distribution of emotions in the dataset. Bar plots are generated to show the frequency of each emotion in the training and validation sets. Example images for each emotion are also plotted to provide a visual representation of the data.

<img width="1187" alt="Screenshot 2023-06-13 at 8 59 13 PM" src="https://github.com/MananAbbott/Facial-Recognition/assets/35071080/dcd51872-dc48-4f2a-96dd-0be64f968437">
<img width="1188" alt="Screenshot 2023-06-13 at 9 00 12 PM" src="https://github.com/MananAbbott/Facial-Recognition/assets/35071080/d3e6540d-ee08-4acb-b9c8-6ccc0a020092">
<img width="1190" alt="Screenshot 2023-06-13 at 9 00 54 PM" src="https://github.com/MananAbbott/Facial-Recognition/assets/35071080/1249facb-f700-4b7a-b776-1cc3cf2fc952">
<img width="1187" alt="Screenshot 2023-06-13 at 9 01 49 PM" src="https://github.com/MananAbbott/Facial-Recognition/assets/35071080/be3004fc-81bb-4a41-bba9-d5a424048bfb">

**4. Model Architecture**
A simple CNN model is defined using the Keras framework. The model consists of convolutional layers with activation functions, max pooling layers, and fully connected layers. The final layer uses the softmax activation function for multi-class classification. The model is compiled with an appropriate optimizer, loss function, and metrics.

**5. Model Training**
The model is trained on the training set using the fit() function. The training data is fed into the model in batches, and the class weights are considered to address the class imbalance issue in the dataset. The validation set is used to evaluate the model's performance during training. The training history, including loss and accuracy values, is recorded.

![output](https://github.com/MananAbbott/Facial-Recognition/assets/35071080/c9148005-edd5-499c-b4e9-905ebe32291e)

![output1](https://github.com/MananAbbott/Facial-Recognition/assets/35071080/a25ef53e-cd44-44a5-aa0e-55afeaf72f18)


**6. Model Evaluation**
The trained model is evaluated on the test set using the evaluate() function. The test accuracy is calculated to assess the model's performance on unseen data.

**7. Predictions and Analysis**
The model is used to predict emotions for test images. The predicted labels are compared with the ground truth labels to analyze the model's accuracy. A function is provided to plot the predicted emotion probabilities and the corresponding image for a specific test sample.

![test](https://github.com/MananAbbott/Facial-Recognition/assets/35071080/3438a064-4476-4893-b5f6-cc69bb5dafd9)
![test1](https://github.com/MananAbbott/Facial-Recognition/assets/35071080/95bc1b36-f9e7-48db-b53e-aa6e827680ce)
![valu](https://github.com/MananAbbott/Facial-Recognition/assets/35071080/bcd79be7-a85b-4d79-b8cd-40e87c0b83f9)
![matrix](https://github.com/MananAbbott/Facial-Recognition/assets/35071080/77bc3328-0b30-4579-bdde-bcd701ed61f8)


**8. Conclusion**
The facial expression recognition model based on CNN architecture achieves a certain level of accuracy in categorizing emotions from images. The model can be further optimized by adjusting hyperparameters, exploring different network architectures, and increasing the diversity and size of the dataset. The project demonstrates the potential of deep learning techniques in facial expression analysis and opens doors for applications in various domains, including emotion detection systems and human-computer interaction.

Overall, the project provides a comprehensive framework for facial expression recognition using CNNs, starting from data preparation and visualization to model training and evaluation. The insights gained from this project can be utilized for further research and development in the field of computer vision and emotion recognition.
