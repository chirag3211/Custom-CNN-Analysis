### Analysis.md contains the results of the experiments done on Resnet-18 architecture for the following settings: (models in Google Drive)
1. Cross-entropy loss + l2 regularization (demo1.ipynb) (resnet18_classifier_setting1.pth)
2. Custom Cross-entropy loss without l2 regularization (demo2.ipynb) (resnet18_classifier_setting2.pth)
3. Softmax focal loss without l2 regularization (demo3.ipynb) (resnet18_classifier_setting3.pth)
4. Custom Cross-entropy loss with Orthogonal Regularization (demo4.ipynb) (resnet18_classifier_setting4.pth)
5. Custom Cross-entropy loss after SMOTE oversampling (smote.py)
6. Custom Cross-entropy loss after Borderline-SMOTE oversampling (blsmote.py) (resnet18_classifier_setting6.pth)

### results.md contains the results of the experiments done for NAS in the following settings:
#### Architectures (models in Google Drive):
1. Arch1 -> Derivative(s) of VGG-13 (model1.pth, model1_heavy.pth)
2. Arch2 -> Derivative(s) of Resnet-18 (model2.pth, model2_2.pth)
3. Arch3 -> Derivative(s) of DenseNet-121 (model3.pth, model3_2.pth, model3_blsmote.pth)
4. Arch4 -> Derivative(s) of MobileNetV2 (model4.pth)

#### Settings: (top-performer settings in Analysis.md)
1. Custom Cross-entropy loss without l2 regularization
2. Custom Cross-entropy loss with Borderline-SMOTE oversampling

#### Arch -> Code:
1. model1.pth -> Arch1.ipynb
2. model2.pth, model2_2.pth -> Arch2.ipynb
3. model3.pth -> Arch3.ipynb
4. model3_2.pth -> Arch3_2.ipynb
5. model3_blsmote.pth -> Arch3_blsmote.ipynb
6. model4.pth -> Arch4.ipynb 

### Arch -> # of parameters:
1. model1.pth -> 52822
2. model1_heavy.pth -> 647934
3. model2.pth -> 21422
4. model2_2.pth -> 173214
5. model3.pth -> 70838
6. model3_2.pth -> 462638
7. model3_blsmote.pth -> 119318
8. model4.pth -> 149566
9. resnet18_classifier_setting1.pth -> 8435070 (trainable parameters: ~130k)

### Other files:
1. demo.ipynb -> Contains the implementation of all the components of the project (all loss functions, regularizers, oversampling techniques, resnet-18 architecture etc.). This is used as a reference for the implementation of the components in the other files.
2. smote.ipynb -> Contains the implementation of SMOTE and Borderline-SMOTE oversampling techniques to save the final balanced dataset.
3. archs.py -> Contains the implementation of some of the architectures used in NAS.
4. Google Drive Link: https://drive.google.com/drive/folders/1Ecdb-c3rAYBmhBbVEFMFBU-MDPYSrzxS?usp=sharing
5. evaluate.ipynb -> Contains the code to evaluate and analyze the models (need to set the models' path if you want to reproduce them). The plots are in plotly so they are not visible in the downloaded file. You can run the notebook in Google Colab to see the plots using the link: https://colab.research.google.com/drive/1-SpEsNEAzcdpCOtSBi-YdxzevnhV8dDn?usp=sharing
    
### Color Interpretation for Filters:
- Bright Colors (High Magnitude Weights): Bright regions in the filters (whether grayscale or color) indicate strong activations or connections for that region, meaning the filter is more sensitive to certain patterns (edges, textures, etc.) in those areas.
- Darker Colors (Low Magnitude Weights): Darker areas represent weaker connections, meaning the filter is less responsive to patterns in those areas.

### Libraries Used:
1. PyTorch
2. NumPy
3. Matplotlib
4. Scikit-learn
5. Imbalanced-learn
6. Plotly
7. Tqdm

### Note:
- In smote and blsmote, the resnet models are training by randomly sampling 50% of the oversampled dataset to fasten the training process.
- Python version used is 3.10.12 (same as google colab's python's version). No need to specify the version of the libraries while installing. Install the latest ones using pip in the same python version.
- The balanced datasets are very big so I didn't upload them on Drive. You can generate them using the code in smote.ipynb.