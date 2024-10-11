# Custom CNN Analysis (Course Assignment)
Explore a new CNN design for personalized handwriting recognition. The dataset to be used is EMNIST. We will focus on: EMNIST ByClass: 814,255 characters. 62 unbalanced classes to solve the following tasks:
1. Custom Loss function Design and evaluation: Extension of cross entropy loss to add a penalty for all classes, not just the true class. Think of how you will give the penalty when you give the penalty and what you will do if some false class is predicted with a very low probability.
2. Regularization Loss: Add a regularizer that promotes orthogonal filters at each layer.
3. Explore Focal Loss and SMOTE and other techniques to handle class imbalance in the dataset.
4. Neural Architecture Search: Since the input is only 28x28 and there are 62 classes, we can work with a smaller network. Our goal is to beat the performance [Accuracy, Precision, Recall, F1 Score] of ResNet with a much smaller number of parameters. Report both micro-average and macro-average performance.
5. Write a report with insights obtained and well-documented code.
