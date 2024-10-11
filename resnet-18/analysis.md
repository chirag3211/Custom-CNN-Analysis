# Analysis of various settings

## Setting 1: Cross-entropy loss + l2 regularization (demo)

Training set size: 593242 <br>
Validation set size: 104690 <br>
Test set size: 116323 <br>
Training Epoch 1: 100%|██████████| 1159/1159 [16:53<00:00,  1.14it/s, accuracy=0.848, loss=4.08] <br>
Validating: 100%|██████████| 205/205 [01:53<00:00,  1.81it/s, accuracy=0.795, val_loss=3.84] <br>
Epoch 1/10, Train Loss: 4.0807, Val Loss: 3.8422, Train Acc: 0.8478, Val Acc: 0.7947, Train Prec: 0.8292, Val Prec: 0.8025, Train F1: 0.8312, Val F1: 0.7627 <br>
Training Epoch 2: 100%|██████████| 1159/1159 [19:18<00:00,  1.00it/s, accuracy=0.859, loss=3.44] <br>
Validating: 100%|██████████| 205/205 [02:52<00:00,  1.19it/s, accuracy=0.814, val_loss=3.53] <br>
Epoch 2/10, Train Loss: 3.4376, Val Loss: 3.5269, Train Acc: 0.8592, Val Acc: 0.8139, Train Prec: 0.8452, Val Prec: 0.8298, Train F1: 0.8418, Val F1: 0.7802 <br>
Training Epoch 3: 100%|██████████| 1159/1159 [24:45<00:00,  1.28s/it, accuracy=0.862, loss=3.31]  <br>
Validating: 100%|██████████| 205/205 [02:21<00:00,  1.44it/s, accuracy=0.841, val_loss=3.37] <br>
Epoch 3/10, Train Loss: 3.3120, Val Loss: 3.3737, Train Acc: 0.8615, Val Acc: 0.8409, Train Prec: 0.8476, Val Prec: 0.8382, Train F1: 0.8443, Val F1: 0.8218 <br>
Training Epoch 4: 100%|██████████| 1159/1159 [21:08<00:00,  1.09s/it, accuracy=0.862, loss=3.27] <br>
Validating: 100%|██████████| 205/205 [02:32<00:00,  1.34it/s, accuracy=0.82, val_loss=3.4]   <br>
Epoch 4/10, Train Loss: 3.2659, Val Loss: 3.4024, Train Acc: 0.8625, Val Acc: 0.8203, Train Prec: 0.8538, Val Prec: 0.8193, Train F1: 0.8454, Val F1: 0.7992 <br>
Training Epoch 5: 100%|██████████| 1159/1159 [21:55<00:00,  1.13s/it, accuracy=0.862, loss=3.25] <br>
Validating: 100%|██████████| 205/205 [02:24<00:00,  1.42it/s, accuracy=0.8, val_loss=3.49]   <br>
Epoch 5/10, Train Loss: 3.2481, Val Loss: 3.4858, Train Acc: 0.8621, Val Acc: 0.7997, Train Prec: 0.8490, Val Prec: 0.8079, Train F1: 0.8453, Val F1: 0.7757 <br>
Testing: 100%|██████████| 228/228 [02:36<00:00,  1.45it/s, accuracy=0.84, loss=3.33]  <br>
Test Loss: 3.3250, Accuracy: 0.84%, Precision: 0.84, F1 Score: 0.82 <br>


## Setting 2: Custom Cross-entropy loss without regularization (demo2)

Training set size: 593242 <br>
Validation set size: 104690 <br>
Test set size: 116323 <br>
Training Epoch 1: 100%|██████████| 2318/2318 [29:15<00:00,  1.32it/s, accuracy=0.858, loss=0.425]   <br>
Validating: 100%|██████████| 409/409 [03:09<00:00,  2.16it/s, accuracy=0.866, val_loss=0.387]  <br>
Epoch 1/10, Train Loss: 0.4251, Val Loss: 0.3873, Train Acc: 0.8582, Val Acc: 0.8659, Train Prec: 0.8466, Val Prec: 0.8656, Train F1: 0.8452, Val F1: 0.8507 <br>
Training Epoch 2: 100%|██████████| 2318/2318 [28:52<00:00,  1.34it/s, accuracy=0.875, loss=0.36]  <br>
Validating: 100%|██████████| 409/409 [03:26<00:00,  1.98it/s, accuracy=0.865, val_loss=0.385]  <br>
Epoch 2/10, Train Loss: 0.3605, Val Loss: 0.3849, Train Acc: 0.8747, Val Acc: 0.8653, Train Prec: 0.8661, Val Prec: 0.8647, Train F1: 0.8634, Val F1: 0.8544 <br>
Training Epoch 3: 100%|██████████| 2318/2318 [26:34<00:00,  1.45it/s, accuracy=0.877, loss=0.355] <br>
Validating: 100%|██████████| 409/409 [01:56<00:00,  3.51it/s, accuracy=0.873, val_loss=0.364]  <br>
Epoch 3/10, Train Loss: 0.3553, Val Loss: 0.3640, Train Acc: 0.8766, Val Acc: 0.8733, Train Prec: 0.8688, Val Prec: 0.8680, Train F1: 0.8657, Val F1: 0.8611 <br>
Training Epoch 4: 100%|██████████| 2318/2318 [18:24<00:00,  2.10it/s, accuracy=0.882, loss=0.335] <br>
Validating: 100%|██████████| 409/409 [02:03<00:00,  3.30it/s, accuracy=0.335, val_loss=4.37] <br>
Epoch 4/10, Train Loss: 0.3353, Val Loss: 4.3662, Train Acc: 0.8821, Val Acc: 0.3353, Train Prec: 0.8751, Val Prec: 0.6056, Train F1: 0.8718, Val F1: 0.3090 <br>
Training Epoch 5: 100%|██████████| 2318/2318 [18:03<00:00,  2.14it/s, accuracy=0.881, loss=0.338] <br>
Validating: 100%|██████████| 409/409 [02:00<00:00,  3.39it/s, accuracy=0.876, val_loss=0.354]  <br>
Epoch 5/10, Train Loss: 0.3379, Val Loss: 0.3538, Train Acc: 0.8813, Val Acc: 0.8764, Train Prec: 0.8745, Val Prec: 0.8718, Train F1: 0.8712, Val F1: 0.8671 <br>
Testing: 100%|██████████| 455/455 [02:45<00:00,  2.76it/s, accuracy=0.877, loss=0.354]  <br>
Test Loss: 0.3540, Accuracy: 0.88%, Precision: 0.87, F1 Score: 0.87 <br>


## Setting 3: Softmax Focal loss without regularization (colab 52) [Colab Timed Out]

Training set size: 593242 <br>
Validation set size: 104690 <br>
Test set size: 116323 <br>
Training Epoch 1: 100%|██████████| 1325/1325 [42:59<00:00,  1.95s/it, accuracy=0.851, loss=0.197] <br>
Validating: 100%|██████████| 234/234 [04:07<00:00,  1.06s/it, accuracy=0.856, val_loss=0.173] <br>
Epoch 1/10, Train Loss: 0.1971, Val Loss: 0.1726, Train Acc: 0.8506, Val Acc: 0.8559, Train Prec: 0.8387, Val Prec: 0.8560, Train F1: 0.8405, Val F1: 0.8441 <br>
Training Epoch 2: 100%|██████████| 1325/1325 [42:28<00:00,  1.92s/it, accuracy=0.869, loss=0.148] <br>
Validating: 100%|██████████| 234/234 [04:05<00:00,  1.05s/it, accuracy=0.843, val_loss=0.189] <br>
Epoch 2/10, Train Loss: 0.1485, Val Loss: 0.1888, Train Acc: 0.8692, Val Acc: 0.8430, Train Prec: 0.8597, Val Prec: 0.8472, Train F1: 0.8594, Val F1: 0.8233 <br>
Training Epoch 3: 100%|██████████| 1325/1325 [42:26<00:00,  1.92s/it, accuracy=0.875, loss=0.136] <br>
Validating: 100%|██████████| 234/234 [04:06<00:00,  1.05s/it, accuracy=0.865, val_loss=0.157] <br>
Epoch 3/10, Train Loss: 0.1362, Val Loss: 0.1574, Train Acc: 0.8749, Val Acc: 0.8654, Train Prec: 0.8668, Val Prec: 0.8647, Train F1: 0.8654, Val F1: 0.8527 <br>
Training Epoch 4: 100%|██████████| 1325/1325 [42:27<00:00,  1.92s/it, accuracy=0.879, loss=0.127]
Validating: 100%|██████████| 234/234 [04:09<00:00,  1.07s/it, accuracy=0.857, val_loss=0.163]
Epoch 4/10, Train Loss: 0.1272, Val Loss: 0.1631, Train Acc: 0.8793, Val Acc: 0.8567, Train Prec: 0.8719, Val Prec: 0.8609, Train F1: 0.8702, Val F1: 0.8431
Testing: 100%|██████████| 260/260 [04:28<00:00,  1.03s/it, accuracy=0.839, loss=0.186]
Test Loss: 0.1865, Accuracy: 0.84%, Precision: 0.85, F1 Score: 0.82


## Setting 4: Custom Cross-entropy loss + Orthogonal Regularization (demo3)

Training set size: 593242 <br>
Validation set size: 104690 <br>
Test set size: 116323 <br>
Training Epoch 1: 100%|██████████| 1159/1159 [27:19<00:00,  1.41s/it, accuracy=0.846, loss=1.31] <br>
Validating: 100%|██████████| 205/205 [03:35<00:00,  1.05s/it, accuracy=0.822, val_loss=0.83] <br>
Epoch 1/10, Train Loss: 1.3111, Val Loss: 0.8299, Train Acc: 0.8460, Val Acc: 0.8217, Train Prec: 0.8284, Val Prec: 0.8288, Train F1: 0.8309, Val F1: 0.7865 <br>
Training Epoch 2: 100%|██████████| 1159/1159 [28:58<00:00,  1.50s/it, accuracy=0.859, loss=0.565] <br>
Validating: 100%|██████████| 205/205 [09:15<00:00,  2.71s/it, accuracy=0.739, val_loss=1.08] <br>
Epoch 2/10, Train Loss: 0.5655, Val Loss: 1.0759, Train Acc: 0.8591, Val Acc: 0.7388, Train Prec: 0.8466, Val Prec: 0.7827, Train F1: 0.8445, Val F1: 0.7190 <br>
Training Epoch 3: 100%|██████████| 1159/1159 [1:04:58<00:00,  3.36s/it, accuracy=0.863, loss=0.492] <br>
Validating: 100%|██████████| 205/205 [07:49<00:00,  2.29s/it, accuracy=0.838, val_loss=0.561] <br>
Epoch 3/10, Train Loss: 0.4921, Val Loss: 0.5609, Train Acc: 0.8635, Val Acc: 0.8384, Train Prec: 0.8526, Val Prec: 0.8503, Train F1: 0.8497, Val F1: 0.8050 <br>
raining Epoch 4: 100%|██████████| 1159/1159 [1:02:36<00:00,  3.24s/it, accuracy=0.866, loss=0.462]
Validating: 100%|██████████| 205/205 [10:47<00:00,  3.16s/it, accuracy=0.786, val_loss=0.735]
Epoch 4/10, Train Loss: 0.4618, Val Loss: 0.7350, Train Acc: 0.8663, Val Acc: 0.7860, Train Prec: 0.8557, Val Prec: 0.8074, Train F1: 0.8532, Val F1: 0.7745
Testing: 100%|██████████| 228/228 [11:03<00:00,  2.91s/it, accuracy=0.793, loss=0.705]
Test Loss: 0.7055, Accuracy: 0.79%, Precision: 0.82, F1 Score: 0.78


## Setting 5: Custom Cross-entropy loss with SMOTE (smote.py)

Training set size: 2022309 <br>
Validation set size: 356879 <br>
Test set size: 116323 <br>
Training Epoch 1: 3950it [1:00:29,  1.09it/s, accuracy=0.827, loss=0.224] <br>
Validating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 698/698 [05:18<00:00,  2.19it/s, accuracy=0.77, val_loss=0.658] <br>
Epoch 1/5, Train Loss: 0.4488, Val Loss: 0.6579, Train Acc: 0.8274, Val Acc: 0.7703, Train Prec: 0.8271, Val Prec: 0.8108, Train F1: 0.8267, Val F1: 0.7499 <br>
Training Epoch 2: 3950it [32:30,  2.02it/s, accuracy=0.867, loss=0.174] <br>
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 698/698 [05:07<00:00,  2.27it/s, accuracy=0.847, val_loss=0.403] <br>
Epoch 2/5, Train Loss: 0.3471, Val Loss: 0.4030, Train Acc: 0.8670, Val Acc: 0.8472, Train Prec: 0.8668, Val Prec: 0.8576, Train F1: 0.8664, Val F1: 0.8436 <br>
Training Epoch 3: 3950it [32:40,  2.01it/s, accuracy=0.898, loss=0.137] <br>
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 698/698 [05:07<00:00,  2.27it/s, accuracy=0.822, val_loss=0.519] <br>
Epoch 3/5, Train Loss: 0.2750, Val Loss: 0.5193, Train Acc: 0.8982, Val Acc: 0.8218, Train Prec: 0.8980, Val Prec: 0.8464, Train F1: 0.8977, Val F1: 0.8112 <br>
Training Epoch 4: 3950it [32:30,  2.03it/s, accuracy=0.879, loss=0.161] <br>
Validating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 698/698 [05:15<00:00,  2.21it/s, accuracy=0.88, val_loss=0.324] <br>
Epoch 4/5, Train Loss: 0.3215, Val Loss: 0.3237, Train Acc: 0.8795, Val Acc: 0.8799, Train Prec: 0.8791, Val Prec: 0.8842, Train F1: 0.8789, Val F1: 0.8793 <br>
Training Epoch 5: 3950it [32:23,  2.03it/s, accuracy=0.913, loss=0.119] <br>
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 698/698 [05:02<00:00,  2.31it/s, accuracy=0.911, val_loss=0.249] <br>
Epoch 5/5, Train Loss: 0.2388, Val Loss: 0.2486, Train Acc: 0.9130, Val Acc: 0.9108, Train Prec: 0.9129, Val Prec: 0.9138, Train F1: 0.9126, Val F1: 0.9100 <br>
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 228/228 [01:57<00:00,  1.95it/s, accuracy=0.844, loss=0.487] <br>
Test Loss: 0.4872, Accuracy: 0.84%, Precision: 0.86, F1 Score: 0.85 <br>

## Setting 6: Custom Cross-entropy loss with Borderline SMOTE (blsmote.py)

Training set size: 2022309 <br>
Validation set size: 356879 <br>
Test set size: 116323 <br>
Training Epoch 1: 3950it [33:20,  1.97it/s, accuracy=0.813, loss=0.235] <br>
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 698/698 [05:09<00:00,  2.26it/s, accuracy=0.782, val_loss=0.575] <br>
Epoch 1/5, Train Loss: 0.4707, Val Loss: 0.5755, Train Acc: 0.8129, Val Acc: 0.7816, Train Prec: 0.8122, Val Prec: 0.8029, Train F1: 0.8118, Val F1: 0.7673 <br>
Training Epoch 2: 3950it [32:25,  2.03it/s, accuracy=0.874, loss=0.165] <br>
Validating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 698/698 [04:58<00:00,  2.34it/s, accuracy=0.86, val_loss=0.371] <br>
Epoch 2/5, Train Loss: 0.3295, Val Loss: 0.3708, Train Acc: 0.8743, Val Acc: 0.8601, Train Prec: 0.8739, Val Prec: 0.8712, Train F1: 0.8737, Val F1: 0.8591 <br>
Training Epoch 3: 3950it [32:36,  2.02it/s, accuracy=0.878, loss=0.164] <br>
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 698/698 [05:04<00:00,  2.29it/s, accuracy=0.807, val_loss=0.493] <br>
Epoch 3/5, Train Loss: 0.3285, Val Loss: 0.4927, Train Acc: 0.8783, Val Acc: 0.8074, Train Prec: 0.8777, Val Prec: 0.8149, Train F1: 0.8777, Val F1: 0.8054 <br>
Training Epoch 4: 3950it [32:27,  2.03it/s, accuracy=0.876, loss=0.164] <br>
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 698/698 [05:06<00:00,  2.28it/s, accuracy=0.884, val_loss=0.314] <br>
Epoch 4/5, Train Loss: 0.3280, Val Loss: 0.3144, Train Acc: 0.8760, Val Acc: 0.8843, Train Prec: 0.8755, Val Prec: 0.8921, Train F1: 0.8755, Val F1: 0.8807 <br>
Training Epoch 5: 3950it [32:22,  2.03it/s, accuracy=0.901, loss=0.138] <br>
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 698/698 [05:05<00:00,  2.29it/s, accuracy=0.873, val_loss=0.342] <br>
Epoch 5/5, Train Loss: 0.2753, Val Loss: 0.3420, Train Acc: 0.9014, Val Acc: 0.8731, Train Prec: 0.9009, Val Prec: 0.8747, Train F1: 0.9009, Val F1: 0.8726 <br>
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 228/228 [01:54<00:00,  1.99it/s, accuracy=0.834, loss=0.455] <br>
Test Loss: 0.4552, Accuracy: 0.83%, Precision: 0.85, F1 Score: 0.84 <br>
