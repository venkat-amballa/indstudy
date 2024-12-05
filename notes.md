baseline gave 84% validation accuracy

lr range test
using class wieghts in cross_entropy
augmenatations:
- img to hsv (epochs 20, validation acc: 65%)
- img to lab (epochs 20, validation acc: 65%)
- different augmentations

use resnet50, inception_resenet_v2
1. Train siamese network on
-  constrastive loss
-  dualms loss
2. Using SamplePairing (ImageFusion) to generate more images for under-represented 
    classes by combining two images from same class
3. Run the code with just transformations


Notes:
1: We need to use large batches while using contrastive loss
2. use LARS, to accomodate for loss of accuracy and the stability of learning rate
3. Requires strong augmentation

              precision    recall  f1-score   support

           0       0.74      0.67      0.70        21
           1       0.94      0.96      0.95       123
           2       0.88      0.93      0.90        15
           3       0.40      0.25      0.31         8
           4       0.70      0.73      0.71        22
           5       0.00      0.00      0.00         1
           6       1.00      1.00      1.00         3

    accuracy                           0.87       193
   macro avg       0.66      0.65      0.65       193
weighted avg       0.86      0.87      0.86       193

Time taken: 23.08 minutes
Validation F1 Score: 0.8598, Validation Accuracy: 0.8653
[I 2024-12-05 07:48:37,967] Trial 1 finished with value: 0.85975483552212 and parameters: {'learning_rate': 8.774912414129264e-05, 'weight_decay': 7.115097287996221e-06, 'temperature': 0.08790493174675384, 'model_type': 'efficientnet_b0'}. Best is trial 1 with value: 0.85975483552212.