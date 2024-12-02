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