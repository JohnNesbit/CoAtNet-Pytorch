# CoAtNet-Pytorch
Pytorch Implementation of CoAtNet. the SOTA image classification model described in https://arxiv.org/abs/2106.04803.

This implementation allows for the easy swapping of CoatNet versions and has clear, abundant comments. As well as having a complete version of coatnet here, this repository also
includes testing of that version on CIFAR-100. The testing uses the data augmentations perscribed by the coatnet paper of label smoothing, data mixup and RandAugment.

The translation equivariant weight is produced within the code by taking advantage of matrix multiplication's inherent order preserving nature. 

![image](https://user-images.githubusercontent.com/49009243/147174385-94829ab8-38e7-4c22-8ab4-48d4971a7d1d.png)

The equation pictured above for finding the index of the respective equivariant weight for an index of the K\*Q number being i-j, the entirety of the indexing for this problem can 
be solved with a diagonally similar matrix of equivarant weight index. For example, if the equivarant weight was (1, 2, 3, 4, 5, 6, 7), the respective additive for K\*Q would be:

![image](https://user-images.githubusercontent.com/49009243/147174530-8c5a4a89-3341-494b-9884-12c5ed6cc1dd.png)

Proving this simplification is, in fact, identical would be a bit labourious for a github readme, so I will leave you to your own mathematical devices to trust myself or to prove
this is not the case yourselves.
