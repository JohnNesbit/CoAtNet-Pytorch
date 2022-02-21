# CoAtNet-Pytorch
Pytorch Implementation of CoAtNet. the SOTA image classification model described in https://arxiv.org/abs/2106.04803.

CoatNet.py contains the coatnet class and accompanying initialization functions for each size of coatnet. 

## translation equivariant weight
The translation equivariant weight is produced within the code by taking advantage of matrix multiplication's inherent order preserving nature. 

![image](https://user-images.githubusercontent.com/49009243/147174385-94829ab8-38e7-4c22-8ab4-48d4971a7d1d.png)

With the equation pictured above for finding the index of the respective equivariant weight for an index of the K\*Q number being i-j, the entirety of the indexing for this 
problem can be solved with a diagonally similar matrix of equivarant weight index. For example, if the equivarant weight was (1, 2, 3, 4, 5, 6, 7), the respective additive for 
K\*Q would be:

![image](https://user-images.githubusercontent.com/49009243/147421614-e8895f1f-970d-4d5a-8abb-56df50237943.png)


This result may seem odd, but really is not when one looks at the respective index for itself:

![image](https://user-images.githubusercontent.com/49009243/147421604-1f16a950-13a5-4302-8d45-0a93d6fe29c1.png)
