# Deep embedding kernel

### Main libraries:
- Python 2.7
- theano
- numpy
- matplotlib
- pandas

*** Update 4/28/2019 -- my paper has been published at https://www.sciencedirect.com/science/article/abs/pii/S0925231219302589

### Summary

This is the sample code and benchmark for my model Deep Embedding Kernel (DEK). DEK combines the advantages of deep learning and kernel methods in a unifed framework. More specifically, DEK is a learnable kernel represented by a newly designed deep architecture. Compared with predefined kernels, this kernel can be explicitly trained to map data to an optimized high-level feature space where data may have favorable features toward the application. Compared with typical deep learning using SoftMax or
logistic regression as the top layer, DEK is expected to be more generalizable to new data. Experimental results show that DEK has superior performance than typical machine learning methods in identity detection and classification, and transfer learning, on different types of data including images, sequences, and regularly structured data.

### The difference between the mappings of DEK and regular kernel functions:

![image](https://user-images.githubusercontent.com/5643444/231931016-d7655ee6-2aeb-4bb3-a7fd-5221ce8023b6.png)

### The architecture of DEK
![image](https://user-images.githubusercontent.com/5643444/231931032-456b61a5-bc01-4b81-9610-a40b8e046fe5.png)


