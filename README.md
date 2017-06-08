# cycleGAN
Coding and learning, an implementation of cycleGAN.

You can find the original paper on:
https://junyanz.github.io/CycleGAN/


## TensorFlow version
tensorflow 1.1.0


## Results
### Test Results
|      input ->  output                  |       input ->   output                       |
|----------------------------------------|-----------------------------------------------|
|          iteration 1000                |                iteration 30000                |
|  ![](test_results/iteration1000.jpg)   |    ![](test_results/iteration30000.jpg)       | 
|          iteration 50000               |                iteration 70000                |
|  ![](test_results/iteration50000.jpg)  |      ![](test_results/iteration70000.jpg)     | 

### Training Results
|  orange ->  apple ->  reconstruction   |     orange ->   apple  ->   reconstruction    |
|----------------------------------------|-----------------------------------------------|
|![](train_results/individualImage2.png) |    ![](train_results/individualImage3.png)    | 




## Files structure
* `main.py`   run this file to train model
* `ops.py`    contains functions which will be used to build model 
* `utils.py`  contains a class which will be used to prepare data


## Notes:
* Different size of input requires different structure of model. input[128x128]: 6 blocks; input[256x256]: 9 blocks.
* Residual block in this paper is a little different than the original one. The small difference is that in original paper the author removes the last relu layer in residual block. 
* In original paper, the author uses regular batch normalization, however, it is recommended to use instance normalization.
* Least square loss was used, and then we need to remove the last sigmoid activation function in discriminator. It turns out to perform better and more stable. In Pix2Pix project, sigmoid layer is required for discriminator.
* Results is sensitive to initialization. Sometimes color of background and object are reversed. You can rerun code to avoid it.
* Residual block is used to ensure properties of previous input layer are available for layer layers. So outputs will not deviate much from original input.
