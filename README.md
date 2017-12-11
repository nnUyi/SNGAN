# SNGAN
  An implement of spectral normalization GAN for tensorflow version, spectral normalization is used in this repo to give some constrants to the weights of discriminator

# Requirements

  - tensorflow 1.3.0

  - python2.7 or python3.6

  - numpy 1.13.*

  - scipy 0.17.0~ 0.19.1
 
# Usages

  (1)download this repo to your own directory
  
    $ git clone https://github.com/nnUyi/SNGAN.git
    $ cd SNGAN
    
  (2)download celebA dataset and store it in the data directory(directory named **data**)
      
   - celebA datasets is cropping into **64*64 size** with *.png or *jpg format, this repo read image format data as input.
      
  (3)training
    
    $ python main_sngan.py --is_training=True
  
  (4)testing
    
   - Anyway, in this repo, testing processing is taken during training, It samples the training results to the sample(named sample) directory, and stores session in the checkpoint(named checkpoint) directory.

# Results

|sample image|sample image|
|:-----------------:|:----------------:|
|![Alt test](/data/train_1.png)|![Alt test](/data/train_2.png)|
|64*64 resolution|64*64 rosolution||

# Contacts

  Email:computerscienceyyz@163.com
