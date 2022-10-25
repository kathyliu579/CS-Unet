# CS-Unet
The codes for the work "Optimizing Vision Transformers for Medical Image Segmentation"(https://arxiv.org/...).


## 1. Prepare data

- The datasets we used are provided by TransUnet's authors. Please go to ["./datasets/README.md"](datasets/README.md) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License).

## 2. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 3. Train/Test

- Run the train script on synapse dataset. The batch size we used is 24.

- Train

```bash
sh my_train.sh 
```

- Test 

```bash
sh my_test.sh 
```

## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
* [SwinUnet]https://github.com/HuCaoFighting/Swin-Unet

## Citation

```bibtex
....
}
```
# CS-Unet
