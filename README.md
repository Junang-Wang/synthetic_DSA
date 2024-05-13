# synthetic_DSA

Digital subtraction angiography (DSA) is a fluoroscopy method primarily used for the diagnosis of cardiovascular diseases.
Deep learning-based DSA (DDSA) is developed to extract DSA-like images directly from fluoroscopic images, which helps in saving dose while improving image quality.
In this work, we developed a method generating synthetic DSA image pairs to train neural network models for DSA image extraction with CT images and simulated vessels.
The synthetic DSA targets are free of typical artifacts and noise commonly found in conventional DSA targets for DDSA model training.
Benefiting from diverse synthetic training data and accurate synthetic DSA targets, models trained on the synthetic data outperform models trained on clinical data in both visual and quantitative assessments.
This approach compensates for the paucity and inadequacy of clinical DSA data.

For more information about this work, please read the [Medical Physics 2024 paper](http://doi.org/10.1002/mp.16973)

> Duan, L., Eulig, E., Knaup, M., Adamus, R., Lell, M., & Kachelrieß, M. "Training of a Deep Learning Based Digital Subtraction Angiography Method using Synthetic Data."


## Requirements
```
python
pytorch >= 2.0
gVirtualXray >=2.0.7
3D-slicer (installed in Ubuntu)
```

## Data

We use Lindenmayer system(L-system) to generate volume pixels(voxel) data of neurovascular structure. Then we employ the 3D-slicer software to convert voxel data into STL format mesh data. Finally, we leverage the gVirtualXray package to projects the mesh data into 2D Xray images. 

 



## Getting started with training
### Train model using synthetic data
coming soooooon...


## Reference
```
@article{duan2024-deepdsa,
  author = {Duan, Lizhen and Eulig, Elias and Knaup, Michael and Adamus, Ralf and Lell, Michael and Kachelrieß, Marc},
  title = {Training of a deep learning based digital subtraction angiography method using synthetic data},
  journal = {Medical Physics},
  volume = {n/a},
  number = {n/a},
  pages = {},
  doi = {https://doi.org/10.1002/mp.16973},
  url = {https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.16973},
  eprint = {https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/mp.16973},
}

@inproceedings{gkioxari2019mesh,
  title={Mesh r-cnn},
  author={Gkioxari, Georgia and Malik, Jitendra and Johnson, Justin},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={9785--9795},
  year={2019}
}
```
