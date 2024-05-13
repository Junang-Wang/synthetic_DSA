# DDSA using Synthetic Training Data

This is the code for generate synthetic vascular projection images

Part of the code is from
**Paper:** Training of a Deep Learning Based Digital Subtraction Angiography Method using Synthetic Data  
**Author:** Duan, Lizhen; Eulig, Elias; Knaup, Michael; Adamus, Ralf; Lell, Michael; Kachelrieß, Marc

## Contents
```
├─ vsystem/                 Folder containing functions for generate vessels with stochastic L-system
├─ bolus/                   Folder containing functions for bolus injection simulation
├─ syntheticDSA.py          Generate and save vessel strings, and generate vascular mesh with simulated bolus injections using  given vessel strings
├─ json2gvxr.py             Generate projections using gVirtualXray.
├─ configuration-03.json    configuration of Xray and detector.
├─ slicer_script.py         3D slicer python script tp convert voxel data to STL format mesh data.
```

## How to use
See example batchDataGenerator.py



## Acknowledgements
- We would like to acknowledge the authors and contributors of the code that we referenced from the GitHub repository [https://github.com/psweens/V-System].
