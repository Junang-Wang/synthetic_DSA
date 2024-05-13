from SimV.syntheticDSA import syntheticDSA
import numpy as np
import os

def main():
    # initial diameter of vascular
    d0_array = np.arange(25,30,0.025) 
    # number of iteration
    niter_array = np.array([8,9])
    # proportion between length & diameter
    epsilon_array = np.array([8,9])
    # ration between d0 to its subbranch
    d_array = np.array([15,20])

    # voxel size
    tissueVolume = [512, 512, 395]
    # number of projections
    nProj = 4

    string_folder = os.path.expanduser("~/syntheticDSA/string")
    stl_folder = os.path.expanduser("~/syntheticDSA/stl")
    images_folder = os.path.expanduser("~/syntheticDSA/images/proj_"+str(nProj).zfill(2))
    XrayConf = "./SimV/configuration-03.json"
    DSA_synth = syntheticDSA(d0_array, niter_array, epsilon_array, d_array, tissueVolume)
    # DSA_synth.string2stl(string_folder, stl_folder)
    DSA_synth.stl2images(nProj, XrayConf, stl_folder, images_folder)

if __name__ == "__main__":
    main()