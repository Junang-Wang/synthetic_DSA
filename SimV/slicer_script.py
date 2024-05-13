import slicer, sys 
# import slicer.util.getNode as getNode
print('import slicer')

def main(volume_path, stl_folder, stl_name):

    # # Load segmentation from .seg.nrrd file (includes segment names and colors)
    sourseVolumeNode = slicer.util.loadVolume(volume_path)
    # sourseVolumeNode = slicer.util.loadVolume("/home/rslsync/Qubot/Codes/synthetic-ddsa/synthetic-ddsa/vessels/volumes/Lnet_d35_dr20_epsilon10_iter8_SD500_v1_t0_512x512x280_nofluid.tiff")



    # import SampleData
    # # sampleDataLogic = SampleData.SampleDataLogic()
    # # masterVolumeNode = sampleDataLogic.downloadMRHead()

    # # Create segmentation
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentationNode.CreateDefaultDisplayNodes() # only needed for display
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(sourseVolumeNode)
    addedSegmentID = segmentationNode.GetSegmentation().AddEmptySegment(stl_name)

    # # Create segment editor to get access to effects
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setSourceVolumeNode(sourseVolumeNode)

    # # Thresholding
    segmentEditorWidget.setActiveEffectByName("Threshold")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("MinimumThreshold","10")
    effect.setParameter("MaximumThreshold","255")
    effect.self().onApply()

    # # Smoothing
    # segmentEditorWidget.setActiveEffectByName("Smoothing")
    # effect = segmentEditorWidget.activeEffect()
    # effect.setParameter("SmoothingMethod", "MEDIAN")
    # effect.setParameter("KernelSizeMm", 3)
    # effect.self().onApply()

    # # Clean up
    segmentEditorWidget = None
    slicer.mrmlScene.RemoveNode(segmentEditorNode)

    # # Make segmentation results visible in 3D
    segmentationNode.CreateClosedSurfaceRepresentation()

    # Write to STL file
    slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsClosedSurfaceRepresentationToFiles(stl_folder, segmentationNode, None, "STL")

    sys.exit()

if __name__ == "__main__":
    volume_path = sys.argv[1]
    stl_folder = sys.argv[2]
    stl_name = sys.argv[3]
    main(volume_path, stl_folder, stl_name)

