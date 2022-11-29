# Run this with paraview python 5.9.1. You may need to adjust the file paths.

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


################################
#          input               #
################################
directories = [
    # variation in mesh size
    "results_air-size=0.8_mesh-size=0.5_winding-d=0.01",
    "results_air-size=0.8_mesh-size=0.25_winding-d=0.01",
    "results_air-size=0.8_mesh-size=1_winding-d=0.01",
    # variation in air size
    "results_air-size=0.4_mesh-size=1_winding-d=0.01",
    "results_air-size=1.6_mesh-size=1_winding-d=0.01",
]

################################
for directory in directories:

    # create a new 'Xdmf3ReaderT'
    solutionxdmf = Xdmf3ReaderT(registrationName='solution.xdmf', FileName=[f'./{directory}/solution.xdmf'])
    solutionxdmf.PointArrays = ['imag_A', 'real_A']
    solutionxdmf.CellArrays = ['imag_varsigma', 'real_varsigma']

    UpdatePipeline(time=0.0, proxy=solutionxdmf)

    # create a new 'Extract Block'
    extractBlock1 = ExtractBlock(registrationName='ExtractBlock1', Input=solutionxdmf)

    # Properties modified on extractBlock1
    extractBlock1.BlockIndices = [1]

    UpdatePipeline(time=0.0, proxy=extractBlock1)

    # create a new 'Plot Over Line'
    plotOverLine1 = PlotOverLine(registrationName='PlotOverLine1', Input=extractBlock1,
        Source='Line')

    # init the 'Line' selected for 'Source'
    plotOverLine1.Source.Point2 = [0.06, 0.0, 0.0]

    UpdatePipeline(time=0.0, proxy=plotOverLine1)

    # save data
    SaveData(f'./{directory}/line-data.csv', proxy=plotOverLine1, PointDataArrays=['arc_length', 'imag_A', 'real_A', 'vtkValidPointMask'],
        Precision=10,
        UseScientificNotation=1)
