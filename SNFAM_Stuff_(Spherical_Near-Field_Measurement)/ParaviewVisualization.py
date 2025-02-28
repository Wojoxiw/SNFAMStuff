'''
Created on 26 feb. 2025

@author: al8032pa
'''

import pyvtk as vtk
import numpy as np
from scipy.constants import pi

def ExportDataForParaview(data):
    '''
    Exports the S21 data (near-field for now, maybe also far-field) into a .vtk file for visualization using Paraview.
    Uses absolute values of each polarization. Currently can just be viewed with TableToPoints, which doesn't look very good...
    :param data: The SNFData
    '''
    print('Exporting', data.name, 'to .vtk... ( shape:',np.shape(data.theta_sphere),')')
    
    
    #===========================================================================
    # ## Try pyvtk (couldnt get this to work, not sure if possible to put this in a structured grid)
    # thetas, phis = data.theta_sphere, data.phi_sphere
    # 
    #  
    # SsPol0 = np.abs(data.S21_sphere[0, 0]) ## [0] to select the 1st freq. point (should be 9.35 GHz on newer datasets)
    # SsPol90 = np.abs(data.S21_sphere[0, 1])
    #     
    # 
    # thetas = vtk.PointData(vtk.Scalars(thetas.flatten(), name = 'theta'))
    # phis = vtk.PointData(vtk.Scalars(phis.flatten(), name = 'phi'))
    #  
    # SsPol0 = vtk.PointData(vtk.Scalars(SsPol0.flatten(), name = 'Spol0'))
    # SsPol90 = vtk.PointData(vtk.Scalars(SsPol90.flatten(), name = 'Spol90'))
    #  
    # vtkData = vtk.VtkData(vtk.StructuredPoints([226, 450]), thetas, phis, SsPol0, SsPol90)
    # vtkData.tofile('vtks/test','ascii')
    #===========================================================================
    
    ## text txt and table import
    SsPol0 = np.abs(data.S21_sphere[0, 0]).flatten() ## [0] to select the 1st freq. point (should be 9.35 GHz on newer datasets)
    SsPol90 = np.abs(data.S21_sphere[0, 1]).flatten()
    thetas = data.theta_sphere.flatten()
    phis = data.phi_sphere.flatten()
    xs = np.sin(pi/180*thetas)*np.cos(pi/180*phis)
    ys = np.sin(pi/180*thetas)*np.sin(pi/180*phis)
    zs = np.cos(pi/180*thetas)
    
    dataFlat = np.transpose(np.vstack((xs, ys, zs, SsPol0, SsPol90)))
    header = 'x, y, z, pol0, pol90'
    np.savetxt('vtks/test.csv', dataFlat, header = header, comments='', delimiter=',') ## this can be imported into paraview, and visualized with TableToPoints
    
    print('Exported')