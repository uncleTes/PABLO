import vtk

points = vtk.vtkPoints()
points.InsertNextPoint(0, 0, 0)
points.InsertNextPoint(1, 0, 0)
points.InsertNextPoint(0, 1, 0)
points.InsertNextPoint(1, 1, 0)
points.InsertNextPoint(0, 2, 0)
points.InsertNextPoint(1, 2, 0)

sg = vtk.vtkStructuredGrid()
sg.SetDimensions(2, 3, 1)
sg.SetPoints(points)

writer = vtk.vtkXMLStructuredGridWriter()
writer.SetFileName("structured_grid.vts")
writer.SetInputData(sg)
writer.Write()


