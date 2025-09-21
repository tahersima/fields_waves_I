#Mohammad H. Tahersima / All Rights Reserved 
#Example code to perform basic vector operations 

import numpy as np 

E1 = np.array([2,3,1])
E2 = np.array([1,4,3])
print ('E1 + E2 = ', np.add(E1 , E2))
print ('E1 - E2 = ', np.subtract(E1 , E2))
print ('E1 . E2 = ', np.dot(E1 , E2))
print ('E1 X E2 = ', np.cross(E1 , E2))
print ('|E1 X E2| = ', np.linalg.norm(np.cross(E1 , E2)))
unit_E1 = E1 / np.linalg.norm(E1)
unit_E2 = E2 / np.linalg.norm(E2)
dot_product = np.dot(unit_E1, unit_E2)
angle = np.arccos(dot_product)
print ('Angle between E1 and E2 = ', np.rad2deg(angle))
