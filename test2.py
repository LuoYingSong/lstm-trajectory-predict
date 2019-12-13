import shapefile
import numpy as np


sf = shapefile.Reader('/home/luoyingsong/Desktop/planet_119.239,26.043_119.409,26.135-shp/shape/roads.shp')
total_lat = []
total_long = []
count = 0
for shape_attribute in sf.shapes():
    lat = np.array(shape_attribute.points)[:,1]
    print(list(lat))
    long = np.array(shape_attribute.points)[:,0]
    print(list(long))
    total_lat += list(lat)
    total_long += list(long)
    count += 1
print(count)
print(len(total_lat))
print(total_long)
print()