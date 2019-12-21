import shapefile
import geopandas

file_path = '/media/luoyingsong/新加卷/GPS数据/fuzhou/Road/Road_polyline.shp'
shp_df = geopandas.GeoDataFrame.from_file(file_path, encoding='gb18030')
shp_df.head()  # 获取表头
shp_df.plot()

print(shp_df.head())