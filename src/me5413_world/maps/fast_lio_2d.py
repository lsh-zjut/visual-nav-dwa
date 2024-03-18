import pcl
import numpy as np
import cv2

# 加载点云
cloud = pcl.load('scans.pcd')
# 创建一个PassThrough过滤器
passthrough = cloud.make_passthrough_filter()

# 设置过滤轴为Z轴
passthrough.set_filter_field_name('z')

# 设置过滤范围，只保留Z坐标大于0.2的点
passthrough.set_filter_limits(1, np.inf)

# 应用过滤器
cloud_filtered = passthrough.filter()


# 算出最大最小值，算的太慢了，直接用下面的值
# min_x, min_y, min_z = cloud[0][0], cloud[0][1], cloud[0][2]
# max_x, max_y, max_z = cloud[0][0], cloud[0][1], cloud[0][2]
# for i in range(cloud.size):
#     min_x = min(min_x, cloud[i][0])
#     min_y = min(min_y, cloud[i][1])
#     min_z = min(min_z, cloud[i][2])
#     max_x = max(max_x, cloud[i][0])
#     max_y = max(max_y, cloud[i][1])
#     max_z = max(max_z, cloud[i][2])

max_x =19.079988479614258; min_x = -1.6874970197677612
max_y = 5.678721904754639; min_y = -15.588150978088379
max_z = 5.464957237243652; min_z = -0.32512667775154114

# 设置网格大小
grid_size=0.01


def convert_to_grid_coordinates(x, y, min_x, max_x, min_y, max_y, grid_size):
    # 计算点在网格中的位置
    grid_x = int(np.floor((x - min_x) / grid_size))
    grid_y = int(np.floor((y - min_y) / grid_size))
    return grid_x, grid_y


# 将点云转换为2D
occupancy_map = np.full((int((max_x - min_x) / grid_size +1), int((max_y - min_y) / grid_size +1)), 255)
print ("map size: " + str(occupancy_map.shape))

for point in pcl.PointCloud(cloud_filtered).to_array():
    # 将3D点转换为2D
    x, y = point[0], point[1]
    # 将2D点转换为占用图的网格坐标
    grid_x, grid_y = convert_to_grid_coordinates(x, y, min_x, max_x, min_y, max_y, grid_size)
    # 标记占用的格子
    occupancy_map[grid_x][grid_y] = 0


#应用形态学操作来去除小的孤立点
occupancy_map = occupancy_map.astype(np.uint8)
kernel = np.ones((3,3),np.uint8)
occupancy_map_morphed = cv2.morphologyEx(occupancy_map, cv2.MORPH_OPEN, kernel)

#倾斜矫正,使用霍夫变换检测直线
edges = cv2.Canny(occupancy_map_morphed, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
angles = []
for rho, theta in lines[0]:
    angle = theta * 180 / np.pi
    angles.append(angle)
average_angle = np.mean(angles)
rotation_angle = 90 - average_angle
height, width = occupancy_map_morphed.shape
M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
occupancy_map_corrected = cv2.warpAffine(occupancy_map_morphed, M, (width, height), borderValue=255)


# 保存占用图
cv2.imwrite('occupancy_map.png', occupancy_map_corrected)
print("successfully saved")

