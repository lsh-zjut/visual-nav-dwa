## Usage

### 1. Gazebo World

This command will launch the gazebo with the project world

```bash
# Launch Gazebo World together with our robot
roslaunch me5413_world world.launch
```

### 2. Navigation

This command will launch the gazebo with the project world

```bash
# Launch Gazebo World together with our robot
roslaunch me5413_world navigation.launch
```

### 3. Modification

现在Navigation.launch里直接用me5413_wolrd里的amcl.launch和move_base.launch文件

# amcl.launch
主要修改了如下部分，解决定位偏移问题：
    <param name="gui_publish_rate" value="20.0"/>
    <param name="laser_max_beams" value="2000"/>
    <param name="laser_min_range" value="-1"/>
    <param name="laser_max_range" value="-1"/>
    <param name="min_particles" value="2000"/>
    <param name="max_particles" value="5000"/>
    <param name="update_min_a" value="0.2"/> 

# move_base.launch
现在直接使用me5413_world/params里的参数
全局costmap参数在costmap_common_params.yaml里面，参数详解：https://blog.csdn.net/weixin_48162479/article/details/118255948

主要修改了膨胀半径（必须大于车的半径）：
inflater_layer:
 inflation_radius: 0.35 

这里最好要根据车的大小来改（不知道怎么看车的大小，还没改）
footprint: [[-0.21, -0.165], [-0.21, 0.165], [0.21, 0.165], [0.21, -0.165]]
footprint_padding: 0.1

另外要换掉local和global planner
    <param name="base_global_planner" type="string" value="navfn/NavfnROS" />
    <param name="base_local_planner" value="base_local_planner/TrajectoryPlannerROS"/>

