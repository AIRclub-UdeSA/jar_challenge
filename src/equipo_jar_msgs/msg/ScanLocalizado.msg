# A scan the localizer matched (docs/WORLD_MODEL.md §3): every valid return, deskewed to
# the scan stamp and placed in the map frame at the matched pose.
std_msgs/Header header           # the scan's stamp; frame: map
geometry_msgs/Point sensor       # the lidar's position at the stamp; z = 0
float32 sigma                    # m, position uncertainty of the pose (largest axis)
float32[] x                      # m, map frame
float32[] y
