# test2-30.mp4 start_frame=300

# (currently not useful) sequence from which the images would be selected from inside the sequences folder in the kitti dataset
sequence=1
start_frame=571
end_frame=1170
use_sift=1
use_ransac=1
live_plot=0

python SVO_underwater.py ${sequence} ${start_frame} ${end_frame} ${use_sift} ${use_ransac} ${live_plot}


# test2-30.mp4 start_frame=300

# (currently not useful) sequence from which the images would be selected from inside the sequences folder in the kitti dataset
#sequence=1
#start_frame=300
##start_frame=920
#end_frame=2200
#use_sift=1
#use_ransac=1
#live_plot=0

#python SVO_custom.py ${sequence} ${start_frame} ${end_frame} ${use_sift} ${use_ransac} ${live_plot}