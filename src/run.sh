# test2-30.mp4 start_frame=300

# (currently not useful) sequence from which the images would be selected from inside the sequences folder in the kitti dataset
sequence=1
start_frame=300
end_frame=2200
use_sift=True
use_ransac=True
live_plot=True

python SVO_custom.py ${sequence} ${start_frame} ${end_frame} ${use_sift} ${use_ransac} ${live_plot}