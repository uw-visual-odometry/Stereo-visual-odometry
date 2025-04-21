import cv2
import numpy as np
import os
import open3d as o3d

def saveDebugImg(imgIn, frmId, tag, points, color=None, postTag=''):
    if not os.path.exists('debugImgs'):
        os.makedirs('debugImgs')

    imgD = imgIn.copy()
    if isinstance(points, list) or isinstance(points, tuple):
        imgD = cv2.drawKeypoints(imgIn, points, imgD, color=color)
    else:
        imgD = cv2.cvtColor(imgD,cv2.COLOR_GRAY2RGB)
        for point in points:
            cv2.circle(imgD, (point[0], point[1]), 2, color=color)

    outFileName = 'debugImgs/' + tag + '_' + str(frmId)
    if postTag != '':
        outFileName = outFileName + '_' + postTag

    outFileName = outFileName + '.png'
    cv2.imwrite(outFileName, imgD)
    return


def visualize_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])

    vis.run()
    vis.destroy_window()

def save_point_cloud(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")


def remap(frame_left, frame_right):

    # Camera parameters to undistort and rectify images
    cv_file = cv2.FileStorage()
    cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

    # Undistort and rectify images
    frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    height, width = frame_left.shape[:2]

    start_y = (height - 900) // 2
    start_x = (width - 1200) // 2

    frame_left = frame_left[start_y:start_y + 900, start_x:start_x + 1200]
    frame_right = frame_right[start_y:start_y + 900, start_x:start_x + 1200]

    return frame_left, frame_right

