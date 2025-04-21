import numpy as np
import time
import os
import pickle


class PointCloudRecorder:
    def __init__(self, save_dir="./point_cloud_data", save_interval=10):

        self.points = []
        self.colors = []
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.last_save_time = time.time()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def add_point(self, point, color=None):
        self.points.append(point)
        if color is not None:
            self.colors.append(color)

    def save_point_cloud(self, filename=None):
        if not self.points:
            return

        if filename is None:
            filename = f"point_cloud_{int(time.time())}.pkl"

        data = {
            "points": np.array(self.points),
        }

        if self.colors and len(self.colors) == len(self.points):
            data["colors"] = np.array(self.colors)

        with open(os.path.join(self.save_dir, filename), 'wb') as f:
            pickle.dump(data, f)

        print(f"Save {len(self.points)} points to {filename}")

    def close(self):
        self.save_point_cloud("final_point_cloud.pkl")


# 示例用法
if __name__ == "__main__":
    recorder = PointCloudRecorder(save_interval=5)  # 每5秒保存一次

    try:
        # 模拟流式接收点数据
        for i in range(100):
            # 这里替换为您实际接收点的逻辑
            point = [np.random.random(), np.random.random(), np.random.random()]
            color = [np.random.random(), np.random.random(), np.random.random()]

            recorder.add_point(point, color)

            # 模拟点的间隔到达
            time.sleep(0.5)  # 每0.5秒一个点

    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        recorder.close()
        print("记录器已关闭")