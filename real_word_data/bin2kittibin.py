import numpy as np
import os

def find_global_max_intensity(directory):
    max_intensity = 0
    for filename in os.listdir(directory):
        if filename.endswith(".bin"):
            bin_file = os.path.join(directory, filename)
            points = np.fromfile(bin_file, dtype=np.float32)
            points = points.reshape(-1, 4)
            max_intensity = max(max_intensity, np.max(points[:, 3]))
    return max_intensity

def load_and_normalize_ouster_bin(bin_file, global_max_intensity):
    points = np.fromfile(bin_file, dtype=np.float32)
    points = points.reshape(-1, 4)  # Assuming each point has 4 values: x, y, z, intensity
    if global_max_intensity != 0:
        points[:, 3] = points[:, 3] / global_max_intensity  # Normalize intensity to 0-1
    return points

def save_normalized_bin(points, output_file):
    points.tofile(output_file)

if __name__ == "__main__":
    input_directory = '/home/auto/9338bin/'  # 修改为你的 bin 文件目录
    output_directory = '/home/auto/9338binnew/'  # 修改为输出的标准化 bin 文件目录

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Step 1: Find the global maximum intensity value
    global_max_intensity = find_global_max_intensity(input_directory)
    print(f"Global max intensity: {global_max_intensity}")

    # Step 2: Normalize all bin files based on the global maximum intensity
    for filename in os.listdir(input_directory):
        if filename.endswith(".bin"):
            input_file = os.path.join(input_directory, filename)
            normalized_points = load_and_normalize_ouster_bin(input_file, global_max_intensity)
            output_file = os.path.join(output_directory, filename)
            save_normalized_bin(normalized_points, output_file)
            print(f"Normalized file saved: {output_file}")

