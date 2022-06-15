#!/usr/bin/ python3
import open3d as o3d
import time


if __name__ == "__main__":
  pc = o3d.geometry.PointCloud()

  pcd = o3d.io.read_point_cloud("fragment.pcd")
