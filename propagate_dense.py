import argparse
import laspy
import numpy as np
from scipy.spatial import cKDTree

def main():
    parser = argparse.ArgumentParser(description="Propagate voxelized TreeLearn predictions to a dense point cloud.")
    parser.add_argument("dense_path", help="Path to the original dense LAS/LAZ file (input_a)")
    parser.add_argument("vox_path", help="Path to the voxelized TreeLearn output LAZ file (input_b)")
    parser.add_argument("out_path", help="Path to save the labeled dense LAZ file (output)")
    
    args = parser.parse_args()

    print(f"Loading dense cloud: {args.dense_path}")
    dense_las = laspy.read(args.dense_path)
    
    print(f"Loading voxelized cloud: {args.vox_path}")
    vox_las = laspy.read(args.vox_path)

    dense_coords = np.vstack((dense_las.x, dense_las.y, dense_las.z)).T
    vox_coords = np.vstack((vox_las.x, vox_las.y, vox_las.z)).T

    print("Building KD-Tree and mapping labels back to dense points (this may take a moment)...")
    tree = cKDTree(vox_coords)
    _, indices = tree.query(dense_coords, k=1)

    # Extract the exact treeID field from the voxelized cloud
    vox_labels = np.array(getattr(vox_las, 'treeID'))

    # Unconditionally add the treeID dimension to the dense cloud
    dense_las.add_extra_dim(laspy.ExtraBytesParams(name='treeID', type=np.int32))

    # Apply the nearest neighbor labels
    dense_las.treeID = vox_labels[indices]
    
    print(f"Saving dense labeled cloud to: {args.out_path}")
    dense_las.write(args.out_path)
    print("Done!")

if __name__ == "__main__":
    main()