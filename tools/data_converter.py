import argparse
import numpy as np
from scipy.spatial import ConvexHull
import json
import os

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.
    tmp=ConvexHull(points).vertices
    hull_points=[]
    # get the convex hull for the points
    for pt in tmp:
        hull_points.append(points[int(pt)])
    hull_points=np.array(hull_points,dtype=np.float32)
    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return np.array(rval,dtype=np.int32)

def collinear(points):
    points.sort()
    if points[0][0]==points[1][0]==points[2][0] or points[0][1]==points[1][1]==points[2][1]:
        return 1
    return 0

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert JSON annotations to ICDAR format'
    )
    parser.add_argument('-label_root','--label_root', help='label root')
    parser.add_argument('-label_des','--label_des', help='label destination')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    path=args.label_root
    destination=args.label_des
    files=os.listdir(path)

    
    for file in files:
        outfile = open(destination+'/'+file.replace('.json','.txt'), "w", encoding="utf-8")
        f=open(path+'/'+file)
        data=json.load(f)
        for dic in data:
            if(collinear(dic['points'])==1):
                continue
            hull=minimum_bounding_rectangle(dic['points'])
            txt=dic['text']
            if txt=='###' or len(txt.split(' '))!=1:
                continue
            for cmp in hull:
                outfile.write(str(cmp[0])+','+str(cmp[1])+',')
            outfile.write(dic['text']+'\n')