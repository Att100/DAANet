import cv2 
import numpy as np 
from PIL import Image
import argparse
import os
from tqdm import tqdm

def label2boundary(path, size):
    img = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    if size is not None:
        img = cv2.resize(img, (size, size))

    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)      
    absY = cv2.convertScaleAbs(y)   

    prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    edge_arr = np.array(prewitt)
    edge_arr[edge_arr >= 123] = 255
    return edge_arr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--label_path', type=str, default='./dataset/Gts', 
        help="path of label (default: ./dataset/Gts)")
    parser.add_argument(
        '--output_path', type=str, default='./dataset/Egs', 
        help="path of output edges image (default: ./dataset/Egs)")
    parser.add_argument(
        '--size', type=str, default='org', 
        help="size of output edges image (default: org)")


    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    size = int(args.size) if args.size != 'org' else None
    for item in tqdm(os.listdir(args.label_path), desc='Processing'):
        path = os.path.join(args.label_path, item)
        edge_arr = label2boundary(path, size)
        Image.fromarray(edge_arr).save(os.path.join(args.output_path, item))