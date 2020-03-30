import cv2
import numpy as np
import torchvision
import torch


def segment(model, x):
    model = model.eval()
    x = cv2.resize(x, (512, 1024))
    x = torchvision.transforms.ToTensor()(x).unsqueeze(0)

    with torch.no_grad():
        x = model(x)

    mask = x[0, 0]
    height = x[0, 1]

    mask = (mask > 0).numpy().astype(np.uint8, copy=False) * 255
    height = height.numpy()
    np.exp(height, out=height)
    height = np.rint(height)
    height = height.astype(np.int, copy=False)

    return mask, height


def detect_lines(x, hmap, im_size):
    fx = im_size[1] / x.shape[1]
    fy = im_size[0] / x.shape[0]

    connect_kernel = np.ones((1, 30 + 1), dtype=np.uint8)
    remove_kernel = np.ones((1, 10 + 1), dtype=np.uint8)

    cv2.dilate(x, connect_kernel, x)
    cv2.erode(x, connect_kernel, x)

    # remove short lines
    cv2.erode(x, remove_kernel, x)
    cv2.dilate(x, remove_kernel, x)

    num_line, segment_lines = cv2.connectedComponents(x)

    boxes = []
    for c in range(num_line):
        line = segment_lines == c
        x = np.sum(line, axis=0)
        x = np.nonzero(x)[0]
        x1, x2 = x[0], x[-1]

        y = np.argmax(line, axis=0)
        y2 = max(y[x1], y[x2])

        h = np.mean(hmap[line])
        h = h + abs(y[x1] - y[x2])
        y1 = max(y2 - h, 0)

        if y2 == 0:
            continue

        boxes.append([
            (int(round(x1*fx)), int(round(y1*fy))),
            (int(round(x2*fx)), int(round(y2*fy)))
        ])

    return boxes
