import cv2
import torch
import unet
import crnn
import sys
import os


def main(img_folder, out_folder='part1_output'):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    unet_model = unet.Model()
    unet_model.load_state_dict(torch.load('weights/unet.pt', map_location='cpu'))

    crnn_model = crnn.Model()
    crnn_model.load_state_dict(torch.load('weights/crnn.pt', map_location='cpu'))

    img_names = os.listdir(img_folder)

    for img_name in img_names:
        im = os.path.join(img_folder, img_name)
        im = cv2.imread(im)

        m, h = unet.segment(unet_model, im)
        boxes = unet.detect_lines(m, h, im.shape)

        label_file = open(os.path.join(out_folder, img_name[:-3] + 'txt'), 'w')

        for box in boxes:
            x1, y1 = box[0]
            x2, y2 = box[1]

            text_line = im[y1:y2, x1:x2]
            text_line = cv2.cvtColor(text_line, cv2.COLOR_BGR2GRAY)
            text = crnn.ocr(crnn_model, text_line)

            if text == '':
                continue

            for c in (x1, y1, x2, y1, x2, y2, x1, y2):
                label_file.write(str(c) + ', ')
            label_file.write(text)
            label_file.write('\n')

        label_file.close()


if __name__ == '__main__':
    IM_FOLDER = sys.argv[1]
    if len(sys.argv) == 3:
        OUT_FOLDER = sys.argv[2]
        main(IM_FOLDER, OUT_FOLDER)
    else:
        main(IM_FOLDER)
