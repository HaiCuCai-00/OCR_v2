import glob
import os
import sys

sys.path.append("../")
import argparse
import time

import cv2
from omegaconf import OmegaConf
from PIL import Image

from configs import config
from models.craft_text_detector import Craft
from modules.ocr.text_classifier import TextClassifier
from modules.post_processing import FormConverter
from modules.pre_processing import DocScanner, RemoveUnnecessaryPart
from utils.image_processing import ocr_input_processing


class Pipeline:
    def __init__(self):
        self.ocr_config_path = "../configs/ocr/transformer_ocr.yaml"
        self.ocr_cfg = OmegaConf.load(self.ocr_config_path)
        self.init_modules()

    def init_modules(self):
        self.det_model = Craft(crop_type="box", cuda=config.CUDA)
        self.ocr_model = TextClassifier(self.ocr_cfg)
        self.form_converter = FormConverter()
        self.remove_unnecessary_part = RemoveUnnecessaryPart()
        self.scanner = DocScanner()

    def start(self, img, output_dir="./results"):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = self.scanner.scan(img, os.path.join(output_dir, "debug_scan.jpg"))
        img = self.remove_unnecessary_part(type, img)
        text_out=''
        try:
            crop_list, regions, image_detect = self.det_model.detect_text(img)
            boxes = regions["boxes"]
            texts = []
            for crop_img in crop_list:
                crop_img = Image.fromarray(crop_img)
                ocr_input = ocr_input_processing(
                    crop_img,
                    self.ocr_cfg.model.input.image_height,
                    self.ocr_cfg.model.input.image_min_width,
                    self.ocr_cfg.model.input.image_max_width,
                )
                text = self.ocr_model.predict(ocr_input)
                print(text)
                texts.append(text)
                text_out += (text +" ") 
            #print(boxes)
            for box in boxes:
                box = box.astype(int)
                cv2.line(img, tuple(box[0]), tuple(box[1]), (0, 0, 255), 5)
                cv2.line(img, tuple(box[1]), tuple(box[2]), (0, 0, 255), 5)
                cv2.line(img, tuple(box[2]), tuple(box[3]), (0, 0, 255), 5)
                cv2.line(img, tuple(box[3]), tuple(box[0]), (0, 0, 255), 5)
            cv2.imwrite("img_with_boxes.png", img)
        except Exception as e:
            text_out=" "
        

        # form = self.form_converter(type, boxes, texts)
        # print(form)
        # return form
        return text_out
    def crop(self, image, x,y,w,h):
        try:
            image =image[y:y+h,x:x+w]
            cv2.imwrite("form.jpg",image)
            return image
        except Exception as e:
            return "box out image"

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Document Extraction")
    parser.add_argument("--input",default='/media/ai-r-d/FaceDB1/life_project/Form/OCR_v2/tools/dong-chi.png', help="Path to single image to be scanned")
    parser.add_argument("--type",default='cmnd', help="Document type (cmnd/ttk/hc/dkkd/sohong...)")
    parser.add_argument("--output", default="./results", help="Path to output folder")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pipeline = Pipeline()

    img = cv2.imread(args.input)
    start_time = time.time()
    imgage=pipeline.crop(img,5,90,355,76)
    res = pipeline.start(imgage, args.output)
    print(res)
    end_time = time.time()
    print(f"Executed in {end_time - start_time} s")
