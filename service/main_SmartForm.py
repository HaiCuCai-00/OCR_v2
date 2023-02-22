import base64
import json
import sys
import time
from typing import List, Dict, Optional, Type
import cv2
from matplotlib.pyplot import flag
import numpy as np
import pydantic
import uvicorn
import requests
# import lib OCR my PDF
import ocrmypdf
from fastapi.staticfiles import StaticFiles
#from config.mapping import Mapping
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel
from starlette.responses import HTMLResponse
import shutil
from pdf2image import convert_from_path
import uuid
sys.path.append("..")
from utils.get_path import isBase64
from tools import Pipeline

app_desc = """Help detect information for some type of identification"""

app = FastAPI(title="OCR Vietnamese documents FastAPI", description=app_desc)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = Pipeline()

class IDClass(BaseModel):
    type: str
    images: List[str]



class Data(BaseModel):
    fieldKey:str
    x:float
    y:float
    w:float
    h:float

class DataPDF(BaseModel):
    fieldKey:str
    page:float
    x:float
    y:float
    w:float
    h:float

class Images(BaseModel):
    image: Optional[List[str]] = pydantic.Field(
        default=None, example=None, description="List of base64 encoded images"
    )
    data:List[Data]

class PDFs(BaseModel):
    pdf:Optional[List[str]]=pydantic.Field(
        default=None, example=None, description="Path download pdf")
    data:List[DataPDF]

class Body(BaseModel):
    __root__: Dict[str, IDClass]

@app.post('/ocrpdf')
def ocrpdf(file :UploadFile=File(...)):
    try:

        with open(f'upload/{file.filename}', 'wb') as fileup:
            shutil.copyfileobj(file.file, fileup)
        file_path=f'upload/{file.filename}'
        file_out = ('static/'f"{file.filename.replace(' ', '')}")
        s = time.time()
        ocrmypdf.ocr(
            input_file=file_path,
            output_file=file_out,
            language=["vie"],
            deskew=True,
            rotate_pages=True,
            jobs=4,
            skip_text=True,
            )
        with open("/proc/stat", "r") as stat:
            (key, user, nice, system, idle, _) = (stat.readline().split(None, 5))
        assert key == "cpu", "'cpu ...' should be the first line in /proc/stat"
        busy = int(user) + int(nice) + int(system)
        CPU = 100 * busy / (busy + int(idle))
        print("time: ", time.time() - s )
        return {"status": True, "path": f"123.25.30.4:3000/static/{file.filename}", "TIME" : time.time() - s, "CPU": CPU}
    except Exception as e:
        return {"status": False, "msg": str(e)}

@app.post("/detect")
async def detect(param:Images):
    try:
        out={}
        if param.image is not None:
            try:
                image=param.image[0]
                if image =="":
                    return {"status": False, "msg": "No image"}
                image_bytes=base64.b64decode(image)
                img =cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
            except Exception as e:
                return {"status": False, "msg": "No detect image base64"}                
            # frame =pipeline.crop(img,int(param.data[0].x ),int(param.data[0].y),int(param.data[0].w),int(param.data[0].h))
            # res = pipeline.start(frame)
            
            for i in range(0,len(param.data)):
                frame =pipeline.crop(img,int(param.data[i].x ),int(param.data[i].y),int(param.data[i].w),int(param.data[i].h))
                res = pipeline.start(frame)
                if res =="can not detect text from form":
                    return {"status": False, "msg": "box not text"}
                out[param.data[i].fieldKey]=res         
            # for k in len(res):
            #     out[param.data[i].fieldKey]+=res[k]
            
            return out
        else:
            return {"status": False, "msg": "Image data not found"}
    except Exception as e:
        return {"status": False, "msg": "box not text"}

@app.post("/detect_image")
async def detect_image(file:UploadFile=File(...),
    meta: Optional[str]=Form("null")):
    out={}
    file_id = uuid.uuid4().hex
    with open(f'Data/{file_id}_{file.filename}', 'wb') as fileup:
        shutil.copyfileobj(file.file, fileup)
    img=cv2.imread(f"Data/{file_id}_{file.filename}")
    meta = json.loads(meta) or {}
    for i in range(0,len(meta["data"])):
        # print(meta["data"][i]["x"])
        # print(meta["data"][i]["y"])
        # print(meta["data"][i]["w"])
        # print(meta["data"][i]["h"])
        frame=pipeline.crop(img,int(meta["data"][i]["x"]),int(meta["data"][i]["y"]),int(meta["data"][i]["w"]),int(meta["data"][i]["h"]))
        res = pipeline.start(frame)
        out[meta["data"][i]["fieldKey"]]=res
    return out
    



# @app.post("/detect_pdf")
# async def detect_pdf(param:PDFs
#     file:UploadFile=File(...),
#     meta: Optional[str]=Form("null")
#     ):
#     file_id = uuid.uuid4().hex
#     with open(f'Data/{file_id}_{file.filename}', 'wb') as fileup:
#         shutil.copyfileobj(file.file, fileup)
#     meta = json.loads(meta) or {}
#     out={}
#     print(meta["x"])
#     images = convert_from_path(f'Data/{file_id}_{file.filename}')
#     images[int(meta["pager"])-1].save(f'Data/{file_id}.jpg')
#     img=cv2.imread(f"Data/{file_id}.jpg")
#     frame=pipeline.crop(img,int(meta["x"]),int(meta["y"]),int(meta["w"]),int(meta["h"]))
#     res = pipeline.start(frame)
#     out[meta["fieldKey"]]=res  
#     return out
#     out={}
#     path=param.pdf[0]
#     response = requests.get(path, verify=False)
#     open(f"Data/{file_id}.pdf", "wb").write(response.content)
#     images = convert_from_path(f'Data/{file_id}.pdf')
#     for i in range(0,len(param.data)):
#         images[int(param.data[i].page)].save(f'Data/{file_id}.jpg')
#         img=cv2.imread(f"Data/{file_id}.jpg")
#         frame=pipeline.crop(img,int(param.data[i].x),int(param.data[i].y),int(param.data[i].w),int(param.data[i].h))
#         res = pipeline.start(frame)
#         out[param.data[i].fieldKey]=res  
#     return out

# @app.post("/detect_file_pdf")
# async def detect_file_pdf(file:UploadFile=File(...),meta: Optional[str]=Form("null")):
#     file_id = uuid.uuid4().hex
#     with open(f'Data/{file_id}_{file.filename}', 'wb') as fileup:
#         shutil.copyfileobj(file.file, fileup)
#     meta = json.loads(meta) or {}
#     out={}
#     print(meta["x"])
#     images = convert_from_path(f'Data/{file_id}_{file.filename}')
#     images[int(meta["page"])-1].save(f'Data/{file_id}.jpg')
#     img=cv2.imread(f"Data/{file_id}.jpg")
#     frame=pipeline.crop(img,int(meta["x"]),int(meta["y"]),int(meta["w"]),int(meta["h"]))
#     res = pipeline.start(frame)
#     out[meta["fieldKey"]]=res  
#     return out

@app.post("/detect_pdf")
async def check(file:UploadFile=File(...),
    meta: Optional[str]=Form("null")):
    out={}

    try:
        file_id = uuid.uuid4().hex
        with open(f'Data/{file_id}_{file.filename}', 'wb') as fileup:
            shutil.copyfileobj(file.file, fileup)
        print(file.filename)
        if not file.filename.endswith('.pdf'):
            return "file is not pdf"
        images = convert_from_path(f'Data/{file_id}_{file.filename}')
        
        meta = json.loads(meta) or {}
        for i in range(0,len(meta["data"])):
            images[int(meta["data"][i]["page"])-1].save(f'Data/{file_id}.jpg')
            #print(meta["data"][i]["x"])
            img=cv2.imread(f"Data/{file_id}.jpg")
            frame=pipeline.crop(img,int(meta["data"][i]["x"]),int(meta["data"][i]["y"]),int(meta["data"][i]["w"]),int(meta["data"][i]["h"]))
            # if frame=="box out image":
            #     return {"status": False, "msg": "box out image"}
            res = pipeline.start(frame)
            out[meta["data"][i]["fieldKey"]]=res
        return out
    except Exception as e:
        return {"status": False, "msg": "box not text {}".format(e)}

if __name__ == "__main__":
    uvicorn.run(app, debug=True, host="0.0.0.0", port=3000)
