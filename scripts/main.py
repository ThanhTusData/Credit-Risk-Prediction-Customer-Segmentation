from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse
import pandas as pd
import io

from predict import predict

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to Credit Risk Prediction API!"}

# API upload file CSV
@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    result_df = predict(df)
    
    # Trả kết quả dưới dạng JSON
    return JSONResponse(content=result_df.to_dict(orient="records"))

# API nhận trực tiếp JSON
@app.post("/predict_json/")
async def predict_json(data: list = Body(...)):
    df = pd.DataFrame(data)
    
    result_df = predict(df)
    
    return JSONResponse(content=result_df.to_dict(orient="records"))
