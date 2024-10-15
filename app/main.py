import pandas as pd
import os
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from app.pipeline import PredictPipeline, CustomData

# Initialize FastAPI app and configure Jinja2 templates for rendering HTML responses
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

pipeline = PredictPipeline()

# Path to temporarily store the predictions CSV file
temp_csv_path = "app/predicted_results.csv"

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Delete the file when the user returns to the home page
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/manual_predict", response_class=HTMLResponse)
def render_manual_form(request: Request):
    return templates.TemplateResponse("manual_predict.html", {"request": request})

@app.post("/manual_predict", response_class=HTMLResponse)
def manual_predict(request: Request, ID_CLIENTE: int = Form(...), SAFRA_REF: str = Form(...), DATA_EMISSAO_DOCUMENTO: str = Form(...),
                    DATA_VENCIMENTO: str = Form(...), VALOR_A_PAGAR: float = Form(...), TAXA: float = Form(...), DATA_CADASTRO: str = Form(...),
                    DDD: int = Form(...), FLAG_PF: str = Form(...), SEGMENTO_INDUSTRIAL: str = Form(...), DOMINIO_EMAIL: str = Form(...),
                    PORTE: str = Form(...), CEP_2_DIG: int = Form(...), RENDA_MES_ANTERIOR: int = Form(...), NO_FUNCIONARIOS: int = Form(...)):
    # Create a CustomData object from the form input
    custom_data = CustomData(ID_CLIENTE, SAFRA_REF, DATA_EMISSAO_DOCUMENTO, DATA_VENCIMENTO, VALOR_A_PAGAR, TAXA, DATA_CADASTRO, DDD, 
                            FLAG_PF, SEGMENTO_INDUSTRIAL, DOMINIO_EMAIL, PORTE, CEP_2_DIG, RENDA_MES_ANTERIOR, NO_FUNCIONARIOS)
    
    # Convert the custom data into a DataFrame, generate predictions and render the template with the prediction results
    data_df = custom_data.get_data_as_dataframe()
    prediction = pipeline.predict(data_df)
    results = pipeline.results_df(data_df, prediction, temp_csv_path)
    return templates.TemplateResponse("manual_predict.html", {"request": request, "results": results})

@app.get("/dataset_predict", response_class=HTMLResponse)
def render_dataset_form(request: Request):
    # Delete the file when the form is rendered again
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)
    return templates.TemplateResponse("dataset_predict.html", {"request": request})

@app.post("/dataset_predict", response_class=HTMLResponse)
def predict_dataset(request: Request, file: UploadFile = File(...)):
    # Read the dataset, generate predictions and render them in the html template. Raise error if it can't process the dataset
    try:
        df = pd.read_csv(file.file)
        processed_df = pipeline.preprocess_dataset(df)
        predictions = pipeline.predict(processed_df)
        results = pipeline.results_df(processed_df, predictions, temp_csv_path)
        return templates.TemplateResponse("dataset_predict.html", {"request": request, "results": results})
    except ValueError as e:
        return templates.TemplateResponse("dataset_predict.html", {"request": request, "error_message": f"Error processing dataset: {e}"})

# Endpoint for downloading the CSV file
@app.get("/download_csv")
def download_csv():
    if os.path.exists(temp_csv_path):
        # Return the CSV file for download
        return FileResponse(temp_csv_path, filename="predicted_results.csv", media_type="text/csv")
    else:
        return {"error": "No file available to download"}