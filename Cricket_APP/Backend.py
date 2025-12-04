from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import pandas as pd
import joblib
app = FastAPI()

Model = joblib.load("trained_model.pkl")
        
class Userinput(BaseModel):
    venue: str 
    batting_team: str
    bowling_team: str
    balls_left: int 
    wicket_left: int 
    Current_Score: int 
    Crr: float
    last_five: int 
    
    
@app.get("/", tags=["root"])
def read_root():
    return {"message": "Model server is running. Use POST /predict to request predictions."}

@app.post("/predict")
def predict_runs(data: Userinput):
    # Build input DataFrame (must match training feature names exactly!)
    input_df = pd.DataFrame([{
        "venue": data.venue,
        "batting_team": data.batting_team,
        "bowling_team": data.bowling_team,
        "balls_left": data.balls_left,
        "wicket_left": data.wicket_left,
        "Current_Score": data.Current_Score, 
        "Crr": data.Crr,   
        "last_five": data.last_five
    }])

    prediction = Model.predict(input_df)[0]

    return JSONResponse(
        status_code=200,
        content={"Predicted runs": int(prediction)}  # ensure JSON serializable
    )
