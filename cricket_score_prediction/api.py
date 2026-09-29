"""REST API for the ODI score model.

Run from the repository root:
    uvicorn api:app --app-dir cricket_score_prediction --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

import cricket_model as cm

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["bundle"] = cm.load_or_train()
    yield
    state.clear()


app = FastAPI(title="ODI Score Predictor", version="1.0.0", lifespan=lifespan)


class MatchState(BaseModel):
    venue: str
    batting_team: str
    bowling_team: str
    balls_left: int = Field(ge=0, le=270, description="Balls remaining (predictions start after 5 overs)")
    wickets_left: int = Field(ge=1, le=10)
    current_score: int = Field(ge=0, le=500)
    last_five: int = Field(ge=0, le=200, description="Runs scored in the last 30 balls")

    @model_validator(mode="after")
    def check_consistency(self):
        if self.batting_team == self.bowling_team:
            raise ValueError("batting_team and bowling_team must differ")
        if self.last_five > self.current_score:
            raise ValueError("last_five cannot exceed current_score")
        return self


class Prediction(BaseModel):
    predicted_score: int
    run_rate_projection: int


@app.get("/")
def health():
    return {"status": "ok", "docs": "/docs"}


@app.get("/options")
def options():
    bundle = state["bundle"]
    return {"venues": bundle["venues"], "teams": bundle["teams"]}


@app.post("/predict", response_model=Prediction)
def predict(match: MatchState):
    bundle = state["bundle"]
    for field, allowed in (("venue", bundle["venues"]), ("batting_team", bundle["teams"]), ("bowling_team", bundle["teams"])):
        if getattr(match, field) not in allowed:
            raise HTTPException(status_code=422, detail=f"Unknown {field}: {getattr(match, field)!r}. See GET /options.")
    features = cm.make_input(**match.model_dump())
    predicted = max(float(bundle["model"].predict(features[cm.FEATURES])[0]), match.current_score)
    return Prediction(
        predicted_score=round(predicted),
        run_rate_projection=round(float(cm.run_rate_projection(features)[0])),
    )
