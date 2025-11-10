from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Interpolator App API",
    description="Endpoints for 5D Interpolator application and simple items CRUD.",
    version="0.1.0",
    contact={"name": "Baron Gracias"}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    name: str
    price: float
    
@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}

@app.get("/")
async def hello():
    return {"message": "Hello 👋"}

@app.post("/items")
async def create(item: Item):
    return {"ok": True, "item": item}