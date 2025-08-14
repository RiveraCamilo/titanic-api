from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

class PredictRequest(BaseModel):
    pclass: int = Field(..., description="Clase del pasajero: 1, 2 o 3")
    age: Optional[float] = Field(None, ge=0, description="Edad en años")
    fare: Optional[float] = Field(None, ge=0, description="Tarifa pagada")
    sibsp: int = Field(..., ge=0, description="cantidad de hermanos/esposo(a) a bordo")
    parch: int = Field(..., ge=0, description="cantidad de padres/hijos a bordo")
    sex: Literal["male", "female"] = Field(..., description='"male" o "female"')
    embarked: Literal["C", "Q", "S"] = Field(..., description='Puerto de embarque: "C", "Q" o "S"')

    @field_validator("pclass")
    @classmethod
    def validate_pclass(cls, v: int) -> int:
        if v not in {1, 2, 3}:
            raise ValueError("pclass debe ser 1, 2 o 3")
        return v

    @field_validator("sex", mode="before")
    @classmethod
    def normalize_sex(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
        if v not in {"male", "female"}:
            raise ValueError("sex debe ser 'male' o 'female'")
        return v

    @field_validator("embarked", mode="before")
    @classmethod
    def normalize_embarked(cls, v):
        if isinstance(v, str):
            v = v.strip().upper()
        if v not in {"C", "Q", "S"}:
            raise ValueError("embarked debe ser 'C', 'Q' o 'S'")
        return v

class PredictResponse(BaseModel):
    prediction: int = Field(..., description="0 = no sobrevive, 1 = sobrevive")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de clase 1")