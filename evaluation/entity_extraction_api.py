from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Set
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger

# Load the Flair NER model once (global for efficiency)
tagger = SequenceTagger.load("flair/ner-english-large")

# Initialize FastAPI app
app = FastAPI()

# Define default entity types
DEFAULT_ENTITY_TYPES = ["PER", "ORG", "LOC"]

# Pydantic models for request & response
class EntityExtractionRequest(BaseModel):
    text: Union[str, List[str]] = Field(..., description="A single string or a list of strings to extract named entities from.")
    allowed_entity_types: List[str] = Field(default=DEFAULT_ENTITY_TYPES, description="List of entity types to extract (e.g., PER, ORG, LOC).")

class EntityExtractionResponse(BaseModel):
    entities: Dict[str, Set[str]] = Field(..., description="A dictionary mapping input text to its extracted named entities.")

# Named Entity Extraction function
def extract_named_entities(text: Union[str, List[str]], allowed_entity_types: List[str]) -> Dict[str, Set[str]]:
    """Extract named entities from one or multiple texts, filtering by entity type."""
    
    # Handle both single string & list input
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text  # Already a list

    # Convert to Pandas Series for vectorized apply
    df = pd.DataFrame({"text": texts})

    # Define extraction function
    def extract(text):
        if pd.isna(text) or not isinstance(text, str):
            return set()
        
        sentence = Sentence(text)
        tagger.predict(sentence)

        return {entity.text for entity in sentence.get_spans('ner') if entity.tag in allowed_entity_types}

    # Vectorized apply
    df["entities"] = df["text"].apply(extract)

    # Convert to dictionary format
    return dict(zip(df["text"], df["entities"]))

# FastAPI endpoint
@app.post("/extract-entities", response_model=EntityExtractionResponse)
async def extract_entities(request: EntityExtractionRequest):
    """API endpoint to extract named entities from input text(s)."""
    entities = extract_named_entities(request.text, request.allowed_entity_types)
    return EntityExtractionResponse(entities=entities)
