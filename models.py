from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class PlannerOutput(BaseModel):
    """The output from the Groq Brain."""
    thought_process: str = Field(description="A brief 1-sentence reasoning of the strategy.")
    steps: List[str] = Field(description="A list of 3 to 5 actionable browser steps.")

class VisionAction(BaseModel):
    """The output from the Gemini Eyes."""
    action: Literal["click", "type", "scroll", "wait", "done"] = Field(description="The type of action to perform.")
    target_selector: str = Field(description="The exact text, role, or placeholder of the element to interact with.")
    value: Optional[str] = Field(default=None, description="The text to type, if action is 'type'.")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0 that the element was found.")
