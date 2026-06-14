import os
from groq import Groq
import json

# Import models from root level
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import PlannerOutput

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

async def generate_plan(task: str) -> PlannerOutput:
    """Uses Groq to break a high-level goal into steps."""
    
    system_prompt = """
    You are an expert browser automation planner. 
    You break down high-level user goals into precise, step-by-step browser actions.
    Keep steps simple and atomic (e.g., "Click the login button", "Type 'hello' into the search box").
    """
    
    completion = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a plan for this task: {task}"}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    raw_json = completion.choices[0].message.content
    return PlannerOutput.model_validate_json(raw_json)
