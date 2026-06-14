import os
import io
import google.generativeai as genai
from PIL import Image
from playwright.async_api import Page
import json

# Import models from root level
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import VisionAction

# Initialize the FREE Gemini Model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_vision = genai.GenerativeModel('gemini-1.5-flash')

async def get_vision_action(page: Page, objective: str) -> VisionAction:
    """Takes a screenshot and asks Gemini what to do next."""
    
    # 1. Capture screenshot as bytes
    screenshot_bytes = await page.screenshot()
    image = Image.open(io.BytesIO(screenshot_bytes))
    
    # 2. Construct the Prompt
    prompt = f"""
    You are an AI browser agent. Your current objective is: "{objective}".
    Look at the attached screenshot of the browser.
    Determine the single next best action to take to achieve the objective.
    
    If the objective is complete, return action: "done".
    Otherwise, identify the exact UI element to click or type into.
    """
    
    # 3. Call Gemini with Image
    response = gemini_vision.generate_content(
        [prompt, image],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=VisionAction.model_json_schema()
        )
    )
    
    # 4. Parse into Pydantic Model
    try:
        # Gemini's native JSON schema enforcement makes this incredibly clean
        return VisionAction.model_validate_json(response.text)
    except Exception as e:
        print(f"⚠️ Vision parsing failed: {e}. Defaulting to wait.")
        return VisionAction(action="wait", target_selector="body", confidence=0.0)
