"""Gemini Vision analysis module for OpenAgent Browser."""

import base64
from typing import Optional, Literal

from google import genai
from google.genai import types
from playwright.async_api import Page
from pydantic import BaseModel, Field


class GeminiAction(BaseModel):
    """Schema for Gemini's predicted action on the screen."""

    action: Literal["click", "type"] = Field(
        ..., description="The action to perform: 'click' or 'type'"
    )
    target_selector: str = Field(
        ..., description="CSS selector of the target element to interact with"
    )
    value: Optional[str] = Field(
        None, description="Text value to type (only required if action is 'type')"
    )
    reasoning: str = Field(
        ..., description="Brief explanation of why this action was chosen"
    )


async def analyze_screen_with_gemini(
    page: Page, objective: str
) -> GeminiAction:
    """
    Take a screenshot of the current page, convert to base64, and send to Gemini 1.5 Flash.

    Args:
        page: Playwright Page object
        objective: The user's overall task objective

    Returns:
        GeminiAction: Parsed action with selector and optional value
    """
    # Initialize Gemini client
    client = genai.Client()

    # Take screenshot and convert to base64
    screenshot_bytes = await page.screenshot(full_page=True)
    screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")

    # Prepare the prompt
    prompt = f"""
You are an AI browser assistant. Analyze this webpage screenshot and determine the next action to achieve the objective.

OBJECTIVE: {objective}

Return a JSON object with:
- action: either "click" or "type"
- target_selector: CSS selector of the element to interact with
- value: text to type (only if action is "type", otherwise null)
- reasoning: brief explanation of your choice

Be precise with CSS selectors. Prefer stable selectors like id, name, or specific class combinations.
"""

    # Prepare image content
    image_content = types.Content(
        parts=[
            types.Part.from_bytes(
                data=screenshot_bytes, mime_type="image/png"
            ),
            types.Part.from_text(text=prompt),
        ]
    )

    # Generate response with structured output
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[image_content],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiAction,
        ),
    )

    # Parse and return the structured response
    if response.parsed:
        return response.parsed
    
    raise ValueError("Gemini did not return a valid structured response")
