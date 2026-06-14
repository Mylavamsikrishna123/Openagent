"""Gemini Vision analysis module for OpenAgent Browser."""

import base64
import re
import json
from typing import Optional, Literal

from google import genai
from google.genai import types
from playwright.async_api import Page, Locator
from pydantic import BaseModel, Field


class SemanticLocator(BaseModel):
    """
    Defines a semantic strategy for locating an element, preferred over raw CSS/XPath.
    """
    strategy: Literal["role", "text", "label", "placeholder", "test_id"] = Field(
        ..., description="The semantic strategy to locate the element"
    )
    value: str = Field(
        ..., description="The value for the semantic locator (e.g., 'button:Submit', 'Search')"
    )
    exact: bool = Field(
        default=True, 
        description="Whether to match exactly or partially"
    )
    
    def to_playwright_locator(self, page: Page) -> Locator:
        """Converts this semantic definition into a Playwright Locator."""
        if self.strategy == "role":
            # Format expected: "role_type:name" e.g., "button:Submit Form"
            parts = self.value.split(":", 1)
            if len(parts) == 2:
                role_type, name = parts
                return page.get_by_role(role_type.strip(), name=name.strip(), exact=self.exact)
            else:
                # Fallback if format is wrong, treat as text
                return page.get_by_text(self.value, exact=self.exact)
        elif self.strategy == "text":
            return page.get_by_text(self.value, exact=self.exact)
        elif self.strategy == "label":
            return page.get_by_label(self.value, exact=self.exact)
        elif self.strategy == "placeholder":
            return page.get_by_placeholder(self.value, exact=self.exact)
        elif self.strategy == "test_id":
            return page.get_by_test_id(self.value)
        
        # Fallback to generic text locator
        return page.locator(f"text={self.value}")


class GeminiAction(BaseModel):
    """Schema for Gemini's predicted action on the screen using semantic locators."""

    action: Literal["click", "type"] = Field(
        ..., description="The action to perform: 'click' or 'type'"
    )
    target: SemanticLocator = Field(
        ..., description="Semantic definition of the element to interact with"
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
    Enforces Pydantic validation by stripping markdown wrappers if present.
    Prefers semantic locators over raw CSS selectors.

    Args:
        page: Playwright Page object
        objective: The user's overall task objective

    Returns:
        GeminiAction: Parsed action with semantic locator and optional value
    """
    # Initialize Gemini client
    client = genai.Client()

    # Take screenshot and convert to base64
    screenshot_bytes = await page.screenshot(full_page=True)
    screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")

    # Prepare the prompt with strict instructions
    prompt = f"""
You are an AI browser assistant. Analyze this webpage screenshot and determine the next action to achieve the objective.

OBJECTIVE: {objective}

CRITICAL INSTRUCTIONS:
1. Output ONLY valid JSON. Do NOT wrap the output in markdown code blocks (```json ... ```).
2. Use SEMANTIC LOCATORS for the target element. Avoid CSS selectors unless absolutely necessary.
   - For buttons: use strategy "role" with value "button:Submit Form"
   - For links: use strategy "role" with value "link:Read More"  
   - For inputs: use strategy "label", "placeholder", or "test_id"
   - For generic text clicks: use strategy "text"
3. Ensure the JSON matches the provided schema exactly.

Return a JSON object with:
- action: either "click" or "type"
- target: object with {{strategy, value, exact}}
- value: text to type (only if action is "type", otherwise null)
- reasoning: brief explanation of your choice
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
    
    # Fallback: manually parse if automatic parsing failed
    if response.text:
        try:
            raw_text = response.text
            
            # Robustly strip markdown wrappers if the LLM ignores instructions
            if raw_text.startswith("```"):
                raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text)
                raw_text = re.sub(r'\s*```$', '', raw_text)
            
            # Parse and validate with Pydantic
            return GeminiAction.model_validate_json(raw_text)
            
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Error manually parsing Gemini response: {e}")
            print(f"Raw response was: {response.text[:500]}...")
    
    raise ValueError("Gemini did not return a valid structured response")
