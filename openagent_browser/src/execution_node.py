"""
LangGraph execution node for the OpenAgent Browser.

This module contains the `execute_action` node which:
1. Retrieves the current objective from the state.
2. Calls `analyze_screen_with_gemini` to determine the next UI action.
3. Executes the action (click or type) using Playwright.
4. Waits for network idle.
5. Updates the state with the result and history.
"""

import asyncio
from typing import Any, Dict, List, Optional
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from src.gemini_vision import analyze_screen_with_gemini, GeminiAction
from src.main import AgentState


async def execute_action(state: AgentState, page: Page) -> AgentState:
    """
    LangGraph node that analyzes the screen and executes the next action.
    
    Args:
        state: The current AgentState containing the objective and history.
        page: The active Playwright Page instance.
        
    Returns:
        Updated AgentState with the result of the action, incremented step index,
        and appended history entry.
    """
    objective = state.get("user_task", "")
    current_step_index = state.get("current_step_index", 0)
    history = state.get("step_results", [])
    
    try:
        # 1. Analyze the screen with Gemini Vision
        action_plan: GeminiAction = await analyze_screen_with_gemini(page, objective)
        
        action_type = action_plan.action
        selector = action_plan.target_selector
        value = action_plan.value
        
        # Prepare log entry for this action
        log_entry: Dict[str, Any] = {
            "step": current_step_index,
            "action": action_type,
            "target_selector": selector,
            "value": value,
            "reasoning": action_plan.reasoning,
            "status": "pending"
        }

        # 2. Execute the action using Playwright
        element = page.locator(selector)
        
        # Check if element exists
        count = await element.count()
        if count == 0:
            raise ValueError(f"Element not found for selector: {selector}")
        
        if action_type == "click":
            await element.click()
            log_entry["status"] = "clicked"
            
        elif action_type == "type":
            if value is None:
                raise ValueError("Value is required for 'type' action but was missing.")
            await element.fill(value)
            log_entry["status"] = "typed"
            
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        # 3. Wait for network idle (with 2 second timeout)
        try:
            await page.wait_for_load_state("networkidle", timeout=2000)
        except PlaywrightTimeoutError:
            # Network activity continued beyond timeout, but we proceed anyway
            pass
        
        # Additional small buffer to ensure UI renders after network activity
        await asyncio.sleep(0.5)

        # Update current URL after action
        current_url = page.url

        # Build updated state
        updated_history = history + [log_entry]
        
        return {
            **state,
            "current_step_index": current_step_index + 1,
            "current_url": current_url,
            "step_results": updated_history,
            "error": None,
            "is_complete": False
        }

    except Exception as e:
        error_msg = str(e)
        log_entry = {
            "step": current_step_index,
            "action": "error",
            "target_selector": "N/A",
            "value": None,
            "error": error_msg,
            "status": "failed"
        }
        
        return {
            **state,
            "error": error_msg,
            "step_results": history + [log_entry],
            "is_complete": True  # Stop on error
        }
