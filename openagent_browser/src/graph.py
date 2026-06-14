"""
LangGraph StateGraph for OpenAgent Browser.

Wires together planner, vision, executor, and self-heal nodes with conditional edges.
"""

import asyncio
from typing import Literal, TypedDict, Optional, List, Any
from playwright.async_api import async_playwright, Page

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from src.main import AgentState
from src.planner_node import plan_task
from src.gemini_vision import analyze_screen_with_gemini, GeminiAction, SemanticLocator
from src.execution_node import execute_action


# Maximum retries before giving up
MAX_RETRIES = 3


async def vision_node(state: AgentState, page: Page) -> AgentState:
    """
    Vision node that analyzes the screen and returns the action plan.
    Stores the action in state for the executor to use.
    """
    objective = state.get("user_task", "")
    
    try:
        action_plan: GeminiAction = await analyze_screen_with_gemini(page, objective)
        
        return {
            **state,
            "current_action": action_plan.model_dump(),
            "error": None,
            "retry_count": 0  # Reset retry count on successful vision analysis
        }
    except Exception as e:
        return {
            **state,
            "error": str(e),
            "retry_count": state.get("retry_count", 0) + 1
        }


async def self_heal_node(state: AgentState, page: Page) -> AgentState:
    """
    Self-healing node that handles failures by retrying vision analysis.
    If max retries exceeded, marks the task as complete with error.
    """
    retry_count = state.get("retry_count", 0)
    error_msg = state.get("error", "Unknown error")
    
    print(f"Self-heal triggered. Retry count: {retry_count}/{MAX_RETRIES}")
    print(f"Error: {error_msg}")
    
    if retry_count >= MAX_RETRIES:
        print(f"Max retries ({MAX_RETRIES}) exceeded. Aborting.")
        return {
            **state,
            "is_complete": True,
            "error": f"Max retries exceeded. Last error: {error_msg}"
        }
    
    # Increment retry count and route back to vision
    return {
        **state,
        "retry_count": retry_count + 1,
        "error": None  # Clear error so vision can try again
    }


def should_continue_after_vision(state: AgentState) -> Literal["executor", "self_heal"]:
    """Conditional edge: route to executor if vision succeeded, else self_heal."""
    if state.get("error"):
        return "self_heal"
    return "executor"


def should_continue_after_executor(state: AgentState) -> Literal["vision", "self_heal", "END"]:
    """
    Conditional edge after execution:
    - If error occurred: route to self_heal
    - If task appears complete (no more actions needed): route to END
    - Otherwise: route back to vision for next step
    """
    if state.get("error"):
        return "self_heal"
    
    # Check if we should stop (for now, we'll run until error or manual limit)
    # In a real implementation, you'd check if the objective is achieved
    step_count = len(state.get("step_results", []))
    max_steps = 10  # Safety limit
    
    if step_count >= max_steps:
        print(f"Reached max steps ({max_steps}). Stopping.")
        return "END"
    
    # Continue loop
    return "vision"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph workflow."""
    
    # Define the graph builder
    builder = StateGraph(AgentState)
    
    # Add nodes (note: we'll pass page via context in the actual run)
    # For now, we define nodes that accept state only
    # We'll need to handle page passing differently
    
    # Since our nodes need both state and page, we'll create wrapper functions
    # that close over the page object
    
    def make_node_wrapper(async_func):
        """Create a wrapper that injects the page object."""
        async def wrapper(state: AgentState):
            # Page will be passed via graph configuration or global context
            # For this demo, we assume page is available in the closure
            return await async_func(state, page)
        return wrapper
    
    # We'll add nodes dynamically when running with a page object
    # For now, define the structure
    
    builder.add_node("planner", make_node_wrapper(plan_task))
    builder.add_node("vision", make_node_wrapper(vision_node))
    builder.add_node("executor", make_node_wrapper(execute_action))
    builder.add_node("self_heal", make_node_wrapper(self_heal_node))
    
    # Set entry point
    builder.set_entry_point("planner")
    
    # Add conditional edges
    builder.add_conditional_edges(
        source="planner",
        path=lambda s: "vision",  # Always go to vision after planning
        target_map=["vision"]
    )
    
    builder.add_conditional_edges(
        source="vision",
        path=should_continue_after_vision,
        target_map=["executor", "self_heal"]
    )
    
    builder.add_conditional_edges(
        source="executor",
        path=should_continue_after_executor,
        target_map=["vision", "self_heal", "END"]
    )
    
    builder.add_conditional_edges(
        source="self_heal",
        path=lambda s: "vision" if s.get("retry_count", 0) < MAX_RETRIES else "END",
        target_map=["vision", "END"]
    )
    
    return builder.compile()


async def run_agent(objective: str, start_url: str = "https://news.ycombinator.com"):
    """
    Main entry point to run the browser agent.
    
    Args:
        objective: The user's task objective
        start_url: The URL to start from
    """
    async with async_playwright() as p:
        # Launch browser (non-headless for visibility during testing)
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Navigate to start URL
        print(f"Navigating to {start_url}...")
        await page.goto(start_url)
        await page.wait_for_load_state("networkidle")
        
        # Build the graph with page closure
        def make_node_wrapper(async_func):
            async def wrapper(state: AgentState):
                return await async_func(state, page)
            return wrapper
        
        builder = StateGraph(AgentState)
        builder.add_node("planner", make_node_wrapper(plan_task))
        builder.add_node("vision", make_node_wrapper(vision_node))
        builder.add_node("executor", make_node_wrapper(execute_action))
        builder.add_node("self_heal", make_node_wrapper(self_heal_node))
        
        builder.set_entry_point("planner")
        
        builder.add_conditional_edges(
            source="planner",
            path=lambda s: "vision",
            target_map=["vision"]
        )
        
        builder.add_conditional_edges(
            source="vision",
            path=should_continue_after_vision,
            target_map=["executor", "self_heal"]
        )
        
        builder.add_conditional_edges(
            source="executor",
            path=should_continue_after_executor,
            target_map=["vision", "self_heal", "END"]
        )
        
        builder.add_conditional_edges(
            source="self_heal",
            path=lambda s: "vision" if s.get("retry_count", 0) < MAX_RETRIES else "END",
            target_map=["vision", "END"]
        )
        
        graph = builder.compile()
        
        # Initialize state
        initial_state: AgentState = {
            "user_task": objective,
            "plan": [],
            "current_step_index": 0,
            "current_url": start_url,
            "step_results": [],
            "retry_count": 0,
            "error": None,
            "is_complete": False,
            "current_action": None
        }
        
        print(f"\nStarting agent with objective: {objective}")
        print("=" * 50)
        
        # Run the graph
        final_state = await graph.ainvoke(initial_state)
        
        print("\n" + "=" * 50)
        print("Agent finished!")
        print(f"Final URL: {final_state.get('current_url')}")
        print(f"Steps executed: {len(final_state.get('step_results', []))}")
        if final_state.get("error"):
            print(f"Error: {final_state['error']}")
        
        # Print step history
        print("\nStep History:")
        for step in final_state.get("step_results", []):
            print(f"  Step {step['step']}: {step['action']} - {step['status']}")
            if step.get("reasoning"):
                print(f"    Reasoning: {step['reasoning']}")
        
        await browser.close()
        
        return final_state


if __name__ == "__main__":
    # Test the agent on Hacker News
    test_objective = "Find and click on the first story link titled 'Show HN'"
    
    result = asyncio.run(run_agent(test_objective))
