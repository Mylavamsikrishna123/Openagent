import asyncio
from typing import List, TypedDict, Literal
from playwright.async_api import async_playwright, Page
from langgraph.graph import StateGraph, END, START

# Import your existing modules
import sys
sys.path.insert(0, '.')
from models import PlannerOutput, VisionAction
from tools.planner import generate_plan
from tools.vision import get_vision_action

# --- State Definition ---
class AgentState(TypedDict):
    task: str
    plan: List[str]
    current_step_index: int
    history: List[str]
    retry_count: int
    error_message: str | None

# --- Nodes ---

async def plan_node(state: AgentState) -> AgentState:
    """Calls Groq to generate the high-level plan."""
    print(f"🧠 Planning task: {state['task']}")
    try:
        plan_output: PlannerOutput = await generate_plan(state['task'])
        return {
            "plan": plan_output.steps,
            "current_step_index": 0,
            "history": [f"Plan generated: {plan_output.thought_process}"],
            "retry_count": 0,
            "error_message": None
        }
    except Exception as e:
        return {
            "history": [f"Planning failed: {str(e)}"],
            "error_message": str(e)
        }

async def execute_node(state: AgentState, page: Page) -> AgentState:
    current_step_idx = state["current_step_index"]
    plan = state["plan"]
    
    if current_step_idx >= len(plan):
        print("✅ Plan completed.")
        return {"history": state["history"] + ["Plan completed"]}

    current_objective = plan[current_step_idx]
    print(f"👁️  Executing step {current_step_idx + 1}/{len(plan)}: {current_objective}")

    try:
        vision_action: VisionAction = await get_vision_action(page, current_objective)
        
        if vision_action.action == "done":
            print("🎉 Objective achieved (Gemini signaled 'done').")
            return {"history": state["history"] + ["Task completed by agent"], "current_step_index": len(plan)}

        locator = None
        
        if vision_action.action == "click":
            locator = page.get_by_role("button", name=vision_action.target_selector)
            if not await locator.count():
                locator = page.get_by_text(vision_action.target_selector)
            if not await locator.count():
                locator = page.locator(f"text={vision_action.target_selector}")
            await locator.click(timeout=5000)
            
        elif vision_action.action == "type":
            if not vision_action.value:
                raise ValueError("Type action requires a 'value'")
            locator = page.get_by_label(vision_action.target_selector)
            if not await locator.count():
                locator = page.get_by_placeholder(vision_action.target_selector)
            if not await locator.count():
                locator = page.locator(f"input, textarea")
            await locator.fill(vision_action.value)
            
        elif vision_action.action == "scroll":
            await page.evaluate("window.scrollBy(0, 500)")
        elif vision_action.action == "wait":
            await page.wait_for_timeout(2000)

        return {
            "current_step_index": current_step_idx + 1,
            "retry_count": 0,
            "error_message": None,
            "history": state["history"] + [f"Executed: {vision_action.action} on {vision_action.target_selector}"]
        }

    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ Execution error: {error_msg}")
        new_retry_count = state["retry_count"] + 1
        
        if new_retry_count > 3:
            print("❌ Max retries reached. Failing step.")
            return {
                "current_step_index": current_step_idx + 1,
                "retry_count": 0,
                "history": state["history"] + [f"Failed step after 3 retries: {error_msg}"]
            }
        
        return {
            "retry_count": new_retry_count,
            "error_message": f"Retry {new_retry_count}: {error_msg}. Look closer at the screen."
        }

def decide_next(state: AgentState):
    if state.get("error_message"):
        if state["retry_count"] <= 3:
            return "retry"
        else:
            return "continue"
    return "continue"

def check_finished(state: AgentState):
    if state["current_step_index"] >= len(state["plan"]):
        return "END"
    return "execute"

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("plan", plan_node)
    
    global browser_page
    async def execute_wrapper(state: AgentState):
        return await execute_node(state, browser_page)

    workflow.add_node("execute", execute_wrapper)
    workflow.add_node("check_done", lambda x: x)

    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "execute")
    
    workflow.add_conditional_edges(
        "execute",
        decide_next,
        {
            "retry": "execute",
            "continue": "check_done"
        }
    )
    
    workflow.add_conditional_edges(
        "check_done",
        check_finished,
        {
            "execute": "execute",
            "END": END
        }
    )

    return workflow.compile()

async def main():
    target_url = "https://news.ycombinator.com"
    user_task = "Find the title of the top story"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        global browser_page
        browser_page = await context.new_page()
        
        await browser_page.goto(target_url)
        
        initial_state = {
            "task": user_task,
            "plan": [],
            "current_step_index": 0,
            "history": [],
            "retry_count": 0,
            "error_message": None
        }

        graph = build_graph()
        
        print(f"🚀 Starting Agent for task: '{user_task}'")
        final_state = await graph.ainvoke(initial_state)
        
        print("\n--- Final History ---")
        for line in final_state["history"]:
            print(line)
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
