from all_imports import *
from backend import * 

def flow_work():
    workflow=StateGraph(AgentState)
    workflow.add_node("Supervisor",function_1)
    workflow.add_node("RAG",function_2)
    workflow.add_node("LLM",function_3)
    workflow.add_node("Web",func4)
    workflow.add_node("Validator",function_6)

    workflow.set_entry_point("Supervisor")
    workflow.add_conditional_edges(
        "Supervisor",
        router,
        {
            "RAG": "RAG",
            "LLM": "LLM",
            "web_crawler": "Web"
        }
    )
    workflow.add_edge("RAG","Validator")
    workflow.add_edge("LLM",END)
    workflow.add_edge("Web",END)
    workflow.add_edge("Validator",END)
    return workflow 

