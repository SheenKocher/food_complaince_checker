from all_imports import *

from langchain_core.vectorstores import VectorStoreRetriever

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    model: ChatGroq
    retriever: VectorStoreRetriever

class SupervisorLabel(BaseModel):
    Topic: Literal["llm_call", "rag_fssai", "web_crawler"]

class ValidationParser(BaseModel):
    Value: Literal["true", "false", "not related to guidelines"] = Field(
        description="Validation criteria: true (valid), false (non-compliant), or not related to guidelines"
    )
    Reasoning: str = Field(
        description="Reasoning behind the validation decision"
    )

parser = PydanticOutputParser(pydantic_object=SupervisorLabel)

def function_1(state: dict) -> dict:
    model = state["model"]
    question = state["messages"][-1]
    print("Question:", question)
    template = """
    You are a query classifier for a food compliance system. 
    Classify the user query into one of the following categories:
    - rag_fssai → factual compliance questions (e.g., 'What % of sucralose is allowed','...Are these ingridients ok to use?')
    - web_crawler → real-time regulation queries (e.g., 'Any recent ban on ashwagandha')
    - llm_call → vague or general queries requiring deeper reasoning
    Respond only with the category name using this format:
    {format_instructions}
    User query: {question}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    # Chain: Prompt → Model → OutputParser
    chain = prompt | model | parser

    response = chain.invoke({"question": question})

    print("Parsed response:", response)

    return {"messages": [response.Topic]}

def router(state:AgentState):
    # print("-> ROUTER ->")
    
    last_message=state["messages"][-1]
    print("last_message:", last_message)
    
    if "llm_call" in last_message.lower():
        return "LLM"
    elif "rag_fssai" in last_message.lower():
        return "RAG"
    else:
        return "web_crawler"
    
# Format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def function_2(state: dict) -> dict:
    print("-> RAG Call ->")
    model = state["model"]
    retriever = state["retriever"]
    question = state["messages"][0]

    # RAG-style compliance-focused prompt
    prompt = PromptTemplate(
        template=(
            "You are an FSSAI compliance assistant. Use the following context to answer the question.\n"
            "- Only answer using the context.\n"
            "- If the answer is not in the context,use the available context to answer the best in your knowledge'\n"
            "- Keep your response concise (max 3 sentences).\n\n"
            "Question: {question}\n"
            "Context:\n{context}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"]
    )

    # RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

    # Final result from chain
    answer = rag_chain.invoke(question)
    return  {"messages": [answer]}

def function_3(state:AgentState):
    print("-> LLM Call ->")
    model = state["model"]
    question = state["messages"][0]
    
    # Normal LLM call
    complete_query = f"""
    You are an intelligent assistant with access to real-world knowledge and FSSAI compliance regulations.
    Your task is to answer the following user question based on your understanding of the real world. Be concise, fact-based, and only include information you're confident about.
    User Question:
    {question}
    """

    response = model.invoke(complete_query)
    return {"messages": [response.content]}
    
def func4(state:AgentState):
    search = DuckDuckGoSearchRun()
    question = state["messages"][0]
    response = search.invoke(question)
    return {"messages": [response]}

parsers = PydanticOutputParser(pydantic_object=ValidationParser)

parsers = PydanticOutputParser(pydantic_object=ValidationParser)

validator_prompt = PromptTemplate(
    template="""
You are an FSSAI compliance validator.

Your job is to assess whether the provided answer, which was generated using retrieved FSSAI documents, is valid and complete under FSSAI regulations.

Please strictly follow the rules below:
- Respond only in the JSON format provided.
- Use "true" if the answer is factually correct and compliant with FSSAI standards.
- Use "false" if the answer is vague, missing key compliance rules, or violates guidelines.
- Use "not related to guidelines" if the user’s question is outside the scope of FSSAI (e.g., business advice, branding, international regulations).

Respond in this exact JSON format:

{format_instructions}

User Query:
{user_query}

RAG Answer:
{rag_answer}
""",
    input_variables=["user_query", "rag_answer"],
    partial_variables={"format_instructions": parsers.get_format_instructions()}
)

def function_6(state: dict) -> dict:
    print("-> FSSAI Validator with Structured Output + Return LLM Output ->")
    model = state["model"]
    user_query = state["messages"][0]
    rag_answer = state["messages"][-1]

    # LLM chain
    chain = validator_prompt | model | parsers
    result = chain.invoke({
        "user_query": user_query,
        "rag_answer": rag_answer
    })

    print("Parsed Output:", result)

    validation_passed = result.Value.lower().strip() == "true"

    return {
        "validation_passed": validation_passed,
        "messages": [rag_answer],
        "llm_output": {
            "Value": result.Value,
            "Reasoning": result.Reasoning
        }
    }