from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

model = OllamaLLM(model="qwen3:4b")

# ------------------------------------------------------------------------------
# 1. RAG SETUP (Qualitative / Specific Lookup)
# ------------------------------------------------------------------------------
rag_template = """
You are a highly knowledgeable and precise assistant specializing in machine maintenance and service records. Your task is to answer questions using only the provided service records below.

Instructions:
- Use only the information from the records to answer. Do not make assumptions or fabricate details.
- If the answer is not found in the records, clearly state: "The answer is not present in the provided records."
- When possible, cite specific details (such as dates, machine IDs, problem types, costs, or service statuses) from the records in your answer.
- If the question asks for a summary, provide a concise and accurate summary based on the records.
- Format your answer in clear, complete sentences.

Relevant machine service records:
{reviews}

Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = rag_prompt | model

# ------------------------------------------------------------------------------
# 2. PANDAS MEMORY LOADING
# ------------------------------------------------------------------------------
df = pd.read_csv("machine_service.csv")

# ------------------------------------------------------------------------------
# 3. DYNAMIC PANDAS ANALYST (Quantitative / Aggregations)
# ------------------------------------------------------------------------------
def query_pandas_agent(question, dataframe):
    """
    Asks the LLM to generate python code to query the dataframe.
    Executes the code and returns the result.
    """
    schema = dataframe.dtypes.to_string()
    
    analyst_template = f"""
You are a Python Pandas expert.
You have a pandas DataFrame named 'df' with the following columns and data types:
{schema}

User Question: {{question}}

Write a snippet of Python code to calculate the answer.
- Assign the final answer to a variable named `result`.
- The answer should be a simple string, number, or list.
- Use `df` directly.
- ONLY output the Python code. No markdown, no explanations.
- Handle potential empty data gracefully.
"""
    analyst_prompt = ChatPromptTemplate.from_template(analyst_template)
    analyst_chain = analyst_prompt | model
    
    try:
        # Generate Code
        code_response = analyst_chain.invoke({"question": question})
        
        # Clean Code
        cleaned_code = code_response.replace("```python", "").replace("```", "").strip()
        
        # Execution Context
        local_context = {"df": dataframe, "pd": pd}
        
        # Execute
        exec(cleaned_code, {}, local_context)
        
        # Retrieve result
        answer = local_context.get("result", None)
        if answer is not None:
            return str(answer)
        return None
        
    except Exception as e:
        # Fail silently to allow fallback to RAG
        return None

# ------------------------------------------------------------------------------
# 4. MAIN INTERACTION LOOP
# ------------------------------------------------------------------------------
print("\nSystem ready. Ask any question about the machine service records.")

while True:
    print("\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n")
    if question.lower() in ["q", "quit", "exit"]:
        break

    # HEURISTIC: Check if it's likely a data aggregation question
    data_keywords = [
        "how many", "count", "total", "sum", "average", "mean", "median",
        "min", "max", "most", "least", "top", "bottom", "list", "which",
        "trend", "cost", "hours", "compare", "frequency", "times", "percentage"
    ]
    
    is_data_query = any(k in question.lower() for k in data_keywords)
    answer_found = False
    
    # Strategy 1: Attempt Logic/Code Answer for Stats
    if is_data_query:
        # print("Analyzing data...")
        pandas_result = query_pandas_agent(question, df)
        if pandas_result:
            print(f"Answer: {pandas_result}")
            # If the answer is short/numeric, we might want to let the RAG elaborate, 
            # but usually a direct stat answer is what the user wants.
            answer_found = True
    
    # Strategy 2: Attempt RAG for specific details or if Analyst failed
    if not answer_found:
        # print("Searching records...")
        reviews = retriever.invoke(question)
        if reviews:
            formatted_reviews = "\n\n".join([d.page_content for d in reviews])
            rag_response = rag_chain.invoke({"reviews": formatted_reviews, "question": question})
            print(rag_response)
        else:
            print("I could not find relevant information in the records.")