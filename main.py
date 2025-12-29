from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import pandas as pd

model = OllamaLLM(model="qwen3:4b")

template = """
You are an expert assistant for answering questions about machine maintenance and service records.

Here are some relevant machine service records:
{reviews}

Please answer the following question using only the information from the records above. If the answer is not present, say so clearly.

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

df = pd.read_csv("machine_service.csv")

def fallback_answer(question):
    # Fallback for frequency/statistics and record count questions
    q = question.lower()
    if ("how many" in q or "total number" in q or "count" in q) and ("record" in q or "service" in q):
        count = len(df)
        return f"There are {count} machine service records in the database."
    if "most frequent" in q or "most common" in q:
        counts = df["Problem_Type"].value_counts()
        if not counts.empty:
            most_common = counts.idxmax()
            freq = counts.max()
            return f"The most frequent problem type is '{most_common}' with {freq} occurrences."
        else:
            return "No problem types found in the data."
    if ("sum" in q or "total" in q) and ("cost" in q or "service cost" in q):
        total_cost = df["Cost"].sum()
        return f"The sum of all service costs in the records is {total_cost:.2f}."
    if ("average" in q or "mean" in q) and ("cost" in q or "service cost" in q):
        avg_cost = df["Cost"].mean()
        return f"The average service cost in the records is {avg_cost:.2f}."
    if ("minimum" in q or "lowest" in q or "smallest" in q) and ("cost" in q or "service cost" in q):
        min_cost = df["Cost"].min()
        return f"The minimum service cost in the records is {min_cost:.2f}."
    if ("maximum" in q or "highest" in q or "largest" in q) and ("cost" in q or "service cost" in q):
        max_cost = df["Cost"].max()
        return f"The maximum service cost in the records is {max_cost:.2f}."
    return None

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    reviews = retriever.invoke(question)
    if not reviews:
        fallback = fallback_answer(question)
        if fallback:
            print(fallback)
            continue
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)