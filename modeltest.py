from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

model = ChatOllama(model="llama3.2", temprature=0.5)

prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant. Keep answers short."),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])

chain = prompt | model

history = []

# def ask(question, history):
#     reply = chain.invoke({"history": history, "question": question})
#     return reply.content

# q1 = "My name is Zaynul and I Love love to paly cricket and i like a girl shes name is Rasmlai."
# a1 = ask(q1, history)
# history += [("human", q1), ("ai", a1)]
# print("Turn 1:",a1)


# q2 = "Which sport do I Love?"
# a2 = ask(q2, history)
# history += [("human", q2), ("ai", a2)]
# print("Turn 2:",a2)


# q3 = "What is my name."
# print("Turn 3 (With history):",ask(q3,history))


# print("Turn 3 (no history):", ask(q3, []))

# q4 = "Which girl do I Love?"
# a4 = ask(q4, history)
# history += [("human", q4), ("ai", a4)]
# print("Turn 2:",a4)

# q5 = "You know who is Rasmlai! she is my girlfriend or not ?"
# a5 = ask(q5, history)
# history += [("human", q5), ("ai", a5)]
# print("Turn 5:",a5)

while True:
    question = input("Ask : ")

    if question.strip().lower() in ("quite", "exit"):
        print("Bye")
        break

    reply = chain.invoke({"history": history, "question": question})
    print("AI..", reply.content, "\n")

    history.append(("human", question))
    history.append(("ai", reply.content))
