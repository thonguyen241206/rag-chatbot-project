from rag.rag_chatbot import ask_rag

while True:
    q = input("Ask: ")
    if q == "exit":
        break

    print(ask_rag(q))
