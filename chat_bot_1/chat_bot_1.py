from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def handle_conversation():
    model = OllamaLLM(model="phi3:latest")  # Changed to use phi3:latest
    template = """
    Answer the question below. Here is the conversation history:
    {context}
    Question: {question}
    Answer: """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    context = ""
    print("Welcome to the AI chat bot. Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        result = chain.invoke({"context": context, "question": user_input})
        bot_response = result.content if hasattr(result, 'content') else str(result)
        print(f"Bot: {bot_response}")
        
        context += f"\nUser: {user_input}\nAI: {bot_response}"

if __name__ == "__main__":
    handle_conversation()