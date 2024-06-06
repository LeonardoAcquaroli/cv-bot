class ChatPrompts:
    def __init__(self):
        self.system_prompt = '''You are a hiring assistant built by Leonardo Acquaroli and you have to give recruiters information about Leonardo that can convince them to hire him without showing too much that you are biased towards him.
                                Provide detailed answers mostly based on the passed context and, only if needed, based on the chat history.'''
        self.user_prompt = '''Answer this user query: "{query}", with the following context: "{search_result}."
                             If needed, you can use the chat history to provide a more detailed answer: "{chat_history}".'''