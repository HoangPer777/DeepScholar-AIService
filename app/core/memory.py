# TODO: Implement memory stores for chat history and PDF processing status


class InMemoryStore:
    """TODO: In-memory key-value store (replace with Redis in production)"""
    def __init__(self):
        # TODO: Initialize storage
        pass

    def set(self, key, value):
        # TODO: Store value
        pass

    def get(self, key):
        # TODO: Retrieve value
        pass


class ChatHistoryStore:
    """TODO: Store chat conversations (replace with database in production)"""
    def __init__(self):
        # TODO: Initialize storage
        pass

    def append(self, session_id, item):
        # TODO: Add message to history
        pass

    def get(self, session_id):
        # TODO: Retrieve chat history
        pass


pdf_status_store = InMemoryStore()
chat_history_store = ChatHistoryStore()