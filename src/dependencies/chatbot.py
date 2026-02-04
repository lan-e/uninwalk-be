from src.services.chatbot_service import ChatbotService

def get_chatbot_service() -> ChatbotService:
    from src.app import chatbot_service
    return chatbot_service