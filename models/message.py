from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

import database
from database import Base, SessionLocal
from models.user import get_user_by_email


# Initialize SQLAlchemy base
# Base = declarative_base()


# Define the Message model
class Message(Base):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True, autoincrement=True)  # Unique ID for each message
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(String, nullable=False)  # The message content
    to = Column(String, nullable=False)  # Recipient ('Chatbot' or 'User')
    from_ = Column(String, nullable=False)  # Sender ('User' or 'Chatbot')
    timestamp = Column(DateTime, nullable=False)  # Timestamp when the message is created

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Define relationship to User model
    user = relationship("User", back_populates="messages")

    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role}, from_={self.from_}, to={self.to}, timestamp={self.timestamp})>"




# Example: Save a new message
def save_message(role, content, to, from_, timestamp, email):
    the_user = get_user_by_email(email)
    new_message = Message(role=role, content=content, to=to, from_=from_, timestamp=timestamp, user=the_user)
    database.Session.add(new_message)
    database.Session.commit()


# Example: Query all messages
def get_all_messages():
    return database.Session.query(Message).all()
