from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship, scoped_session

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

    # Foreign key for session
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    session = relationship("Session", back_populates="messages")

    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role}, from_={self.from_}, to={self.to}, timestamp={self.timestamp})>"




# Example: Save a new message
def save_message(role, content, to, from_, timestamp, email, session_id):
    the_user = get_user_by_email(email)
    dbsession = scoped_session(SessionLocal)
    try:
        new_message = Message(role=role, content=content, to=to, from_=from_, timestamp=timestamp, user=the_user, session_id=session_id)
        dbsession.add(new_message)
        dbsession.commit()
    except:
        dbsession.rollback()
        raise
    finally:
        dbsession.remove()

# def save_message(role, content, to, from_, timestamp, email, session_id):
#     the_user = get_user_by_email(email)
#     new_message = Message(role=role, content=content, to=to, from_=from_, timestamp=timestamp, user=the_user, session_id=session_id)
#     database.Session.add(new_message)
#     database.Session.commit()


# # Example: Query all messages
# def get_all_messages():
#     return database.Session.query(Message).all()
