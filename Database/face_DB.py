from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
engine = create_engine('sqlite:///kosen_face_db.sqlite3', echo=True)

class User(Base):
    __tablename__ = 'faces'
    id = Column(Integer, primary_key=True)
    student_name = Column(String)
    course = Column(String)
    embedding = Column(LargeBinary)
    sex = Column(String)
    #変更

    def save_face_data(name, course, embedding, sex):
        new_face = User(student_name=name, course=course, embedding=embedding, sex=sex)
        session.add(new_face)
        session.commit()

    def get_all_faces():
        return session.query(User).all()

    def update_face_embedding(id,embedding):
        face = session.query(User).filter_by(id=id).first()
        if face:
            face.add_embedding(new_embedding)
            session.commit()
Base.metadata.create_all(engine)