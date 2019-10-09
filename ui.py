import cv2

class Text:

    def draw(self, string: str, frame: object, color: object) -> object:
        string = str(string)
        cv2.putText(frame, string, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 1, cv2.LINE_4)
