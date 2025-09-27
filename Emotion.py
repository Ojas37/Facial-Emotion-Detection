import cv2
from deepface import DeepFace

def main():
    # Load OpenCV's Haar cascade for face detection (use absolute path)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise IOError("Failed to load haarcascade_frontalface_default.xml")
    
    # Open webcam video stream
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            
            try:
                # Analyze emotions on detected face region
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, silent=True)
                
                # Handle the case where result might be a list or dict
                if isinstance(result, list):
                    result = result[0]  # Take the first result if it's a list
                
                dominant_emotion = result['dominant_emotion']
                confidence = result['emotion'][dominant_emotion]
                
                # Draw rectangle and label on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{dominant_emotion} ({confidence:.1f}%)"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            except Exception as e:
                # Handle errors without breaking the loop
                print(f"Emotion analysis error: {str(e)}")
                # Draw rectangle without emotion label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Processing...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow('Emotion Detection', frame)
        
        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
