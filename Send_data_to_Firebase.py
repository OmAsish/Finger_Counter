import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase Admin SDK
cred = credentials.Certificate("C:\\Users\\Purna\\Desktop\\Old project clone\\Finger_Counter\\new-project-2c91b-firebase-adminsdk-ujwez-e475c8599f.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://new-project-2c91b-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Replace with your database URL
})

# Function to send an array to Firebase Realtime Database
def send_array_to_firebase(path, array):
    try:
        ref = db.reference(path)  # Define the path in the database
        ref.set(array)  # Send the array to the database
        print("Array sent successfully!")
    except Exception as e:
        print("Error:", e)

# Example usage
if __name__ == "__main__":
    # Define the array of integers
    integer_array = [10, 20, 30, 40, 55]
    
    # Path in the database where data will be stored
    database_path = "my_data/integers"
    
    # Send the array to Firebase
    send_array_to_firebase(database_path, integer_array)

