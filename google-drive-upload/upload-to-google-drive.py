import os

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Go to script location
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)

gauth = GoogleAuth()
drive = GoogleDrive(gauth)

file1 = drive.CreateFile(
    {"title": "Hello.txt"}
)  # Create GoogleDriveFile instance with title 'Hello.txt'.
file1.SetContentString(
    "Hello World!"
)  # Set content of the file from given string.
file1.Upload()
