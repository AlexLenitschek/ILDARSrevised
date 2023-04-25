import os

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Go to script location
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)

gauth = GoogleAuth()
drive = GoogleDrive(gauth)


# helper function for comparing the ages of google drive folders
def creation_date(folder):
    return folder["createdDate"]


# hardcoded id to google drive folder that is accessed
RESULTS_FOLDER_ID = "1A_pZ9KxUDk7Q6K4Y2dNfAh1B-LV-NafP"

# Create Folder with most current exported csv folder
res_folders = os.listdir("../results")
latest_res = max(
    [f"../results/{folder}" for folder in res_folders], key=os.path.getmtime
)
dirname = os.path.basename(latest_res)
tl_folder_data = {
    "title": dirname,
    "parents": [{"id": RESULTS_FOLDER_ID}],
    "mimeType": "application/vnd.google-apps.folder",
}
parent_folder = drive.CreateFile(tl_folder_data)
parent_folder.Upload()
res_folders_gdrive = drive.ListFile(
    {"q": f"'{RESULTS_FOLDER_ID}' in parents and trashed=false"}
).GetList()
res_folders_gdrive.sort(key=creation_date, reverse=True)
data_folder_id = None
for folder in res_folders_gdrive:
    if folder["title"] == dirname:
        data_folder_id = folder["id"]
        break

# parse all result csvs and upload them to drive
csv_sources = os.listdir(latest_res)
for csv in csv_sources:
    name = os.path.basename(csv)
    drive_file = drive.CreateFile(
        {"title": name, "parents": [{"id": data_folder_id}]}
    )
    drive_file.SetContentFile(f"../results/{dirname}/{csv}")
    drive_file.Upload()
