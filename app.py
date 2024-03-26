from flask_ngrok import run_with_ngrok
from flask import Flask, render_template
import csv

template_folder = "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR-Modified/template"
static_folder = "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR-Modified/static"
csv_file_path = "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR-Modified/license_plate_info.csv"
app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
run_with_ngrok(app)

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        # Skip header row
        next(csv_reader)
        for row in csv_reader:
            # Convert speed value to float
            try:
                speed = float(row[3])
            except ValueError:
                # Skip if speed cannot be converted to float
                continue
            # Check if speed is greater than 80km/h
            if speed > 80:
                data.append(row)  # Append entire row if speed is greater than 80km/h
    return data

@app.route("/")
def home():
    # Read data from CSV file with speed > 80km/h
    data_from_csv = read_csv_file(csv_file_path)
    # Extract label, timestamp1, timestamp2, and speed from filtered data
    label = [row[0] for row in data_from_csv]
    timestamp1 = [row[1] for row in data_from_csv]
    timestamp2 = [row[2] for row in data_from_csv]
    speed = [row[3] for row in data_from_csv]
    # Pass data to template
    return render_template('index.html', label=label, timestamp1=timestamp1, timestamp2=timestamp2, speed=speed, l=len(label))

if __name__ == "__main__":
    app.run()
