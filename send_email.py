import csv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def get_license_plates(csv_file):
    license_plates = []
    # Open the CSV file and read the license plates from the "License Plate" column
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            license_plates.append(row['License Plate'])
    return license_plates

def send_email(sender_email, receiver_emails, password, violation_details):
    # Email configuration
    subject = 'Speed Violation Alert'

    # Sending the email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Gmail SMTP server and port
        server.starttls()
        server.login(sender_email, password)

        for receiver_email, details in zip(receiver_emails, violation_details):
            # Email content
            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = receiver_email
            message['Subject'] = subject

            body = f"Dear User,\n\nYou have violated speed rules.\n\nViolation Details:\n{details}\n\nPlease drive safely.\n\nRegards,\nYour Traffic Management Team"
            message.attach(MIMEText(body, 'plain'))

            text = message.as_string()
            server.sendmail(sender_email, receiver_email, text)
            print(f"Email sent successfully to {receiver_email}!")
    except Exception as e:
        print(f"Failed to send emails: {e}")
    finally:
        server.quit()

# Example usage:
if __name__ == "__main__":
    sender_email = 'sjn4433@gmail.com'
    receiver_emails = ['sajinsjn4433@gmail.com', 'sajin6427@gmail.com']  # List of receiver emails
    password = 'ghkr snfe wupy xfxh'
    csv_file = "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR-Modified/license_plate_info.csv"
    
    # Fetch license plate numbers from CSV file
    license_plates = get_license_plates(csv_file)
    
    # Create violation details for each license plate
    violation_details = [f'Vehicle with license plate {plate} exceeded the speed limit.' for plate in license_plates]
    
    send_email(sender_email, receiver_emails, password, violation_details)
