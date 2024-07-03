### Standard Operating Procedure (SOP) for Capturing and Uploading Photos on Raspberry Pi

#### Purpose:
This SOP outlines the steps to operate a Raspberry Pi setup for capturing photos and uploading them to Google Drive using the rclone tool. The procedure involves pushing a button to capture images and waiting for an LED indication to confirm upload completion.

#### Equipment Needed:
- Raspberry Pi with necessary peripherals
- Button connected to GPIO
- Red, Blue, and Green LEDs connected to GPIO
- Camera module compatible with Picamera2
- Internet connection for Google Drive upload

#### Software Requirements:
- Python
- gpiod library
- picamera2 library
- rclone configured for Google Drive

#### Pre-setup Instructions:
1. **Install necessary libraries and tools:**
   ```sh
   sudo apt update
   sudo apt install -y python3 python3-pip gpiod python3-gpiozero
   pip3 install picamera2
   ```

2. **Install rclone and configure with Google Drive:**
   ```sh
   curl https://rclone.org/install.sh | sudo bash
   rclone config
   ```
   Follow the prompts to configure rclone with your Google Drive account.

#### Script Setup:
**Save the Python script `main_pilot.py` to your Home folder and create a `main_pilot.log` file to accompany it. **


#### Running the Script on Boot:
To ensure the script runs on boot, use `crontab`.

1. **Open the crontab editor:**
   ```sh
   crontab -e
   ```

2. **Add the following line to run the script on boot:**
   ```sh
   @reboot /usr/bin/python3 /home/amir/main_pilot.py
   ```

#### Operating Procedure:
1. **Start the Raspberry Pi:**
   - The script will run automatically if configured via `crontab`. Otherwise, you can manually run it using:
     ```sh
     python3 /home/amir/main_pilot.py
     ```

2. **Capture Photos:**
   - **First Button Press:** Press the button once to capture the initial photo.
   - **Create Fog:** Create the desired fog effect.
   - **Second Button Press:** Press the button again to capture the second photo.

3. **Wait for Upload Completion:**
   - **Blue LED:** Wait for the blue LED to turn on, indicating that the upload to Google Drive is in progress. Once it turns off, the upload is complete.

4. **Error Indication:**
   - **Red LED:** If the red LED turns on, an error has occurred. Check the logs for details.

#### Note:
- Ensure your Raspberry Pi has a stable internet connection for successful uploads.
- Regularly check the `pilot_main.log` file for any errors or warnings.