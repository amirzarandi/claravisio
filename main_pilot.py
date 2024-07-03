import gpiod
import subprocess, sys
import os
import logging
from time import sleep
from datetime import datetime, timedelta
from picamera2 import Picamera2

sleep(5)

log_file = '/home/amir/main_pilot.log'
logging.basicConfig(
	filename=log_file,
	level=logging.INFO,
	format='%(asctime) s - %(levelname)s - %(message) s'
)

def filter_log_file(file_path, max_age_hours=24):
    """Filter out log entries older than max_age_hours."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Calculate the cutoff time
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        filtered_lines = []

        for line in lines:
            try:
                # Extract the timestamp from the log line
                timestamp_str = line.split(' - ')[0]
                log_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                # Keep the line if it is newer than the cutoff time
                if log_time > cutoff_time:
                    filtered_lines.append(line)
            except ValueError:
                # If timestamp parsing fails, keep the line (to avoid losing any unexpected format)
                filtered_lines.append(line)

        # Write the filtered lines back to the log file
        with open(file_path, 'w') as f:
            f.writelines(filtered_lines)
		

def main():
	filter_log_file(log_file)
	logging.info('Pilot started')
	
	try:
		RED_LED_PIN = 17
		BLUE_LED_PIN = 27
		GREEN_LED_PIN = 22
		BUTTON_PIN = 10

		command = "rclone copy -v /home/amir/Pictures ClaraDrive:claravisio_images"

		picam0 = Picamera2(0)
		picam0.start()

		chip = gpiod.Chip('gpiochip4')

		red_led_line = chip.get_line(RED_LED_PIN)
		blue_led_line = chip.get_line(BLUE_LED_PIN)
		green_led_line = chip.get_line(GREEN_LED_PIN)
		button_line = chip.get_line(BUTTON_PIN)

		red_led_line.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
		blue_led_line.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
		green_led_line.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
		button_line.request(consumer="Button", type=gpiod.LINE_REQ_DIR_IN)
		
		blue_led_line.set_value(1)
		green_led_line.set_value(1)
		red_led_line.set_value(1)
		sleep(1)
		blue_led_line.set_value(0)
		green_led_line.set_value(0)
		red_led_line.set_value(0)
		

		current_date = datetime.now().strftime('%Y-%m-%d')
		run_number = 1
		folder_name = f"/home/amir/Pictures/{current_date}_RUN{run_number}"

		while os.path.exists(folder_name):
			run_number += 1
			folder_name = f"/home/amir/Pictures/{current_date}_RUN{run_number}"

		os.makedirs(folder_name)
		os.makedirs(f"{folder_name}/A")
		os.makedirs(f"{folder_name}/B")
		logging.info(f"Folder created: {current_date}_RUN{run_number}")

		photo_number = 1
		is_A_or_B = True

		while True:
			green_led_line.set_value(0)
			blue_led_line.set_value(0)
			button_state = button_line.get_value()
			if button_state == 1:
				if(is_A_or_B):
					picam0.capture_file(f"{folder_name}/A/{current_date}_RUN{run_number}__{photo_number}.png")
					is_A_or_B = False
				else:
					picam0.capture_file(f"{folder_name}/B/{current_date}_RUN{run_number}__{photo_number}.png")
					is_A_or_B = True
					photo_number += 1
				green_led_line.set_value(1)
				sleep(0.5)
				green_led_line.set_value(0)
				try: 
					result = subprocess.run(command, check=True, text=True, shell=True, capture_output=True)
					logging.info(result.stderr)
					blue_led_line.set_value(1)
					sleep(0.5)
					blue_led_line.set_value(0)
				except subprocess.CalledProcessError as cpe:
					logging.info(cpe.stderr)
					red_led_line.set_value(1)


	except Exception as e:
		logging.error(f'An error occurred: {e}')
		red_led_line.set_value(1)
		sleep(10)
		
	finally:
		green_led_line.set_value(0)
		red_led_line.set_value(0)
		blue_led_line.set_value(0)
		red_led_line.release()
		green_led_line.release()
		blue_led_line.release()
		button_line.release()
		picam0.stop()
		picam0.stop_preview()
		logging.info('Program exited')

if __name__ == "__main__":
	main()
