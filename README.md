# Pupil Labs Neon Syntalos Module

Streams video from Pupil Labs Neon glasses to Syntalos.

## Known issues

Keep the phone screen on!
In princle the Companion App will keep the phone awake while the App is opened.
Turning off the screen of the phone while a stream is running seemed to cause the Neon Companion App to crash ocasionally.
When it happens, it requires to manually close the App on the phone and restart it. Confirm the glasses are recognized and the camera is working in App before attempting to run the Syntalos module again.

## Connecting to the glasses

Enter the IP address of the phone running the Neon Companion app in the module settings before starting acquisition.
To find the IP address, open the Companion app on the main screen and press the phone icon under the settings icon.

## Running in a Virtual Machine

At least for UTM VMs you have to use a Bridged Network device instead of Shared Network.
Accessing the streamed frames did not work with Shared Network. Some issue with the RTC protocol used by the Neon. Using a Bridged Network should put the VM on the same network as the host and the glasses should be auto-discoverable too.
