# Pupil Labs Neon Syntalos Module

Streams video from Pupil Labs Neon glasses to Syntalos.

## Known issues

Keep the phone screen on!
In princle the Companion App will keep the phone awake while the App is opened.
Turning off the screen of the phone while a stream is running seemed to cause the Neon Companion App to crash ocasionally.
When it happens, it requires to manually close the App on the phone and restart it. Confirm the glasses are recognized and the camera is working in App before attempting to run the Syntalos module again.

## Connecting to the glasses

The module will try to automatically discover the Neon companion app running on a Smartphone connected to the glasses.
This can fails on certain networks, you can enter the IP address of the glasses manually in the module settings.
To find the IP address of the Smartphone running the Neon companion app, open the app on the main screen and press the phone icon under the settings icon.

## Running in a Virtual Machine

At least for UTM VMs you have to use a Bridged Network device instead of Shared Network.
Accessing the streamed frames did not work with Shared Network. Some issue with the RTC protocol used by the Neon. Using a Bridged Network should put the VM on the same network as the host and the glasses should be auto-discoverable too.
