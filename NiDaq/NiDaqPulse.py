import time
import nidaqmx.system


class NiDaqPulse(nidaqmx.Task):
    """Sets up Nidaq board to send a pulse, via digital output line"""
    def __init__(self, device_name="Dev2/port1/line7"):
        nidaqmx.Task.__init__(self)
        self.do_channels.add_do_chan(device_name)
        self.SLEEP_BETWEEN_PULSES_SEC = 0.01

    def give_pulse(self):
        """Write true, wait, write false. This will create a square pulse of value 1 (waiting controls the width)."""
        self.start()
        self.write(True)
        time.sleep(self.SLEEP_BETWEEN_PULSES_SEC)
        self.write(False)
        self.stop()

    @staticmethod
    def write_system_devices():
        """Help method. Show the devices exist on the system for each channel"""
        system = nidaqmx.system.System.local()
        if len(system.devices) == 1:
            MY_COMPUTER_DEVICE = system.devices[0].name  # should be like 'Dev1' etc
            print("Found only one device: ", MY_COMPUTER_DEVICE, ". Show specific physical channels for it:")
        else:
            print("Found several devices: ", system.devices)
            return

        print("Analog-input: ", [_ for _ in system.devices[MY_COMPUTER_DEVICE].ai_physical_chans])
        print("Analog-output: ", [_ for _ in system.devices[MY_COMPUTER_DEVICE].ao_physical_chans])
        print("Counter-input: ", [_ for _ in system.devices[MY_COMPUTER_DEVICE].ci_physical_chans])
        print("Counter-output: ", [_ for _ in system.devices[MY_COMPUTER_DEVICE].co_physical_chans])
        print("Digital-input: ", [_ for _ in system.devices[MY_COMPUTER_DEVICE].di_lines])
        print("Digital-output: ", [_ for _ in system.devices[MY_COMPUTER_DEVICE].do_ports])
