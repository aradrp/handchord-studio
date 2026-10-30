"""Optional MIDI output (mido + python-rtmidi). Silently disabled if not installed."""


class MidiOut:
    def __init__(self, port_name=None, enable=True):
        self.port = None
        self.label = "off"
        self._on = set()
        if not enable:
            return
        try:
            import mido
        except ImportError:
            print("[!] mido is not installed: pip install mido python-rtmidi")
            return
        try:
            names = mido.get_output_names()
            chosen = None
            if port_name:
                matches = [n for n in names if port_name.lower() in n.lower()]
                chosen = matches[0] if matches else None
                if chosen is None:
                    print(f"[!] Port '{port_name}' not found. Available ports: {names}")
            elif names:
                chosen = names[0]
            if chosen:
                self.port = mido.open_output(chosen)
                self.label = chosen
            else:
                # On macOS/Linux a virtual port can be created (Windows: use loopMIDI)
                self.port = mido.open_output("HandChord", virtual=True)
                self.label = "HandChord (virtual)"
        except Exception as e:  # noqa
            print(f"[!] Could not open MIDI: {e}")
            self.port = None

    @property
    def enabled(self):
        return self.port is not None

    @staticmethod
    def list_ports():
        try:
            import mido
            return mido.get_output_names()
        except Exception:  # noqa
            return []

    def set_notes(self, notes, velocity=90):
        if not self.enabled:
            return
        import mido
        new = set(notes)
        for n in self._on - new:
            self.port.send(mido.Message("note_off", note=int(n)))
        for n in new - self._on:
            self.port.send(mido.Message("note_on", note=int(n), velocity=int(velocity)))
        self._on = new

    def all_off(self):
        self.set_notes([])

    def close(self):
        if self.enabled:
            self.all_off()
            self.port.close()
