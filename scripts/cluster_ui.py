import sys

import gi
from gi.repository import GLib, Gtk

gi.require_version("Gtk", "4.0")


class ClusterUI(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.ClusterUI")
        GLib.set_application_name("Terrain Cluster UI")

    def do_activate(self):
        # Create a window
        window = Gtk.ApplicationWindow(application=self, title="Terrain Cluster")

        # Create a vertical box layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)
        window.set_child(vbox)

        ### Pickle file selector
        spf = SelectPickleFile(window)
        vbox.append(spf.get_component())

        # Add a spacer
        spacer = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        vbox.append(spacer)
        
        smf = SelectModelFile(window)
        vbox.append(smf.get_component())


        # Add another text entry box
        entry2 = Gtk.Entry()
        entry2.set_placeholder_text("Enter more text here...")
        vbox.append(entry2)

        # Add the first button
        button1 = Gtk.Button(label="Button 1")
        button1.connect("clicked", self.on_button1_clicked)
        vbox.append(button1)

        # Add the second button
        button2 = Gtk.Button(label="Button 2")
        button2.connect("clicked", self.on_button2_clicked)
        vbox.append(button2)

        # Show the window
        window.present()

    def on_button1_clicked(self, button):
        print("Button 1 clicked!")

    def on_button2_clicked(self, button):
        print("Button 2 clicked!")

class SelectPickleFile:
    def __init__(self, parent_window):
        self.parent_window = parent_window
        self.data_pkl_path = None

    def get_component(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)

        vbox.append(Gtk.Label(label="Data pickle file selected:"))

        # Pickle file status
        self.label = Gtk.Label(label="None")
        vbox.append(self.label)

        # Pickle file chooser button
        file_chooser_button = Gtk.Button(label="Open")
        file_chooser_button.connect("clicked", self.on_file_chooser_button_clicked)
        vbox.append(file_chooser_button)
        
        return vbox

    def on_file_chooser_button_clicked(self, button):
        dialog = Gtk.FileChooserDialog(
            title="Select Data Pickle File",
            transient_for=self.parent_window,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons("_Cancel", Gtk.ResponseType.CANCEL, "_Open", Gtk.ResponseType.ACCEPT)
        dialog.connect("response", self.on_file_chooser_response)
        dialog.show()

    def on_file_chooser_response(self, dialog, response):
        if response == Gtk.ResponseType.ACCEPT:
            file_path = dialog.get_file().get_path()
            if not file_path.endswith(".pkl"):
                self.label.set_markup(
                    "<span foreground='red'>Error: Selected file is not a .pkl file</span>"
                )
                dialog.destroy()
                return

            self.data_pkl_path = file_path
            self.label.set_markup(f"<span foreground='green'>{self.data_pkl_path}</span>")
        dialog.destroy()

class SelectModelFile:
    def __init__(self, parent_window):
        self.parent_window = parent_window
        self.model_path = None

    def get_component(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)

        vbox.append(Gtk.Label(label="Model file selected:"))

        # Pickle file status
        self.label = Gtk.Label(label="None")
        vbox.append(self.label)

        # Pickle file chooser button
        file_chooser_button = Gtk.Button(label="Open")
        file_chooser_button.connect("clicked", self.on_file_chooser_button_clicked)
        vbox.append(file_chooser_button)
        
        return vbox

    def on_file_chooser_button_clicked(self, button):
        dialog = Gtk.FileChooserDialog(
            title="Select PyTorch Model File",
            transient_for=self.parent_window,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons("_Cancel", Gtk.ResponseType.CANCEL, "_Open", Gtk.ResponseType.ACCEPT)
        dialog.connect("response", self.on_file_chooser_response)
        dialog.show()

    def on_file_chooser_response(self, dialog, response):
        if response == Gtk.ResponseType.ACCEPT:
            file_path = dialog.get_file().get_path()
            if not file_path.endswith(".pt"):
                self.label.set_markup(
                    "<span foreground='red'>Error: Selected file is not a .pt file</span>"
                )
                dialog.destroy()
                return

            self.model_path = file_path
            self.label.set_markup(f"<span foreground='green'>{self.model_path}</span>")
        dialog.destroy()

app = ClusterUI()
exit_status = app.run(sys.argv)
sys.exit(exit_status)
