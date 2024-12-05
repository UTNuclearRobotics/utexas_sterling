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

        # Add a text entry box
        vbox.append(Gtk.Label(label="Data pickle file selected:"))
        
        self.data_pkl_label = Gtk.Label(label="None")
        vbox.append(self.data_pkl_label)
        
        # Add a file chooser button
        file_chooser_button = Gtk.Button(label="Select a File")
        file_chooser_button.connect("clicked", self.on_file_chooser_button_clicked)
        vbox.append(file_chooser_button)

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
        
    def on_file_chooser_button_clicked(self, button):
        dialog = Gtk.FileChooserDialog(
            title="Select data file",
            transient_for=self.get_active_window(),
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            "_Cancel", Gtk.ResponseType.CANCEL,
            "_Open", Gtk.ResponseType.ACCEPT
        )
        dialog.connect("response", self.on_file_chooser_response)
        dialog.show()

    def on_file_chooser_response(self, dialog, response):
        if response == Gtk.ResponseType.ACCEPT:
            file_path = dialog.get_file().get_path()
            if not file_path.endswith('.pkl'):
                self.data_pkl_label.set_markup("<span foreground='red'>Error: Selected file is not a .pkl file</span>")
                dialog.destroy()
                return
            
            self.data_pkl = file_path
            self.data_pkl_label.set_text(self.data_pkl)
        dialog.destroy()

    def on_button1_clicked(self, button):
        print("Button 1 clicked!")

    def on_button2_clicked(self, button):
        print("Button 2 clicked!")


app = ClusterUI()
exit_status = app.run(sys.argv)
sys.exit(exit_status)