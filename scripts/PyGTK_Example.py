import sys

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk


class MyApplication(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.MyGtkApplication")
        GLib.set_application_name('My Gtk Application')

    def do_activate(self):
        # Create a window
        window = Gtk.ApplicationWindow(application=self, title="Hello World")

        # Create a vertical box layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)
        window.set_child(vbox)

        # Add a text entry box
        entry1 = Gtk.Entry()
        entry1.set_placeholder_text("Enter text here...")
        vbox.append(entry1)

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


app = MyApplication()
exit_status = app.run(sys.argv)
sys.exit(exit_status)


'''
[
    []
    []
    []
]

At top
[Text box - # Clusters]
[Button - Generate Clusters]

Let's suppose that each cluster has a row of data, picture each of these horizontally
[Text box - Name (think, road, sidewalk, grass, dirt)] 
[Picture x <However many, assume 5>] OR
[Picture 1]
[Picture 2]
[Picture ...]
[Text box - Preference Ranking]
'''