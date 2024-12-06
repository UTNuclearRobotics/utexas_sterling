import sys

import gi
from gi.repository import GLib, Gtk, GdkPixbuf
from cluster import Cluster, PatchRenderer

gi.require_version("Gtk", "4.0")


class ClusterUI(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.ClusterUI")
        GLib.set_application_name("Terrain Cluster UI")

    def do_activate(self):
        # Create a window
        window = Gtk.ApplicationWindow(application=self, title="Terrain Cluster")
        window.set_resizable(True)
        window.set_default_size(800, 600)

        # Create a scrolled window
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        window.set_child(scrolled_window)

        # Create a vertical box layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)
        scrolled_window.set_child(vbox)

        spf = SelectPickleFile(window)
        vbox.append(spf.get_component())

        vbox.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        smf = SelectModelFile(window)
        vbox.append(smf.get_component())

        vbox.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        gc = GenerateClusters(window, spf, smf)
        vbox.append(gc.get_component())

        # Show the window
        window.present()


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
                self.label.set_markup("<span foreground='red'>Error: Selected file is not a .pkl file</span>")
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
                self.label.set_markup("<span foreground='red'>Error: Selected file is not a .pt file</span>")
                dialog.destroy()
                return

            self.model_path = file_path
            self.label.set_markup(f"<span foreground='green'>{self.model_path}</span>")
        dialog.destroy()


class GenerateClusters:
    def __init__(self, parent_window, spf, smf):
        self.parent_window = parent_window
        self.spf = spf
        self.smf = smf

        # self.vbox
        # self.entry_clusters
        # self.entry_iterations

    def get_component(self):
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.vbox.set_margin_top(10)
        self.vbox.set_margin_bottom(10)
        self.vbox.set_margin_start(10)
        self.vbox.set_margin_end(10)

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

        label = Gtk.Label(label="Number of Clusters:")
        hbox.append(label)

        self.entry_clusters = Gtk.Entry()
        self.entry_clusters.set_placeholder_text("Enter number of clusters...")
        self.entry_clusters.set_text("5")
        hbox.append(self.entry_clusters)

        self.vbox.append(hbox)

        hbox_iterations = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

        label_iterations = Gtk.Label(label="Number of Iterations:")
        hbox_iterations.append(label_iterations)

        self.entry_iterations = Gtk.Entry()
        self.entry_iterations.set_placeholder_text("Enter number of iterations...")
        self.entry_iterations.set_text("100")
        hbox_iterations.append(self.entry_iterations)

        self.vbox.append(hbox_iterations)

        button = Gtk.Button(label="Generate Clusters")
        button.connect("clicked", self.on_button_clicked)
        self.vbox.append(button)

        return self.vbox

    def on_button_clicked(self, button):
        data_pkl_path = self.spf.data_pkl_path
        model_path = self.smf.model_path

        if data_pkl_path is None or model_path is None:
            error_dialog = Gtk.MessageDialog(
                transient_for=self.parent_window,
                modal=True,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text="Please select both a data pickle file and a model file before generating clusters.",
            )
            error_dialog.show()
            error_dialog.connect("response", lambda dialog, response: dialog.destroy())
            return

        num_clusters = self.entry_clusters.get_text()
        num_iterations = self.entry_iterations.get_text()

        try:
            num_clusters = int(num_clusters)
            num_iterations = int(num_iterations)
            if num_clusters <= 0 or num_iterations <= 0:
                raise ValueError
        except ValueError:
            error_dialog = Gtk.MessageDialog(
                transient_for=self.parent_window,
                modal=True,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text="Please enter valid positive numbers for clusters and iterations.",
            )
            error_dialog.show()
            error_dialog.connect("response", lambda dialog, response: dialog.destroy())
            return

        # Generate clusters
        cluster = Cluster(data_pkl_path, model_path)
        all_cluster_image_indices = cluster.generate_clusters(num_clusters, num_iterations)

        # Render clusters
        rendered_clusters = PatchRenderer.render_clusters(all_cluster_image_indices, cluster.patches)

        images = []
        for i, cluster in enumerate(rendered_clusters):
            images.append(PatchRenderer.image_grid(cluster))

        def pil_to_pixbuf(pil_image):
            data = pil_image.tobytes()
            width, height = pil_image.size
            return GdkPixbuf.Pixbuf.new_from_data(
                data,
                GdkPixbuf.Colorspace.RGB,
                pil_image.mode == "RGBA",
                8,
                width,
                height,
                pil_image.mode == "RGBA" and width * 4 or width * 3,
            )

        hbox_images = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

        for i, image in enumerate(images):
            pixbuf = pil_to_pixbuf(image)
            image_widget = Gtk.Image.new_from_pixbuf(pixbuf)
            image_widget.set_size_request(200, 200)  # Set the desired width and height

            text_field = Gtk.Entry()
            text_field.set_placeholder_text(f"Label cluster{i + 1}...")
            text_field.set_text(f"cluster{i + 1}")

            ranking_field = Gtk.Entry()
            ranking_field.set_placeholder_text(f"Rank cluster{i + 1}...")
            ranking_field.set_text("0")

            vbox_image = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
            vbox_image.append(image_widget)
            vbox_image.append(Gtk.Label(label=f"Label {i + 1}"))
            vbox_image.append(text_field)
            vbox_image.append(Gtk.Label(label="Preference"))
            vbox_image.append(ranking_field)

            hbox_images.append(vbox_image)

        self.vbox.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))
        self.vbox.append(hbox_images)


app = ClusterUI()
exit_status = app.run(sys.argv)
sys.exit(exit_status)
