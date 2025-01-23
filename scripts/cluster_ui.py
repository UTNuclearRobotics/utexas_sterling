import sys
import os

import gi
from gi.repository import GLib, Gtk, GdkPixbuf
from cluster import Cluster, PatchRenderer

gi.require_version("Gtk", "4.0")

script_dir = os.path.dirname(os.path.realpath(__file__))


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

        self.generated_flag = False

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
        
        save_path = os.path.join(os.path.dirname(data_pkl_path), "clusters")
        if not os.path.exists(save_path):
            # Make directory if it doesn't exist
            os.makedirs(save_path)

        # Generate clusters
        cluster = Cluster(data_pkl_path, model_path)
        all_cluster_image_indices = cluster.generate_clusters(
            num_clusters,
            num_iterations,
            save_model_path=os.path.join(os.path.dirname(data_pkl_path), "clusters", "kmeans_model.pkl"),
            save_scaler_path=os.path.join(os.path.dirname(data_pkl_path), "clusters", "scaler.pkl"),
        )

        # Dynamic grid size
        min_length = min(len(list) for index, list in enumerate(all_cluster_image_indices))
        grid_size = 10
        while min_length < grid_size**2:
            grid_size -= 1

        # Render clusters
        rendered_clusters = PatchRenderer.render_clusters(all_cluster_image_indices, cluster.patches)

        self.images = []
        for i, cluster in enumerate(rendered_clusters):
            self.images.append(PatchRenderer.image_grid(cluster, grid_size))

        def numpy_to_pixbuf(array):
            """
            Convert a NumPy array to a GdkPixbuf.Pixbuf object.
            Supports RGB and RGBA formats.
            """
            # Ensure input is a 3D array
            if array.ndim not in (3,):
                raise ValueError("Input must be a 3D NumPy array with shape (height, width, channels)")

            height, width, channels = array.shape

            # Validate channels
            if channels not in (3, 4):  # RGB or RGBA
                raise ValueError("Array must have 3 (RGB) or 4 (RGBA) channels")

            # Convert array to bytes
            data = array.tobytes()

            # Define rowstride (bytes per row)
            rowstride = width * channels

            # Create and return GdkPixbuf
            return GdkPixbuf.Pixbuf.new_from_data(
                data,
                GdkPixbuf.Colorspace.RGB,
                channels == 4,  # Has alpha if RGBA
                8,  # Bits per sample
                width,
                height,
                rowstride,
            )

        hbox_images = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

        for i, image in enumerate(self.images):
            pixbuf = numpy_to_pixbuf(image)
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

        # Check if the last item is a VBox containing the save button
        if self.generated_flag:
            for i in range(3):
                self.vbox.remove(self.vbox.get_last_child())
        self.generated_flag = True

        # Add components to the VBox
        self.vbox.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))
        self.vbox.append(hbox_images)

        save_button = Gtk.Button(label="Save Labels and Preferences")
        save_button.connect("clicked", self.on_save_button_clicked)
        self.vbox.append(save_button)

    def on_save_button_clicked(self, button):
        data_pkl_path = self.spf.data_pkl_path
        save_path = os.path.join(os.path.dirname(data_pkl_path), "clusters")

        if not os.path.exists(save_path):
            # Make directory if it doesn't exist
            os.makedirs(save_path)
        else:
            # Remove all .jpg and .yaml files in clusters directory
            for file in os.listdir(save_path):
                file_path = os.path.join(save_path, file)
                if os.path.isfile(file_path) and (file_path.endswith(".jpg") or file_path.endswith(".yaml")):
                    os.unlink(file_path)

        labels_and_rankings = []
        vbox = get_children(self.vbox)
        hbox = get_children(vbox[-2])
        for vbox_image in hbox:
            entries = get_children(vbox_image)
            if len(entries) == 5:
                label_entry = entries[2]
                label = label_entry.get_text()

                ranking_entry = entries[4]
                ranking = ranking_entry.get_text()

                labels_and_rankings.append((label, ranking))

        for i, image in enumerate(self.images):
            label, _ = labels_and_rankings[i]
            image_path = os.path.join(save_path, f"{label}.jpg")
            from PIL import Image
            import numpy as np

            # Convert NumPy array to PIL image
            pil_image = Image.fromarray(np.uint8(image))

            # Save the PIL image as a JPG file
            pil_image.save(image_path, "JPEG")

        import yaml
        # Save labels and rankings to a YAML file
        labels = [label for label, _ in labels_and_rankings]
        preferences = [int(ranking) for _, ranking in labels_and_rankings]
        data = {"labels": labels, "preferences": preferences}
        with open(os.path.join(save_path, "terrain_preferences.yaml"), "w") as file:
            yaml.dump(data, file)
            
        print("Labels:", labels)
        print("Preferences:", preferences)

        success_dialog = Gtk.MessageDialog(
            transient_for=self.parent_window,
            modal=True,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Labels and preferences have been saved successfully.",
        )
        success_dialog.show()
        success_dialog.connect("response", lambda dialog, response: dialog.destroy())


def get_children(box):
    children = []
    child = box.get_first_child()
    while child:
        children.append(child)
        # print(type(child))  # Print the type of each child widget
        child = child.get_next_sibling()
    return children


app = ClusterUI()
exit_status = app.run(sys.argv)
sys.exit(exit_status)
