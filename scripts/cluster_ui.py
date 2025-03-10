import sys
import os
import yaml
import numpy as np
import pickle
from PIL import Image
import torch
from torch.utils.data import DataLoader
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

        spf = SelectVicregFile(window)
        vbox.append(spf.get_component())

        vbox.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        ssf = SelectSyncedFile(window)
        vbox.append(ssf.get_component())

        vbox.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        smf = SelectModelFile(window)
        vbox.append(smf.get_component())

        vbox.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        gc = GenerateClusters(window, spf, ssf, smf)
        vbox.append(gc.get_component())

        # Show the window
        window.present()


class SelectVicregFile:
    def __init__(self, parent_window):
        self.parent_window = parent_window
        self.data_pkl_path = None

    def get_component(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)

        vbox.append(Gtk.Label(label="Vicreg pickle file selected:"))

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

class SelectSyncedFile:
    def __init__(self, parent_window):
        self.parent_window = parent_window
        self.synced_pkl_path = None

    def get_component(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)

        vbox.append(Gtk.Label(label="Synced pickle file selected:"))

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

            self.synced_pkl_path= file_path
            self.label.set_markup(f"<span foreground='green'>{self.synced_pkl_path}</span>")
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
    def __init__(self, parent_window, spf, ssf, smf):
        self.parent_window = parent_window
        self.spf = spf
        self.ssf = ssf
        self.smf = smf

        self.generated_flag = False
        self.cluster = None
        self.all_cluster_image_indices = None
        self.cluster_labels = None
        self.dataset = None

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
        synced_pkl_path = self.ssf.synced_pkl_path
        model_path = self.smf.model_path

        if data_pkl_path is None or model_path is None or synced_pkl_path is None:
            error_dialog = Gtk.MessageDialog(
                transient_for=self.parent_window,
                modal=True,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text="Please select both the pickle file and a model file before generating clusters.",
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
            os.makedirs(save_path)

        # Generate clusters
        self.cluster = Cluster(data_pkl_path, synced_pkl_path, model_path)
        self.all_cluster_image_indices = self.cluster.generate_clusters(
            num_clusters,
            num_iterations,
            save_model_path=os.path.join(save_path, "kmeans_model.pkl"),
        )

        # Store the dataset and cluster labels for later use
        self.dataset = self.cluster.dataset
        self.cluster_labels = np.zeros(len(self.dataset), dtype=int)
        for cluster_idx, indices in enumerate(self.all_cluster_image_indices):
            for idx in indices:
                self.cluster_labels[idx] = cluster_idx

        # Dynamic grid size
        min_length = min(len(lst) for lst in self.all_cluster_image_indices)
        grid_size = 10
        while min_length < grid_size**2:
            grid_size -= 1

        # Render clusters
        rendered_clusters = PatchRenderer.render_clusters(self.all_cluster_image_indices, self.cluster.patches)

        self.images = []
        for i, cluster in enumerate(rendered_clusters):
            self.images.append(PatchRenderer.image_grid(cluster))

        def numpy_to_pixbuf(array):
            """
            Convert a NumPy array to a GdkPixbuf.Pixbuf object.
            Supports RGB and RGBA formats.
            """
            if array.ndim not in (3,):
                raise ValueError("Input must be a 3D NumPy array with shape (height, width, channels)")

            height, width, channels = array.shape
            if channels not in (3, 4):
                raise ValueError("Array must have 3 (RGB) or 4 (RGBA) channels")

            data = array.tobytes()
            rowstride = width * channels
            return GdkPixbuf.Pixbuf.new_from_data(
                data,
                GdkPixbuf.Colorspace.RGB,
                channels == 4,
                8,
                width,
                height,
                rowstride,
            )

        hbox_images = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

        self.labels_and_rankings = []
        for i, image in enumerate(self.images):
            pixbuf = numpy_to_pixbuf(image)
            image_widget = Gtk.Image.new_from_pixbuf(pixbuf)
            image_widget.set_size_request(400, 400)

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
            self.labels_and_rankings.append((text_field, ranking_field))

        if self.generated_flag:
            for i in range(3):
                self.vbox.remove(self.vbox.get_last_child())
        self.generated_flag = True

        self.vbox.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))
        self.vbox.append(hbox_images)

        save_button = Gtk.Button(label="Save Labels and Preferences")
        save_button.connect("clicked", self.on_save_button_clicked)
        self.vbox.append(save_button)

    def on_save_button_clicked(self, button):
        data_pkl_path = self.spf.data_pkl_path
        save_path = os.path.join(os.path.dirname(data_pkl_path), "clusters")
        config_path = os.path.join(save_path, "config.yaml")

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            for file in os.listdir(save_path):
                file_path = os.path.join(save_path, file)
                if os.path.isfile(file_path) and file_path.endswith(".jpg"):
                    os.unlink(file_path)  # Remove only .jpg files, preserve config.yaml

        # Collect user-provided labels and preferences
        user_labels_and_rankings = []
        for text_field, ranking_field in self.labels_and_rankings:
            label = text_field.get_text()
            ranking = ranking_field.get_text()
            try:
                ranking = float(ranking)
            except ValueError:
                error_dialog = Gtk.MessageDialog(
                    transient_for=self.parent_window,
                    modal=True,
                    message_type=Gtk.MessageType.ERROR,
                    buttons=Gtk.ButtonsType.OK,
                    text=f"Invalid preference value: {ranking}. Please enter a valid number.",
                )
                error_dialog.show()
                error_dialog.connect("response", lambda dialog, response: dialog.destroy())
                return
            user_labels_and_rankings.append((label, ranking))

        # Save cluster images
        for i, image in enumerate(self.images):
            label, _ = user_labels_and_rankings[i]
            image_path = os.path.join(save_path, f"{label}.jpg")
            pil_image = Image.fromarray(np.uint8(image))
            pil_image.save(image_path, "JPEG")

        # Read existing config.yaml
        existing_config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                existing_config = yaml.safe_load(file) or {}
        else:
            existing_config = {}

        # Create new terrains section with cluster-level data only
        new_terrains = [
            {'name': label, 'label': idx, 'preference': preference}
            for idx, (label, preference) in enumerate(user_labels_and_rankings)
        ]

        # Map cluster labels to terrain labels and preferences for internal use
        cluster_to_terrain = {
            cluster_idx: {'terrain_label': terrain_label, 'preference': preference}
            for cluster_idx, (terrain_label, preference) in enumerate(user_labels_and_rankings)
        }

        # Assign terrain labels to each sample based on its cluster
        terrain_labels = np.zeros(len(self.dataset), dtype=object)
        preferences = np.zeros(len(self.dataset), dtype=float)
        for idx in range(len(self.dataset)):
            cluster_label = self.cluster_labels[idx]
            terrain_info = cluster_to_terrain.get(cluster_label, {'terrain_label': 'unknown', 'preference': 0.0})
            terrain_labels[idx] = terrain_info['terrain_label']
            preferences[idx] = terrain_info['preference']

        # Update the dataset with cluster and terrain labels
        self.dataset.cluster_labels = self.cluster_labels
        self.dataset.terrain_labels = terrain_labels
        self.dataset.preferences = preferences

        # Create a mapping of sample indices to their data and labels for the pickle file
        labeled_data = []
        for idx in range(len(self.dataset)):
            sample = {
                'index': idx,
                'patch': self.cluster.patches[idx],  # Visual data
                'inertial': self.cluster.inertial[idx],  # Inertial data
                'cluster_label': int(self.cluster_labels[idx]),
                'terrain_label': terrain_labels[idx],
                'preference': float(preferences[idx])
            }
            labeled_data.append(sample)

        # Save the dataset to a pickle file
        dataset_save_path = os.path.join(save_path, "labeled_dataset.pkl")
        with open(dataset_save_path, "wb") as file:
            pickle.dump(self.dataset, file)
        print(f"Saved labeled dataset to: {dataset_save_path}")

        # Update the existing config: replace terrains only (samples optional)
        existing_config['terrains'] = new_terrains
        # Optional: Include samples section without preferences
        new_samples = [
            {'sample_id': idx, 'terrain': terrain_labels[idx]}
            for idx in range(len(self.dataset))
        ]
        #existing_config['samples'] = new_samples  # Comment out if samples not needed in YAML

        # Save updated config.yaml
        with open(config_path, "w") as file:
            yaml.dump(existing_config, file)

        #print("Labels:", terrain_labels.tolist())
        #print("Preferences:", preferences.tolist())
        print(f"Updated config at: {config_path}")

        success_dialog = Gtk.MessageDialog(
            transient_for=self.parent_window,
            modal=True,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Existing terrains replaced, samples updated in config.yaml, and dataset saved to labeled_dataset.pkl.",
        )
        success_dialog.show()
        success_dialog.connect("response", lambda dialog, response: dialog.destroy())

def get_children(box):
    children = []
    child = box.get_first_child()
    while child:
        children.append(child)
        child = child.get_next_sibling()
    return children

app = ClusterUI()
exit_status = app.run(sys.argv)
sys.exit(exit_status)
